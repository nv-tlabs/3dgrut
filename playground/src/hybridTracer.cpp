// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgrt/tensorBuffering.h>
#include <3dgrt/cuoptixMacros.h>
#include <3dgrt/optixTracer.h>
#include <3dgrt/particlePrimitives.h>
#include <3dgrt/pipelineParameters.h>
#include <playground/hybridTracer.h>
#include <playground/meshBuffers.h>
#include <playground/pipelineParameters.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <algorithm>
#include <fstream>
#include <optix.h>


//------------------------------------------------------------------------------
// HybridOptixTracer
//------------------------------------------------------------------------------

HybridOptixTracer::HybridOptixTracer(
    const std::string& threedgrtPath,
    const std::string& playgroundPath,
    const std::string& cuda_path,
    const std::string& pipeline,
    const std::string& backwardPipeline,
    const std::string& primitive,
    float particleKernelDegree,
    float particleKernelMinResponse,
    bool particleKernelDensityClamping,
    int particleRadianceSphDegree,
    bool enableNormals,
    bool enableHitCounts) : OptixTracer(threedgrtPath, cuda_path, pipeline, backwardPipeline, primitive,
                                        particleKernelDegree, particleKernelMinResponse, particleKernelDensityClamping,
                                        particleRadianceSphDegree, enableNormals, enableHitCounts){

    char log[2048]; // For error reporting from OptiX creation functions
    std::vector<std::string> defines = generateDefines(particleKernelDegree, particleKernelDensityClamping,
                                                        particleRadianceSphDegree, enableNormals, enableHitCounts);
    const uint32_t sharedFlags =
        (_state->gPrimType == MOGTracingSphere ?
                              PipelineFlag_SpherePrim : ((_state->gPrimType == MOGTracingCustom) || (_state->gPrimType == MOGTracingInstances) ? PipelineFlag_HasIS : 0));

    _playgroundState = new PlaygroundState();
    memset(_playgroundState, 0, sizeof(PlaygroundState));
    // Use same context as gaussians program
    _playgroundState->context = _state->context;
    _playgroundState->fNum = 0;
    _playgroundState->vNum = 0;

    _playgroundState->moduleTriGSTracing = nullptr;
    _playgroundState->pipelineTriGSTracing = nullptr;
    _playgroundState->sbtTriGSTracing = {};
    createPipeline(
        _playgroundState->context, playgroundPath, threedgrtPath, cuda_path, defines, "playgroundKernel",
        sharedFlags | PipelineFlag_HasRG | PipelineFlag_HasCH | PipelineFlag_HasAH | PipelineFlag_HasMS,
        &_playgroundState->moduleTriGSTracing,
        &_playgroundState->pipelineTriGSTracing,
        _playgroundState->sbtTriGSTracing,
        32,
        {threedgrtPath + "/include", playgroundPath + "/src/kernels/cuda/"} // extra includes
    );
}

HybridOptixTracer::~HybridOptixTracer(void){

    if (_playgroundState)
        _playgroundState->denoiser.release(); // Release if allocated

    OPTIX_CHECK(optixPipelineDestroy(_playgroundState->pipelineTriGSTracing));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->sbtTriGSTracing.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->sbtTriGSTracing.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->sbtTriGSTracing.hitgroupRecordBase)));
    OPTIX_CHECK(optixModuleDestroy(_playgroundState->moduleTriGSTracing));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->fPrimVrt)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->fPrimTri)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->gasBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->gasBufferTmp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_playgroundState->paramsDevice)));

    delete _playgroundState;
}

void HybridOptixTracer::reallocateMeshPrimGeomBuffer(cudaStream_t cudaStream) {
    if (_playgroundState) {
        reallocateBuffer(&_playgroundState->fPrimVrt, _playgroundState->fPrimVrtSz, sizeof(float3) * 3 * _playgroundState->fNum, cudaStream);
        reallocateBuffer(&_playgroundState->fPrimTri, _playgroundState->fPrimTriSz, sizeof(int3) * _playgroundState->fNum, cudaStream);
    }
}

void HybridOptixTracer::reallocatePlaygroundParamsDevice(size_t sz, cudaStream_t cudaStream) {
    if (_playgroundState) {
        reallocateBuffer(&_playgroundState->paramsDevice, _playgroundState->paramsDeviceSz, sz, cudaStream);
    }
}

void HybridOptixTracer::buildMeshBVH(
    torch::Tensor meshVerts,
    torch::Tensor meshFaces,
    unsigned int rebuild,
    bool allow_update) {

    const uint32_t vNum = meshVerts.size(0);
    const uint32_t fNum = meshFaces.size(0);
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    if (!rebuild && (_playgroundState->fNum != fNum)) {
        std::cerr << "ERROR:: cannot refit GAS with a different number of mesh faces" << std::endl;
        rebuild = 1;
    }
    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    _playgroundState->vNum = vNum;
    _playgroundState->fNum = fNum;
    reallocateMeshPrimGeomBuffer(cudaStream);

    computeMeshFaceBuffer(fNum,
                          getPtr<float3>(meshVerts),
                          getPtr<int3>(meshFaces),
                          reinterpret_cast<float3*>(_playgroundState->fPrimVrt),
                          reinterpret_cast<int3*>(_playgroundState->fPrimTri),
                          cudaStream);
    CUDA_CHECK_LAST();

    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        if (allow_update) {
            accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        }
        accel_options.operation = rebuild ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

        // Our build input is a simple list of non-indexed triangle vertices
        OptixBuildInput prim_input = {};
        uint32_t prim_input_flags                 = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        prim_input.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        prim_input.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
        prim_input.triangleArray.numVertices      = vNum;
        prim_input.triangleArray.vertexBuffers    = &_playgroundState->fPrimVrt;
        prim_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        prim_input.triangleArray.numIndexTriplets = fNum;
        prim_input.triangleArray.indexBuffer      = _playgroundState->fPrimTri;
        prim_input.triangleArray.flags            = &prim_input_flags;
        prim_input.triangleArray.numSbtRecords    = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(_playgroundState->context, &accel_options, &prim_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));
        if (_playgroundState->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_playgroundState->gasBufferTmp), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_playgroundState->gasBufferTmp), gas_buffer_sizes.tempSizeInBytes, cudaStream));
            _playgroundState->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
        }

        if (rebuild && (_playgroundState->gasBufferSz < gas_buffer_sizes.outputSizeInBytes)) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_playgroundState->gasBuffer), cudaStream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_playgroundState->gasBuffer),
                                       gas_buffer_sizes.outputSizeInBytes, cudaStream));
            _playgroundState->gasBufferSz = gas_buffer_sizes.outputSizeInBytes;
        }

        OPTIX_CHECK(optixAccelBuild(_playgroundState->context,
                                    cudaStream, // CUDA stream
                                    &accel_options, &prim_input,
                                    1, // num build inputs
                                    _playgroundState->gasBufferTmp, gas_buffer_sizes.tempSizeInBytes, _playgroundState->gasBuffer,
                                    gas_buffer_sizes.outputSizeInBytes, &_playgroundState->gasHandle,
                                    nullptr, // emitted property list
                                    0        // num emitted properties
                                    ));
    }

    CUDA_CHECK_LAST();
}

void HybridOptixTracer::setMaterialTextures(
    PlaygroundPipelineParameters& params,
    unsigned int textureId,
    CudaTexture2DFloat4Object& diffuseTexObject,
    CudaTexture2DFloat4Object& emissiveTexObject,
    CudaTexture2DFloat2Object& metallicRoughnessTexObject,
    CudaTexture2DFloat4Object& normalTexObject,
    const CPBRMaterial& cmat)
{
    PBRMaterial* paramsMat = nullptr;
    switch (textureId)
    {
        case 0: paramsMat = &params.mat0; break;
        case 1: paramsMat = &params.mat1; break;
        case 2: paramsMat = &params.mat2; break;
        case 3: paramsMat = &params.mat3; break;
        case 4: paramsMat = &params.mat4; break;
        case 5: paramsMat = &params.mat5; break;
        case 6: paramsMat = &params.mat6; break;
        case 7: paramsMat = &params.mat7; break;
        case 8: paramsMat = &params.mat8; break;
        case 9: paramsMat = &params.mat9; break;
        case 10: paramsMat = &params.mat10; break;
        case 11: paramsMat = &params.mat11; break;
        case 12: paramsMat = &params.mat12; break;
        case 13: paramsMat = &params.mat13; break;
        case 14: paramsMat = &params.mat14; break;
        case 15: paramsMat = &params.mat15; break;
        case 16: paramsMat = &params.mat16; break;
        case 17: paramsMat = &params.mat17; break;
        case 18: paramsMat = &params.mat18; break;
        case 19: paramsMat = &params.mat19; break;
        case 20: paramsMat = &params.mat20; break;
        case 21: paramsMat = &params.mat21; break;
        case 22: paramsMat = &params.mat22; break;
        case 23: paramsMat = &params.mat23; break;
        case 24: paramsMat = &params.mat24; break;
        case 25: paramsMat = &params.mat25; break;
        case 26: paramsMat = &params.mat26; break;
        case 27: paramsMat = &params.mat27; break;
        case 28: paramsMat = &params.mat28; break;
        case 29: paramsMat = &params.mat29; break;
        case 30: paramsMat = &params.mat30; break;
        case 31: paramsMat = &params.mat31; break;
        default: break;
    }

    if (paramsMat)
    {
        paramsMat->useDiffuseTexture = diffuseTexObject.isTexInitialized();
        paramsMat->diffuseTexture = diffuseTexObject.tex();
        paramsMat->useEmissiveTexture = emissiveTexObject.isTexInitialized();
        paramsMat->emissiveTexture = emissiveTexObject.tex();
        paramsMat->useMetallicRoughnessTexture = metallicRoughnessTexObject.isTexInitialized();
        paramsMat->metallicRoughnessTexture = metallicRoughnessTexObject.tex();
        paramsMat->useNormalTexture = normalTexObject.isTexInitialized();
        paramsMat->normalTexture = normalTexObject.tex();
        float* diffuseFactor = cmat.diffuseFactor.data_ptr<float>();
        paramsMat->diffuseFactor = make_float4(diffuseFactor[0], diffuseFactor[1], diffuseFactor[2], diffuseFactor[3]);
        float* emissiveFactor = cmat.emissiveFactor.data_ptr<float>();
        paramsMat->emissiveFactor = make_float3(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]);
        paramsMat->metallicFactor = cmat.metallicFactor;
        paramsMat->roughnessFactor = cmat.roughnessFactor;
        paramsMat->alphaMode = cmat.alphaMode;
        paramsMat->alphaCutoff = cmat.alphaCutoff;
        paramsMat->transmissionFactor = cmat.transmissionFactor;
        paramsMat->ior = cmat.ior;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> HybridOptixTracer::traceHybrid(
    uint32_t frameNumber,
    torch::Tensor rayToWorld,
    torch::Tensor rayOri,
    torch::Tensor rayDir,
    torch::Tensor particleDensity,
    torch::Tensor particleRadiance,
    int sphDegree,
    float minTransmittance,
    torch::Tensor rayMaxT,
    uint32_t playgroundOpts,
    torch::Tensor triangles,
    torch::Tensor vNormals,
    torch::Tensor vTangents,
    torch::Tensor vHasTangents,
    torch::Tensor primType,
    torch::Tensor matUV,
    torch::Tensor matID,
    const std::vector<CPBRMaterial>& materials,
    torch::Tensor refractiveIndex,
    torch::Tensor backgroundColor,
    torch::Tensor envmap,
    bool enableEnvmap,
    bool useEnvmapAsBackground,
    const unsigned int maxPBRBounces) {

    // ----- 3dgrt launch params -----
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const torch::TensorOptions int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor rayRad = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayDns = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayHit = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 2}, opts);
    torch::Tensor rayNrm = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayHitsCount = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor traceState = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, int_opts);

    PlaygroundPipelineParameters paramsHost;
    paramsHost.handle = _state->gasHandle;
    paramsHost.aabb = _state->gasAABB;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber = frameNumber;
    paramsHost.gPrimNumTri = _state->gPrimNumTri;

    paramsHost.minTransmittance = minTransmittance;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold = 1.0f / 255.0f;
    paramsHost.sphDegree = sphDegree;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));
    paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    paramsHost.particleDensity      = getPtr<const ParticleDensity>(particleDensity);
    paramsHost.particleRadiance     = getPtr<const float>(particleRadiance);
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);

    paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);
    paramsHost.rayHitsCount   = packed_accessor32<float, 4>(rayHitsCount);

    // ----- Playground launch params -----
    paramsHost.rayMaxT = packed_accessor32<float, 3>(rayMaxT);
    paramsHost.triangles = packed_accessor32<int32_t, 2>(triangles);
    paramsHost.vNormals = packed_accessor32<float, 2>(vNormals);
    paramsHost.vTangents = packed_accessor32<float, 2>(vTangents);
    paramsHost.vHasTangents = packed_accessor32<bool, 2>(vHasTangents);
    paramsHost.primType = packed_accessor32<int32_t, 2>(primType);
    paramsHost.matUV = packed_accessor32<float, 2>(matUV);
    paramsHost.matID =  packed_accessor32<int32_t, 2>(matID);
    paramsHost.refractiveIndex = packed_accessor32<float, 2>(refractiveIndex);
    paramsHost.maxPBRBounces = maxPBRBounces;
    paramsHost.playgroundOpts = playgroundOpts;
    paramsHost.triHandle = _playgroundState->gasHandle;
    paramsHost.trace_state = packed_accessor32<int32_t, 4>(traceState);
    paramsHost.backgroundColor = make_float3(
        backgroundColor[0].item<float>(), backgroundColor[1].item<float>(), backgroundColor[2].item<float>()
    );
    paramsHost.useEnvmap = false;
    paramsHost.useEnvmapAsBackground = useEnvmapAsBackground;
    int envmapHeight = envmap.size(0);
    int envmapWidth = envmap.size(1);
    CudaTexture2DFloat4Object cuEnvMap = CudaTexture2DFloat4Object();
    if (envmapHeight > 0 && envmapWidth > 0)
    {
        cuEnvMap.reset(envmap.data_ptr<float>(), envmapHeight, envmapWidth);
        paramsHost.useEnvmap = enableEnvmap;
        paramsHost.envmap = cuEnvMap.tex();
    }

    // Material Textures
    size_t numMaterials = materials.size();
    std::vector<CudaTexture2DFloat4Object> cuDiffuseTexture(numMaterials);
    std::vector<CudaTexture2DFloat4Object> cuEmissiveTexture(numMaterials);
    std::vector<CudaTexture2DFloat2Object> cuMetallicRoughnessTexture(numMaterials);
    std::vector<CudaTexture2DFloat4Object> cuNormalTexture(numMaterials);

    for (int i = 0; i < numMaterials; ++i) {
        torch::Tensor texDiffuse = materials[i].diffuseMap;
        int diffuseTexHeight = texDiffuse.size(0);
        int diffuseTexWidth = texDiffuse.size(1);
        cuDiffuseTexture[i] = CudaTexture2DFloat4Object();
        if (diffuseTexHeight > 0 && diffuseTexWidth > 0)
        {
            cuDiffuseTexture[i].reset(texDiffuse.data_ptr<float>(), diffuseTexHeight, diffuseTexWidth);
        }

        torch::Tensor texEmissive = materials[i].emissiveMap;
        int emissiveTexHeight = texEmissive.size(0);
        int emissiveTexWidth = texEmissive.size(1);
        cuEmissiveTexture[i] = CudaTexture2DFloat4Object();
        if (emissiveTexHeight > 0 && emissiveTexWidth > 0)
        {
            cuEmissiveTexture[i].reset(texEmissive.data_ptr<float>(), emissiveTexHeight, emissiveTexWidth);
        }

        torch::Tensor texMetallicRoughness = materials[i].metallicRoughnessMap;
        int metallicRoughnessTexHeight = texMetallicRoughness.size(0);
        int metallicRoughnessTexWidth = texMetallicRoughness.size(1);
        cuMetallicRoughnessTexture[i] = CudaTexture2DFloat2Object();
        if (metallicRoughnessTexHeight > 0 && metallicRoughnessTexWidth > 0)
        {
            cuMetallicRoughnessTexture[i].reset(texMetallicRoughness.data_ptr<float>(),
                                                metallicRoughnessTexHeight, metallicRoughnessTexWidth);
        }

        torch::Tensor texNormal = materials[i].normalMap;
        int normalTexHeight = texNormal.size(0);
        int normalTexWidth = texNormal.size(1);
        cuNormalTexture[i] = CudaTexture2DFloat4Object();
        if (normalTexHeight > 0 && normalTexWidth > 0)
        {
            cuNormalTexture[i].reset(texNormal.data_ptr<float>(), normalTexHeight, normalTexWidth);
        }

        setMaterialTextures(paramsHost, i,
                            cuDiffuseTexture[i], cuEmissiveTexture[i],
                            cuMetallicRoughnessTexture[i], cuNormalTexture[i],
                            materials[i]);
    }

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    reallocatePlaygroundParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(_playgroundState->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream)
    );

    OPTIX_CHECK(optixLaunch(_playgroundState->pipelineTriGSTracing,
                        cudaStream, _playgroundState->paramsDevice,
                        sizeof(PlaygroundPipelineParameters), &_playgroundState->sbtTriGSTracing,
                        rayRad.size(2),
                        rayRad.size(1),
                        rayRad.size(0)
    ));

    CUDA_CHECK_LAST();

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(rayRad, rayDns, rayHit, rayNrm, rayHitsCount);
}

torch::Tensor HybridOptixTracer::denoise(torch::Tensor rayRad) {

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto& denoiser = _playgroundState->denoiser;
    denoiser.setup(rayRad.size(2), rayRad.size(1), _playgroundState->context, cudaStream);

    // Output buffer
    torch::Tensor rayRadDenoised = torch::empty({rayRad.size(0), rayRad.size(1), rayRad.size(2), 3}, opts);

    OptixDenoiserParams denoiserParams = {};
#if OPTIX_VERSION > 70700
    denoiserParams.temporalModeUsePreviousLayers = 0;
#elif OPTIX_VERSION > 70500
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    denoiserParams.blendFactor = 0.0f;

    // -------------------------------------------------------
    OptixImage2D inputLayer = {};
    inputLayer.data = reinterpret_cast<CUdeviceptr>(getPtr<float3>(rayRad));
    /// Width of the image (in pixels)
    inputLayer.width = rayRad.size(2);
    /// Height of the image (in pixels)
    inputLayer.height = rayRad.size(1);
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = rayRad.size(2) * sizeof(float3);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float3);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = reinterpret_cast<CUdeviceptr>(getPtr<float3>(rayRadDenoised));
    /// Width of the image (in pixels)
    outputLayer.width = rayRad.size(2);
    /// Height of the image (in pixels)
    outputLayer.height = rayRad.size(1);
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = rayRad.size(2) * sizeof(float3);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float3);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    // -------------------------------------------------------
    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(denoiser._denoiser,
                                    cudaStream,
                                    &denoiserParams,
                                    denoiser._denoiserStatePtr,
                                    denoiser._denoiserStateSz,
                                    &denoiserGuideLayer,
                                    &denoiserLayer, 1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    denoiser._denoiserScratchPtr,
                                    denoiser._denoiserScratchSz));

    return rayRadDenoised;
}

// Denoiser

void OptixDenoiserWrapper::setup(unsigned int width, unsigned int height, OptixDeviceContext optixContext, cudaStream_t stream)
{
    if ((width == _width) && (height == _height))
    {
        return;
    }
    release(stream);
    _width = width;
    _height = height;
    OptixDenoiserOptions denoiserOptions = {};
    OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &_denoiser));
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(_denoiser, _width, _height, &denoiserReturnSizes));
    _denoiserScratchSz = std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&_denoiserScratchPtr), _denoiserScratchSz, stream));
    _denoiserStateSz = denoiserReturnSizes.stateSizeInBytes;
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&_denoiserStatePtr), _denoiserStateSz, stream));
    OPTIX_CHECK(optixDenoiserSetup(_denoiser, 0,
                                   _width, _height,
                                   _denoiserStatePtr,
                                   _denoiserStateSz,
                                   _denoiserScratchPtr,
                                   _denoiserScratchSz));
}

void OptixDenoiserWrapper::release(cudaStream_t stream)
{
    if (_denoiser)
    {
        OPTIX_CHECK(optixDenoiserDestroy(_denoiser));
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(_denoiserScratchPtr), stream));
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(_denoiserStatePtr), stream));
    }
    _denoiser = nullptr;
    _width = _height = _denoiserScratchSz = _denoiserStateSz = _denoiserScratchPtr = _denoiserStatePtr = 0;
}