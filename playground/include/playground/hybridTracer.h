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

#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <string>

#include <3dgrt/pipelineDefinitions.h>
#include <3dgrt/optixTracer.h>
#include <playground/pipelineDefinitions.h>
#include <playground/pipelineParameters.h>

// intermediate struct to pass pbr material data from python
// when tracing this struct gets converted to PBRMaterial, see playground_params.h
struct CPBRMaterial {
    unsigned int material_id;
    torch::Tensor diffuseMap;
    torch::Tensor emissiveMap;
    torch::Tensor metallicRoughnessMap;
    torch::Tensor normalMap;
    torch::Tensor diffuseFactor;
    torch::Tensor emissiveFactor;
    float metallicFactor;
    float roughnessFactor;
    unsigned int alphaMode;
    float alphaCutoff;
    float transmissionFactor;
    float ior;
};

struct OptixDenoiserWrapper
{
    OptixDenoiser _denoiser = nullptr;
    CUdeviceptr _denoiserScratchPtr = 0;
    size_t _denoiserScratchSz = 0;
    CUdeviceptr _denoiserStatePtr = 0;
    size_t _denoiserStateSz = 0;
    unsigned int _width = 0;
    unsigned int _height = 0;

    void setup(unsigned int width, unsigned int height, OptixDeviceContext optixContext, cudaStream_t stream);
    void release(cudaStream_t stream = 0);
};

class HybridOptixTracer: public OptixTracer {

protected:
    struct PlaygroundState
    {
        OptixDeviceContext context;
        OptixTraversableHandle gasHandle; /// Handle to the acceleration structure after it is built
        CUdeviceptr gasBuffer;            /// Scratch buffer for building acceleration structure
        size_t gasBufferSz;
        CUdeviceptr gasBufferTmp;         /// Holds the acceleration structure mem after it is built
        size_t gasBufferTmpSz;

        CUdeviceptr paramsDevice;
        size_t paramsDeviceSz;

        // The following are used in the construction of the BVH and tracing
        uint32_t vNum;            /// Number of mesh vertices
        uint32_t fNum;            /// Number of indexed mesh faces
        CUdeviceptr fPrimVrt;     /// Buffer containing mesh vertices
        size_t fPrimVrtSz;
        CUdeviceptr fPrimTri;     /// Buffer containing indexed triangles
        size_t fPrimTriSz;

        OptixPipeline pipelineTriGSTracing;
        OptixShaderBindingTable sbtTriGSTracing;
        OptixModule moduleTriGSTracing;

        // Denoiser
        OptixDenoiserWrapper denoiser;
    }* _playgroundState;

    void reallocateMeshPrimGeomBuffer(cudaStream_t stream);
    void reallocatePlaygroundParamsDevice(size_t sz, cudaStream_t stream);

public:
    HybridOptixTracer(
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
        bool enableHitCounts);

    ~HybridOptixTracer() override;

    void buildMeshBVH(torch::Tensor meshVerts,
                            torch::Tensor meshFaces,
                            unsigned int rebuild,
                            bool allow_update);

    void setMaterialTextures(PlaygroundPipelineParameters& params,
                             unsigned int textureId,
                             CudaTexture2DFloat4Object& diffuseTexObject,
                             CudaTexture2DFloat4Object& emissiveTexObject,
                             CudaTexture2DFloat2Object& metallicRoughnessTexObject,
                             CudaTexture2DFloat4Object& normalTexObject,
                             const CPBRMaterial& cmat);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    traceHybrid(uint32_t frameNumber,
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
                const unsigned int maxPBRBounces);

    torch::Tensor denoise(torch::Tensor rayRad);
};
