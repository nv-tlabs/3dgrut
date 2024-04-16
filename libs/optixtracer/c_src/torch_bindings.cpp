// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL)   \
    {                                      \
        cudaError_t err = CUDA_CALL;       \
        AT_CUDA_CHECK(cudaGetLastError()); \
    }
#define NVDR_CHECK_GL_ERROR(GL_CALL)                                                                   \
    {                                                                                                  \
        GL_CALL;                                                                                       \
        GLenum err = glGetError();                                                                     \
        TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); \
    }
#define CHECK_TENSOR(X, DIMS, CHANNELS)                                                                              \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor")                                                            \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions")                                               \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "mogTracing/params.h"
#include "mogTracingInd/params.h"
#include "optix_utils.h"
#include "optix_wrapper.h"

#include <optix_stubs.h>

namespace
{
    inline uint32_t div_round_up(uint32_t val, uint32_t divisor)
    {
        return (val + divisor - 1) / divisor;
    }

    inline float slabSpacingFromAABB(const OptixAabb &aabb, const uint32_t maxNumSlabs)
    {
        const float aabbDiag_x = aabb.maxX - aabb.minX;
        const float aabbDiag_y = aabb.maxY - aabb.minY;
        const float aabbDiag_z = aabb.maxZ - aabb.minZ;
        return maxNumSlabs > 0 ? sqrt(aabbDiag_x * aabbDiag_x + aabbDiag_y * aabbDiag_y + aabbDiag_z * aabbDiag_z) / maxNumSlabs : 1e9;
    }

    inline float minGaussianResponse(const float s)
    {
        return exp(-0.5f * s * s);
    }

    class TraceFile
    {
    public:
        FILE *f = nullptr;
        const int rate = 0;
        uint32_t num = 0;

        TraceFile(const char *file, int srate = -1) : rate(srate)
        {
            if (rate >= 0)
            {
                f = fopen(file, "wt");
            }
        }

        ~TraceFile()
        {
            if (f)
            {
                fclose(f);
            }
        }

        inline void sample() { num++; }
        inline bool trace() { return (rate >= 0) && (!(num % rate)); }
    };
    std::unique_ptr<TraceFile> tFilePtr;

} // namespace name

void computeGaussianEnclosingIcosaHedron(uint32_t gNum,
                                         torch::Tensor gPos,
                                         torch::Tensor gRot,
                                         torch::Tensor gScl,
                                         float sigmaSclTh,
                                         float3 *gPrimVrt,
                                         int3 *gPrimTri,
                                         OptixAabb *gPrimAABB,
                                         cudaStream_t stream);

void computeGaussianEnclosingOctaHedron(uint32_t gNum,
                                        torch::Tensor gPos,
                                        torch::Tensor gRot,
                                        torch::Tensor gScl,
                                        float sigmaSclTh,
                                        float3 *gPrimVrt,
                                        int3 *gPrimTri,
                                        OptixAabb *gPrimAABB,
                                        cudaStream_t stream);

void computeGaussianEnclosingTriHexa(uint32_t gNum,
                                     torch::Tensor gPos,
                                     torch::Tensor gRot,
                                     torch::Tensor gScl,
                                     float sigmaSclTh,
                                     float3 *gPrimVrt,
                                     int3 *gPrimTri,
                                     OptixAabb *gPrimAABB,
                                     cudaStream_t stream);

void computeGaussianEnclosingTriSurfel(uint32_t gNum,
                                       torch::Tensor gPos,
                                       torch::Tensor gRot,
                                       torch::Tensor gScl,
                                       float sigmaSclTh,
                                       float3 *gPrimVrt,
                                       int3 *gPrimTri,
                                       OptixAabb *gPrimAABB,
                                       cudaStream_t stream);

void computeGaussianEnclosingTetraHedron(uint32_t gNum,
                                         torch::Tensor gPos,
                                         torch::Tensor gRot,
                                         torch::Tensor gScl,
                                         float sigmaSclTh,
                                         float3 *gPrimVrt,
                                         int3 *gPrimTri,
                                         OptixAabb *gPrimAABB,
                                         cudaStream_t stream);

void computeGaussianEnclosingDiamond(uint32_t gNum,
                                     torch::Tensor gPos,
                                     torch::Tensor gRot,
                                     torch::Tensor gScl,
                                     float sigmaSclTh,
                                     float3 *gPrimVrt,
                                     int3 *gPrimTri,
                                     OptixAabb *gPrimAABB,
                                     cudaStream_t stream);

void computeGaussianEnclosingSphere(uint32_t gNum,
                                    torch::Tensor gPos,
                                    torch::Tensor gRot,
                                    torch::Tensor gScl,
                                    float sigmaSclTh,
                                    float3 *gPrimCenter,
                                    float *gPrimRadius,
                                    OptixAabb *gPrimAABB,
                                    cudaStream_t stream);

void computeGaussianEnclosingAABB(uint32_t gNum,
                                  torch::Tensor gPos,
                                  torch::Tensor gRot,
                                  torch::Tensor gScl,
                                  float sigmaSclTh,
                                  OptixAabb *gPrimAABB,
                                  OptixAabb *gAABB,
                                  cudaStream_t stream);

void copyGaussianEnclosingPrimitives(uint32_t gNum,
                                     uint32_t gNumVertices,
                                     uint32_t gNumTri,
                                     torch::Tensor gPrimVrtTs,
                                     torch::Tensor gPrimTriTs,
                                     const float3 *gPrimVrt,
                                     const int3 *gPrimTri,
                                     cudaStream_t stream);

void generatePinholeCameraRays(int2 resolution,
                               float2 tanFoV,
                               const float4 *invViewMatrix,
                               float3 *rayOri,
                               float3 *rayDir,
                               cudaStream_t stream);

void build_mog_bvh(OptiXStateWrapper &stateWrapper,
                   torch::Tensor mogPos,
                   torch::Tensor mogRot,
                   torch::Tensor mogScl,
                   unsigned int rebuild)
{
    const uint32_t gNum = mogPos.size(0);
    
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    if (!rebuild && (stateWrapper.pState->gNum != gNum))
    {
        std::cerr << "ERROR:: cannot refit GAS with a different number of gaussian" << std::endl;
        rebuild = 1;
    }
    stateWrapper.pState->gNum = gNum;

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    // Create enclosing geometry primitives from 3d gaussians
    if ((stateWrapper.pState->pipeline == MOGTracingPipelineIS) || (stateWrapper.pState->gPrimType == MOGTracingCustom))
    {
        if (stateWrapper.pState->gPrimAABBSz < sizeof(OptixAabb) * gNum)
        {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(stateWrapper.pState->gPrimAABB), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void **>(&stateWrapper.pState->gPrimAABB), sizeof(OptixAabb) * gNum, cudaStream));
            stateWrapper.pState->gPrimAABBSz = sizeof(OptixAabb) * gNum;
        }

        if (!stateWrapper.pState->optixAabbPtr)
        {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&stateWrapper.pState->optixAabbPtr), sizeof(OptixAabb), cudaStream));
        }

        OptixAabb hostOptixAabb{1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(stateWrapper.pState->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

        stateWrapper.pState->gPrimNumVert = 0;
        stateWrapper.pState->gPrimNumTri = 0;

        computeGaussianEnclosingAABB(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                     reinterpret_cast<OptixAabb *>(stateWrapper.pState->gPrimAABB),
                                     reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr), cudaStream);

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(&stateWrapper.pState->gasAABB),
                                   reinterpret_cast<void *>(stateWrapper.pState->optixAabbPtr), sizeof(OptixAabb), cudaMemcpyDeviceToHost,
                                   cudaStream));
        // std::cout << "AABB = [ (" << stateWrapper.pState->gasAABB.minX << " , " << stateWrapper.pState->gasAABB.minY
        //           << " , " << stateWrapper.pState->gasAABB.minZ << " ) ( " << stateWrapper.pState->gasAABB.maxX << "
        //           , "
        //           << stateWrapper.pState->gasAABB.maxY << " , " << stateWrapper.pState->gasAABB.maxZ << " ) ] "
        //           << std::endl;
    }
    else
    {
        if (!stateWrapper.pState->optixAabbPtr)
        {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&stateWrapper.pState->optixAabbPtr), sizeof(OptixAabb), cudaStream));
        }

        OptixAabb hostOptixAabb{1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(stateWrapper.pState->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

        if (stateWrapper.pState->gPrimType == MOGTracingIcosaHedron)
        {
            stateWrapper.pState->gPrimNumVert = 12;
            stateWrapper.pState->gPrimNumTri = 20;
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingIcosaHedron(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                                reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                                reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri),
                                                reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr), cudaStream);
        }
        else if (stateWrapper.pState->gPrimType == MOGTracingOctraHedron)
        {
            stateWrapper.pState->gPrimNumVert = 6;
            stateWrapper.pState->gPrimNumTri = 8;
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingOctaHedron(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                               reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                               reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri),
                                               reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr),
                                               cudaStream);
            CUDA_CHECK_LAST();
        }
        else if (stateWrapper.pState->gPrimType == MOGTracingTriHexa)
        {
            stateWrapper.pState->gPrimNumVert = 6;
            stateWrapper.pState->gPrimNumTri = 6;
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTriHexa(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                            reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                            reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri),
                                            reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr),
                                            cudaStream);
            CUDA_CHECK_LAST();
        }
        else if (stateWrapper.pState->gPrimType == MOGTracingTriSurfel)
        {
            stateWrapper.pState->gPrimNumVert = 4;
            stateWrapper.pState->gPrimNumTri = 2;
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTriSurfel(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                              reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                              reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri),
                                              reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr),
                                              cudaStream);
            CUDA_CHECK_LAST();
        }
        else if (stateWrapper.pState->gPrimType == MOGTracingTetraHedron)
        {
            stateWrapper.pState->gPrimNumVert = 4;
            stateWrapper.pState->gPrimNumTri = 4;
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTetraHedron(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                                reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                                reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri),
                                                reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr), cudaStream);
        }
        else if (stateWrapper.pState->gPrimType == MOGTracingSphere)
        {
            stateWrapper.pState->gPrimNumVert = 0;
            stateWrapper.pState->gPrimNumTri = 1; // number of primtive per gaussians
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingSphere(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                           reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                           reinterpret_cast<float *>(stateWrapper.pState->gPrimTri),
                                           reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr), cudaStream);
        }
        else
        {
            stateWrapper.pState->gPrimNumVert = 5;
            stateWrapper.pState->gPrimNumTri = 6;
            stateWrapper.reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingDiamond(gNum, mogPos, mogRot, mogScl, stateWrapper.pState->gaussianSigmaThreshold,
                                            reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                            reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri),
                                            reinterpret_cast<OptixAabb *>(stateWrapper.pState->optixAabbPtr), cudaStream);
        }
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(&stateWrapper.pState->gasAABB),
                                   reinterpret_cast<void *>(stateWrapper.pState->optixAabbPtr), sizeof(OptixAabb), cudaMemcpyDeviceToHost,
                                   cudaStream));
        // std::cout << "AABB = [ (" << stateWrapper.pState->gasAABB.minX << " , " << stateWrapper.pState->gasAABB.minY
        //           << " , " << stateWrapper.pState->gasAABB.minZ << " ) ( " << stateWrapper.pState->gasAABB.maxX << ", "
        //           << stateWrapper.pState->gasAABB.maxY << " , " << stateWrapper.pState->gasAABB.maxZ << " ) ] "
        //           << std::endl;
    }

    // Clear BVH GPU memory
    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        accel_options.operation = rebuild ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

        OptixBuildInput prim_input = {};

        if ((stateWrapper.pState->pipeline == MOGTracingPipelineIS) ||
            (stateWrapper.pState->gPrimType == MOGTracingCustom))
        {
            const uint32_t prim_input_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            prim_input.customPrimitiveArray.numPrimitives = gNum;
            prim_input.customPrimitiveArray.aabbBuffers = &stateWrapper.pState->gPrimAABB;
            prim_input.customPrimitiveArray.strideInBytes = 0;
            prim_input.customPrimitiveArray.flags = prim_input_flags;
            prim_input.customPrimitiveArray.numSbtRecords = 1;
        }
        else if (stateWrapper.pState->gPrimType == MOGTracingSphere)
        {
            const uint32_t prim_input_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            prim_input.sphereArray.vertexBuffers = &stateWrapper.pState->gPrimVrt;
            prim_input.sphereArray.vertexStrideInBytes = 0;
            prim_input.sphereArray.numVertices = gNum;
            prim_input.sphereArray.radiusBuffers = &stateWrapper.pState->gPrimTri;
            prim_input.sphereArray.radiusStrideInBytes = 0;
            prim_input.sphereArray.singleRadius = 0;
            prim_input.sphereArray.flags = prim_input_flags;
            prim_input.sphereArray.numSbtRecords = 1;
        }
        else
        {
            // Our build input is a simple list of non-indexed triangle vertices
            const uint32_t prim_input_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            prim_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            prim_input.triangleArray.numVertices = stateWrapper.pState->gPrimNumVert * gNum;
            prim_input.triangleArray.vertexBuffers = &stateWrapper.pState->gPrimVrt;
            prim_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            prim_input.triangleArray.numIndexTriplets = stateWrapper.pState->gPrimNumTri * gNum;
            prim_input.triangleArray.indexBuffer = stateWrapper.pState->gPrimTri;
            prim_input.triangleArray.flags = prim_input_flags;
            prim_input.triangleArray.numSbtRecords = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(stateWrapper.pState->context, &accel_options, &prim_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));
        if (stateWrapper.pState->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes)
        {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(stateWrapper.pState->gasBufferTmp), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void **>(&stateWrapper.pState->gasBufferTmp), gas_buffer_sizes.tempSizeInBytes, cudaStream));
            stateWrapper.pState->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
        }

        if (rebuild && (stateWrapper.pState->gasBufferSz < gas_buffer_sizes.outputSizeInBytes))
        {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(stateWrapper.pState->gasBuffer), cudaStream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&stateWrapper.pState->gasBuffer),
                                       gas_buffer_sizes.outputSizeInBytes, cudaStream));
            stateWrapper.pState->gasBufferSz = gas_buffer_sizes.outputSizeInBytes;
        }

        OPTIX_CHECK(optixAccelBuild(stateWrapper.pState->context,
                                    cudaStream, // CUDA stream
                                    &accel_options, &prim_input,
                                    1, // num build inputs
                                    stateWrapper.pState->gasBufferTmp, gas_buffer_sizes.tempSizeInBytes, stateWrapper.pState->gasBuffer,
                                    gas_buffer_sizes.outputSizeInBytes, &stateWrapper.pState->gasHandle,
                                    nullptr, // emitted property list
                                    0        // num emitted properties
                                    ));
    }

    CUDA_CHECK_LAST();
}

template <class T, int N, template <typename U> class PtrTraits = DefaultPtrTraits>
PackedTensorAccessor32<T, N> packed_accessor32(torch::Tensor tensor)
{
    return PackedTensorAccessor32<T, N, PtrTraits>(static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
                                                   tensor.sizes().data(), tensor.strides().data());
}

std::tuple<torch::Tensor, torch::Tensor> get_mog_primitives(OptiXStateWrapper &stateWrapper)
{
    torch::Tensor gPrimVert = torch::empty({stateWrapper.pState->gNum * stateWrapper.pState->gPrimNumVert, 3},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor gPrimTri = torch::empty({stateWrapper.pState->gNum * stateWrapper.pState->gPrimNumTri, 3},
                                          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    copyGaussianEnclosingPrimitives(stateWrapper.pState->gNum, stateWrapper.pState->gPrimNumVert,
                                    stateWrapper.pState->gPrimNumTri, gPrimVert, gPrimTri,
                                    reinterpret_cast<float3 *>(stateWrapper.pState->gPrimVrt),
                                    reinterpret_cast<int3 *>(stateWrapper.pState->gPrimTri), cudaStream);

    return std::tuple<torch::Tensor, torch::Tensor>(gPrimVert, gPrimTri);
}

std::tuple<torch::Tensor, torch::Tensor> create_camera_rays(
    int width,
    int height,
    float tanfovx,
    float tanfovy,
    torch::Tensor invViewMatrix)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rayOri = torch::empty({1, width, height, 3}, opts);
    torch::Tensor rayDir = torch::empty({1, width, height, 3}, opts);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    generatePinholeCameraRays(
        make_int2(width, height),
        make_float2(tanfovx, tanfovy),
        reinterpret_cast<const float4 *>(invViewMatrix.contiguous().data_ptr()),
        reinterpret_cast<float3 *>(rayOri.contiguous().data_ptr()),
        reinterpret_cast<float3 *>(rayDir.contiguous().data_ptr()),
        cudaStream);

    CUDA_CHECK_LAST();

    return std::tuple<torch::Tensor, torch::Tensor>(rayOri, rayDir);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> trace_mog(OptiXStateWrapper &stateWrapper,
                                                                                                               uint32_t frameNumber,
                                                                                                               uint32_t renderOpts,
                                                                                                               torch::Tensor rayOri,
                                                                                                               torch::Tensor rayDir,
                                                                                                               torch::Tensor mogPos,
                                                                                                               torch::Tensor mogRot,
                                                                                                               torch::Tensor mogScl,
                                                                                                               torch::Tensor mogDns,
                                                                                                               torch::Tensor mogSph)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rayRad = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayDns = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayHit = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayNrm = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayHitsCount = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor mogWeightSum = torch::zeros({mogDns.size(0), mogDns.size(1)}, opts);

    MoGTracingParams paramsHost;
    paramsHost.handle = stateWrapper.pState->gasHandle;
    paramsHost.rayOri = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDir = packed_accessor32<float, 4>(rayDir);
    paramsHost.mogPos = packed_accessor32<float, 2>(mogPos);
    paramsHost.mogRot = packed_accessor32<float, 2>(mogRot);
    paramsHost.mogScl = packed_accessor32<float, 2>(mogScl);
    paramsHost.mogDns = packed_accessor32<float, 2>(mogDns);
    paramsHost.mogSph = packed_accessor32<float, 2>(mogSph);
    paramsHost.rayRad = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDns = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHit = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNrm = packed_accessor32<float, 4>(rayNrm);
    paramsHost.rayHitsCount = packed_accessor32<float, 4>(rayHitsCount);
    paramsHost.mogWeightSum = packed_accessor32<float, 2>(mogWeightSum);

    paramsHost.minTransmittance = stateWrapper.pState->minTransmittance;
    paramsHost.hitMinGaussianResponse = minGaussianResponse(stateWrapper.pState->gaussianSigmaThreshold);
    paramsHost.alphaMaxValue = 0.99f;
    paramsHost.alphaMinThreshold = 1.0f / 255.0f;
    paramsHost.renderOpts = renderOpts;

    paramsHost.aabb = stateWrapper.pState->gasAABB;
    paramsHost.slabSpacing = slabSpacingFromAABB(paramsHost.aabb, stateWrapper.pState->maxNumSlabs);
    paramsHost.sphDegree = stateWrapper.pState->sphDegree;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber = frameNumber;
    paramsHost.gPrimNumTri = stateWrapper.pState->gPrimNumTri;

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    stateWrapper.reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(stateWrapper.pState->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    if (stateWrapper.pState->pipeline == MOGTracingPipelineCH)
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingCH, cudaStream, stateWrapper.pState->paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingCH, rayOri.size(2),
                                rayOri.size(1), rayOri.size(0)));
    }
    else if (stateWrapper.pState->pipeline == MOGTracingPipelineIS)
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingIS, cudaStream, stateWrapper.pState->paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingIS, rayRad.size(2),
                                rayRad.size(1), rayRad.size(0)));
    }
    else if (stateWrapper.pState->pipeline == MOGTracingPipelineMLAT)
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingMLAT, cudaStream, stateWrapper.pState->paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingMLAT, rayRad.size(2),
                                rayRad.size(1), rayRad.size(0)));
    }
    else if (stateWrapper.pState->pipeline == MOGTracingPipelineMBOIT)
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingMBOIT, cudaStream, stateWrapper.pState->paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingMBOIT, rayRad.size(2),
                                rayRad.size(1), rayRad.size(0)));
    }
    else
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingAH, cudaStream, stateWrapper.pState->paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingAH,
                                div_round_up(rayRad.size(2), stateWrapper.pState->patchSize),
                                div_round_up(rayRad.size(1), stateWrapper.pState->patchSize), rayRad.size(0)));
    }

    CUDA_CHECK_LAST();

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(rayRad, rayDns, rayHit, rayNrm, rayHitsCount, mogWeightSum);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> trace_mog_bwd(
    OptiXStateWrapper &stateWrapper,
    uint32_t frameNumber,
    uint32_t renderOpts,
    torch::Tensor rayOri,
    torch::Tensor rayDir,
    torch::Tensor rayRad,
    torch::Tensor rayDns,
    torch::Tensor rayHit,
    torch::Tensor rayNrm,
    torch::Tensor mogPos,
    torch::Tensor mogRot,
    torch::Tensor mogScl,
    torch::Tensor mogDns,
    torch::Tensor mogSph,
    torch::Tensor rayRadGrd,
    torch::Tensor rayDnsGrd,
    torch::Tensor rayHitGrd,
    torch::Tensor rayNrmGrd,
    torch::Tensor rayError)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor mogPosGrd = torch::zeros({mogPos.size(0), mogPos.size(1)}, opts);
    torch::Tensor mogRotGrd = torch::zeros({mogRot.size(0), mogRot.size(1)}, opts);
    torch::Tensor mogSclGrd = torch::zeros({mogScl.size(0), mogScl.size(1)}, opts);
    torch::Tensor mogDnsGrd = torch::zeros({mogDns.size(0), mogDns.size(1)}, opts);
    torch::Tensor mogSphGrd = torch::zeros({mogSph.size(0), mogSph.size(1)}, opts);
    torch::Tensor mogErrorBack = torch::zeros({mogDns.size(0), 1}, opts);

    MoGTracingBwdParams paramsHost;
    paramsHost.handle = stateWrapper.pState->gasHandle;
    paramsHost.rayOri = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDir = packed_accessor32<float, 4>(rayDir);
    paramsHost.rayRad = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDns = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHit = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNrm = packed_accessor32<float, 4>(rayNrm);
    paramsHost.mogPos = packed_accessor32<float, 2>(mogPos);
    paramsHost.mogRot = packed_accessor32<float, 2>(mogRot);
    paramsHost.mogScl = packed_accessor32<float, 2>(mogScl);
    paramsHost.mogDns = packed_accessor32<float, 2>(mogDns);
    paramsHost.mogSph = packed_accessor32<float, 2>(mogSph);
    paramsHost.rayRadGrd = packed_accessor32<float, 4>(rayRadGrd);
    paramsHost.rayDnsGrd = packed_accessor32<float, 4>(rayDnsGrd);
    paramsHost.rayHitGrd = packed_accessor32<float, 4>(rayHitGrd);
    paramsHost.rayNrmGrd = packed_accessor32<float, 4>(rayNrmGrd);
    paramsHost.rayError = packed_accessor32<float, 4>(rayError);
    paramsHost.mogPosGrd = packed_accessor32<float, 2>(mogPosGrd);
    paramsHost.mogRotGrd = packed_accessor32<float, 2>(mogRotGrd);
    paramsHost.mogSclGrd = packed_accessor32<float, 2>(mogSclGrd);
    paramsHost.mogDnsGrd = packed_accessor32<float, 2>(mogDnsGrd);
    paramsHost.mogSphGrd = packed_accessor32<float, 2>(mogSphGrd);
    paramsHost.mogErrorBack = packed_accessor32<float, 2>(mogErrorBack);

    paramsHost.minTransmittance = stateWrapper.pState->minTransmittance;
    paramsHost.hitMinGaussianResponse = minGaussianResponse(stateWrapper.pState->gaussianSigmaThreshold);
    paramsHost.alphaMaxValue = 0.99f;
    paramsHost.alphaMinThreshold = 1.0f / 255.0f;
    paramsHost.renderOpts = renderOpts;

    paramsHost.aabb = stateWrapper.pState->gasAABB;
    paramsHost.slabSpacing = slabSpacingFromAABB(paramsHost.aabb, stateWrapper.pState->maxNumSlabs);
    paramsHost.sphDegree = stateWrapper.pState->sphDegree;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber = frameNumber;
    paramsHost.gPrimNumTri = stateWrapper.pState->gPrimNumTri;

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    stateWrapper.reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(stateWrapper.pState->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    if (stateWrapper.pState->pipeline == MOGTracingPipelineCH)
    {
    }
    else if (stateWrapper.pState->pipeline == MOGTracingPipelineIS)
    {
    }
    else if (stateWrapper.pState->pipeline == MOGTracingPipelineMLAT)
    {
    }
    else
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingAHBwd, cudaStream, stateWrapper.pState->paramsDevice,
                                sizeof(MoGTracingBwdParams), &stateWrapper.pState->sbtMoGTracingAHBwd,
                                div_round_up(rayRad.size(2), stateWrapper.pState->patchSize),
                                div_round_up(rayRad.size(1), stateWrapper.pState->patchSize), rayRad.size(0)));
    }

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        mogPosGrd, mogRotGrd, mogSclGrd, mogDnsGrd, mogSphGrd, mogErrorBack);
}

std::tuple<torch::Tensor> trace_mog_inds(OptiXStateWrapper &stateWrapper,
                                         torch::Tensor rayOri,
                                         torch::Tensor rayDir,
                                         torch::Tensor mogPos,
                                         torch::Tensor mogRot,
                                         torch::Tensor mogScl,
                                         torch::Tensor mogDns)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor rayHitInd =
        torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), stateWrapper.pState->maxHitsReturned}, opts);

    MoGIndTracingParams paramsHost;
    paramsHost.handle = stateWrapper.pState->gasHandle;
    paramsHost.rayOri = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDir = packed_accessor32<float, 4>(rayDir);
    paramsHost.mogPos = packed_accessor32<float, 2>(mogPos);
    paramsHost.mogRot = packed_accessor32<float, 2>(mogRot);
    paramsHost.mogScl = packed_accessor32<float, 2>(mogScl);
    paramsHost.mogDns = packed_accessor32<float, 2>(mogDns);
    paramsHost.rayHitInd = packed_accessor32<int, 4>(rayHitInd);

    paramsHost.minTransmittance = stateWrapper.pState->minTransmittance;
    paramsHost.hitMinGaussianResponse = minGaussianResponse(stateWrapper.pState->gaussianSigmaThreshold);
    paramsHost.alphaMaxValue = 0.99f;
    paramsHost.alphaMinThreshold = 1.0f / 255.0f;

    paramsHost.aabb = stateWrapper.pState->gasAABB;
    paramsHost.slabSpacing = slabSpacingFromAABB(paramsHost.aabb, stateWrapper.pState->maxNumSlabs);
    paramsHost.sphDegree = stateWrapper.pState->sphDegree;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber = 0;
    paramsHost.gPrimNumTri = stateWrapper.pState->gPrimNumTri;

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    stateWrapper.reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(stateWrapper.pState->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingInd, cudaStream, stateWrapper.pState->paramsDevice,
                            sizeof(MoGIndTracingParams), &stateWrapper.pState->sbtMoGTracingInd, rayHitInd.size(2),
                            rayHitInd.size(1), rayHitInd.size(0)));

    return std::tuple<torch::Tensor>(rayHitInd);
}

torch::Tensor count_mog_hits(OptiXStateWrapper &stateWrapper,
                             torch::Tensor rayOri,
                             torch::Tensor rayDir,
                             torch::Tensor mogPos,
                             torch::Tensor mogRot,
                             torch::Tensor mogScl,
                             torch::Tensor mogDns)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor mogHitCount = torch::zeros({mogDns.size(0), mogDns.size(1)}, opts);

    MoGTracingParams paramsHost;
    paramsHost.handle = stateWrapper.pState->gasHandle;
    paramsHost.rayOri = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDir = packed_accessor32<float, 4>(rayDir);
    paramsHost.mogPos = packed_accessor32<float, 2>(mogPos);
    paramsHost.mogRot = packed_accessor32<float, 2>(mogRot);
    paramsHost.mogScl = packed_accessor32<float, 2>(mogScl);
    paramsHost.mogDns = packed_accessor32<float, 2>(mogDns);
    paramsHost.mogHitCount = packed_accessor32<float, 2>(mogHitCount);

    paramsHost.minTransmittance = stateWrapper.pState->minTransmittance;
    paramsHost.hitMinGaussianResponse = minGaussianResponse(stateWrapper.pState->gaussianSigmaThreshold);
    paramsHost.alphaMaxValue = 0.99f;
    paramsHost.alphaMinThreshold = 1.0f / 255.0f;

    paramsHost.aabb = stateWrapper.pState->gasAABB;
    paramsHost.slabSpacing = slabSpacingFromAABB(paramsHost.aabb, stateWrapper.pState->maxNumSlabs);
    paramsHost.sphDegree = stateWrapper.pState->sphDegree;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber = 0;
    paramsHost.gPrimNumTri = stateWrapper.pState->gPrimNumTri;

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    stateWrapper.reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(stateWrapper.pState->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingHC, cudaStream, stateWrapper.pState->paramsDevice,
                            sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingHC,
                            div_round_up(rayOri.size(2), stateWrapper.pState->patchSize),
                            div_round_up(rayOri.size(1), stateWrapper.pState->patchSize), rayOri.size(0)));

    return mogHitCount;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // State classes.
    pybind11::class_<OptiXStateWrapper>(m, "OptiXStateWrapper")
        .def(pybind11::init<const std::string &, const std::string &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            bool, uint32_t, uint32_t, float, float, uint32_t>())
        .def("set_sph_degree", &OptiXStateWrapper::setSphDegree, R"()", py::arg("degree"))
        .def("set_pipeline", &OptiXStateWrapper::setPipeline, R"()", py::arg("pipeline"));
    m.def("build_mog_bvh", &build_mog_bvh, "build_mog_bvh");
    m.def("trace_mog", &trace_mog, "trace_mog");
    m.def("trace_mog_bwd", &trace_mog_bwd, "trace_mog_bwd");
    m.def("trace_mog_inds", &trace_mog_inds, "trace_mog_inds");
    m.def("count_mog_hits", &count_mog_hits, "count_mog_hits");
    m.def("get_mog_primitives", &get_mog_primitives, "get_mog_primitives");
    m.def("create_camera_rays", &create_camera_rays, "create_camera_rays");
}
