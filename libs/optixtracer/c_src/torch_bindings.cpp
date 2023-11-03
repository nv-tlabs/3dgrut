// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifdef _MSC_VER
#    pragma warning(push, 0)
#    include <torch/extension.h>
#    pragma warning(pop)
#else
#    include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL)                                                                               \
    {                                                                                                                  \
        cudaError_t err = CUDA_CALL;                                                                                   \
        AT_CUDA_CHECK(cudaGetLastError());                                                                             \
    }
#define NVDR_CHECK_GL_ERROR(GL_CALL)                                                                                   \
    {                                                                                                                  \
        GL_CALL;                                                                                                       \
        GLenum err = glGetError();                                                                                     \
        TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]");                 \
    }
#define CHECK_TENSOR(X, DIMS, CHANNELS)                                                                                \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor")                                                              \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16")   \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions")                                                 \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "mogTracing/params.h"
#include "mogTracingBwd/params.h"
#include "optix_utils.h"
#include "optix_wrapper.h"

#include <optix_stubs.h>

namespace
{

inline float slabSpacingFromAABB(const OptixAabb& aabb, const uint32_t maxNumSlabs = 16)
{
    const float aabbDiag_x = aabb.maxX - aabb.minX;
    const float aabbDiag_y = aabb.maxY - aabb.minY;
    const float aabbDiag_z = aabb.maxZ - aabb.minZ;
    return sqrt(aabbDiag_x * aabbDiag_x + aabbDiag_y * aabbDiag_y + aabbDiag_z * aabbDiag_z) / maxNumSlabs;
}

} // namespace name


void computeGaussianEnclosingOctaHedron(uint32_t gNum,
                                        torch::Tensor gPos,
                                        torch::Tensor gRot,
                                        torch::Tensor gScl,
                                        float sigmaSclTh,
                                        float3* gPrimVrt,
                                        int3* gPrimTri,
                                        OptixAabb* gPrimAABB);

void build_mog_bvh(OptiXStateWrapper& stateWrapper,
                   torch::Tensor mogPos,
                   torch::Tensor mogRot,
                   torch::Tensor mogScl,
                   float enclosingSigmaFactorThreshold,
                   unsigned int rebuild)
{
    const uint32_t gNum = mogPos.size(0);
    const uint32_t gPrimNumVert = MOGPrimNumVert;
    const uint32_t gPrimNumTri = MOGPrimNumTri;

    // Create enclosing geometry primitives from 3d gaussians
    {
        // TODO : reuse the same buffer if same size + async function
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(stateWrapper.pState->gPrimVrt)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(stateWrapper.pState->gPrimTri)));

        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&stateWrapper.pState->gPrimVrt), sizeof(float3) * gPrimNumVert * gNum));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&stateWrapper.pState->gPrimTri), sizeof(int3) * gPrimNumTri * gNum));

        CUdeviceptr optixAabbPtr = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&optixAabbPtr), sizeof(OptixAabb)));

        stateWrapper.pState->gPrimNumTri = gPrimNumTri;

        computeGaussianEnclosingOctaHedron(gNum, mogPos, mogRot, mogScl, enclosingSigmaFactorThreshold,
                                           reinterpret_cast<float3*>(stateWrapper.pState->gPrimVrt),
                                           reinterpret_cast<int3*>(stateWrapper.pState->gPrimTri),
                                           reinterpret_cast<OptixAabb*>(optixAabbPtr));

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(&stateWrapper.pState->gasAABB),
                              reinterpret_cast<void*>(optixAabbPtr), sizeof(OptixAabb), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(optixAabbPtr)));

        // std::cout << "AABB = [ (" << stateWrapper.pState->gasAABB.minX << " , " << stateWrapper.pState->gasAABB.minY
                //   << " , " << stateWrapper.pState->gasAABB.minZ << " ) ( " << stateWrapper.pState->gasAABB.maxX << " , "
                //   << stateWrapper.pState->gasAABB.maxY << " , " << stateWrapper.pState->gasAABB.maxZ << " ) ] "
                //   << std::endl;
    }

    // Clear BVH GPU memory
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;

        if (rebuild > 0)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(stateWrapper.pState->gasBuffer)));
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        }
        else
        {
            accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
        }

        const uint32_t sbt_index[] = { 0, 1 };
        CUdeviceptr d_sbt_index;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), sizeof(sbt_index)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbt_index), sbt_index, sizeof(sbt_index), cudaMemcpyHostToDevice));


        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = gPrimNumVert * gNum;
        triangle_input.triangleArray.vertexBuffers = &stateWrapper.pState->gPrimVrt;
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets = gPrimNumTri * gNum;
        triangle_input.triangleArray.indexBuffer = stateWrapper.pState->gPrimTri;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;
        // triangle_input.triangleArray.numSbtRecords = 2;
        // triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_sbt_index;
        // triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        // triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );


        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(stateWrapper.pState->context, &accel_options, &triangle_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

        if (rebuild > 0)
        {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&stateWrapper.pState->gasBuffer), gas_buffer_sizes.outputSizeInBytes));
        }

        OPTIX_CHECK(optixAccelBuild(stateWrapper.pState->context,
                                    0, // CUDA stream
                                    &accel_options, &triangle_input,
                                    1, // num build inputs
                                    d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, stateWrapper.pState->gasBuffer,
                                    gas_buffer_sizes.outputSizeInBytes, &stateWrapper.pState->gasHandle,
                                    nullptr, // emitted property list
                                    0 // num emitted properties
                                    ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));

        // printf("Built OptiX BVH\n");
    }
}

template <class T, int N, template <typename U> class PtrTraits = DefaultPtrTraits>
PackedTensorAccessor32<T, N> packed_accessor32(torch::Tensor tensor)
{
    return PackedTensorAccessor32<T, N, PtrTraits>(static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
                                                   tensor.sizes().data(), tensor.strides().data());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> trace_mog(OptiXStateWrapper& stateWrapper,
                                                                  torch::Tensor rayOri,
                                                                  torch::Tensor rayDir,
                                                                  torch::Tensor mogPos,
                                                                  torch::Tensor mogRot,
                                                                  torch::Tensor mogScl,
                                                                  torch::Tensor mogDns,
                                                                  torch::Tensor mogSph)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rayRad = torch::empty({ rayOri.size(0), rayOri.size(1), rayOri.size(2), 3 }, opts);
    torch::Tensor rayDns = torch::empty({ rayOri.size(0), rayOri.size(1), rayOri.size(2), 1 }, opts);
    torch::Tensor rayHit = torch::empty({ rayOri.size(0), rayOri.size(1), rayOri.size(2), 1 }, opts);

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
    paramsHost.minTransmittance = MOGTracingDefaultMinTransmittance;
    paramsHost.aabb = stateWrapper.pState->gasAABB;
    paramsHost.slabSpacing = slabSpacingFromAABB(paramsHost.aabb);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    CUdeviceptr paramsDevice;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&paramsDevice), sizeof(MoGTracingParams)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice));

    if (MOGTracingDefaultPipeline == MOGTracingPipelineCH)
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingCH, cudaStream, paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingCH, rayOri.size(2),
                                rayOri.size(1), rayOri.size(0)));
    }
    else
    {
        OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingAH, cudaStream, paramsDevice,
                                sizeof(MoGTracingParams), &stateWrapper.pState->sbtMoGTracingAH, rayRad.size(2),
                                rayRad.size(1), rayRad.size(0)));
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(rayRad, rayDns, rayHit);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> trace_mog_bwd(
    OptiXStateWrapper& stateWrapper,
    torch::Tensor rayOri,
    torch::Tensor rayDir,
    torch::Tensor rayRad,
    torch::Tensor mogPos,
    torch::Tensor mogRot,
    torch::Tensor mogScl,
    torch::Tensor mogDns,
    torch::Tensor mogSph,
    torch::Tensor rayRadGrd,
    torch::Tensor rayDnsGrd,
    torch::Tensor rayHitGrd)
{
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor mogPosGrd = torch::zeros({ mogPos.size(0), mogPos.size(1) }, opts);
    torch::Tensor mogRotGrd = torch::zeros({ mogRot.size(0), mogRot.size(1) }, opts);
    torch::Tensor mogSclGrd = torch::zeros({ mogScl.size(0), mogScl.size(1) }, opts);
    torch::Tensor mogDnsGrd = torch::zeros({ mogDns.size(0), mogDns.size(1) }, opts);
    torch::Tensor mogSphGrd = torch::zeros({ mogSph.size(0), mogSph.size(1) }, opts);

    MoGTracingBwdParams paramsHost;
    paramsHost.handle = stateWrapper.pState->gasHandle;
    paramsHost.rayOri = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDir = packed_accessor32<float, 4>(rayDir);
    paramsHost.rayRad = packed_accessor32<float, 4>(rayRad);
    paramsHost.mogPos = packed_accessor32<float, 2>(mogPos);
    paramsHost.mogRot = packed_accessor32<float, 2>(mogRot);
    paramsHost.mogScl = packed_accessor32<float, 2>(mogScl);
    paramsHost.mogDns = packed_accessor32<float, 2>(mogDns);
    paramsHost.mogSph = packed_accessor32<float, 2>(mogSph);
    paramsHost.rayRadGrd = packed_accessor32<float, 4>(rayRadGrd);
    paramsHost.rayDnsGrd = packed_accessor32<float, 4>(rayDnsGrd);
    paramsHost.rayHitGrd = packed_accessor32<float, 4>(rayHitGrd);
    paramsHost.mogPosGrd = packed_accessor32<float, 2>(mogPosGrd);
    paramsHost.mogRotGrd = packed_accessor32<float, 2>(mogRotGrd);
    paramsHost.mogSclGrd = packed_accessor32<float, 2>(mogSclGrd);
    paramsHost.mogDnsGrd = packed_accessor32<float, 2>(mogDnsGrd);
    paramsHost.mogSphGrd = packed_accessor32<float, 2>(mogSphGrd);
    paramsHost.minTransmittance = MOGTracingDefaultMinTransmittance;
    paramsHost.aabb = stateWrapper.pState->gasAABB;
    paramsHost.slabSpacing = slabSpacingFromAABB(paramsHost.aabb);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    CUdeviceptr paramsDevice;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&paramsDevice), sizeof(MoGTracingBwdParams)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(stateWrapper.pState->pipelineMoGTracingAHBwd, cudaStream, paramsDevice,
                            sizeof(MoGTracingBwdParams), &stateWrapper.pState->sbtMoGTracingAHBwd, rayOri.size(2),
                            rayOri.size(1), rayOri.size(0)));

    CUDA_CHECK(cudaStreamSynchronize(cudaStream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
        mogPosGrd, mogRotGrd, mogSclGrd, mogDnsGrd, mogSphGrd);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // State classes.
    pybind11::class_<OptiXStateWrapper>(m, "OptiXStateWrapper")
        .def(pybind11::init<const std::string&, const std::string&>());
    m.def("build_mog_bvh", &build_mog_bvh, "build_mog_bvh");
    m.def("trace_mog", &trace_mog, "trace_mog");
    m.def("trace_mog_bwd", &trace_mog_bwd, "trace_mog_bwd");
}
