// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include "math_utils.h"

namespace
{

constexpr float oneOverCosQuarterPi = 1.41421356237; // 1 / cos(pi/4)
constexpr uint32_t octaHedronNumVrt = 6;
constexpr uint32_t octaHedronNumTri = 8;

template <typename scalar_t>
__global__ void computeGaussianEnclosingOctaHedronKernel(
    const uint32_t gNum,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gPos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gRot,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gScl,
    const float sigmaSclTh,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum)
    {
        const uint32_t sVertIdx = octaHedronNumVrt * idx;
        const uint32_t sTriIdx = octaHedronNumTri * idx;

        float33 rot;
        invRotationMatrix(make_float4(gRot[idx][0], gRot[idx][1], gRot[idx][2], gRot[idx][3]), rot);
        const float3 scl = make_float3(gScl[idx][0], gScl[idx][1], gScl[idx][2]) * sigmaSclTh;
        const float3 trans = make_float3(gPos[idx][0], gPos[idx][1], gPos[idx][2]);

        const float3 octaHedronVrt[octaHedronNumVrt] = {
            make_float3(0, 0, -oneOverCosQuarterPi), make_float3(0, oneOverCosQuarterPi, 0),
            make_float3(-oneOverCosQuarterPi, 0, 0), make_float3(0, -oneOverCosQuarterPi, 0),
            make_float3(oneOverCosQuarterPi, 0, 0),  make_float3(0, 0, oneOverCosQuarterPi)
        };

#pragma unroll
        for (int i = 0; i < octaHedronNumVrt; ++i)
        {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt = (octaHedronVrt[i] * scl) * rot + trans;
            if (gPrimAABB)
            {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 octaHedronTri[octaHedronNumTri] = { make_int3(2, 1, 0), make_int3(1, 4, 0), make_int3(4, 3, 0),
                                                       make_int3(3, 2, 0), make_int3(4, 1, 5), make_int3(3, 4, 5),
                                                       make_int3(2, 3, 5), make_int3(1, 2, 5) };
        const int3 triIdxOffset = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < octaHedronNumTri; ++i)
        {
            gPrimTri[sTriIdx + i] = octaHedronTri[i] + triIdxOffset;
        }
    }
}

inline __host__ uint32_t div_round_up(uint32_t val, uint32_t divisor)
{
    return (val + divisor - 1) / divisor;
}
}

void computeGaussianEnclosingOctaHedron(uint32_t gNum,
                                        torch::Tensor gPos,
                                        torch::Tensor gRot,
                                        torch::Tensor gScl,
                                        float sigmaSclTh,
                                        float3* gPrimVrt,
                                        int3* gPrimTri,
                                        OptixAabb* gPrimAABB)
{
    const uint32_t threads = 1024;
    const uint32_t blocks = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gPos.type(), "computeGaussianEnclosingOctaHedron", ([&] {
                                            computeGaussianEnclosingOctaHedronKernel<scalar_t><<<blocks, threads>>>(
                                                gNum, gPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                gRot.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                gScl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                sigmaSclTh, gPrimVrt, gPrimTri, gPrimAABB);
                                        }));
}
