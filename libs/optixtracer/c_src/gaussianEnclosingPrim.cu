// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include "math_utils.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace
{
constexpr float twosqrt2third =  1.63299316186;
constexpr uint32_t octaHedronNumVrt = 6;
constexpr uint32_t octaHedronNumTri = 8;
constexpr float twoInvSqrt3 = 1.15470053838;// 2 / sqrt(3)
constexpr uint32_t tetraHedronNumVrt = 4;
constexpr uint32_t tetraHedronNumTri = 4;
constexpr uint32_t aabbNumVrt = 8;

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
            make_float3(0, 0, -twosqrt2third), make_float3(0, twosqrt2third, 0),
            make_float3(-twosqrt2third, 0, 0), make_float3(0, -twosqrt2third, 0),
            make_float3(twosqrt2third, 0, 0),  make_float3(0, 0, twosqrt2third)
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

//
// enclosing tetraHedra 
//     1              Y
//    / \            |__X
//   / 2 \            \
//  0-----3            Z
//       
//
// r = radius of the inscribed circle = 1
// s = edge length = 6 /sqrt(3)
// h = height = 3
// q = h - r = 2        
template <typename scalar_t>
__global__ void computeGaussianEnclosingTetraHedronKernel(
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

        const float3 tetraHedronVrt[tetraHedronNumVrt] = { make_float3(-twoInvSqrt3, -1, -1),
                                                           make_float3(0, 2, -1),
                                                           make_float3(0, 0, 2),
                                                           make_float3(twoInvSqrt3, -1, -1) };

#pragma unroll
        for (int i = 0; i < tetraHedronNumVrt; ++i)
        {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt = (tetraHedronVrt[i] * scl) * rot + trans;
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

        const int3 tetraHedronTri[tetraHedronNumTri] = { make_int3(0, 2, 1), make_int3(0, 3, 2), make_int3(0, 1, 3),
                                                         make_int3(1, 2, 3) };
        const int3 triIdxOffset = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < tetraHedronNumTri; ++i)
        {
            gPrimTri[sTriIdx + i] = tetraHedronTri[i] + triIdxOffset;
        }
    }
}

template <typename scalar_t>
__global__ void copyGaussianEnclosingPrimitivesKernel(
    const uint32_t gNum,
    const uint32_t gNumVert,
    const uint32_t gNumTri,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gPrimVertTs,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> gPrimTriTs,
    const float3* __restrict__ gPrimVrt,
    const int3* __restrict__ gPrimTri)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum)
    {
        const uint32_t sVertIdx = gNumVert * idx;
        
        for (int i = 0; i < gNumVert; ++i)
        {
            const float3 vrt = gPrimVrt[sVertIdx + i];
            gPrimVertTs[sVertIdx + i][0] = vrt.x;
            gPrimVertTs[sVertIdx + i][1] = vrt.y;
            gPrimVertTs[sVertIdx + i][2] = vrt.z;
        }
        
        const uint32_t sTriIdx = gNumTri * idx;

        for (int i = 0; i < gNumTri; ++i)
        {
            const int3 faceIdx = gPrimTri[sTriIdx + i];
            gPrimTriTs[sTriIdx + i][0] = faceIdx.x;
            gPrimTriTs[sTriIdx + i][1] = faceIdx.y;
            gPrimTriTs[sTriIdx + i][2] = faceIdx.z;
        }
    }
}


template <typename scalar_t>
__global__ void computeGaussianEnclosingAABBKernel(
    const uint32_t gNum,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gPos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gRot,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gScl,
    const float sigmaSclTh,
    OptixAabb* __restrict__ gPrimAABB,
    OptixAabb* gAABB)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum)
    {
        float33 rot;
        invRotationMatrix(make_float4(gRot[idx][0], gRot[idx][1], gRot[idx][2], gRot[idx][3]), rot);
        const float3 scl = make_float3(gScl[idx][0], gScl[idx][1], gScl[idx][2]) * sigmaSclTh;
        const float3 trans = make_float3(gPos[idx][0], gPos[idx][1], gPos[idx][2]);

        const float3 aabbVrt[aabbNumVrt] = { make_float3(-1, -1, -1), make_float3(-1, -1, 1), make_float3(-1, 1, -1),
                                             make_float3(-1, 1, 1),   make_float3(1, -1, -1), make_float3(1, -1, 1),
                                             make_float3(1, 1, -1),   make_float3(1, 1, 1) };

        OptixAabb& aabb = gPrimAABB[idx];
#pragma unroll
        for (int i = 0; i < aabbNumVrt; ++i)
        {
            const float3 vrt = (aabbVrt[i] * scl) * rot + trans;
            if (i == 0)
            {
                aabb.minX = vrt.x;
                aabb.minY = vrt.y;
                aabb.minZ = vrt.z;
                aabb.maxX = vrt.x;
                aabb.maxY = vrt.y;
                aabb.maxZ = vrt.z;
            }
            else
            {
                aabb.minX = fminf(aabb.minX, vrt.x);
                aabb.minY = fminf(aabb.minY, vrt.y);
                aabb.minZ = fminf(aabb.minZ, vrt.z);
                aabb.maxX = fmaxf(aabb.maxX, vrt.x);
                aabb.maxY = fmaxf(aabb.maxY, vrt.y);
                aabb.maxZ = fmaxf(aabb.maxZ, vrt.z);
            }
        }

        if (gAABB)
        {
            atomicMinFloat(&gAABB[0].minX, aabb.minX);
            atomicMinFloat(&gAABB[0].minY, aabb.minY);
            atomicMinFloat(&gAABB[0].minZ, aabb.minZ);
            atomicMaxFloat(&gAABB[0].maxX, aabb.maxX);
            atomicMaxFloat(&gAABB[0].maxY, aabb.maxY);
            atomicMaxFloat(&gAABB[0].maxZ, aabb.maxZ);
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
                                        OptixAabb* gPrimAABB,
                                        cudaStream_t stream)
{
    const uint32_t threads = 1024;
    const uint32_t blocks = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        gPos.type(), "computeGaussianEnclosingOctaHedron", ([&] {
            computeGaussianEnclosingOctaHedronKernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(gNum, gPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                 gRot.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                 gScl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                 sigmaSclTh, gPrimVrt, gPrimTri, gPrimAABB);
        }));
}

void computeGaussianEnclosingTetraHedron(uint32_t gNum,
                                         torch::Tensor gPos,
                                         torch::Tensor gRot,
                                         torch::Tensor gScl,
                                         float sigmaSclTh,
                                         float3* gPrimVrt,
                                         int3* gPrimTri,
                                         OptixAabb* gPrimAABB,
                                         cudaStream_t stream)
{
    const uint32_t threads = 1024;
    const uint32_t blocks = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        gPos.type(), "computeGaussianEnclosingTetraHedron", ([&] {
            computeGaussianEnclosingTetraHedronKernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(gNum, gPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                 gRot.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                 gScl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                 sigmaSclTh, gPrimVrt, gPrimTri, gPrimAABB);
        }));
}

void copyGaussianEnclosingPrimitives(uint32_t gNum,
                                     uint32_t gNumVert,
                                     uint32_t gNumTri,
                                     torch::Tensor gPrimVertTs,
                                     torch::Tensor gPrimTriTs,
                                     const float3* gPrimVrt,
                                     const int3* gPrimTri,
                                     cudaStream_t stream)
{
    const uint32_t threads = 1024;
    const uint32_t blocks = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        gPrimVertTs.type(), "copyGaussianEnclosingPrimitives", ([&] {
            copyGaussianEnclosingPrimitivesKernel<scalar_t><<<blocks, threads, 0, stream>>>(
                gNum, gNumVert, gNumTri, gPrimVertTs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                gPrimTriTs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), gPrimVrt, gPrimTri);
        }));
}

void computeGaussianEnclosingAABB(uint32_t gNum,
                                  torch::Tensor gPos,
                                  torch::Tensor gRot,
                                  torch::Tensor gScl,
                                  float sigmaSclTh,
                                  OptixAabb* gPrimAABB,
                                  OptixAabb* gAABB,
                                  cudaStream_t stream)
{
    const uint32_t threads = 1024;
    const uint32_t blocks = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gPos.type(), "computeGaussianEnclosingAABB", ([&] {
                                            computeGaussianEnclosingAABBKernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                                gNum, gPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                gRot.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                gScl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                sigmaSclTh, gPrimAABB, gAABB);
                                        }));
}
