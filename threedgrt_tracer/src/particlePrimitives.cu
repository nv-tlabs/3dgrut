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

#include <3dgrt/mathUtils.h>
#include <3dgrt/pipelineDefinitions.h>
// clang-format on

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace {
__device__ inline float kernelScale(float density, float modulatedMinResponse, uint32_t opts, float kernelDegree) {
    const float responseModulation = opts & MOGRenderAdaptiveKernelClamping ? density : 1.0f;
    const float minResponse        = fminf(modulatedMinResponse / responseModulation, 0.97f);

    // bump kernel
    if (kernelDegree < 0) {
        const float k = fabsf(kernelDegree);
        const float s     = 1.0 / powf(3.0, k);
        const float ks = powf((1.f / (logf(minResponse) - 1.f) + 1.f) / s, 1.f / k);
        return ks;
    }

    // linear kernel
    if (kernelDegree == 0) {
        return ((1.0f - minResponse) / 3.0f) / -0.329630334487f;
    }

    /// generalized gaussian of degree b : scaling a = -4.5/3^b
    /// e^{a*|x|^b}
    const float b = kernelDegree;
    const float a = -4.5f / powf(3.0f, static_cast<float>(b));
    /// find distance r (>0) st e^{a*r^b} = minResponse
    /// TODO : reshuffle the math to call powf only once
    return powf(logf(minResponse) / a, 1.0f / b);

}

constexpr uint32_t octaHedronNumVrt = 6;
constexpr uint32_t octaHedronNumTri = 8;
//
// enclosing regular octahedron
//
// r = radius of the inscribed circle = 1
// s = edge length = 6 / sqrt(6) = 2.4494897427831783
// h = height = sqrt(2/3) * s = 2
// V = 2 * s^2 * h / 3 = 8.0
constexpr float octaHedraDiag = 1.7320508075688774; // s / sqrt(2)
__global__ void computeGaussianEnclosingOctaHedronKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        const uint32_t sVertIdx = octaHedronNumVrt * idx;
        const uint32_t sTriIdx  = octaHedronNumTri * idx;

        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 octaHedronVrt[octaHedronNumVrt] = {
            make_float3(0, 0, -octaHedraDiag), make_float3(0, octaHedraDiag, 0), make_float3(-octaHedraDiag, 0, 0),
            make_float3(0, -octaHedraDiag, 0), make_float3(octaHedraDiag, 0, 0), make_float3(0, 0, octaHedraDiag)};

        const float3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < octaHedronNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = (octaHedronVrt[i] * kscl) * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 octaHedronTri[octaHedronNumTri] = {make_int3(2, 1, 0), make_int3(1, 4, 0), make_int3(4, 3, 0),
                                                      make_int3(3, 2, 0), make_int3(4, 1, 5), make_int3(3, 4, 5),
                                                      make_int3(2, 3, 5), make_int3(1, 2, 5)};
        const int3 triIdxOffset                    = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < octaHedronNumTri; ++i) {
            gPrimTri[sTriIdx + i] = octaHedronTri[i] + triIdxOffset;
        }
    }
}

constexpr float triHexaDiag      = 1.4142135623730951; // sqrt(2)
constexpr uint32_t triHexaNumVrt = 6;
constexpr uint32_t triHexaNumTri = 6;

__global__ void computeGaussianEnclosingTriHexaKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 triHexaVrt[triHexaNumVrt] = {
            make_float3(0, 0, -triHexaDiag), make_float3(0, triHexaDiag, 0), make_float3(-triHexaDiag, 0, 0),
            make_float3(0, -triHexaDiag, 0), make_float3(triHexaDiag, 0, 0), make_float3(0, 0, triHexaDiag)};

        const uint32_t sVertIdx = triHexaNumVrt * idx;
        const uint32_t sTriIdx  = triHexaNumTri * idx;
        const float3 kscl       = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < triHexaNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = (triHexaVrt[i] * kscl) * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 triHexaTri[triHexaNumTri] = {
            make_int3(0, 1, 5), make_int3(0, 5, 3),
            make_int3(0, 5, 4), make_int3(0, 2, 5),
            make_int3(4, 1, 3), make_int3(2, 1, 3)};
        const int3 triIdxOffset = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < triHexaNumTri; ++i) {
            gPrimTri[sTriIdx + i] = triHexaTri[i] + triIdxOffset;
        }
    }
}

constexpr float triSurfelDiag      = 1.4142135623730951; // sqrt(2)
constexpr uint32_t triSurfelNumVrt = 4;
constexpr uint32_t triSurfelNumTri = 2;

template <bool minScaleAxis = false>
__global__ void computeGaussianEnclosingTriSurfelKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    float4* __restrict__ gNormalDensity,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const uint8_t axis = minScaleAxis ? (scl.y < scl.x ? (scl.z < scl.y ? 2 : 1) : (scl.z < scl.x ? 2 : 0)) : 2;

        const float3 triSurfelVrt[3][triSurfelNumVrt] = {
            {make_float3(0, triSurfelDiag, 0), make_float3(0, -triSurfelDiag, 0), make_float3(0, 0, triSurfelDiag), make_float3(0, 0, -triSurfelDiag)},
            {make_float3(0, 0, triSurfelDiag), make_float3(0, 0, -triSurfelDiag), make_float3(triSurfelDiag, 0, 0), make_float3(-triSurfelDiag, 0, 0)},
            {make_float3(triSurfelDiag, 0, 0), make_float3(-triSurfelDiag, 0, 0), make_float3(0, triSurfelDiag, 0), make_float3(0, -triSurfelDiag, 0)}};

        const float density = gDns[idx];

        const uint32_t sVertIdx = triSurfelNumVrt * idx;
        const uint32_t sTriIdx  = triSurfelNumTri * idx;
        const float3 kscl       = kernelScale(density, kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < triSurfelNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = (triSurfelVrt[axis][i] * kscl) * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 triSurfelTri[triSurfelNumTri] = {make_int3(0, 1, 2), make_int3(0, 1, 3)};
        const int3 triIdxOffset                  = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < triSurfelNumTri; ++i) {
            gPrimTri[sTriIdx + i] = triSurfelTri[i] + triIdxOffset;
        }

        const float3 normal = safe_normalize(cross(gPrimVrt[sVertIdx + 1] - gPrimVrt[sVertIdx + 0], gPrimVrt[sVertIdx + 2] - gPrimVrt[sVertIdx + 0]));
        gNormalDensity[idx] = make_float4(normal.x, normal.y, normal.z, density);
    }
}

constexpr uint32_t triBaryNumVrt = 3;
constexpr uint32_t triBaryNumTri = 1;
constexpr float triBaryCosPi6    = 0.8660254037844387; // cos(pi/6)
constexpr float triBarySinPi6    = 0.5;                // sin(pi/6)

template <typename scalar_t, bool minScaleAxis = false>
__global__ void computeGaussianEnclosingTriBaryKernel(
    const uint32_t gNum,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gPos,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gRot,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gScl,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        float33 rot;
        invRotationMatrix(make_float4(gRot[idx][0], gRot[idx][1], gRot[idx][2], gRot[idx][3]), rot);
        const float3 scl   = make_float3(gScl[idx][0], gScl[idx][1], gScl[idx][2]);
        const float3 trans = make_float3(gPos[idx][0], gPos[idx][1], gPos[idx][2]);

        const float3 triBaryVrt[triBaryNumVrt] = {
            scl.x * make_float3(-triBaryCosPi6, -triBarySinPi6, 0),
            scl.y * make_float3(0, 1.0f, 0),
            scl.z * make_float3(triBaryCosPi6, -triBarySinPi6, 0)};

        const uint32_t sVertIdx = triBaryNumVrt * idx;
        const uint32_t sTriIdx  = triBaryNumTri * idx;
#pragma unroll
        for (int i = 0; i < triBaryNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = triBaryVrt[i] * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 triBaryTri[triBaryNumTri] = {make_int3(0, 1, 2)};
        const int3 triIdxOffset              = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < triBaryNumTri; ++i) {
            gPrimTri[sTriIdx + i] = triBaryTri[i] + triIdxOffset;
        }
    }
}

constexpr uint32_t tetraHedronNumVrt = 4;
constexpr uint32_t tetraHedronNumTri = 4;
//
// enclosing regular tetraHedra
//     1              Y
//    / \            |__X
//   / 2 \            \
//  0-----3            Z
//
//
// r = radius of the inscribed circle = 1
// s = edge length = sqrt(24) ~ 4.89
// h = height = sqrt(2/3) * s = sqrt(2/3) * sqrt(24) = sqrt(16) = 4
// q = h - r = 4 -1 = 3
// V = s^3 / (6*sqrt(2)) = 13.856406460551014
constexpr float tetraHedraInRadius     = 1;
constexpr float tetraHedraEdge         = 4.898979485566356;  // sqrt(24)
constexpr float tetraHedraHeight       = 4;                  // tetraHedraEdge * sqrt(2/3)
constexpr float tetraHedraFaceHeight   = 4.242640687119285;  //  tetraHedraEdge * sqrt(3) / 2
constexpr float tetraHedraFaceInRadius = 1.4142135623730951; //  tetraHedraEdge * sqrt(3) / 6 = sqrt(2)
__global__ void computeGaussianEnclosingTetraHedronKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        const uint32_t sVertIdx = tetraHedronNumVrt * idx;
        const uint32_t sTriIdx  = tetraHedronNumTri * idx;

        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 tetraHedronVrt[tetraHedronNumVrt] = {
            make_float3(-0.5 * tetraHedraEdge, -tetraHedraFaceInRadius, -1),
            make_float3(0, tetraHedraFaceHeight - tetraHedraFaceInRadius, -1),
            make_float3(0, 0, tetraHedraHeight - tetraHedraInRadius),
            make_float3(0.5 * tetraHedraEdge, -tetraHedraFaceInRadius, -1)};

        const float3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < tetraHedronNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = (tetraHedronVrt[i] * kscl) * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 tetraHedronTri[tetraHedronNumTri] = {make_int3(0, 2, 1), make_int3(0, 3, 2), make_int3(0, 1, 3),
                                                        make_int3(1, 2, 3)};
        const int3 triIdxOffset                      = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < tetraHedronNumTri; ++i) {
            gPrimTri[sTriIdx + i] = tetraHedronTri[i] + triIdxOffset;
        }
    }
}

constexpr uint32_t diamondNumVrt = 5;
constexpr uint32_t diamondNumTri = 6;
//
// enclosing triangular diamond
//
//              0
//             / \
//            2-3-4
//             \ /
//              1
// r = radius of the inscribed circle = 1
// s = edge length = 6 * r / sqrt(3)
// h = height = sqrt(2/3) * s
// V = 2 * s^3 / (6*sqrt(2)) = 9.797958971132713
// constexpr float diamondInRadius   = 1;
constexpr float diamondEdge       = 3.464101615137755;  // 6 / sqrt(3)
constexpr float diamondHeight     = 2.8284271247461903; // tetraHedraEdge * sqrt(2/3) = 2 * sqrt(2)
constexpr float diamondFaceHeight = 3;                  //  tetraHedraEdge * sqrt(3) / 2
__global__ void computeGaussianEnclosingDiamondKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        const uint32_t sVertIdx = diamondNumVrt * idx;
        const uint32_t sTriIdx  = diamondNumTri * idx;

        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 diamondVrt[diamondNumVrt] = {make_float3(0, diamondHeight, 0), make_float3(0, -diamondHeight, 0),
                                                  make_float3(-0.5 * diamondEdge, 0, -1),
                                                  make_float3(0, 0, diamondFaceHeight - 1),
                                                  make_float3(0.5 * diamondEdge, 0, -1)};

        const float3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < diamondNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = (diamondVrt[i] * kscl) * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 diamondTri[diamondNumTri] = {make_int3(0, 2, 3), make_int3(0, 4, 2), make_int3(0, 3, 4),
                                                make_int3(1, 3, 2), make_int3(1, 2, 4), make_int3(1, 4, 3)};
        const int3 triIdxOffset              = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < diamondNumTri; ++i) {
            gPrimTri[sTriIdx + i] = diamondTri[i] + triIdxOffset;
        }
    }
}

//
// enclosing sphere
__global__ void computeGaussianEnclosingSphereKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimCenter,
    float* __restrict__ gPrimRadius,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        gPrimCenter[idx] = gPos[idx];
        gPrimRadius[idx] = fmaxf(gScl[idx].x, fmaxf(gScl[idx].y, gScl[idx].z)) * kernelScale(gDns[idx], kernelMinResponse, opts, degree);

        if (gPrimAABB) {
            atomicMinFloat(&gPrimAABB[0].minX, gPrimCenter[idx].x - gPrimRadius[idx]);
            atomicMinFloat(&gPrimAABB[0].minY, gPrimCenter[idx].y - gPrimRadius[idx]);
            atomicMinFloat(&gPrimAABB[0].minZ, gPrimCenter[idx].z - gPrimRadius[idx]);
            atomicMaxFloat(&gPrimAABB[0].maxX, gPrimCenter[idx].x + gPrimRadius[idx]);
            atomicMaxFloat(&gPrimAABB[0].maxY, gPrimCenter[idx].y + gPrimRadius[idx]);
            atomicMaxFloat(&gPrimAABB[0].maxZ, gPrimCenter[idx].z + gPrimRadius[idx]);
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
    const int3* __restrict__ gPrimTri) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        const uint32_t sVertIdx = gNumVert * idx;

        for (int i = 0; i < gNumVert; ++i) {
            const float3 vrt             = gPrimVrt[sVertIdx + i];
            gPrimVertTs[sVertIdx + i][0] = vrt.x;
            gPrimVertTs[sVertIdx + i][1] = vrt.y;
            gPrimVertTs[sVertIdx + i][2] = vrt.z;
        }

        const uint32_t sTriIdx = gNumTri * idx;

        for (int i = 0; i < gNumTri; ++i) {
            const int3 faceIdx         = gPrimTri[sTriIdx + i];
            gPrimTriTs[sTriIdx + i][0] = faceIdx.x;
            gPrimTriTs[sTriIdx + i][1] = faceIdx.y;
            gPrimTriTs[sTriIdx + i][2] = faceIdx.z;
        }
    }
}

constexpr uint32_t icosaHedronNumVrt = 12;
constexpr uint32_t icosaHedronNumTri = 20;
//
// enclosing regular octahedron
//
// phi = golden ratio = (1 + sqrt(5)) / 2
// r = radius of the inscribed circle = 1 = (phi^2 * s) / ( 2 * sqrt(3))
// s = edge length = ( 2 * sqrt(3) ) / phi^2
// V = (5/12) * ( 3 + sqrt(5) ) * s^3 = 8.0
constexpr float goldenRatio   = 1.618033988749895;
constexpr float icosaEdge     = 1.323169076499215;
constexpr float icosaVrtScale = 0.5 * icosaEdge;
__global__ void computeGaussianEnclosingIcosaHedronKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    float3* __restrict__ gPrimVrt,
    int3* __restrict__ gPrimTri,
    OptixAabb* gPrimAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        const uint32_t sVertIdx = icosaHedronNumVrt * idx;
        const uint32_t sTriIdx  = icosaHedronNumTri * idx;

        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 icosaHedronVrt[icosaHedronNumVrt] = {
            make_float3(-1, goldenRatio, 0), make_float3(1, goldenRatio, 0), make_float3(0, 1, -goldenRatio),
            make_float3(-goldenRatio, 0, -1), make_float3(-goldenRatio, 0, 1), make_float3(0, 1, goldenRatio),
            make_float3(goldenRatio, 0, 1), make_float3(0, -1, goldenRatio), make_float3(-1, -goldenRatio, 0),
            make_float3(0, -1, -goldenRatio), make_float3(goldenRatio, 0, -1), make_float3(1, -goldenRatio, 0)};

        const float3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl * icosaVrtScale;
#pragma unroll
        for (int i = 0; i < icosaHedronNumVrt; ++i) {
            float3& vrt = gPrimVrt[sVertIdx + i];
            vrt         = (icosaHedronVrt[i] * kscl) * rot + trans;
            if (gPrimAABB) {
                atomicMinFloat(&gPrimAABB[0].minX, vrt.x);
                atomicMinFloat(&gPrimAABB[0].minY, vrt.y);
                atomicMinFloat(&gPrimAABB[0].minZ, vrt.z);
                atomicMaxFloat(&gPrimAABB[0].maxX, vrt.x);
                atomicMaxFloat(&gPrimAABB[0].maxY, vrt.y);
                atomicMaxFloat(&gPrimAABB[0].maxZ, vrt.z);
            }
        }

        const int3 icosaHedronTri[icosaHedronNumTri] = {
            make_int3(0, 1, 2), make_int3(0, 2, 3), make_int3(0, 3, 4), make_int3(0, 4, 5), make_int3(0, 5, 1),
            make_int3(6, 1, 5), make_int3(6, 5, 7), make_int3(6, 7, 11), make_int3(6, 11, 10), make_int3(6, 10, 1),
            make_int3(8, 4, 3), make_int3(8, 3, 9), make_int3(8, 9, 11), make_int3(8, 11, 7), make_int3(8, 7, 4),
            make_int3(9, 3, 2), make_int3(9, 2, 10), make_int3(9, 10, 11),
            make_int3(5, 4, 7), make_int3(1, 10, 2)};
        const int3 triIdxOffset = make_int3(sVertIdx, sVertIdx, sVertIdx);

#pragma unroll
        for (int i = 0; i < icosaHedronNumTri; ++i) {
            gPrimTri[sTriIdx + i] = icosaHedronTri[i] + triIdxOffset;
        }
    }
}

constexpr uint32_t aabbNumVrt = 8;

__global__ void computeGaussianEnclosingAABBKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    OptixAabb* __restrict__ gPrimAABB,
    OptixAabb* gAABB) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 aabbVrt[aabbNumVrt] = {make_float3(-1, -1, -1), make_float3(-1, -1, 1), make_float3(-1, 1, -1),
                                            make_float3(-1, 1, 1), make_float3(1, -1, -1), make_float3(1, -1, 1),
                                            make_float3(1, 1, -1), make_float3(1, 1, 1)};

        OptixAabb& aabb   = gPrimAABB[idx];
        const float3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < aabbNumVrt; ++i) {
            const float3 vrt = (aabbVrt[i] * kscl) * rot + trans;
            if (i == 0) {
                aabb.minX = vrt.x;
                aabb.minY = vrt.y;
                aabb.minZ = vrt.z;
                aabb.maxX = vrt.x;
                aabb.maxY = vrt.y;
                aabb.maxZ = vrt.z;
            } else {
                aabb.minX = fminf(aabb.minX, vrt.x);
                aabb.minY = fminf(aabb.minY, vrt.y);
                aabb.minZ = fminf(aabb.minZ, vrt.z);
                aabb.maxX = fmaxf(aabb.maxX, vrt.x);
                aabb.maxY = fmaxf(aabb.maxY, vrt.y);
                aabb.maxZ = fmaxf(aabb.maxZ, vrt.z);
            }
        }

        if (gAABB) {
            atomicMinFloat(&gAABB[0].minX, aabb.minX);
            atomicMinFloat(&gAABB[0].minY, aabb.minY);
            atomicMinFloat(&gAABB[0].minZ, aabb.minZ);
            atomicMaxFloat(&gAABB[0].maxX, aabb.maxX);
            atomicMaxFloat(&gAABB[0].maxY, aabb.maxY);
            atomicMaxFloat(&gAABB[0].maxZ, aabb.maxZ);
        }
    }
}

__global__ void computeGaussianEnclosingInstancesKernel(
    const uint32_t gNum,
    const float3* __restrict__ gPos,
    const float4* __restrict__ gRot,
    const float3* __restrict__ gScl,
    const float* __restrict__ gDns,
    const float kernelMinResponse,
    const uint32_t opts,
    const float degree,
    OptixTraversableHandle ias,
    OptixInstance* __restrict__ gPrimInstances,
    OptixAabb* gAABB) {

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {

        float33 rot;
        invRotationMatrix(gRot[idx], rot);
        const float3 scl   = gScl[idx];
        const float3 trans = gPos[idx];

        const float3 aabbVrt[aabbNumVrt] = {make_float3(-1, -1, -1), make_float3(-1, -1, 1), make_float3(-1, 1, -1),
                                            make_float3(-1, 1, 1), make_float3(1, -1, -1), make_float3(1, -1, 1),
                                            make_float3(1, 1, -1), make_float3(1, 1, 1)};

        OptixAabb aabb;
        const float3 kscl = kernelScale(gDns[idx], kernelMinResponse, opts, degree) * scl;
#pragma unroll
        for (int i = 0; i < aabbNumVrt; ++i) {
            const float3 vrt = (aabbVrt[i] * kscl) * rot + trans;
            if (i == 0) {
                aabb.minX = vrt.x;
                aabb.minY = vrt.y;
                aabb.minZ = vrt.z;
                aabb.maxX = vrt.x;
                aabb.maxY = vrt.y;
                aabb.maxZ = vrt.z;
            } else {
                aabb.minX = fminf(aabb.minX, vrt.x);
                aabb.minY = fminf(aabb.minY, vrt.y);
                aabb.minZ = fminf(aabb.minZ, vrt.z);
                aabb.maxX = fmaxf(aabb.maxX, vrt.x);
                aabb.maxY = fmaxf(aabb.maxY, vrt.y);
                aabb.maxZ = fmaxf(aabb.maxZ, vrt.z);
            }
        }

        if (gAABB) {
            atomicMinFloat(&gAABB[0].minX, aabb.minX);
            atomicMinFloat(&gAABB[0].minY, aabb.minY);
            atomicMinFloat(&gAABB[0].minZ, aabb.minZ);
            atomicMaxFloat(&gAABB[0].maxX, aabb.maxX);
            atomicMaxFloat(&gAABB[0].maxY, aabb.maxY);
            atomicMaxFloat(&gAABB[0].maxZ, aabb.maxZ);
        }

        OptixInstance instance;
        instance.instanceId        = idx;
        instance.sbtOffset         = 0;
        instance.visibilityMask    = 255;
        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = ias;
        instance.transform[0]      = rot[0].x * kscl.x;
        instance.transform[1]      = rot[0].y * kscl.y;
        instance.transform[2]      = rot[0].z * kscl.z;
        instance.transform[3]      = trans.x;
        instance.transform[4]      = rot[1].x * kscl.x;
        instance.transform[5]      = rot[1].y * kscl.y;
        instance.transform[6]      = rot[1].z * kscl.z;
        instance.transform[7]      = trans.y;
        instance.transform[8]      = rot[2].x * kscl.x;
        instance.transform[9]      = rot[2].y * kscl.y;
        instance.transform[10]     = rot[2].z * kscl.z;
        instance.transform[11]     = trans.z;

        gPrimInstances[idx] = instance;
    }
}

__global__ void generatePinholeCameraRaysKernel(int2 resolution, float2 tanFoV, const float4* __restrict__ invViewMatrix, float3* __restrict__ rayOri, float3* __restrict__ rayDir) {
    const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if ((x < resolution.x) && (y < resolution.y)) {
        uint32_t idx = x * resolution.y + y; // CHECK HWC vs WHC
        float3 dir   = safe_normalize(float3{(x - 0.5f * resolution.x) * tanFoV.x, (y - 0.5f * resolution.y) * tanFoV.y, 1.0f});
        rayDir[idx]  = float3{
            invViewMatrix[0].x * dir.x + invViewMatrix[0].y * dir.y + invViewMatrix[0].z * dir.z,
            invViewMatrix[1].x * dir.x + invViewMatrix[1].y * dir.y + invViewMatrix[1].z * dir.z,
            invViewMatrix[2].x * dir.x + invViewMatrix[2].y * dir.y + invViewMatrix[2].z * dir.z};
        rayOri[idx] = make_float3(invViewMatrix[0].w, invViewMatrix[1].w, invViewMatrix[2].w);
    }
}

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__device__ inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
__device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = expand_bits(x);
    uint32_t yy = expand_bits(y);
    uint32_t zz = expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

__global__ void generateMorton3DMoGLayoutKernel(int gridResolution,
                                                OptixAabb mogAABB,
                                                int gNum,
                                                const float3* __restrict__ mogPos,
                                                int* m3dGridCell,
                                                int2* m3dMoGCell) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gNum) {
        const float3 minAABB  = make_float3(mogAABB.minX, mogAABB.minY, mogAABB.minZ);
        const float3 maxAABB  = make_float3(mogAABB.maxX, mogAABB.maxY, mogAABB.maxZ);
        const float3 normPos  = (mogPos[idx] - minAABB) / (maxAABB - minAABB);
        const uint32_t m3dIdx = morton3D(normPos.x * gridResolution, normPos.y * gridResolution, normPos.z * gridResolution);
        const int cellIdx     = atomicAdd(&m3dGridCell[m3dIdx + 1], 1); // we add an offset for the cumsum
        m3dMoGCell[idx]       = make_int2(m3dIdx, cellIdx);
    }
}

inline __host__ uint32_t div_round_up(uint32_t val, uint32_t divisor) {
    return (val + divisor - 1) / divisor;
}

} // namespace

void computeGaussianEnclosingOctaHedron(uint32_t gNum,
                                        const float3* gPos,
                                        const float4* gRot,
                                        const float3* gScl,
                                        const float* gDns,
                                        float kernelMinResponse,
                                        uint32_t opts,
                                        const float degree,
                                        float3* gPrimVrt,
                                        int3* gPrimTri,
                                        OptixAabb* gPrimAABB,
                                        cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingOctaHedronKernel<<<blocks, threads, 0, stream>>>(
        gNum, gPos,
        gRot,
        gScl,
        gDns,
        kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gPrimAABB);
}

void computeGaussianEnclosingIcosaHedron(uint32_t gNum,
                                         const float3* gPos,
                                         const float4* gRot,
                                         const float3* gScl,
                                         const float* gDns,
                                         float kernelMinResponse,
                                         uint32_t opts,
                                         const float degree,
                                         float3* gPrimVrt,
                                         int3* gPrimTri,
                                         OptixAabb* gPrimAABB,
                                         cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingIcosaHedronKernel<<<blocks, threads, 0, stream>>>(gNum,
                                                                              gPos,
                                                                              gRot,
                                                                              gScl,
                                                                              gDns,
                                                                              kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gPrimAABB);
}

void computeGaussianEnclosingTetraHedron(uint32_t gNum,
                                         const float3* gPos,
                                         const float4* gRot,
                                         const float3* gScl,
                                         const float* gDns,
                                         float kernelMinResponse,
                                         uint32_t opts,
                                         const float degree,
                                         float3* gPrimVrt,
                                         int3* gPrimTri,
                                         OptixAabb* gPrimAABB,
                                         cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingTetraHedronKernel<<<blocks, threads, 0, stream>>>(gNum,
                                                                              gPos,
                                                                              gRot,
                                                                              gScl,
                                                                              gDns,
                                                                              kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gPrimAABB);
}

void computeGaussianEnclosingDiamond(uint32_t gNum,
                                     const float3* gPos,
                                     const float4* gRot,
                                     const float3* gScl,
                                     const float* gDns,
                                     float kernelMinResponse,
                                     uint32_t opts,
                                     const float degree,
                                     float3* gPrimVrt,
                                     int3* gPrimTri,
                                     OptixAabb* gPrimAABB,
                                     cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingDiamondKernel<<<blocks, threads, 0, stream>>>(gNum,
                                                                          gPos,
                                                                          gRot,
                                                                          gScl,
                                                                          gDns,
                                                                          kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gPrimAABB);
}

void computeGaussianEnclosingSphere(uint32_t gNum,
                                    const float3* gPos,
                                    const float4* gRot,
                                    const float3* gScl,
                                    const float* gDns,
                                    float kernelMinResponse,
                                    uint32_t opts,
                                    const float degree,
                                    float3* gPrimCenter,
                                    float* gPrimRadius,
                                    OptixAabb* gPrimAABB,
                                    cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingSphereKernel<<<blocks, threads, 0, stream>>>(gNum,
                                                                         gPos,
                                                                         gRot,
                                                                         gScl,
                                                                         gDns,
                                                                         kernelMinResponse, opts, degree, gPrimCenter, gPrimRadius, gPrimAABB);
}

void copyGaussianEnclosingPrimitives(uint32_t gNum,
                                     uint32_t gNumVert,
                                     uint32_t gNumTri,
                                     torch::Tensor gPrimVertTs,
                                     torch::Tensor gPrimTriTs,
                                     const float3* gPrimVrt,
                                     const int3* gPrimTri,
                                     cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        gPrimVertTs.type(), "copyGaussianEnclosingPrimitives", ([&] { copyGaussianEnclosingPrimitivesKernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                                                          gNum, gNumVert, gNumTri, gPrimVertTs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                          gPrimTriTs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), gPrimVrt, gPrimTri); }));
}

void computeGaussianEnclosingAABB(uint32_t gNum,
                                  const float3* gPos,
                                  const float4* gRot,
                                  const float3* gScl,
                                  const float* gDns,
                                  float kernelMinResponse,
                                  uint32_t opts,
                                  const float degree,
                                  OptixAabb* gPrimAABB,
                                  OptixAabb* gAABB,
                                  cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingAABBKernel<<<blocks, threads, 0, stream>>>(
        gNum,
        gPos,
        gRot,
        gScl,
        gDns,
        kernelMinResponse, opts, degree, gPrimAABB, gAABB);
}

void computeGaussianEnclosingInstances(uint32_t gNum,
                                       const float3* gPos,
                                       const float4* gRot,
                                       const float3* gScl,
                                       const float* gDns,
                                       float kernelMinResponse,
                                       uint32_t opts,
                                       const float degree,
                                       OptixTraversableHandle ias,
                                       OptixInstance* gPrimInstances,
                                       OptixAabb* gAABB,
                                       cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingInstancesKernel<<<blocks, threads, 0, stream>>>(
        gNum,
        gPos,
        gRot,
        gScl,
        gDns,
        kernelMinResponse, opts, degree, ias, gPrimInstances, gAABB);
}

void computeGaussianEnclosingTriHexa(uint32_t gNum,
                                     const float3* gPos,
                                     const float4* gRot,
                                     const float3* gScl,
                                     const float* gDns,
                                     float kernelMinResponse,
                                     uint32_t opts,
                                     const float degree,
                                     float3* gPrimVrt,
                                     int3* gPrimTri,
                                     OptixAabb* gPrimAABB,
                                     cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingTriHexaKernel<<<blocks, threads, 0, stream>>>(gNum,
                                                                          gPos,
                                                                          gRot,
                                                                          gScl,
                                                                          gDns,
                                                                          kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gPrimAABB);
}

void computeGaussianEnclosingTriSurfel(uint32_t gNum,
                                       const float3* gPos,
                                       const float4* gRot,
                                       const float3* gScl,
                                       const float* gDns,
                                       float kernelMinResponse,
                                       uint32_t opts,
                                       const float degree,
                                       float3* gPrimVrt,
                                       int3* gPrimTri,
                                       OptixAabb* gPrimAABB,
                                       float4* gNormalDensity,
                                       cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    computeGaussianEnclosingTriSurfelKernel<<<blocks, threads, 0, stream>>>(gNum,
                                                                            gPos,
                                                                            gRot,
                                                                            gScl,
                                                                            gDns,
                                                                            kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gNormalDensity, gPrimAABB);
}

void computeGaussianEnclosingTriBary(uint32_t gNum,
                                     torch::Tensor gPos,
                                     torch::Tensor gRot,
                                     torch::Tensor gScl,
                                     torch::Tensor gDns,
                                     float kernelMinResponse,
                                     uint32_t opts,
                                     const float degree,
                                     float3* gPrimVrt,
                                     int3* gPrimTri,
                                     OptixAabb* gPrimAABB,
                                     cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        gPos.type(), "computeGaussianEnclosingTriBary", ([&] { computeGaussianEnclosingTriBaryKernel<scalar_t>
                                                                   <<<blocks, threads, 0, stream>>>(gNum, gPos.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                                    gRot.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                                    gScl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                                    gDns.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                                    kernelMinResponse, opts, degree, gPrimVrt, gPrimTri, gPrimAABB); }));
}

void generatePinholeCameraRays(int2 resolution, float2 tanFoV, const float4* invViewMatrix, float3* rayOri, float3* rayDir, cudaStream_t stream) {
    const dim3 threads = {32, 32, 1};
    const dim3 blocks  = {
        div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1};
    generatePinholeCameraRaysKernel<<<blocks, threads, 0, stream>>>(resolution, tanFoV, invViewMatrix, rayOri, rayDir);
}

void generateMorton3DMoGLayout(int gridResolution,
                               OptixAabb mogAABB,
                               int gNum,
                               const float3* mogPos,
                               int* m3dGridCell,
                               int2* m3dMoGCell,
                               cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(gNum), threads);

    generateMorton3DMoGLayoutKernel<<<blocks, threads, 0, stream>>>(gridResolution, mogAABB, gNum, mogPos, m3dGridCell, m3dMoGCell);
}
