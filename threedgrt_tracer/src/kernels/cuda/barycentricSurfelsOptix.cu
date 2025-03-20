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

#include <3dgrt/pipelineParameters.h>
#include <3dgrt/kernels/cuda/gaussianParticles.cuh>
#include <cuda_fp16.h>
// clang-format on

extern "C" {
__constant__ PipelineParameters params;
}

static constexpr float epsT                 = 1e-9;
static constexpr uint32_t MaxNumHitPerTrace = 10;
struct RayHit {
    unsigned int particleId;
    float distance;
    float particleSquaredDistance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e6f;
};
using RayPayload = RayHit[MaxNumHitPerTrace];

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir) {
    const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)) - epsT), fminf(tmax.x, fminf(tmax.y, tmax.z)) + epsT};
}

static __device__ __inline__ void trace(
    RayPayload& rayPayload,
    const float3& rayOri,
    const float3& rayDir,
    const float tstart,
    const float tmin,
    const float tmax) {

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;
    r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = RayHit::InvalidParticleId;
    r10 = r11 = r12 = r13 = r14 = r15 = r16 = r17 = r18 = r19 = __float_as_uint(RayHit::InfiniteDistance);
    r20 = r21 = r22 = r23 = r24 = r25 = r26 = r27 = r28 = r29 = __float_as_uint(0.f);

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tstart,                   // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29);

    rayPayload[0] = {r0, __uint_as_float(r10), __uint_as_float(r20)};
    rayPayload[1] = {r1, __uint_as_float(r11), __uint_as_float(r21)};
    rayPayload[2] = {r2, __uint_as_float(r12), __uint_as_float(r22)};
    rayPayload[3] = {r3, __uint_as_float(r13), __uint_as_float(r23)};
    rayPayload[4] = {r4, __uint_as_float(r14), __uint_as_float(r24)};
    rayPayload[5] = {r5, __uint_as_float(r15), __uint_as_float(r25)};
    rayPayload[6] = {r6, __uint_as_float(r16), __uint_as_float(r26)};
    rayPayload[7] = {r7, __uint_as_float(r17), __uint_as_float(r27)};
    rayPayload[8] = {r8, __uint_as_float(r18), __uint_as_float(r28)};
    rayPayload[9] = {r9, __uint_as_float(r19), __uint_as_float(r29)};
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return;
    }

    const float4* particlesNormalsDensity = reinterpret_cast<const float4*>(params.particleExtendedData);

    float3 rayOrigin    = params.rayWorldOrigin(idx);
    float3 rayDirection = params.rayWorldDirection(idx);

    float3 rayRadiance     = make_float3(0.0f);
    float rayTransmittance = 1.0f;
    float rayHitDistance   = 0.f;
#ifdef ENABLE_NORMALS
    float3 rayNormal = make_float3(0.f);
#endif
#ifdef ENABLE_HIT_COUNTS
    float rayHitsCount = 0.f;
#endif

    const float2 minMaxT     = intersectAABB(params.aabb, rayOrigin, rayDirection);
    float rayLastHitDistance = minMaxT.x;
    RayPayload rayPayload;

    const float particleScaleMinResponse =
        PipelineParameters::ClampedPrimitive || (PipelineParameters::ParticleKernelDegree == 0) ? params.hitMinGaussianResponse : logf(params.hitMinGaussianResponse);

    while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance)) {
        trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.x, minMaxT.y);
        if (rayPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < MaxNumHitPerTrace; i++) {
            const RayHit rayHit = rayPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > params.minTransmittance)) {

                const float4 particleNormalsDensity = particlesNormalsDensity[rayHit.particleId];

                const float rayParticleHitDistance = rayHit.distance;
                const float rayParticleHitKernelResponse =
                    particleScaledResponse<PipelineParameters::ParticleKernelDegree, PipelineParameters::ClampedPrimitive>(
                        rayHit.particleSquaredDistance, particleScaleMinResponse, particleNormalsDensity.w);

                const float rayParticleAlpha = fminf(0.99f, rayParticleHitKernelResponse * particleNormalsDensity.w);
                if ((rayParticleHitKernelResponse > params.hitMinGaussianResponse) && (rayParticleAlpha > params.alphaMinThreshold)) {
                    const float rayParticleWeight = rayParticleAlpha * rayTransmittance;

                    // radiance from sph coefficients
                    float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
                    fetchParticleSphCoefficients(
                        rayHit.particleId,
                        params.particleRadiance,
                        &sphCoefficients[0]);
                    const float3 rayParticleRadiance = radianceFromSpH(params.sphDegree, &sphCoefficients[0], rayDirection);

                    rayRadiance += rayParticleRadiance * rayParticleWeight;
                    rayTransmittance *= (1 - rayParticleAlpha);
                    rayHitDistance += rayParticleHitDistance * rayParticleWeight;

#ifdef ENABLE_NORMALS
                    // fetch the normals from the precomputed normals buffer
                    const float3 rayParticleNormal = make_float3(particleNormalsDensity.x, particleNormalsDensity.y, particleNormalsDensity.z);
                    rayNormal += (dot(rayParticleNormal, rayDirection) < 0 ? -1.f : 1.f) * rayParticleNormal * rayParticleWeight;
#endif

#ifdef ENABLE_HIT_COUNTS
                    rayHitsCount += 1.0f;
#endif
                }

                rayLastHitDistance = fmaxf(rayLastHitDistance, rayParticleHitDistance);
            }
        }
    }

    params.rayRadiance[idx.z][idx.y][idx.x][0]    = rayRadiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = rayRadiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = rayRadiance.z;
    params.rayDensity[idx.z][idx.y][idx.x][0]     = 1 - rayTransmittance;
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = rayHitDistance;
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = rayLastHitDistance;
#ifdef ENABLE_NORMALS
    params.rayNormal[idx.z][idx.y][idx.x][0] = rayNormal.x;
    params.rayNormal[idx.z][idx.y][idx.x][1] = rayNormal.y;
    params.rayNormal[idx.z][idx.y][idx.x][2] = rayNormal.z;
#endif
#ifdef ENABLE_HIT_COUNTS
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = rayHitsCount;
#endif
}

static inline float computeTrisurfelSquaredDistance(
    const float2& triangleBarycentrics) {
    constexpr float triSurfelDiag = 1.4142135623730951; // sqrt(2)
    const float2 hitPos =
        ((1.0f - triangleBarycentrics.x - triangleBarycentrics.y) * make_float2(triSurfelDiag, 0) +
         triangleBarycentrics.x * make_float2(-triSurfelDiag, 0) +
         triangleBarycentrics.y * make_float2(0, triSurfelDiag));
    return dot(hitPos, hitPos);
}

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance, i_particleSquaredDistance)                      \
    {                                                                                                        \
        if (hit.distance < __uint_as_float(optixGetPayload_##i_distance##())) {                              \
            const RayHit hitCopy        = hit;                                                               \
            hit.particleId              = optixGetPayload_##i_id##();                                        \
            hit.distance                = __uint_as_float(optixGetPayload_##i_distance##());                 \
            hit.particleSquaredDistance = __uint_as_float(optixGetPayload_##i_particleSquaredDistance##());  \
            optixSetPayload_##i_id##(hitCopy.particleId);                                                    \
            optixSetPayload_##i_distance##(__float_as_uint(hitCopy.distance));                               \
            optixSetPayload_##i_particleSquaredDistance##(__float_as_uint(hitCopy.particleSquaredDistance)); \
        }                                                                                                    \
    }

extern "C" __global__ void __anyhit__ah() {

    if (optixGetRayTmax() < __uint_as_float(optixGetPayload_19())) {

        RayHit hit = RayHit{
            static_cast<uint32_t>(optixGetPrimitiveIndex() / params.gPrimNumTri),
            optixGetRayTmax(),
            computeTrisurfelSquaredDistance(optixGetTriangleBarycentrics())};

        compareAndSwapHitPayloadValue(hit, 0, 10, 20);
        compareAndSwapHitPayloadValue(hit, 1, 11, 21);
        compareAndSwapHitPayloadValue(hit, 2, 12, 22);
        compareAndSwapHitPayloadValue(hit, 3, 13, 23);
        compareAndSwapHitPayloadValue(hit, 4, 14, 24);
        compareAndSwapHitPayloadValue(hit, 5, 15, 25);
        compareAndSwapHitPayloadValue(hit, 6, 16, 26);
        compareAndSwapHitPayloadValue(hit, 7, 17, 27);
        compareAndSwapHitPayloadValue(hit, 8, 18, 28);
        compareAndSwapHitPayloadValue(hit, 9, 19, 29);

        // ignore all inserted hits, expect if the last one
        if (__uint_as_float(optixGetPayload_19()) > optixGetRayTmax()) {
            optixIgnoreIntersection();
        }
    }
}
