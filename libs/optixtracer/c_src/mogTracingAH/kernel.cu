// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include "../mogTracing/params.h"
#include "../mogTracing/utils.h"
#include "../optix_utils.h"
#include "../random_utils.h"
#include "../ray_data.h"

extern "C"
{
    __constant__ MoGTracingParams params;
}

static constexpr bool MoGSurfPrimitive = (MOGTRACING_PRIMITIVE_TYPE == 6) || (MOGTRACING_PRIMITIVE_TYPE == 7);
static constexpr bool MoGCustomPrimitive = (MOGTRACING_PRIMITIVE_TYPE == 5);
static constexpr unsigned int MoGTracingHitMode = MOGTRACING_HIT_MODE;
static constexpr unsigned int MoGTracingAHMaxNumHitPerSlab = MOGTRACING_MAXNUMHITS_PER_SLAB;
static constexpr unsigned int MOGTracingPatchSize = MOGTRACING_PATCH_SIZE;
struct RayPayload
{
    unsigned int ahNumHits; // number of valid hits in ahHitTable
#if MOGTRACING_SAMPLING_MODE
    unsigned int rndSeed;
#endif
    float2 ahHitTable[MoGTracingAHMaxNumHitPerSlab]; // hit data : x = hitT, y = primId
};

static __device__ __inline__ float getGaussianIndex(uint32_t primitiveIndex)
{
    return MoGCustomPrimitive ? primitiveIndex : primitiveIndex / params.gPrimNumTri;
}

static __device__ __inline__ float getRayGaussianHit(const float3 &gro, const float3 &grd, const float3 &gscl)
{
    const float3 grds = gscl * grd * dot(grd, -1 * gro);
    return sqrtf(dot(grds, grds));
}

static __device__ __inline__ float2 intersectAABB(const OptixAabb &aabb, const float3 &rayOri, const float3 &rayDir)
{
    const float3 t0 = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1 = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = maxf3(t0, t1);
    const float3 tmin = minf3(t0, t1);
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ void trace(
    RayPayload *p, const float3 &rayOri, const float3 &rayDir, const float tmin, const float tmax)
{
    const unsigned long long ahHitTablePtr = reinterpret_cast<unsigned long long>(&p->ahHitTable);
    unsigned int ahHitTablePtr0 = ahHitTablePtr >> 32;
    unsigned int ahHitTablePtr1 = ahHitTablePtr & 0x00000000ffffffff;

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (MoGSurfPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               reinterpret_cast<unsigned int &>(p->ahNumHits), ahHitTablePtr0, ahHitTablePtr1
#if MOGTRACING_SAMPLING_MODE
               ,
               reinterpret_cast<unsigned int &>(p->rndSeed)
#endif
    );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 rayOri[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayDir[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayRad[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayTrm[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayHit[MOGTracingPatchSize][MOGTracingPatchSize];
#if MOGTRACING_WITH_NORMALS
    float3 rayNrm[MOGTracingPatchSize][MOGTracingPatchSize];
#endif
#if MOGTRACING_WITH_HITCOUNTS
    float rayHitsCount[MOGTracingPatchSize][MOGTracingPatchSize];
#endif
    const int startIdxX = idx.x * MOGTracingPatchSize;
    const int startIdxY = idx.y * MOGTracingPatchSize;

    float2 minMaxT = make_float2(1e9, -1e9);

#pragma unroll
    for (int j = 0; j < MOGTracingPatchSize; ++j)
    {
        const int y = fminf(startIdxY + j, params.frameBounds.y);
#pragma unroll
        for (int i = 0; i < MOGTracingPatchSize; ++i)
        {
            const int x = fminf(startIdxX + i, params.frameBounds.x);
            rayOri[j][i] =
                make_float3(params.rayOri[idx.z][y][x][0], params.rayOri[idx.z][y][x][1], params.rayOri[idx.z][y][x][2]);
            rayDir[j][i] =
                make_float3(params.rayDir[idx.z][y][x][0], params.rayDir[idx.z][y][x][1], params.rayDir[idx.z][y][x][2]);
            rayRad[j][i] = make_float3(0);
            rayTrm[j][i] = 1;
            rayHit[j][i] = 0;
#if MOGTRACING_WITH_NORMALS
            rayNrm[j][i] = make_float3(0);
#endif
#if MOGTRACING_WITH_HITCOUNTS
            rayHitsCount[j][i] = 0;
#endif

            const float2 sampleMinMaxT = intersectAABB(params.aabb, rayOri[j][i], rayDir[j][i]);
            minMaxT.x = fminf(minMaxT.x, sampleMinMaxT.x);
            minMaxT.y = fmaxf(minMaxT.y, sampleMinMaxT.y);
        }
    }

    // ray- aabb intersection to determine number of segments
    constexpr float epsT = 1e-9;
    const float slabSpacing = params.slabSpacing;
    float startT = fmaxf(0.0f, minMaxT.x - epsT);

    const float minTransmittance = params.minTransmittance;
    const int sphDegree = params.sphDegree;
    const bool useGWeights = params.renderOpts & MOGRenderUseGWeights;

    float transmit = 1.0f;
    RayPayload p;

#if MOGTRACING_SAMPLING_MODE
    p.rndSeed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);
#endif

    while ((startT <= minMaxT.y) && (transmit > minTransmittance))
    {
        p.ahNumHits = 0;
#pragma unroll
        for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
        {
            p.ahHitTable[i] = make_float2(1e9, 1e9);
        }

        // TODO : add a GOOD ray jittering scheme over the patch (not bounded by the convex hull of the rayDirs if
        // possible)
        const float3 sampleRayOri = rayOri[MOGTracingPatchSize / 2][MOGTracingPatchSize / 2];
        const float3 sampleRayDir = rayDir[MOGTracingPatchSize / 2][MOGTracingPatchSize / 2];

        trace(&p, sampleRayOri, sampleRayDir, startT + epsT, startT + slabSpacing);
        if (p.ahNumHits == 0)
        {
            startT += slabSpacing;
            continue;
        }

#ifndef MOGTRACING_TOPK_HITS
        // in case we got more hits than available slots, start the next ray from the last hit
        if (p.ahNumHits == MoGTracingAHMaxNumHitPerSlab)
        {
            startT = p.ahHitTable[p.ahNumHits - 1].x;
        }
        else
#endif
        {
            startT += slabSpacing;
        }

        for (int i = 0; (i < p.ahNumHits) && (transmit > minTransmittance); i++)
        {
            transmit = 0.0f;

            const uint32_t primId = float_as_uint(p.ahHitTable[i].y);
            const uint32_t gId = getGaussianIndex(primId);

            float gdns = 1.0f;
#if MOGTRACING_SAMPLING_MODE
            if (!params.renderOpts & MOGRenderDnsHitSampling)
            {
                gdns = params.mogDns[gId][0];
            }
#else
            gdns = params.mogDns[gId][0];
#endif
            const float3 gpos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
            const float4 grot =
                make_float4(params.mogRot[gId][0], params.mogRot[gId][1], params.mogRot[gId][2], params.mogRot[gId][3]);
            const float3 gscl = make_float3(params.mogScl[gId][0], params.mogScl[gId][1], params.mogScl[gId][2]);

            float3 sphCoefficients[MOGTRACING_MAX_NUM_RADIANCE_SPH_COEFFS];
#pragma unroll
            for (unsigned int j = 0; j < MOGTRACING_MAX_NUM_RADIANCE_SPH_COEFFS; ++j)
            {
                const int off = j * 3;
                sphCoefficients[j] = make_float3(params.mogSph[gId][off + 0], params.mogSph[gId][off + 1], params.mogSph[gId][off + 2]);
            }

            // project ray in the gaussian
            float33 grotMat;
            rotationMatrix(make_float4(grot.x, grot.y, grot.z, grot.w), grotMat);
            float33 invGrotMat;
            invRotationMatrix(make_float4(grot.x, grot.y, grot.z, grot.w), invGrotMat);
            const float3 giscl = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);

            // NB : this is an approximation : we are using the sampleRayOri instead of the real rayOri we factorizing
            // the sph computation
            // const float3 grad = computeColorFromSH(sphDegree, gpos, sampleRayOri, gId, params);

#pragma unroll
            for (int j = 0; j < MOGTracingPatchSize; ++j)
            {
#pragma unroll
                for (int k = 0; k < MOGTracingPatchSize; ++k)
                {

                    if (rayTrm[k][j] > minTransmittance)
                    {
                        const float3 gposc = (rayOri[k][j] - gpos);
                        const float3 gposcr = (gposc * grotMat);
                        const float3 gro = giscl * gposcr;
                        const float3 rayDirR = rayDir[k][j] * grotMat;
                        const float3 grdu = giscl * rayDirR;
                        const float3 grd = safe_normalize(grdu);

                        float hitT;
                        float grayDist = 0;
                        if (MoGSurfPrimitive)
                        {
                            const float3 surfelNm = fetchSurfelNm<MOGTRACING_PRIMITIVE_TYPE>(primId % params.gPrimNumTri);
                            const float ghitT = -dot(surfelNm, gro) / dot(surfelNm, grd);
                            const float3 grds = gscl * grd * ghitT;
                            hitT = sqrtf(dot(grds, grds));
                            const float3 ghitPos = gro + grd * ghitT;
                            grayDist = dot(ghitPos, ghitPos);
                        }
                        else
                        {
                            const float3 gcrod = cross(grd, gro);
                            grayDist = dot(gcrod, gcrod);
                        }

#if MOGTRACING_TESSERACTIC_KERNEL
                        const float gres = expf(-0.0555f * grayDist * grayDist);
#else
                        const float gres = expf(-0.5f * grayDist);
#endif
                        const float galpha = fminf(gres * gdns, params.alphaMaxValue);

                        if ((gres > params.hitMinGaussianResponse) && (galpha > params.alphaMinThreshold))
                        {
                            const float weight = galpha * rayTrm[k][j];

                            if (!MoGSurfPrimitive)
                            {
                                // Distance to the gaussian center projection on the ray
                                hitT = getRayGaussianHit(gro, grd, gscl);
                            }

                            if (useGWeights)
                            {
                                atomicAdd(&params.mogWeightSum[gId][0], weight);
                            }

                            const float3 grad = computeColorFromSH(sphDegree, &sphCoefficients[0], rayDir[k][j], rayOri[k][j]);

                            rayRad[k][j] += grad * weight;
                            rayTrm[k][j] *= (1 - galpha);
                            rayHit[k][j] += hitT * weight;

#if MOGTRACING_WITH_NORMALS
                            if (MoGSurfPrimitive)
                            {
                                float3 surfelNm = fetchSurfelNm<MOGTRACING_PRIMITIVE_TYPE>(primId % params.gPrimNumTri);
                                // resolve direction ambiguities
                                if (dot(surfelNm, grd) > 0)
                                {
                                    surfelNm *= -1.0f;
                                }
                                rayNrm[k][j] += safe_normalize(surfelNm * gscl * invGrotMat) * weight;
                            }
                            else
                            {
                                constexpr float ellispoidSqRadius = 9.0f;
                                rayNrm[k][j] += safe_normalize((gro + grd * (dot(grd, -1 * gro) - sqrtf(ellispoidSqRadius - grayDist))) * gscl * invGrotMat) * weight;
                            }
#endif
#if MOGTRACING_WITH_HITCOUNTS
                            rayHitsCount[k][j] += 1.0f;
#endif
                        }
                    }

                    transmit = fmaxf(transmit, rayTrm[k][j]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < MOGTracingPatchSize; ++j)
    {
        const int y = fminf(startIdxY + j, params.frameBounds.y);
#pragma unroll
        for (int i = 0; i < MOGTracingPatchSize; ++i)
        {
            const int x = fminf(startIdxX + i, params.frameBounds.x);
            params.rayRad[idx.z][y][x][0] = rayRad[j][i].x;
            params.rayRad[idx.z][y][x][1] = rayRad[j][i].y;
            params.rayRad[idx.z][y][x][2] = rayRad[j][i].z;
            params.rayDns[idx.z][y][x][0] = 1 - rayTrm[j][i];
            params.rayHit[idx.z][y][x][0] = rayHit[j][i];
#if MOGTRACING_WITH_NORMALS
            rayNrm[j][i] = safe_normalize(rayNrm[j][i]);
            params.rayNrm[idx.z][y][x][0] = rayNrm[j][i].x;
            params.rayNrm[idx.z][y][x][1] = rayNrm[j][i].y;
            params.rayNrm[idx.z][y][x][2] = rayNrm[j][i].z;
#endif
#if MOGTRACING_WITH_HITCOUNTS
            params.rayHitsCount[idx.z][y][x][0] = rayHitsCount[j][i];
#endif
        }
    }
}

extern "C" __global__ void __intersection__is()
{
    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();

    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    const unsigned int gId = getGaussianIndex(optixGetPrimitiveIndex());

    const float3 gPos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
    const float4 gRot =
        make_float4(params.mogRot[gId][0], params.mogRot[gId][1], params.mogRot[gId][2], params.mogRot[gId][3]);
    const float3 gScl = make_float3(params.mogScl[gId][0], params.mogScl[gId][1], params.mogScl[gId][2]);

    float33 rot;
    rotationMatrix(make_float4(gRot.x, gRot.y, gRot.z, gRot.w), rot);
    const float3 giscl = make_float3(1 / gScl.x, 1 / gScl.y, 1 / gScl.z);
    const float3 gro = giscl * ((ro - gPos) * rot);
    const float3 grdu = giscl * ((rd * rot));
    const float t = fabs(dot(gro, grdu) / dot(grdu, grdu));

    if (t > tmin && t < tmax)
    {
        const float3 grd = safe_normalize(grdu);
        const float gdns = params.mogDns[gId][0];
        const float3 gcrod = cross(grd, gro);
        const float gres = expf(-0.5 * dot(gcrod, gcrod));
        if (gres > params.hitMinGaussianResponse)
        {
            optixReportIntersection(t, 0);
        }
    }
}

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int ahHitTablePtr0 = optixGetPayload_1();
    const unsigned int ahHitTablePtr1 = optixGetPayload_2();
    float2 *ahHitTablePtr =
        reinterpret_cast<float2 *>(static_cast<unsigned long long>(ahHitTablePtr0) << 32 | ahHitTablePtr1);
    const unsigned int gId = getGaussianIndex(optixGetPrimitiveIndex());
    const float hitT = (MoGSurfPrimitive || MoGCustomPrimitive || MoGTracingHitMode != MOGTracingGaussianHit) ? optixGetRayTmax() : computeGHitDistance(gId, optixGetWorldRayOrigin(), optixGetWorldRayDirection(), params);
    if (hitT < ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x)
    {
        unsigned int ahNumHits = optixGetPayload_0();

#if MOGTRACING_SAMPLING_MODE
        if (params.renderOpts & MOGRenderDnsHitSampling)
        {
            unsigned int rndSeed = optixGetPayload_3();
            const float sple = rnd(rndSeed);
            optixSetPayload_3(rndSeed);
            const float dns = params.mogDns[gId][0];
            if (dns < sple)
            {
                optixIgnoreIntersection();
                return;
            }
        }
#endif

        float2 ahHit = {hitT, uint_as_float(optixGetPrimitiveIndex())};
#pragma unroll
        for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
        {
            if (ahHit.x < ahHitTablePtr[i].x)
            {
                const float2 swapHit = ahHitTablePtr[i];
                ahHitTablePtr[i] = ahHit;
                ahHit = swapHit;
            }
        }
        if (ahNumHits < MoGTracingAHMaxNumHitPerSlab)
        {
            optixSetPayload_0(ahNumHits + 1);
        }
        // report the last entry
        if (ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x != hitT)
        {
            optixIgnoreIntersection();
        }
    }
}
