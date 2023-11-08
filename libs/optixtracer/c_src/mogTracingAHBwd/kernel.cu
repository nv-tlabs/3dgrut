// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include "../mogTracing/utils.h"
#include "../mogTracingBwd/params.h"
#include "../optix_utils.h"
#include "../random_utils.h"
#include "../ray_data.h"

extern "C"
{
    __constant__ MoGTracingBwdParams params;
}

static constexpr unsigned int MoGTracingHitMode = MOGTRACING_HIT_MODE;
static constexpr unsigned int MoGTracingAHMaxNumHitPerSlab = MOGTRACING_MAXNUMHITS_PER_SLAB;
static constexpr unsigned int MOGTracingPatchSize = MOGTRACING_PATCH_SIZE;

struct RayPayload
{
    unsigned int ahNumHits; // number of valid hits in ahHitTable
    float2 ahHitTable[MoGTracingAHMaxNumHitPerSlab]; // hit data : x = hitT, y = gId
};

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir)
{
    const float3 t0 = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1 = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = maxf3(t0, t1);
    const float3 tmin = minf3(t0, t1);
    return float2{ fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z)) };
}

static __device__ __inline__ void trace(
    RayPayload* p, const float3& rayOri, const float3& rayDir, const float tmin, const float tmax)
{
    const unsigned long long ahHitTablePtr = reinterpret_cast<unsigned long long>(&p->ahHitTable);
    unsigned int ahHitTablePtr0 = ahHitTablePtr >> 32;
    unsigned int ahHitTablePtr1 = ahHitTablePtr & 0x00000000ffffffff;

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin, // Min intersection distance
               tmax, // Max intersection distance
               0.0f, // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               reinterpret_cast<unsigned int&>(p->ahNumHits), ahHitTablePtr0, ahHitTablePtr1);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    unsigned int rndSeed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);

    float3 rayOri[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayDir[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayTrm[MOGTracingPatchSize][MOGTracingPatchSize];

    float3 totalRayRad[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayRadGrd[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 accumulatedRayRad[MOGTracingPatchSize][MOGTracingPatchSize];

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
            rayTrm[j][i] = 1;

            const float2 sampleMinMaxT = intersectAABB(params.aabb, rayOri[j][i], rayDir[j][i]);
            minMaxT.x = fminf(minMaxT.x, sampleMinMaxT.x);
            minMaxT.y = fmaxf(minMaxT.y, sampleMinMaxT.y);

            totalRayRad[j][i] =
                make_float3(params.rayRad[idx.z][y][x][0], params.rayRad[idx.z][y][x][1], params.rayRad[idx.z][y][x][2]);
            rayRadGrd[j][i] =
                make_float3(params.rayRad[idx.z][y][x][0], params.rayRad[idx.z][y][x][1], params.rayRad[idx.z][y][x][2]);
            accumulatedRayRad[j][i] = make_float3(0);
        }
    }

    // ray- aabb intersection to determine number of segments
    constexpr float epsT = 1e-9;
    const float slabSpacing = params.slabSpacing;
    float startT = fmaxf(0.0f, minMaxT.x - epsT);

    const float minTransmittance = params.minTransmittance;
    const int sphDegree = params.sphDegree;

    uint32_t numHits = 0;
    float transmit = 1.0f;
    RayPayload p;

    while ((startT <= minMaxT.y) && (transmit > minTransmittance))
    {
        p.ahNumHits = 0;
#pragma unroll
        for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
        {
            p.ahHitTable[i] = make_float2(1e9, 1e9);
        }

        // TOOD : rework this jitter scheme, it is biased toward the center of the patch
        float3 sampleRayOri = make_float3(0);
        float3 sampleRayDir = make_float3(0);
        {
            float3 sampleRayOriWeight = make_float3(0);
            float3 sampleRayDirWeight = make_float3(0);
#pragma unroll
            for (int j = 0; j < MOGTracingPatchSize; ++j)
            {
#pragma unroll
                for (int i = 0; i < MOGTracingPatchSize; ++i)
                {
                    const float3 sro = rnd3(rndSeed) + 1e-6;
                    sampleRayOri += sro * rayOri[j][i];
                    sampleRayOriWeight += sro;

                    const float3 srd = rnd3(rndSeed) + 1e-6;
                    sampleRayDir += srd * rayDir[j][i];
                    sampleRayDirWeight += srd;
                }
            }
            sampleRayOri /= sampleRayOriWeight;
            sampleRayDir = safe_normalize(sampleRayDir / sampleRayDirWeight);
        }

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

            const uint32_t gId = float_as_uint(p.ahHitTable[i].y);

            const float gdns = params.mogDns[gId][0];
            const float3 gpos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
            const float4 grot =
                make_float4(params.mogRot[gId][0], params.mogRot[gId][1], params.mogRot[gId][2], params.mogRot[gId][3]);
            const float3 gscl = make_float3(params.mogScl[gId][0], params.mogScl[gId][1], params.mogScl[gId][2]);

            float33 grotMat;
            rotationMatrix(make_float4(grot.x, grot.y, grot.z, grot.w), grotMat);
            const float3 giscl = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);

#pragma unroll
            for (int j = 0; j < MOGTracingPatchSize; ++j)
            {
#pragma unroll
                for (int k = 0; k < MOGTracingPatchSize; ++k)
                {
                    if (rayTrm[k][j] > minTransmittance)
                    {
                        // project ray in the gaussian
                        const float3 gposc = (rayOri[k][j] - gpos);
                        const float3 gposcr = (gposc * grotMat);
                        const float3 gro = giscl * gposcr;
                        const float3 rayDirR = rayDir[k][j] * grotMat;
                        const float3 grdu = giscl * rayDirR;
                        const float3 grd = safe_normalize(grdu);
                        const float3 gcrod = cross(grd, gro);
                        const float grayDist = dot(gcrod, gcrod);
                        const float gres = expf(-0.5f * grayDist);
                        const float galpha = gres * gdns;

                        const float weight = galpha * rayTrm[k][j];

                        // NB : no gradient wrt d_rayDns assert(rayDnsGrd==0)

                        // gradient computation wrt rayRad

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // compute the gradient wrt to the sph coefficients and position (through the sph view
                        // direction)
                        const float3 grad =
                            computeColorFromSHBwd(sphDegree, rayOri[k][j], gId, gpos, weight, rayRadGrd[k][j], params);

                        // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
                        const float3 rayRad = weight * grad;
                        accumulatedRayRad[k][j] += rayRad;
                        const float nextTransmit = (1 - galpha) * rayTrm[k][j];
                        const float3 residualRayRad = maxf3((nextTransmit <= minTransmittance ?
                                                                 make_float3(0) :
                                                                 (totalRayRad[k][j] - accumulatedRayRad[k][j]) / nextTransmit),
                                                            make_float3(0));

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1-galpha) * transmit *
                        // residualRayRad
                        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1-gdns*gres) *
                        //                  transmit * residualRayRad
                        // ===> d_rayRad / d_gdns = gres * transmit * grad - gres * transmit * residualRayRad
                        atomicAdd(
                            &params.mogDnsGrd[gId][0], gres * rayTrm[k][j] * (grad.x - residualRayRad.x) * rayRadGrd[k][j].x +
                                                           gres * transmit * (grad.y - residualRayRad.y) * rayRadGrd[k][j].y +
                                                           gres * transmit * (grad.z - residualRayRad.z) * rayRadGrd[k][j].z);

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1 - galpha) * transmit *
                        // residualRayRad
                        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1 - gdns * gres) *
                        //                  transmit * residualRayRad
                        // ===> d_rayRad / d_gres = gdns * transmit * grad - gdns * transmit * residualRayRad
                        const float gresGrd = gdns * rayTrm[k][j] * (grad.x - residualRayRad.x) * rayRadGrd[k][j].x +
                                              gdns * rayTrm[k][j] * (grad.y - residualRayRad.y) * rayRadGrd[k][j].y +
                                              gdns * rayTrm[k][j] * (grad.z - residualRayRad.z) * rayRadGrd[k][j].z;

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> gres = exp(-0.5 * grayDist)
                        // ===> d_gres / d_grayDist = -0.5 * exp(-0.5 * grayDist)
                        //                          = -0.5 * gres
                        const float grayDistGrd = -0.5f * gres * gresGrd;

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> grayDist = dot(gcrod, gcrod)
                        //               = gcrod.x^2 + gcrod.y^2 + gcrod.z^2
                        // ===> d_grayDist / d_gcrod = 2*gcrod
                        const float3 gcrodGrd = 2 * gcrod * grayDistGrd;

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> gcrod = cross(grd, gro)
                        // ---> gcrod.x = grd.y * gro.z - grd.z * gro.y
                        // ---> gcrod.y = grd.z * gro.x - grd.x * gro.z
                        // ---> gcrod.z = grd.x * gro.y - grd.y * gro.x
                        const float3 grdGrd = make_float3(gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                                                          gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                                                          gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);
                        const float3 groGrd = make_float3(gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                                                          gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                                                          gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> gro = (1/gscl)*gposcr
                        // ===> d_gro / d_gscl = -gposcr/(gscl*gscl)
                        // ===> d_gro / d_gposcr = (1/gscl)
                        const float3 gsclGrdGro = make_float3((-gposcr.x / (gscl.x * gscl.x)) * groGrd.x,
                                                              (-gposcr.y / (gscl.y * gscl.y)) * groGrd.y,
                                                              (-gposcr.z / (gscl.z * gscl.z)) * groGrd.z);
                        const float3 gposcrGrd = giscl * groGrd;

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> gposcr = matmul(gposc, grotMat)
                        // ===> d_gposcr / d_gposc = matmul_bw_vec(grotMat)
                        // ===> d_gposcr / d_grotmat = matmul_bw_mat(gposc)
                        const float3 gposcGrd = matmul_bw_vec(grotMat, gposcrGrd);
                        const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> gposc = rayOri - gpos
                        // ===> d_gposc / d_gpos = -1
                        atomicAdd(&params.mogPosGrd[gId][0], -gposcGrd.x);
                        atomicAdd(&params.mogPosGrd[gId][1], -gposcGrd.y);
                        atomicAdd(&params.mogPosGrd[gId][2], -gposcGrd.z);

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> grd = safe_normalize(grdu)
                        // ===> d_grd / d_grdu = safe_normalize_bw(grd)
                        const float3 grduGrd = safe_normalize_bw(grdu, grdGrd);

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> grdu = (1/gscl)*rayDirR
                        // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
                        // ===> d_grdu / d_rayDirR = (1/gscl)
                        atomicAdd(&params.mogSclGrd[gId][0], gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
                        atomicAdd(&params.mogSclGrd[gId][1], gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
                        atomicAdd(&params.mogSclGrd[gId][2], gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);
                        const float3 rayDirRGrd = giscl * grduGrd;

                        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        // ---> rayDirR = matmul(rayDir, grotMat)
                        // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
                        const float4 grotGrdRayDirR = matmul_bw_quat(rayDir[k][j], rayDirRGrd, grot);
                        atomicAdd(&params.mogRotGrd[gId][0], grotGrdPoscr.x + grotGrdRayDirR.x);
                        atomicAdd(&params.mogRotGrd[gId][1], grotGrdPoscr.y + grotGrdRayDirR.y);
                        atomicAdd(&params.mogRotGrd[gId][2], grotGrdPoscr.z + grotGrdRayDirR.z);
                        atomicAdd(&params.mogRotGrd[gId][3], grotGrdPoscr.w + grotGrdRayDirR.w);

                        rayTrm[k][j] = nextTransmit;
                        transmit = fmaxf(transmit, rayTrm[k][j]);
                    }
                }
            }

            numHits++;
        }
    }
}

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int ahHitTablePtr0 = optixGetPayload_1();
    const unsigned int ahHitTablePtr1 = optixGetPayload_2();
    float2* ahHitTablePtr =
        reinterpret_cast<float2*>(static_cast<unsigned long long>(ahHitTablePtr0) << 32 | ahHitTablePtr1);
    unsigned int ahNumHits = optixGetPayload_0();
    const unsigned int gId = optixGetPrimitiveIndex() / MOGPrimNumTri;
    const float hitT = MoGTracingHitMode == MOGTracingGaussianHit ?
                           computeGHitDistance(gId, optixGetWorldRayOrigin(), optixGetWorldRayDirection(), params) :
                           optixGetRayTmax();
    if (hitT < ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x)
    {
        float2 ahHit = { hitT, uint_as_float(gId) };
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
