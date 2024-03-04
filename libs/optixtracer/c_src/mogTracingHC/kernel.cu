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

    // Need for potential jittering
    // unsigned int rndSeed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);

    float3 rayOri[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayDir[MOGTracingPatchSize][MOGTracingPatchSize];
    float3 rayRad[MOGTracingPatchSize][MOGTracingPatchSize];
    float rayTrm[MOGTracingPatchSize][MOGTracingPatchSize];
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

        // TODO : add a GOOD ray jittering scheme over the patch (not bounded by the convex hull of the rayDirs if possible)
        const float3 sampleRayOri = rayOri[MOGTracingPatchSize/2][MOGTracingPatchSize/2];
        const float3 sampleRayDir = rayDir[MOGTracingPatchSize/2][MOGTracingPatchSize/2];

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

            // project ray in the gaussian
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
                        const float3 gposc = (rayOri[k][j] - gpos);
                        const float3 gposcr = (gposc * grotMat);
                        const float3 gro = giscl * gposcr;
                        const float3 rayDirR = rayDir[k][j] * grotMat;
                        const float3 grdu = giscl * rayDirR;
                        const float3 grd = safe_normalize(grdu);
                        const float3 gcrod = cross(grd, gro);
                        const float grayDir = dot(gcrod, gcrod);
                        const float gres = expf(-0.5 * grayDir);
                        const float galpha = gres * gdns;

                        const float weight = galpha * rayTrm[k][j];

                        rayTrm[k][j] *= (1 - galpha);

                        atomicAdd(&params.mogHitCount[gId][0],1);

                        transmit = fmaxf(transmit, rayTrm[k][j]);
                    }
                }
            }
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
    const unsigned int gId = optixGetPrimitiveIndex() / params.gPrimNumTri;
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
