// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#include "../mogTracing/params.h"
#include "../random_utils.h"
#include "../mogTracing/utils.h"
#include "../optix_utils.h"
#include "../ray_data.h"

extern "C"
{
    __constant__ MoGTracingParams params;
}

struct RayPayload
{
    unsigned int ahNumHits; // number of valid hits in ahHitTable
    float2 ahHitTable[MoGTracingAHMaxNumHitPerSlab]; // hit data : x = hitT, y = triId
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

    unsigned int rndSeed = tea<16>(dim.x*idx.y+idx.x, params.frameNumber);

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

    float hitDistance = 0;
    uint32_t numHits = 0;
    float transmit = 1.0f;
    RayPayload p;

    while ((startT <= minMaxT.y) && (transmit > params.minTransmittance) && (numHits < params.maxNumHits))
    {
        p.ahNumHits = 0;

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
            sampleRayDir = safe_normalize(sampleRayDir/sampleRayDirWeight); 
        }

        trace(&p, sampleRayOri, sampleRayDir, startT + epsT, startT + slabSpacing);
        if (p.ahNumHits == 0)
        {
            startT += slabSpacing;
            continue;
        }

        // in case we got more hits than available slots, start the next ray from the last hit
        if (p.ahNumHits == MoGTracingAHMaxNumHitPerSlab)
        {
            startT = p.ahHitTable[p.ahNumHits - 1].x;
        }
        else
        {
            startT += slabSpacing;
        }

        for (int i = 0; (i < p.ahNumHits) && (transmit > params.minTransmittance); i++)
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

            // NB : this is an approximation : we are using the sampleRayOri instead of the real rayOri we factorizing
            // the sph computation
            const float3 grad = computeColorFromSH(params.sphDegree, gpos, sampleRayOri, gId, params);

#pragma unroll
            for (int j = 0; j < MOGTracingPatchSize; ++j)
            {
#pragma unroll
                for (int k = 0; k < MOGTracingPatchSize; ++k)
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

                    rayRad[k][j] += grad * weight;
                    hitDistance += p.ahHitTable[i].x * weight;
                    rayTrm[k][j] *= (1 - galpha);

                    transmit = fmaxf(transmit, rayTrm[k][j]);
                }
            }

            numHits++;
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
            params.rayHit[idx.z][y][x][0] = numHits;
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
    const float hitT = MOGTracingDefaultMode & MOGTracingGaussianHit ?
                           computeGHitDistance(gId, optixGetWorldRayOrigin(), optixGetWorldRayDirection(), params) :
                           optixGetRayTmax();
    float2 ahHit = { hitT, uint_as_float(gId) };
    if ((ahNumHits < MoGTracingAHMaxNumHitPerSlab) || (ahHit.x < ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x))
    {
        ahNumHits = min(ahNumHits + 1, MoGTracingAHMaxNumHitPerSlab); // increment num hit
        int i = ahNumHits - 1; // assert(ahNumHits-1 is garbage)
        for (; (i > 0) && (ahHit.x < ahHitTablePtr[i - 1].x); --i)
        {
            ahHitTablePtr[i] = ahHitTablePtr[i - 1];
        }
        // assert(i==0 || (ahHit.x >= ahHitTablePtr[i-1].x))
        ahHitTablePtr[i] = ahHit;
        optixSetPayload_0(ahNumHits);
        // report the last entry
        if (ahHitTablePtr[MoGTracingAHMaxNumHitPerSlab - 1].x != hitT)
        {
            optixIgnoreIntersection();
        }
    }
}
