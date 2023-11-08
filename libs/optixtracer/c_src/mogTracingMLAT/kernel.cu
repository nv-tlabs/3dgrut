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

struct RayPayload
{
    float3 ahHitTable[MoGTracingAHMaxNumHitPerSlab]; // hit data : x = hitT, y = gId, z = galpha
};

static __device__ __inline__ float getRayGaussianHit(const float3& gro, const float3& grd, const float3& gscl)
{
    const float3 grds = gscl * grd * dot(grd, -1 * gro);
    return sqrtf(dot(grds, grds));
}

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
               ahHitTablePtr0, ahHitTablePtr1);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // unsigned int rndSeed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);

    float3 rayOri = make_float3(params.rayOri[idx.z][idx.y][idx.x][0], params.rayOri[idx.z][idx.y][idx.x][1],
                                params.rayOri[idx.z][idx.y][idx.x][2]);
    float3 rayDir = make_float3(params.rayDir[idx.z][idx.y][idx.x][0], params.rayDir[idx.z][idx.y][idx.x][1],
                                params.rayDir[idx.z][idx.y][idx.x][2]);
    float3 rayRad = make_float3(0);

    const float2 minMaxT = intersectAABB(params.aabb, rayOri, rayDir);

    const float minTransmittance = params.minTransmittance;
    const int sphDegree = params.sphDegree;

    float hitDistance = 0;
    uint32_t numHits = 0;
    float transmit = 1.0f;

    RayPayload p;
#pragma unroll
    for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
    {
        p.ahHitTable[i] = make_float3(minMaxT.y, -1, 0);
    }

    trace(&p, rayOri, rayDir, minMaxT.x, minMaxT.y);

#pragma unroll
    for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
    {
        if ((p.ahHitTable[i].x < minMaxT.y) && (transmit > minTransmittance))
        {
            const uint32_t gId = float_as_uint(p.ahHitTable[i].y);

            const float3 gpos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
            const float galpha = p.ahHitTable[i].z;

            const float weight = galpha * transmit;

            // NB : this is an approximation : we are using the sampleRayOri instead of the real rayOri we factorizing
            // the sph computation
            const float3 grad = computeColorFromSH(sphDegree, gpos, rayOri, gId, params);

            rayRad += grad * weight;
            hitDistance += p.ahHitTable[i].x * weight;
            transmit *= (1 - galpha);

            numHits++;
        }
    }

    params.rayRad[idx.z][idx.y][idx.x][0] = rayRad.x;
    params.rayRad[idx.z][idx.y][idx.x][1] = rayRad.y;
    params.rayRad[idx.z][idx.y][idx.x][2] = rayRad.z;
    params.rayDns[idx.z][idx.y][idx.x][0] = 1 - transmit;
    params.rayHit[idx.z][idx.y][idx.x][0] = numHits;
}

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int gId = optixGetPrimitiveIndex() / MOGPrimNumTri;

    const float3 rayOri = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();

    const float3 gpos = make_float3(params.mogPos[gId][0], params.mogPos[gId][1], params.mogPos[gId][2]);
    const float4 grot =
        make_float4(params.mogRot[gId][0], params.mogRot[gId][1], params.mogRot[gId][2], params.mogRot[gId][3]);
    const float3 gscl = make_float3(params.mogScl[gId][0], params.mogScl[gId][1], params.mogScl[gId][2]);

    float33 grotMat;
    rotationMatrix(make_float4(grot.x, grot.y, grot.z, grot.w), grotMat);
    const float3 giscl = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);
    const float3 gro = giscl * ((rayOri - gpos) * grotMat);
    const float3 grd = safe_normalize(giscl * ((rayDir * grotMat)));

    const float3 gcrod = cross(grd, gro);
    const float gres = expf(-0.5 * dot(gcrod, gcrod));
    
    if (gres < params.hitMinGaussianResponse)
    {
        optixIgnoreIntersection();
    }
    else
    {
        const float gdns = params.mogDns[gId][0];
        const float hitT =
            MoGTracingHitMode == MOGTracingGaussianHit ? getRayGaussianHit(gro, grd, gscl) : optixGetRayTmax();
        const float galpha = gres * gdns;

        const unsigned int ahHitTablePtr0 = optixGetPayload_0();
        const unsigned int ahHitTablePtr1 = optixGetPayload_1();
        float3* ahHitTablePtr =
            reinterpret_cast<float3*>(static_cast<unsigned long long>(ahHitTablePtr0) << 32 | ahHitTablePtr1);

        float3 ahHit = { hitT, uint_as_float(gId), galpha };
        float transmit = 1.0;
        float surfHit = -1.0;
#pragma unroll
        for (int i = 0; i < MoGTracingAHMaxNumHitPerSlab; ++i)
        {
            if (ahHit.x < ahHitTablePtr[i].x)
            {
                const float3 swapHit = ahHitTablePtr[i];
                ahHitTablePtr[i] = ahHit;
                ahHit = swapHit;
            }
            if (surfHit < 0)
            {
                transmit *= (1 - ahHitTablePtr[i].z);
                if (transmit < params.minTransmittance)
                {
                    surfHit = ahHitTablePtr[i].z;
                }
            }
        }
        if ((surfHit < 0) || (hitT > surfHit))
        {
            optixIgnoreIntersection();
        }
        // TODO MERGE
    }
}
