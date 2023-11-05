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
#include "../ray_data.h"

extern "C"
{
    __constant__ MoGTracingParams params;
}

static constexpr unsigned int MoGTracingAHMaxNumHitPerSlab = 32;

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

    const float3 rayOri = make_float3(params.rayOri[idx.z][idx.y][idx.x][0], params.rayOri[idx.z][idx.y][idx.x][1],
                                      params.rayOri[idx.z][idx.y][idx.x][2]);

    const float3 rayDir = make_float3(params.rayDir[idx.z][idx.y][idx.x][0], params.rayDir[idx.z][idx.y][idx.x][1],
                                      params.rayDir[idx.z][idx.y][idx.x][2]);

    // ray- aabb intersection to determine number of segments
    const float2 minMaxT = intersectAABB(params.aabb, rayOri, rayDir);
    constexpr float epsT = 1e-9;
    //const float slabSpacing = MoGTracingAHMaxNumHitPerSlab * params.expectedDistanceBetweenHit + epsT;
    const float slabSpacing = params.slabSpacing;
    float startT = fmaxf(0.0f, minMaxT.x - epsT);
    
    float hitDistance = 0;
    float3 radiance = make_float3(0);
    uint32_t numHits = 0;
    float transmit = 1.0f;
    RayPayload p;

    while ((startT <= minMaxT.y) && (transmit > params.minTransmittance))
    {
        p.ahNumHits = 0;
        trace(&p, rayOri, rayDir, startT + epsT, startT + slabSpacing);
        if (p.ahNumHits == 0)
        {
            startT += slabSpacing;
            continue;
        }

        // in case we got more hits than available slots, start the next ray from the last hit
        if (p.ahNumHits == MoGTracingAHMaxNumHitPerSlab)
        {
            startT = p.ahHitTable[MoGTracingAHMaxNumHitPerSlab - 1].x;
        }
        else
        {
            startT += slabSpacing;
        }

        for (int i = 0; (i < p.ahNumHits) && (transmit > params.minTransmittance); i++)
        {
            const uint32_t gId = float_as_uint(p.ahHitTable[i].y);

            const float gdns = params.mogDns[gId][0];
            const float3 gpos = make_float3(params.mogPos[gId][0],params.mogPos[gId][1],params.mogPos[gId][2]);
            const float4 grot = make_float4(params.mogRot[gId][0],params.mogRot[gId][1],params.mogRot[gId][2],params.mogRot[gId][3]);
            const float3 gscl = make_float3(params.mogScl[gId][0],params.mogScl[gId][1],params.mogScl[gId][2]);

            // project ray in the gaussian
            float33 grotMat;
            rotationMatrix(make_float4(grot.x,grot.y,grot.z,grot.w), grotMat);
            const float3 giscl = make_float3(1/gscl.x,1/gscl.y,1/gscl.z);
            const float3 gposc = (rayOri - gpos);
            const float3 gposcr = (gposc*grotMat); 
            const float3 gro =  giscl*gposcr;
            const float3 rayDirR = rayDir*grotMat;
            const float3 grdu = giscl*rayDirR; 
            const float3 grd = safe_normalize(grdu);
            const float3 gcrod = cross(grd, gro);
            const float grayDir = dot(gcrod,gcrod);
            const float gres = expf(-0.5 * grayDir);
            const float galpha = gres * gdns; 

            const float3 gradu = SH_C0 * make_float3(params.mogSph[gId][0], params.mogSph[gId][1], params.mogSph[gId][2]) + make_float3(0.5);
            const float3 grad =
                make_float3(gradu.x > SHRadMinBound ? gradu.x : expf(gradu.x - SHRadMinBound) * SHRadMinBound,
                            gradu.y > SHRadMinBound ? gradu.y : expf(gradu.y - SHRadMinBound) * SHRadMinBound,
                            gradu.z > SHRadMinBound ? gradu.z : expf(gradu.z - SHRadMinBound) * SHRadMinBound);

            const float weight = galpha * transmit;

            radiance += grad * weight;
            hitDistance += p.ahHitTable[i].x * weight;
            transmit *= (1-galpha);

            numHits++;
        }
    }

    params.rayRad[idx.z][idx.y][idx.x][0] = radiance.x;
    params.rayRad[idx.z][idx.y][idx.x][1] = radiance.y;
    params.rayRad[idx.z][idx.y][idx.x][2] = radiance.z;
    params.rayDns[idx.z][idx.y][idx.x][0] = 1 - transmit;
    params.rayHit[idx.z][idx.y][idx.x][0] = numHits;
}

extern "C" __global__ void __miss__ms()
{
    // TODO : fetch background or generate random background (maybe done outside of tracer)
}

extern "C" __global__ void __intersection__is()
{
    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();
    const unsigned int gId = optixGetPrimitiveIndex();
    const float t = computeGHitDistance(gId, optixGetWorldRayOrigin(), optixGetWorldRayDirection(), params);
    if (t > tmin && t < tmax)
    {
        optixReportIntersection(t, 0);
    }
}

// extern "C" __global__ void __intersection__is_()
// {
//     const float3 ro = optixGetWorldRayOrigin();
//     const float3 rd = optixGetWorldRayDirection();
//     const float tmin = optixGetRayTmin();
//     const float tmax = optixGetRayTmax();
//     const unsigned int gId = optixGetPrimitiveIndex();

//     const float3 gPos = make_float3(params.mogPos[gId][0],params.mogPos[gId][1],params.mogPos[gId][2]);
//     const float4 gRot = make_float4(params.mogRot[gId][0],params.mogRot[gId][1],params.mogRot[gId][2],params.mogRot[gId][3]);
//     const float3 gScl = make_float3(params.mogScl[gId][0],params.mogScl[gId][1],params.mogScl[gId][2]);
    
//     float33 rot;
//     rotationMatrix(make_float4(gRot.x,gRot.y,gRot.z,gRot.w), rot);
//     const float3 giscl = make_float3(1/gScl.x,1/gScl.y,1/gScl.z);
//     const float3 gro =  iscl*((ro - gPos)*rot);
//     const float3 grd = safe_normalize(iscl*((rd*rot)));

//     const float t = dot(gro,gro) / dot(grd,gro);
//     if (t > tmin && t < tmax)
//     {
//         const float gdns = params.mogDns[gId][0];
//         const float3 gcrod = cross(grd, gro);
//         const float gres = expf(-0.5 * dot(gcrod,gcrod));
//         const float galpha = gres * gdns; 
//         optixReportIntersection(t, 0);
//     }
// }

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int ahHitTablePtr0 = optixGetPayload_1();
    const unsigned int ahHitTablePtr1 = optixGetPayload_2();
    float2* ahHitTablePtr =
        reinterpret_cast<float2*>(static_cast<unsigned long long>(ahHitTablePtr0) << 32 | ahHitTablePtr1);
    unsigned int ahNumHits = optixGetPayload_0();
    const unsigned int gId = optixGetPrimitiveIndex();
    //float2 ahHit = { computeGHitDistance(gId, optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmax(), params), uint_as_float(gId) };
    float2 ahHit = { optixGetRayTmax(), uint_as_float(gId) };
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
        optixIgnoreIntersection();
    }
}
