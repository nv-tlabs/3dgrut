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

struct RayPayload
{
    float hitT;
    unsigned int triId;
    float2 padding;
};

static __device__ __inline__ void setRayPayload(const RayPayload& p)
{
    optixSetPayload_0(float_as_int(p.hitT));
    optixSetPayload_1(p.triId);
}

static __device__ __inline__ void trace(
    RayPayload& p, const float3& rayOri, const float3& rayDir, const float tmin, const float tmax)
{
    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, rayOri, rayDir,
               tmin, // Min intersection distance
               tmax, // Max intersection distance
               0.0f, // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               reinterpret_cast<unsigned int&>(p.hitT), reinterpret_cast<unsigned int&>(p.triId));
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();

    const float3 rayOri = make_float3(params.rayOri[idx.z][idx.y][idx.x][0], params.rayOri[idx.z][idx.y][idx.x][1],
                                      params.rayOri[idx.z][idx.y][idx.x][2]);

    const float3 rayDir = make_float3(params.rayDir[idx.z][idx.y][idx.x][0], params.rayDir[idx.z][idx.y][idx.x][1],
                                      params.rayDir[idx.z][idx.y][idx.x][2]);

    float hitDistance = 0;
    float3 radiance = make_float3(0);
    uint32_t numHits = 0;
    float density = 0.0f;

    RayPayload p;
    p.hitT = 0;

    while (density < (1 - params.minTransmittance))
    {
        trace(p, rayOri, rayDir, p.hitT + 1e-9, 1e16);
        if (p.hitT < 0)
        {
            break;
        }
        const uint32_t gId = p.triId / MOGPrimNumTri;

        const float4 gRadAlpha = computeGRadAlpha(gId, rayOri, rayDir, p.hitT, params);
        // const float4 gRadAlpha = make_float4(params.mogSph[gId][0],params.mogSph[gId][1],params.mogSph[gId][2],1.);
        // //params.mogDns[gId][0]); const float4 gRadAlpha =
        // make_float4(p.triId&1,p.triId&2,p.triId&4,params.mogDns[gId][0]); const float4 gRadAlpha =
        // make_float4(1,1,0,0.1);

        if (gRadAlpha.w <= 0.0f)
        {
            continue;
        }

        const float weight = gRadAlpha.w * (1.0f - density);
        radiance = radiance + make_float3(gRadAlpha) * weight;
        hitDistance += p.hitT * weight;
        density += weight;

        // radiance += make_float3(gRadAlpha) * gRadAlpha.w;
        // hitDistance = p.hitT;
        // density += gRadAlpha.w;

        numHits++;
    }

    params.rayRad[idx.z][idx.y][idx.x][0] = radiance.x;
    params.rayRad[idx.z][idx.y][idx.x][1] = radiance.y;
    params.rayRad[idx.z][idx.y][idx.x][2] = radiance.z;
    params.rayDns[idx.z][idx.y][idx.x][0] = density;
    params.rayHit[idx.z][idx.y][idx.x][0] = numHits;
}

extern "C" __global__ void __miss__ms()
{
    RayPayload p;
    p.triId = 0;
    p.hitT = -1.0;
    setRayPayload(p);
    // TODO : fetch background or generate random background (maybe done outside of tracer)
}

extern "C" __global__ void __closesthit__ch()
{
    RayPayload p;
    p.hitT = optixGetRayTmax();
    p.triId = optixGetPrimitiveIndex();
    setRayPayload(p);
}
