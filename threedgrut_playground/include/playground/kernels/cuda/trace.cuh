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

#pragma once
#ifdef __PLAYGROUND__MODE__

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif

#include <optix.h>
#include <3dgrt/kernels/cuda/3dgrtTracer.cuh>
#include <playground/pipelineDefinitions.h>
#include <playground/pipelineParameters.h>
#include <playground/kernels/cuda/mathUtils.cuh>

extern "C"
{
    #ifndef __PLAYGROUND__PARAMS__
    __constant__ PlaygroundPipelineParameters params;
    #define __PLAYGROUND__PARAMS__ 1
    #endif
}

struct HybridRayPayload {
    float t_hit;               // ray t of latest intersection
    float3 rayOri;             // next ray origin to use
    float3 rayDir;             // next ray dir to use, if ray was reflected, refracted, etc
    unsigned int numBounces;   // current number of reflectance bounces
    unsigned int rndSeed;      // random seed for current ray

    // PBR params
    float3 accumulatedColor;    // Amount of RGB color accumulated so far by ray
    float  accumulatedAlpha;    // Amount of density accumulated by the ray so far. Solid mesh faces count as opaque.
    float3 directLight;         // Total light to be reflected off the surfaces seen so far
    unsigned int pbrNumBounces; // Current number of PBR ray iterations, to limit reflections, refractions, etc
    bool rayMissed;             // True if ray missed

    float blockingRadiance;     // Total radiance accumulated only by volumetric radiance integration so far
    float lastPBRTransmittance; // Transmittance of last PBR surface hit, accumulated together with volumetric density
    float3 lastRayOri;          // Last ray origin used to trace gaussians
    float3 lastRayDir;          // Last ray direction used to trace gaussians

    RayData* rayData;
};

constexpr float epsT = 1e-9;                   // Minimal offset to ray t to avoid zero t
constexpr float TRACE_MESH_TMIN = 1e-5;
constexpr float TRACE_MESH_TMAX = 1e5;

static __device__ __forceinline__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __device__ __forceinline__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __device__ __forceinline__ unsigned int getNextTraceState()
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

    return params.trace_state[idx.z][ry][rx][0];
}

static __device__ __forceinline__ HybridRayPayload* getRayPayload()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<HybridRayPayload*>(unpackPointer(u0, u1));
}

static __device__ __forceinline__ void setNextTraceState(unsigned int traceState)
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels
    params.trace_state[idx.z][ry][rx][0] = traceState;
}

static __device__ __forceinline__ void clearOutputBuffers()
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

    params.rayRadiance[idx.z][ry][rx][0] = 0.0f;
    params.rayRadiance[idx.z][ry][rx][1] = 0.0f;
    params.rayRadiance[idx.z][ry][rx][2] = 0.0f;
    params.rayDensity[idx.z][ry][rx][0] = 0.0f;
}

static __device__ __forceinline__ void writeRadianceDensityToOutputBuffer(float4 radiance)
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

    params.rayRadiance[idx.z][ry][rx][0] = radiance.x;
    params.rayRadiance[idx.z][ry][rx][1] = radiance.y;
    params.rayRadiance[idx.z][ry][rx][2] = radiance.z;
    params.rayDensity[idx.z][ry][rx][0] = radiance.w;
}

static __device__ __forceinline__ void accumulateRadianceToOutputBuffer(float3 radiance)
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

    params.rayRadiance[idx.z][ry][rx][0] += radiance.x;
    params.rayRadiance[idx.z][ry][rx][1] += radiance.y;
    params.rayRadiance[idx.z][ry][rx][2] += radiance.z;
}

static __device__ __forceinline__ void accumulateRadianceDensityToOutputBuffer(float4 radiance)
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

    params.rayRadiance[idx.z][ry][rx][0] += radiance.x;
    params.rayRadiance[idx.z][ry][rx][1] += radiance.y;
    params.rayRadiance[idx.z][ry][rx][2] += radiance.z;
    params.rayDensity[idx.z][ry][rx][0] += radiance.w;
}

static __device__ __forceinline__ void writeUpdatedRaysToBuffer(const float3 rayOri, const float3 rayDir)
{
    const uint3 idx = optixGetLaunchIndex();
    const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
    const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels
    params.rayOrigin[idx.z][ry][rx][0] = rayOri.x;
    params.rayOrigin[idx.z][ry][rx][1] = rayOri.y;
    params.rayOrigin[idx.z][ry][rx][2] = rayOri.z;
    params.rayDirection[idx.z][ry][rx][0] = rayDir.x;
    params.rayDirection[idx.z][ry][rx][1] = rayDir.y;
    params.rayDirection[idx.z][ry][rx][2] = rayDir.z;
}

static __device__ __forceinline__ void traceMesh(const float3 rayOri, const float3 rayDir, HybridRayPayload* payload)
{
    setNextTraceState(PGRNDTracePrimitivesPass);

    unsigned int p0, p1;
    packPointer(payload, p0, p1);
    optixTrace(
        params.triHandle,
        rayOri,
        rayDir,
        TRACE_MESH_TMIN,          // Min intersection distance
        TRACE_MESH_TMAX,          // Max intersection distance
        0.0f,                     // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, // | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        0, // SBT offset   -- See SBT discussion
        1, // SBT stride   -- See SBT discussion
        0, // missSBTIndex -- See SBT discussion
        p0, p1
    );
}

static __device__ __forceinline__ float4 traceGaussians(
    RayData& rayData,
    const float3& rayOrigin,
    const float3& rayDirection,
    const float tmin,
    const float tmax,
    HybridRayPayload* payload) {

   const uint3 idx = optixGetLaunchIndex();
   const int rx = fminf(idx.x, params.frameBounds.x);  // Ray coordinates in pixels
   const int ry = fminf(idx.y, params.frameBounds.y);  // Ray coordinates in pixels

   if (params.playgroundOpts & PGRNDRenderDisableGaussianTracing)
       return make_float4(0.0);

   // Copy RayData, to avoid writing the output buffer by this pass
   RayData prevRayData = rayData;
   setNextTraceState(PGRNDTraceRTGaussiansPass);
   traceVolumetricGS(rayData, rayOrigin, rayDirection, tmin, tmax);

   // The difference in the output buffer is the result of this trace path
   float4 accumulated_radiance = make_float4(
        rayData.radiance.x - prevRayData.radiance.x,
        rayData.radiance.y - prevRayData.radiance.y,
        rayData.radiance.z - prevRayData.radiance.z,
        rayData.density - prevRayData.density
   );

    payload->lastRayOri = rayOrigin;
    payload->lastRayDir = rayDirection;

   return accumulated_radiance;
}


static __device__ __forceinline__ float3 getBackgroundColor(const float3 rayDir)
{
    if (!params.useEnvmap) {
        return params.backgroundColor;
    }
    else {
        float theta = atan2(rayDir.x, rayDir.z);
        float phi = M_PIf * 0.5f - acosf(rayDir.y);
        float u = (theta + M_PIf) * (0.5f * M_1_PIf);
        float v = 0.5f * (1.0f + sin(phi));
        float4 env = tex2D<float4>(params.envmap, u, v);
        return make_float3(env.x, env.y, env.z);
    }
}

#endif