// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "../math_utils.h"
#include "gsplat_sph_utils.h"
#include "params.h"

#ifdef __CUDACC__

static constexpr float GAlphaMax = 0.99f;
static constexpr float GAlphaMin = 0.005f;

static constexpr int SphDegree = 0;
static constexpr int SphNumCoeffs = 1;

static __device__ inline float computeGRayResponse(
    const float3& gPos, 
    const float4& gRot, 
    const float3& gScl, 
    const float3& rayOri, 
    const float3& rayDir
)
{
    // project ray in the gaussian
    float33 rot;
    rotationMatrix(make_float4(gRot.x,gRot.y,gRot.z,gRot.w), rot);
    const float3 iscl = make_float3(1/gScl.x,1/gScl.y,1/gScl.z);
    const float3 ro =  iscl*((rayOri - gPos)*rot);
    const float3 rd = safe_normalize(iscl*((rayDir*rot)));
    const float3 crod = cross(rd, ro);
    return expf(-0.5 * dot(crod,crod));
}

template <typename T>
static __device__ inline float computeGHitDistance(uint32_t gId, const float3& rayOri, const float3& rayDir, const T& params)
{
    const float3 gPos = make_float3(params.mogPos[gId][0],params.mogPos[gId][1],params.mogPos[gId][2]);
    const float4 gRot = make_float4(params.mogRot[gId][0],params.mogRot[gId][1],params.mogRot[gId][2],params.mogRot[gId][3]);
    const float3 gScl = make_float3(params.mogScl[gId][0],params.mogScl[gId][1],params.mogScl[gId][2]);
    
    float33 rot;
    rotationMatrix(make_float4(gRot.x,gRot.y,gRot.z,gRot.w), rot);
    const float3 iscl = make_float3(1/gScl.x,1/gScl.y,1/gScl.z);
    const float3 ro =  iscl*((rayOri - gPos)*rot);
    const float3 rd = safe_normalize(iscl*((rayDir*rot)));

    return dot(ro,ro) / dot(rd,ro);
}

#endif