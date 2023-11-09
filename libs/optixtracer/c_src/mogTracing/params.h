// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include "../accessor.h"

#include <optix.h>

enum MOGTracingMode
{
    MOGTracingModeNone = 0,
    MOGTracingGaussianHit = 1<<0, ///< use the position on the ray with highest gaussian response
    MOGTracingDefaultMode = MOGTracingModeNone
};

static constexpr int MOGPrimNumVert = 6; ///< octaHedron have 6 vertices
static constexpr int MOGPrimNumTri = 8; ///< octaHedron have 8 triangles

enum MOGTracingPipeline
{
    MOGTracingPipelineCH = 0,
    MOGTracingPipelineAH = 1,
    MOGTracingPipelineIS = 2,
    MOGTracingPipelineMLAT = 3,
    MOGTracingPipelineMBOIT = 4,
    MOGTracingPipelineHC = 5
};

struct MoGTracingParams
{
    PackedTensorAccessor32<float, 4> rayOri; ///< ray origin
    PackedTensorAccessor32<float, 4> rayDir; ///< ray direction
    PackedTensorAccessor32<float, 2> mogPos; ///< gaussians position
    PackedTensorAccessor32<float, 2> mogRot; ///< gaussians rotation (quaternions)
    PackedTensorAccessor32<float, 2> mogScl; ///< gaussians scale
    PackedTensorAccessor32<float, 2> mogDns; ///< gaussians density (opacity)
    PackedTensorAccessor32<float, 2> mogSph; ///< gaussians spherical harmonics coeffs
    PackedTensorAccessor32<float, 4> rayRad; ///< output integrated ray radiance
    PackedTensorAccessor32<float, 4> rayDns; ///< output integrated ray density
    PackedTensorAccessor32<float, 4> rayHit; ///< output estimated ray hit distance
    PackedTensorAccessor32<float, 2> mogHitCount; ///< output (only in HC pipeline) number of ray hits per mog
    OptixTraversableHandle handle;

    OptixAabb aabb;
    float minTransmittance;
    float slabSpacing;

    float hitMinGaussianResponse;
    float sphDegree;
    uint2 frameBounds;
    
    unsigned int frameNumber;
    float3 padding;
};
