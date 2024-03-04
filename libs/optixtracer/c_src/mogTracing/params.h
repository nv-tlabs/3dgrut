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
#include "paramDefs.h"

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
    PackedTensorAccessor32<float, 2> mogWeightSum; ///< output (only in AH pipeline) sum of all weights that a gaussian contributed during render pass
    OptixTraversableHandle handle;

    OptixAabb aabb;
    float minTransmittance;
    float slabSpacing;

    float hitMinGaussianResponse;
    unsigned int sphDegree;
    uint2 frameBounds;
    
    unsigned int frameNumber;
    int gPrimNumTri;
    float alphaMaxValue;
    float alphaMinThreshold;
};

struct MoGTracingBwdParams
{
    PackedTensorAccessor32<float, 4> rayOri; ///< ray origin
    PackedTensorAccessor32<float, 4> rayDir; ///< ray direction
    PackedTensorAccessor32<float, 4> rayRad; ///< ray radiance (as computed by the forward pass)
    PackedTensorAccessor32<float, 4> rayDns; ///< ray density (as computed by the forward pass)
    PackedTensorAccessor32<float, 4> rayHit; ///< ray hit distance (as computed by the forward pass)
    PackedTensorAccessor32<float, 2> mogPos; ///< gaussians position
    PackedTensorAccessor32<float, 2> mogRot; ///< gaussians rotation (quaternions)
    PackedTensorAccessor32<float, 2> mogScl; ///< gaussians scale
    PackedTensorAccessor32<float, 2> mogDns; ///< gaussians density (opacity)
    PackedTensorAccessor32<float, 2> mogSph; ///< gaussians spherical harmonics coeffs
    PackedTensorAccessor32<float, 4> rayRadGrd; ///< integrated ray radiance gradient
    PackedTensorAccessor32<float, 4> rayDnsGrd; ///< integrated ray density gradient
    PackedTensorAccessor32<float, 4> rayHitGrd; ///< estimated ray hit distance gradient
    PackedTensorAccessor32<float, 4> rayError;  ///< integrated error on each ray (NOT A GRADIENT, we abuse the backward shader to compute compute something that is not a gradient)
    PackedTensorAccessor32<float, 2> mogPosGrd; ///< output gaussians position gradient
    PackedTensorAccessor32<float, 2> mogRotGrd; ///< output gaussians rotation (quaternions) gradient
    PackedTensorAccessor32<float, 2> mogSclGrd; ///< output gaussians scale gradient
    PackedTensorAccessor32<float, 2> mogDnsGrd; ///< output gaussians density (opacity) gradient
    PackedTensorAccessor32<float, 2> mogSphGrd; ///< output gaussians spherical harmonics coeffs gradient
    PackedTensorAccessor32<float, 2> mogErrorBack; ///< output gaussians rayError*weight
    OptixTraversableHandle handle;

    OptixAabb aabb;
    float minTransmittance;
    float slabSpacing;

    float hitMinGaussianResponse;
    unsigned int sphDegree;
    uint2 frameBounds;
    
    unsigned int frameNumber;
    int gPrimNumTri;
    float alphaMaxValue;
    float alphaMinThreshold;
  
};
