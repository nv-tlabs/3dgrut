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

struct MoGIndTracingParams
{
    PackedTensorAccessor32<float, 4> rayOri;    ///< ray origin
    PackedTensorAccessor32<float, 4> rayDir;    ///< ray direction
    PackedTensorAccessor32<float, 2> mogPos;    ///< gaussians position
    PackedTensorAccessor32<float, 2> mogRot;    ///< gaussians rotation (quaternions)
    PackedTensorAccessor32<float, 2> mogScl;    ///< gaussians scale
    PackedTensorAccessor32<float, 2> mogDns;    ///< gaussians density (opacity)
    PackedTensorAccessor32<int, 4> rayHitInd; ///< hits for the ray
    OptixTraversableHandle handle;

    OptixAabb aabb;
    float minTransmittance;
    float slabSpacing;

    float hitMinGaussianResponse;
    unsigned int sphDegree;
    uint2 frameBounds;
    
    unsigned int frameNumber;
    float alphaMaxValue;
    float alphaMinThreshold;
    float padding;
};
