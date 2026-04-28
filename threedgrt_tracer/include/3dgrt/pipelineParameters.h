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

#include <optix.h>

#include <3dgrt/particleDensity.h>
#include <3dgrt/pipelineDefinitions.h>
#include <3dgrt/tensorAccessor.h>

// Per-particle feature storage element type. fp16 when PARTICLE_FEATURE_HALF=1, else fp32.
// Gradient buffer for per-particle features is always fp32.
#ifndef PARTICLE_FEATURE_HALF
#define PARTICLE_FEATURE_HALF 0
#endif
// Per-ray integrated feature output element type. fp16 when FEATURE_OUTPUT_HALF=1, else fp32.
// Gradient buffer for the integrated feature output is always fp32.
#ifndef FEATURE_OUTPUT_HALF
#define FEATURE_OUTPUT_HALF 0
#endif
#if PARTICLE_FEATURE_HALF || FEATURE_OUTPUT_HALF
#include <cuda_fp16.h>
#endif
#if PARTICLE_FEATURE_HALF
using TParticleFeatureElem = __half;
#else
using TParticleFeatureElem = float;
#endif
#if FEATURE_OUTPUT_HALF
using TRayFeatureElem = __half;
#else
using TRayFeatureElem = float;
#endif

struct PipelineParameters {
    float4 rayToWorld[3];                          ///< float3x4 ray to world transformation (row-major)
    PackedTensorAccessor32<float, 4> rayOrigin;    ///< ray origin
    PackedTensorAccessor32<float, 4> rayDirection; ///< ray direction

    const ParticleDensity* particleDensity;         ///< position, scale, quaternions, density
    const TParticleFeatureElem* particleFeatures;   ///< per-particle features (fp16 when PARTICLE_FEATURE_HALF)
    const void* particleExtendedData;               ///< pipeline specific particle data
    int32_t* particleVisibility;                    ///< pipeline specific particle data

    PackedTensorAccessor32<TRayFeatureElem, 4> rayFeatures; ///< integrated ray features (fp16 when FEATURE_OUTPUT_HALF)
    PackedTensorAccessor32<float, 4> rayDensity;     ///< output integrated ray density
    PackedTensorAccessor32<float, 4> rayHitDistance; ///< output integrated ray hit distance
    PackedTensorAccessor32<float, 4> rayNormal;      ///< output integrated ray normal
    PackedTensorAccessor32<float, 4> rayHitsCount;   ///< output (only in AH pipeline) number of hits per ray

    OptixTraversableHandle handle;
    OptixAabb aabb;

    float minTransmittance;
    float hitMinGaussianResponse;
    static constexpr float hitMaxParticleSquaredDistance = 9.f; ///< by design
    float alphaMinThreshold;
    unsigned int sphDegree;

    // Feature-based radiance dimensions
    static constexpr unsigned int ParticleFeatureDim   = PARTICLE_FEATURE_DIM;   ///< Total feature dim per particle (buffer stride K)
    static constexpr unsigned int RayFeatureDim        = RAY_FEATURE_DIM;       ///< Per-ray (decoder input N)
    static constexpr unsigned int FeatureTransformType = FEATURE_TRANSFORM_TYPE; ///< 0=SH, 1=learned

    uint2 frameBounds;
    unsigned int frameNumber;
    int gPrimNumTri;

    static constexpr unsigned int MaxNumHitPerTrace = 16;

#ifdef PARTICLE_PRIMITIVE_TYPE
    static constexpr bool CustomPrimitive   = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingCustom);
    static constexpr bool InstancePrimitive = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingInstances);
    static constexpr bool SurfelPrimitive   = (PARTICLE_PRIMITIVE_TYPE == MOGPrimitiveTypes::MOGTracingTriSurfel);
    static constexpr bool ClampedPrimitive  = PARTICLE_PRIMITIVE_CLAMPED;
#endif

#ifdef PARTICLE_KERNEL_DEGREE
    static constexpr int ParticleKernelDegree = PARTICLE_KERNEL_DEGREE;
#endif

#ifdef __CUDACC__
    inline __device__ float3 rayWorldOrigin(const uint3& idx) const {
        float3 origin = make_float3(rayOrigin[idx.z][idx.y][idx.x][0],
                                    rayOrigin[idx.z][idx.y][idx.x][1],
                                    rayOrigin[idx.z][idx.y][idx.x][2]);
        return make_float3(
            rayToWorld[0].x * origin.x + rayToWorld[0].y * origin.y + rayToWorld[0].z * origin.z + rayToWorld[0].w,
            rayToWorld[1].x * origin.x + rayToWorld[1].y * origin.y + rayToWorld[1].z * origin.z + rayToWorld[1].w,
            rayToWorld[2].x * origin.x + rayToWorld[2].y * origin.y + rayToWorld[2].z * origin.z + rayToWorld[2].w);
    }

    inline __device__ float3 rayWorldDirection(const uint3& idx) const {
        float3 direction = make_float3(rayDirection[idx.z][idx.y][idx.x][0],
                                       rayDirection[idx.z][idx.y][idx.x][1],
                                       rayDirection[idx.z][idx.y][idx.x][2]);
        return make_float3(
            rayToWorld[0].x * direction.x + rayToWorld[0].y * direction.y + rayToWorld[0].z * direction.z,
            rayToWorld[1].x * direction.x + rayToWorld[1].y * direction.y + rayToWorld[1].z * direction.z,
            rayToWorld[2].x * direction.x + rayToWorld[2].y * direction.y + rayToWorld[2].z * direction.z);
    }
#endif
};

struct PipelineBackwardParameters : PipelineParameters {
    PackedTensorAccessor32<float, 4> rayFeaturesGrad;    ///< integrated ray features gradient (fp32)
    PackedTensorAccessor32<float, 4> rayDensityGrad;     ///< integrated ray density gradient
    PackedTensorAccessor32<float, 4> rayHitDistanceGrad; ///< integrated ray hit distance gradient
    PackedTensorAccessor32<float, 4> rayNormalGrad;      ///< integrated ray hit distance gradient

    ParticleDensity* particleDensityGrad; ///< output position, scale, quaternions, density gradient
    float* particleFeaturesGrad;          ///< per-particle features gradient (fp32)
};
