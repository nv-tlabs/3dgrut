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

#include <3dgut/kernels/cuda/common/cudaMath.cuh>
#include <3dgut/kernels/cuda/common/random.cuh>
#include <3dgut/renderer/renderParameters.h>

template <int FeatN>
struct RayPayload {
    static constexpr uint32_t FeatDim = FeatN;

    threedgut::TTimestamp timestamp;
    tcnn::vec3 origin;
    tcnn::vec3 direction;
    tcnn::vec2 tMinMax;
    float hitT;
    float transmittance;
    enum {
        Default             = 0,
        Valid               = 1 << 0,
        Alive               = 1 << 2,
        BackHit             = 1 << 3,
        BackHitProxySurface = 1 << 4,
        FrontHit            = 1 << 5
    };
    uint32_t flags;
    uint32_t idx;
    tcnn::vec<FeatN> features;

    __device__ __inline__ bool isAlive() const {
        return flags & Alive;
    }
    __device__ __inline__ void kill() {
        flags &= ~Alive;
    }
    __device__ __inline__ bool isValid() const {
        return flags & Valid;
    }
    __device__ __inline__ bool isFrontHit() const {
        return flags & FrontHit;
    }
    __device__ __inline__ void hitFront() {
        flags |= FrontHit;
    }
};

template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRay(const threedgut::RenderParameters& params,
                                                const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                const float* __restrict__ worldHitDistancePtr,
                                                const tcnn::mat4x3& sensorToWorldTransform) {
    const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    RayPayloadT ray;
    ray.flags = RayPayloadT::Default;
    if ((x >= params.resolution.x) || (y >= params.resolution.y)) {
        return ray;
    }
    ray.idx           = x + params.resolution.x * y;
    ray.hitT          = 0.0f;
    ray.transmittance = 1.0f;
    ray.features      = tcnn::vec<RayPayloadT::FeatDim>::zero();

    ray.origin    = sensorToWorldTransform * tcnn::vec4(sensorRayOriginPtr[ray.idx], 1.0f);
    ray.direction = tcnn::mat3(sensorToWorldTransform) * sensorRayDirectionPtr[ray.idx];

    ray.tMinMax   = params.objectAABB.ray_intersect(ray.origin, ray.direction);
    ray.tMinMax.x = fmaxf(ray.tMinMax.x, 0.0f);

    if (ray.tMinMax.y > ray.tMinMax.x) {
        ray.flags |= RayPayloadT::Valid | RayPayloadT::Alive;
    }

    return ray;
}

template <typename TRayPayload>
__device__ __inline__ void finalizeRay(const TRayPayload& ray,
                                       const threedgut::RenderParameters& params,
                                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                       float* __restrict__ worldHitDistancePtr,
                                       tcnn::vec4* __restrict__ radianceDensityPtr,
                                       const tcnn::mat4x3& sensorToWorldTransform) {
    if (!ray.isValid()) {
        return;
    }

    radianceDensityPtr[ray.idx] = {ray.features[0], ray.features[1], ray.features[2],  (1.0f - ray.transmittance)};
    if (ray.isFrontHit()) {
        worldHitDistancePtr[ray.idx] = ray.hitT;
    }
}
