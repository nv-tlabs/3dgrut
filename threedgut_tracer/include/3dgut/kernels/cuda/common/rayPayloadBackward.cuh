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

#include <3dgut/kernels/cuda/common/rayPayload.cuh>

template <int FeatN>
struct RayPayloadBackward : public RayPayload<FeatN> {
    float transmittanceBackward;
    float transmittanceGradient;
    float hitTBackward;
    float hitTGradient;
    tcnn::vec<FeatN> featuresBackward;
    tcnn::vec<FeatN> featuresGradient;
};

template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeBackwardRay(const threedgut::RenderParameters& params,
                                                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                        const float* __restrict__ worldHitDistancePtr,
                                                        const float* __restrict__ worldHitDistanceGradientPtr,
                                                        const TFeatureDensityElem* __restrict__ featuresDensityPtr,
                                                        const float* __restrict__ featuresDensityGradientPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform) {

    // NB : no backpropagation through the forward ray initialization / finalization
    RayPayloadT ray = initializeRay<RayPayloadT>(params,
                                                 sensorRayOriginPtr,
                                                 sensorRayDirectionPtr,
                                                 sensorToWorldTransform);

    if (ray.isAlive()) {
        constexpr uint32_t stride = RayPayloadT::FeatDim + 1;
        const uint32_t base = ray.idx * stride;
        // Forward features: fp16 when FEATURE_OUTPUT_HALF=1, fp32 otherwise.
        // Gradient buffer: always fp32 — keeps backward numerically stable regardless of forward dtype.
#if FEATURE_OUTPUT_HALF
        #pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            ray.featuresBackward[i] = __half2float(featuresDensityPtr[base + i]);
            ray.featuresGradient[i] = featuresDensityGradientPtr[base + i];
        }
        ray.transmittanceBackward = 1.f - __half2float(featuresDensityPtr[base + RayPayloadT::FeatDim]);
        ray.transmittanceGradient = -1.f * featuresDensityGradientPtr[base + RayPayloadT::FeatDim];
#else
        #pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            ray.featuresBackward[i] = featuresDensityPtr[base + i];
            ray.featuresGradient[i] = featuresDensityGradientPtr[base + i];
        }
        ray.transmittanceBackward = 1.f - featuresDensityPtr[base + RayPayloadT::FeatDim];
        ray.transmittanceGradient = -1.f * featuresDensityGradientPtr[base + RayPayloadT::FeatDim];
#endif
        ray.hitTBackward = worldHitDistancePtr[ray.idx];
        ray.hitTGradient = worldHitDistanceGradientPtr[ray.idx];
    }

    return ray;
}
