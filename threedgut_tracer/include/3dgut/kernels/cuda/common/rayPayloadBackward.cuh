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
                                                        const tcnn::vec<RayPayloadT::FeatDim + 1>* __restrict__ featuresDensityPtr,
                                                        const tcnn::vec<RayPayloadT::FeatDim + 1>* __restrict__ featuresDensityGradientPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform) {

    // NB : no backpropagation through the forward ray initialization / finalization
    RayPayloadT ray = initializeRay<RayPayloadT>(params,
                                                 sensorRayOriginPtr,
                                                 sensorRayDirectionPtr,
                                                 worldHitDistancePtr,
                                                 sensorToWorldTransform);

    if (ray.isAlive()) {
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensity         = featuresDensityPtr[ray.idx];
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensityGradient = featuresDensityGradientPtr[ray.idx];
        ray.transmittanceBackward                                         = 1.f - featuresDensity[RayPayloadT::FeatDim];
        ray.transmittanceGradient                                         = -1.f * featuresDensityGradient[RayPayloadT::FeatDim];
        ray.hitTBackward                                                  = worldHitDistancePtr[ray.idx];
        ray.hitTGradient                                                  = worldHitDistanceGradientPtr[ray.idx];
        ray.featuresBackward                                              = threedgut::sliceVec<0, RayPayloadT::FeatDim>(featuresDensity);
        ray.featuresGradient                                              = threedgut::sliceVec<0, RayPayloadT::FeatDim>(featuresDensityGradient);
    }

    return ray;
}
