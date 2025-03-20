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
#include <3dgut/renderer/gutRendererParameters.h>

__global__ void projectOnTiles(tcnn::uvec2 tileGrid,
                               uint32_t numParticles,
                               tcnn::ivec2 resolution,
                               threedgut::TSensorModel sensorModel,
                               tcnn::vec3 sensorWorldPosition,
                               tcnn::mat4x3 sensorViewMatrix,
                               threedgut::TSensorState sensorShutterState,
                               uint32_t* __restrict__ particlesTilesOffsetPtr,
                               tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                               float* __restrict__ particlesGlobalDepthPtr,
                               float* __restrict__ particlesPrecomputedFeaturesPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles) {

    TGUTProjector::eval(tileGrid,
                        numParticles,
                        resolution,
                        sensorModel,
                        sensorWorldPosition,
                        sensorViewMatrix,
                        sensorShutterState,
                        particlesTilesOffsetPtr,
                        particlesProjectedPositionPtr,
                        particlesProjectedConicOpacityPtr,
                        particlesProjectedExtentPtr,
                        particlesGlobalDepthPtr,
                        particlesPrecomputedFeaturesPtr,
                        {parameterMemoryHandles});
}

__global__ void expandTileProjections(tcnn::uvec2 tileGrid,
                                      uint32_t numParticles,
                                      tcnn::ivec2 resolution,
                                      threedgut::TSensorModel sensorModel,
                                      threedgut::TSensorState sensorState,
                                      const uint32_t* __restrict__ particlesTilesOffsetPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                      const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                      const float* __restrict__ particlesGlobalDepthPtr,
                                      const uint64_t* __restrict__ parameterMemoryHandles,
                                      uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                      uint32_t* __restrict__ unsortedTileParticleIdxPtr) {

    TGUTProjector::expand(tileGrid,
                          numParticles,
                          resolution,
                          sensorModel,
                          sensorState,
                          particlesTilesOffsetPtr,
                          particlesProjectedPositionPtr,
                          particlesProjectedConicOpacityPtr,
                          particlesProjectedExtentPtr,
                          particlesGlobalDepthPtr,
                          {parameterMemoryHandles},
                          unsortedTileDepthKeysPtr,
                          unsortedTileParticleIdxPtr);
}

__global__ void render(threedgut::RenderParameters params,
                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                       const uint32_t* __restrict__ sortedTileDataPtr,
                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                       const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                       tcnn::mat4x3 sensorToWorldTransform,
                       float* __restrict__ worldHitDistancePtr,
                       tcnn::vec4* __restrict__ radianceDensityPtr,
                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                       const float* __restrict__ particlesGlobalDepthPtr,
                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                       const uint64_t* __restrict__ parameterMemoryHandles) {

    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr, worldHitDistancePtr, sensorToWorldTransform);

    TGUTRenderer::eval(params,
                       ray,
                       sortedTileRangeIndicesPtr,
                       sortedTileDataPtr,
                       particlesProjectedPositionPtr,
                       particlesProjectedConicOpacityPtr,
                       particlesGlobalDepthPtr,
                       particlesPrecomputedFeaturesPtr,
                       {parameterMemoryHandles});

    TGUTModel::eval(params, ray, {parameterMemoryHandles});

    // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
    finalizeRay(ray, params, sensorRayOriginPtr, worldHitDistancePtr, radianceDensityPtr, sensorToWorldTransform);
}

__global__ void renderBackward(threedgut::RenderParameters params,
                               const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                               const uint32_t* __restrict__ sortedTileDataPtr,
                               const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                               const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                               tcnn::mat4x3 sensorToWorldTransform,
                               const float* __restrict__ worldHitDistancePtr,
                               const float* __restrict__ worldHitDistanceGradientPtr,
                               const tcnn::vec4* __restrict__ radianceDensityPtr,
                               const tcnn::vec4* __restrict__ radianceDensityGradientPtr,
                               tcnn::vec3* __restrict__ wordlRayOriginGradientPtr,
                               tcnn::vec3* __restrict__ worldRayDirectionGradientPtr,
                               const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               const float* __restrict__ particlesGlobalDepthPtr,
                               const float* __restrict__ particlesPrecomputedFeaturesPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles,
                               tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                               float* __restrict__ particlesGlobalDepthGradPtr,
                               float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                               const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    auto ray = initializeBackwardRay<TGUTRenderer::TRayPayloadBackward>(params,
                                                                        sensorRayOriginPtr,
                                                                        sensorRayDirectionPtr,
                                                                        worldHitDistancePtr,
                                                                        worldHitDistanceGradientPtr,
                                                                        radianceDensityPtr,
                                                                        radianceDensityGradientPtr,
                                                                        sensorToWorldTransform);

    TGUTModel::evalBackward(params, ray, {parameterMemoryHandles}, {parameterGradientMemoryHandles});

    TGUTBackwardRenderer::eval(params,
                               ray,
                               sortedTileRangeIndicesPtr,
                               sortedTileDataPtr,
                               particlesProjectedPositionPtr,
                               particlesProjectedConicOpacityPtr,
                               particlesGlobalDepthPtr,
                               particlesPrecomputedFeaturesPtr,
                               {parameterMemoryHandles},
                               particlesProjectedPositionGradPtr,
                               particlesProjectedConicOpacityGradPtr,
                               particlesGlobalDepthGradPtr,
                               particlesPrecomputedFeaturesGradPtr,
                               {parameterGradientMemoryHandles});
}

__global__ void projectBackward(tcnn::uvec2 tileGrid,
                                uint32_t numParticles,
                                tcnn::ivec2 resolution,
                                threedgut::TSensorModel sensorModel,
                                tcnn::vec3 sensorWorldPosition,
                                tcnn::mat4x3 sensorViewMatrix,
                                const uint32_t* __restrict__ particlesTilesCountPtr,
                                const uint64_t* __restrict__ parameterMemoryHandles,
                                const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                const float* __restrict__ particlesGlobalDepthGradPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    TGUTProjector::evalBackward(tileGrid,
                                numParticles,
                                resolution,
                                sensorModel,
                                sensorWorldPosition,
                                sensorViewMatrix,
                                particlesTilesCountPtr,
                                {parameterMemoryHandles},
                                particlesProjectedPositionGradPtr,
                                particlesProjectedConicOpacityGradPtr,
                                particlesGlobalDepthGradPtr,
                                particlesPrecomputedFeaturesPtr,
                                particlesPrecomputedFeaturesGradPtr,
                                {parameterGradientMemoryHandles});
}
