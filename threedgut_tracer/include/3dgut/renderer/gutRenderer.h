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

#include <3dgut/renderer/renderParameters.h>
#include <3dgut/utils/cuda/cudaBuffer.h>

#include <json/json.hpp>

#include <memory>

namespace threedgut {

class GUTRenderer {
public:
    struct GutRenderForwardContext;

    struct Parameters {
        struct {
            uint32_t numParticles;
            int radianceSphDegree;
        } values;

        struct {
            void* dptrValuesBuffer;
            void* dptrDensityParameters;
            void* dptrRadianceParameters;
        } parameters;

        struct {
            void* dptrDensityGradients;
            void* dptrRadianceGradients;
        } gradients;

        threedgut::CudaBuffer parametersBuffer;
        threedgut::CudaBuffer gradientsBuffer;
        threedgut::CudaBuffer valuesBuffer;

        uint64_t* m_dptrParametersBuffer = nullptr;
        uint64_t* m_dptrGradientsBuffer  = nullptr;
    };

private:
    Logger m_logger;
    std::unique_ptr<GutRenderForwardContext> m_forwardContext;

public:
    GUTRenderer(const nlohmann::json& config, const Logger& logger);
    virtual ~GUTRenderer();

    /// march the scene according to the given camera and composite the result into the given cuda arrays
    Status renderForward(const RenderParameters& params,
                         const tcnn::vec3* wordlRayOriginCudaPtr,
                         const tcnn::vec3* worldRayDirectionCudaPtr,
                         float* worldHitDistanceCudaPtr,
                         tcnn::vec4* radianceDensityCudaPtr,
                         Parameters& parameters,
                         int cudaDeviceIndex,
                         cudaStream_t cudaStream);

    Status renderBackward(const RenderParameters& params,
                          const tcnn::vec3* wordlRayOriginCudaPtr,
                          const tcnn::vec3* worldRayDirectionCudaPtr,
                          const float* worldHitDistanceCudaPtr,
                          const float* worldHitDistanceGradientCudaPtr,
                          const tcnn::vec4* radianceDensityCudaPtr,
                          const tcnn::vec4* radianceDensityGradientCudaPtr,
                          tcnn::vec3* wordlRayOriginGradientCudaPtr,
                          tcnn::vec3* worldRayDirectionGradientCudaPtr,
                          Parameters& parameters,
                          int cudaDeviceIndex,
                          cudaStream_t cudaStream);
};

} // namespace threedgut
