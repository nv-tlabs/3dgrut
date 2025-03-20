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

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/utils/logger.h>

#include <json/json.hpp>
#include <pybind11_json/pybind11_json.hpp>

#include <deque>
#include <string>

using TTimestamp = int64_t;

class SplatRaster final {
private:
    uint8_t m_logLevel;
    threedgut::Logger m_logger;

    std::unique_ptr<threedgut::GUTRenderer> m_renderer;

    threedgut::GUTRenderer::Parameters m_parameters;

    class CudaTimer;

    bool m_enableKernelTimings  = false;
    std::map<std::string, float> m_timings;

    const size_t m_maxNumTimers = 256; // We only keep the most recent 256 timers
    std::deque<std::shared_ptr<CudaTimer>> m_timers;

public:
    SplatRaster(const nlohmann::json& config);

    ~SplatRaster();

    std::tuple<torch::Tensor, torch::Tensor>
    trace(uint32_t frameNumber, int numActiveFeatures,
          // Particles
          torch::Tensor particleDensity,
          torch::Tensor particleRadiance,
          // Rays
          torch::Tensor rayOrigin,
          torch::Tensor rayDirection,
          torch::Tensor rayTimestamp,
          // Sensor
          threedgut::TSensorModel sensor,
          TTimestamp startTimestamp,
          TTimestamp endTimestamp,
          torch::Tensor sensorsStartPose,
          torch::Tensor sensorsEndPose);

    std::tuple<torch::Tensor, torch::Tensor>
    traceBwd(uint32_t frameNumber, int numActiveFeatures,
             // Particles
             torch::Tensor particleDensity,
             torch::Tensor particleRadiance,
             // Rays
             torch::Tensor rayOrigin,
             torch::Tensor rayDirection,
             torch::Tensor rayTimestamp,
             // Sensor
             threedgut::TSensorModel sensorModel,
             TTimestamp startTimestamp,
             TTimestamp endTimestamp,
             torch::Tensor sensorsStartPose,
             torch::Tensor sensorsEndPose,
             // Gradients
             torch::Tensor rayRadianceDensity,
             torch::Tensor rayRadianceDensityGradient,
             torch::Tensor rayHitDistance,
             torch::Tensor rayHitDistanceGradient);

    std::map<std::string, float>
    collectTimes();
};
