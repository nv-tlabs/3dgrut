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

template <typename TParticles>
struct ShGaussian {

    using Particles = TParticles;

    template <typename TRay>
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       threedgut::MemoryHandles parameters) {
        if (ray.isValid()) {

            // mark the ray as front hit if the traversed volume is sufficiently opaque
            if (ray.transmittance < params.hitTransmittance) {
                ray.hitFront();
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackward(const threedgut::RenderParameters& params,
                                               TRay& ray,
                                               threedgut::MemoryHandles parameters,
                                               threedgut::MemoryHandles parametersGradient) {
        // NOOP
    }
};