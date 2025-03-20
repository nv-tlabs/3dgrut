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

#include <threedgutSlang.cuh>

struct model_InternalParams {
    static constexpr int GlobalParametersValueBufferIndex = 0;
    static constexpr int DensityRawParametersBufferIndex  = 1;
    static constexpr int FeaturesRawParametersBufferIndex = 2;

    static constexpr int DensityRawParametersGradientBufferIndex  = 0;
    static constexpr int FeaturesRawParametersGradientBufferIndex = 1;

    static constexpr int FeatureShDegreeValueOffset = 4; // in bytes, offset in the global parameters buffer
};

struct model_ExternalParams {
    static constexpr int FeaturesDim                 = 3;
    static constexpr float AlphaThreshold            = GAUSSIAN_PARTICLE_MIN_ALPHA;          // = 1.0/255.0
    static constexpr float MinTransmittanceThreshold = GAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD; // = 0.0001
    static constexpr int KernelDegree                = GAUSSIAN_PARTICLE_KERNEL_DEGREE;
    static constexpr float MinParticleKernelDensity  = GAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY;
    static const int RadianceMaxNumSphCoefficients   = PARTICLE_RADIANCE_NUM_COEFFS;
};

#include <3dgut/kernels/cuda/models/shRadiativeGaussianParticles.cuh>

using model_Particles = ShRadiativeGaussianVolumetricFeaturesParticles<gaussianParticle_Parameters_0,
                                                                       shRadiativeParticle_Parameters_0,
                                                                       model_InternalParams,
                                                                       model_ExternalParams,
                                                                       1>;

#include <3dgut/kernels/cuda/models/shGaussianModel.cuh>

using model_ = ShGaussian<model_Particles>;

struct TGUTProjectorParams {
    static constexpr float ParticleMinSensorZ    = 0.2;
    static constexpr float CovarianceDilation    = 0.3;
    static constexpr float AlphaThreshold        = model_::Particles::AlphaThreshold;
    static constexpr bool TightOpacityBounding   = GAUSSIAN_TIGHT_OPACITY_BOUNDING;
    static constexpr bool RectBounding           = GAUSSIAN_RECT_BOUNDING;
    static constexpr bool TileCulling            = GAUSSIAN_TILE_BASED_CULLING;
    static constexpr bool PerRayParticleFeatures = false;
    static constexpr float MaxDepthValue         = 3.4028235e+38;
    static constexpr bool GlobalZOrder           = GAUSSIAN_GLOBAL_Z_ORDER;
    static constexpr bool BackwardProjection     = false; // m_settings.renderMode == Settings::Splat
    static constexpr bool MipSplattingScaling    = true;
};

struct TGUTProjectionParams {
    static constexpr int NRollingShutterIterations = GAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS;
    static constexpr int D                         = 3;
    static constexpr float Alpha                   = GAUSSIAN_UT_ALPHA;
    static constexpr float Beta                    = GAUSSIAN_UT_BETA;
    static constexpr float Kappa                   = GAUSSIAN_UT_KAPPA;
    static constexpr float Delta                   = GAUSSIAN_UT_DELTA; ///< sqrt(Alpha*Alpha*(D+Kappa))
    static constexpr float ImageMarginFactor       = GAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR;
    static constexpr bool RequireAllSigmaPoints    = GAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID;
};

static_assert(TGUTProjectionParams::RequireAllSigmaPoints == false, "RequireAllSigmaPoints must be false");

#include <3dgut/kernels/cuda/renderers/gutProjector.cuh>

using TGUTProjector = GUTProjector<model_::Particles, TGUTProjectorParams, TGUTProjectionParams>;

struct TGUTRendererParams {
    static constexpr bool PerRayParticleFeatures = TGUTProjectorParams::PerRayParticleFeatures;
    static constexpr int KHitBufferSize          = GAUSSIAN_K_BUFFER_SIZE;
    static constexpr bool CustomBackward         = false;
};

#include <3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh>

using TGUTRenderer         = GUTKBufferRenderer<model_::Particles, TGUTRendererParams>;
using TGUTBackwardRenderer = GUTKBufferRenderer<model_::Particles, TGUTRendererParams, true>;

using TGUTModel = model_;

#include <3dgut/kernels/cuda/renderers/gutRenderer.cuh>
