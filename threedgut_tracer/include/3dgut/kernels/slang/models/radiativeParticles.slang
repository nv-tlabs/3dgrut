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

// Radiative Particles Wrapper
// Conditionally includes the appropriate feature implementation based on FEATURE_TRANSFORM_TYPE

#if FEATURE_TRANSFORM_TYPE == 0
    // Spherical Harmonics mode
    #include <3dgut/kernels/slang/models/shRadiativeParticles.slang>
#elif FEATURE_TRANSFORM_TYPE == 1
    // Post-MLP radiance mode
    #include <3dgut/kernels/slang/models/neuralHarmonicFeaturesParticle.slang>
#else
    #error "Unknown FEATURE_TRANSFORM_TYPE. Must be 0 (SH) or 1 (neural_harmonic_features)"
#endif
