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

#include <3dgut/kernels/cuda/models/gaussianParticles.cuh>
#include <3dgut/renderer/renderParameters.h>
template <typename TBuffer, bool TDifferentiable>
struct ShRadiativeGaussianParticlesBuffer {
    TBuffer* ptr = nullptr;
};

template <typename TBuffer>
struct ShRadiativeGaussianParticlesBuffer<TBuffer, true> {
    TBuffer* ptr     = nullptr;
    TBuffer* gradPtr = nullptr;
};

template <typename TBuffer, bool TDifferentiable, bool Enabled>
struct ShRadiativeGaussianParticlesOptionalBuffer {
};

template <typename TBuffer, bool TDifferentiable>
struct ShRadiativeGaussianParticlesOptionalBuffer<TBuffer, TDifferentiable, true> : ShRadiativeGaussianParticlesBuffer<TBuffer, TDifferentiable> {
};

template <typename Params,
          typename ExtParams,
          bool TDifferentiable = true>
struct ShRadiativeGaussianVolumetricFeaturesParticles : Params, public ExtParams {

    using DensityParameters    = threedgut::ParticeFetchedDensity;
    using DensityRawParameters = threedgut::ParticleDensity;

    __forceinline__ __device__ void initializeDensity(threedgut::MemoryHandles parameters) {
        static_assert(sizeof(DensityRawParameters) == sizeof(gaussianParticle_RawParameters_0), "Sizes must match for binary compatibility");
        static_assert(sizeof(DensityParameters) == sizeof(gaussianParticle_Parameters_0), "Sizes must match for binary compatibility");
        m_densityRawParameters.ptr =
            parameters.bufferPtr<DensityRawParameters>(Params::DensityRawParametersBufferIndex);
    }

    __forceinline__ __device__ void initializeDensityGradient(threedgut::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            m_densityRawParameters.gradPtr =
                parametersGradient.bufferPtr<DensityRawParameters>(Params::DensityRawParametersGradientBufferIndex);
        }
    };

    __forceinline__ __device__ DensityRawParameters fetchDensityRawParameters(uint32_t particleIdx) const {
        return m_densityRawParameters.ptr[particleIdx];
    }

    __forceinline__ __device__ DensityParameters fetchDensityParameters(uint32_t particleIdx) const {
        const auto parameters = particleDensityParameters(
            particleIdx,
            {reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr), nullptr});
        return *reinterpret_cast<const DensityParameters*>(&parameters);
    }

    __forceinline__ __device__ tcnn::vec3 fetchPosition(uint32_t particleIdx) const {
        return *(reinterpret_cast<const tcnn::vec3*>(&m_densityRawParameters.ptr[particleIdx].position));
    }

    __forceinline__ __device__ const tcnn::vec3& position(const DensityParameters& parameters) const {
        return *(reinterpret_cast<const tcnn::vec3*>(&parameters.position));
    }

    __forceinline__ __device__ const tcnn::vec3& scale(const DensityParameters& parameters) const {
        return *(reinterpret_cast<const tcnn::vec3*>(&parameters.scale));
    }

    __forceinline__ __device__ const tcnn::mat3& rotation(const DensityParameters& parameters) const {
        // slang uses row-major order (tcnn uses column-major order), so we return the rotation (not transposed)
        return *(reinterpret_cast<const tcnn::mat3*>(&parameters.rotationT));
    }

    __forceinline__ __device__ const float& opacity(const DensityParameters& parameters) const {
        return parameters.density;
    }

    __forceinline__ __device__ bool densityHit(const tcnn::vec3& rayOrigin,
                                               const tcnn::vec3& rayDirection,
                                               const DensityParameters& parameters,
                                               float& alpha,
                                               float& depth,
                                               tcnn::vec3* normal = nullptr) const {

        return particleDensityHit(*reinterpret_cast<const float3*>(&rayOrigin),
                                  *reinterpret_cast<const float3*>(&rayDirection),
                                  reinterpret_cast<const gaussianParticle_Parameters_0&>(parameters),
                                  &alpha,
                                  &depth,
                                  normal != nullptr,
                                  reinterpret_cast<float3*>(normal));
    }

    __forceinline__ __device__ float densityIntegrateHit(float alpha,
                                                         float& transmittance,
                                                         float depth,
                                                         float& integratedDepth,
                                                         const tcnn::vec3* normal     = nullptr,
                                                         tcnn::vec3* integratedNormal = nullptr) const {
        return particleDensityIntegrateHit(alpha,
                                           &transmittance,
                                           depth,
                                           &integratedDepth,
                                           normal != nullptr,
                                           normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(&normal),
                                           reinterpret_cast<float3*>(integratedNormal));
    }

    __forceinline__ __device__ float densityProcessHitFwdFromBuffer(const tcnn::vec3& rayOrigin,
                                                                    const tcnn::vec3& rayDirection,
                                                                    uint32_t particleIdx,
                                                                    float& transmittance,
                                                                    float& integratedDepth,
                                                                    tcnn::vec3* integratedNormal = nullptr) const {
        return particleDensityProcessHitFwdFromBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                      *reinterpret_cast<const float3*>(&rayDirection),
                                                      particleIdx,
                                                      {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr), nullptr, true}},
                                                      &transmittance,
                                                      &integratedDepth,
                                                      integratedNormal != nullptr,
                                                      reinterpret_cast<float3*>(integratedNormal));
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void densityProcessHitBwdToBuffer(const tcnn::vec3& rayOrigin,
                                                                 const tcnn::vec3& rayDirection,
                                                                 uint32_t particleIdx,
                                                                 float alpha,
                                                                 float alphaGrad,
                                                                 float& transmittance,
                                                                 float& transmittanceGrad,
                                                                 float depth,
                                                                 float& integratedDepth,
                                                                 float& integratedDepthGrad,
                                                                 const tcnn::vec3* normal         = nullptr,
                                                                 tcnn::vec3* integratedNormal     = nullptr,
                                                                 tcnn::vec3* integratedNormalGrad = nullptr

    ) const {
        if constexpr (TDifferentiable) {
            particleDensityProcessHitBwdToBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                 *reinterpret_cast<const float3*>(&rayDirection),
                                                 particleIdx,
                                                 {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr),
                                                   reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.gradPtr),
                                                   exclusiveGradient}},
                                                 alpha,
                                                 alphaGrad,
                                                 &transmittance,
                                                 &transmittanceGrad,
                                                 depth,
                                                 &integratedDepth,
                                                 &integratedDepthGrad,
                                                 normal != nullptr,
                                                 normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                                 reinterpret_cast<float3*>(integratedNormal),
                                                 reinterpret_cast<float3*>(integratedNormalGrad));
        }
    }

    __forceinline__ __device__ bool densityHitCustom(const tcnn::vec3& rayOrigin,
                                                     const tcnn::vec3& rayDirection,
                                                     uint32_t particleIdx,
                                                     float minHitDistance,
                                                     float maxHitDistance,
                                                     float maxParticleSquaredDistance,
                                                     float& hitDistance) const {
        return particleDensityHitCustom(*reinterpret_cast<const float3*>(&rayOrigin),
                                        *reinterpret_cast<const float3*>(&rayDirection),
                                        particleIdx,
                                        {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr), nullptr, true}},
                                        minHitDistance,
                                        maxHitDistance,
                                        maxParticleSquaredDistance,
                                        &hitDistance);
    }

    __forceinline__ __device__ bool densityHitInstance(const tcnn::vec3& canonicalRayOrigin,
                                                       const tcnn::vec3& canonicalUnormalizedRayDirection,
                                                       float minHitDistance,
                                                       float maxHitDistance,
                                                       float maxParticleSquaredDistance,
                                                       float& hitDistance

    ) const {
        return particleDensityHitInstance(*reinterpret_cast<const float3*>(&canonicalRayOrigin),
                                          *reinterpret_cast<const float3*>(&canonicalUnormalizedRayDirection),
                                          minHitDistance,
                                          maxHitDistance,
                                          maxParticleSquaredDistance,
                                          &hitDistance);
    }

    __forceinline__ __device__ tcnn::vec3 densityIncidentDirection(const DensityParameters& parameters,
                                                                   const tcnn::vec3& sourcePosition)

    {
        const auto incidentDirection = particleDensityIncidentDirection(reinterpret_cast<const gaussianParticle_Parameters_0&>(parameters),
                                                                        *reinterpret_cast<const float3*>(&sourcePosition));
        return *reinterpret_cast<const tcnn::vec3*>(&incidentDirection);
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void densityIncidentDirectionBwdToBuffer(uint32_t particlesIdx,
                                                                        const tcnn::vec3& sourcePosition)

    {
        particleDensityIncidentDirectionBwdToBuffer(particlesIdx,
                                                    {{reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.ptr),
                                                      reinterpret_cast<gaussianParticle_RawParameters_0*>(m_densityRawParameters.gradPtr),
                                                      exclusiveGradient}},
                                                    *reinterpret_cast<const float3*>(&sourcePosition));
    }

    using FeaturesParameters = shRadiativeParticle_Parameters_0;
    using TFeaturesVec       = typename tcnn::vec<ExtParams::FeaturesDim>;

    inline __device__ void initializeFeatures(threedgut::MemoryHandles parameters) {
        static_assert(ExtParams::FeaturesDim == 3, "Hardcoded 3-dimensional radiance because of Slang-Cuda interop");
        m_featureRawParameters.ptr = parameters.bufferPtr<float3>(Params::FeaturesRawParametersBufferIndex);
        m_featureActiveShDegree    = *reinterpret_cast<int*>(parameters.bufferPtr<uint8_t>(Params::GlobalParametersValueBufferIndex) + Params::FeatureShDegreeValueOffset);
    };

    inline __device__ void initializeFeaturesGradient(threedgut::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            m_featureRawParameters.gradPtr = parametersGradient.bufferPtr<float3>(Params::FeaturesRawParametersGradientBufferIndex);
        }
    };

    __forceinline__ __device__ TFeaturesVec featuresFromBuffer(uint32_t particleIdx,
                                                               const tcnn::vec3& incidentDirection) const {

        const auto features = particleFeaturesFromBuffer(particleIdx,
                                                         {{m_featureRawParameters.ptr, nullptr, true}, m_featureActiveShDegree},
                                                         *reinterpret_cast<const float3*>(&incidentDirection));
        return *reinterpret_cast<const TFeaturesVec*>(&features);
    }

    template <bool Clamped = true>
    __forceinline__ __device__ TFeaturesVec featuresCustomFromBuffer(uint32_t particleIdx,
                                                                     const tcnn::vec3& incidentDirection) const {
        const float3 gradu = threedgut::radianceFromSpH(m_featureActiveShDegree,
                                                        reinterpret_cast<const float3*>(&m_featureRawParameters.ptr[particleIdx * ExtParams::RadianceMaxNumSphCoefficients]),
                                                        *reinterpret_cast<const float3*>(&incidentDirection),
                                                        Clamped);
        return *reinterpret_cast<const TFeaturesVec*>(&gradu);
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void featuresBwdToBuffer(uint32_t particleIdx,
                                                        const TFeaturesVec& featuresGrad,
                                                        const tcnn::vec3& incidentDirection) const {

        particleFeaturesBwdToBuffer(particleIdx,
                                    {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                    *reinterpret_cast<const float3*>(&featuresGrad),
                                    *reinterpret_cast<const float3*>(&incidentDirection));
    }

    template <bool Atomic = false>
    __forceinline__ __device__ void featuresBwdCustomToBuffer(uint32_t particleIdx,
                                                              const TFeaturesVec& features,
                                                              const TFeaturesVec& featuresGrad,
                                                              const tcnn::vec3& incidentDirection) const {
        threedgut::radianceFromSpHBwd<Atomic>(m_featureActiveShDegree,
                                      *reinterpret_cast<const float3*>(&incidentDirection),
                                      *reinterpret_cast<const float3*>(&featuresGrad),
                                      reinterpret_cast<float3*>(&m_featureRawParameters.gradPtr[particleIdx * ExtParams::RadianceMaxNumSphCoefficients]),
                                      *reinterpret_cast<const float3*>(&features));
    }

    __forceinline__ __device__ void featureIntegrateFwd(float weight,
                                                        const TFeaturesVec& features,
                                                        TFeaturesVec& integratedFeatures) const {

        particleFeaturesIntegrateFwd(weight,
                                     *reinterpret_cast<const float3*>(&features),
                                     reinterpret_cast<float3*>(&integratedFeatures));
    }

    __forceinline__ __device__ void featuresIntegrateFwdFromBuffer(const tcnn::vec3& incidentDirection,
                                                                   float weight,
                                                                   uint32_t particleIdx, TFeaturesVec integratedFeatures) const {

        particleFeaturesIntegrateFwdFromBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                               weight,
                                               particleIdx,
                                               {{m_featureRawParameters.ptr, nullptr, true}, m_featureActiveShDegree},
                                               reinterpret_cast<float3*>(&integratedFeatures));
    }

    __forceinline__ __device__ void featuresIntegrateBwd(float alpha,
                                                         float& alphaGrad,
                                                         const TFeaturesVec& features,
                                                         TFeaturesVec& featuresGrad,
                                                         TFeaturesVec& integratedFeatures,
                                                         TFeaturesVec& integratedFeaturesGrad) const {
        if (TDifferentiable) {
            particleFeaturesIntegrateBwd(alpha,
                                         &alphaGrad,
                                         *reinterpret_cast<const float3*>(&features),
                                         reinterpret_cast<float3*>(&featuresGrad),
                                         reinterpret_cast<float3*>(&integratedFeatures),
                                         reinterpret_cast<float3*>(&integratedFeaturesGrad));
        }
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void featuresIntegrateBwdToBuffer(const tcnn::vec3& incidentDirection,
                                                                 float alpha,
                                                                 float& alphaGrad,
                                                                 uint32_t particleIdx,
                                                                 const TFeaturesVec& features,
                                                                 TFeaturesVec& integratedFeatures,
                                                                 TFeaturesVec& integratedFeaturesGrad) const {

        if (TDifferentiable) {
            particleFeaturesIntegrateBwdToBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                                 alpha,
                                                 &alphaGrad,
                                                 particleIdx,
                                                 {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                                 *reinterpret_cast<const float3*>(&features),
                                                 reinterpret_cast<float3*>(&integratedFeatures),
                                                 reinterpret_cast<float3*>(&integratedFeaturesGrad));
        }
    }

    template <bool PerRayRadiance>
    __forceinline__ __device__ bool processHitFwd(const tcnn::vec3& rayOrigin,
                                                  const tcnn::vec3& rayDirection,
                                                  uint32_t particleIdx,
                                                  const TFeaturesVec* particleFeaturesPtr,
                                                  float& transmittance,
                                                  TFeaturesVec& features,
                                                  float& hitT) const {
        return threedgut::processHitFwd<ExtParams::KernelDegree, false, PerRayRadiance>(
            reinterpret_cast<const float3&>(rayOrigin),
            reinterpret_cast<const float3&>(rayDirection),
            particleIdx,
            m_densityRawParameters.ptr,
            PerRayRadiance ? reinterpret_cast<const float*>(m_featureRawParameters.ptr) : reinterpret_cast<const float*>(particleFeaturesPtr),
            ExtParams::MinParticleKernelDensity,
            ExtParams::AlphaThreshold,
            m_featureActiveShDegree,
            &transmittance,
            reinterpret_cast<float3*>(&features),
            &hitT,
            nullptr);
    }

    template <bool PerRayRadiance>
    __forceinline__ __device__ void processHitBwd(const tcnn::vec3& rayOrigin,
                                                  const tcnn::vec3& rayDirection,
                                                  uint32_t particleIdx,
                                                  const DensityRawParameters& densityRawParameters,
                                                  DensityRawParameters* densityRawParametersGrad,
                                                  const TFeaturesVec& particleFeatures,
                                                  TFeaturesVec* particleFeaturesGradPtr,
                                                  float& transmittance,
                                                  float transmittanceBackward,
                                                  float transmittanceGradient,
                                                  TFeaturesVec& features,
                                                  const TFeaturesVec& featuresBackward,
                                                  const TFeaturesVec& featuresGradient,
                                                  float& hitT,
                                                  float hitTBackward,
                                                  float hitTGradient) const {

        threedgut::processHitBwd<ExtParams::KernelDegree, false, PerRayRadiance>(
            reinterpret_cast<const float3&>(rayOrigin),
            reinterpret_cast<const float3&>(rayDirection),
            particleIdx,
            reinterpret_cast<const threedgut::ParticleDensity&>(densityRawParameters),
            reinterpret_cast<threedgut::ParticleDensity*>(densityRawParametersGrad),
            PerRayRadiance ? reinterpret_cast<const float*>(m_featureRawParameters.ptr) : reinterpret_cast<const float*>(particleFeatures.data()),
            PerRayRadiance ? reinterpret_cast<float*>(m_featureRawParameters.gradPtr) : reinterpret_cast<float*>(particleFeaturesGradPtr),
            ExtParams::MinParticleKernelDensity,
            ExtParams::AlphaThreshold,
            ExtParams::MinTransmittanceThreshold,
            m_featureActiveShDegree,
            transmittanceBackward,
            transmittance,
            transmittanceGradient,
            reinterpret_cast<const float3&>(featuresBackward),
            reinterpret_cast<float3&>(features),
            reinterpret_cast<const float3&>(featuresGradient),
            hitTBackward,
            hitT,
            hitTGradient);
    }

    template <bool synchedThread = true>
    __forceinline__ __device__ void processHitBwdUpdateFeaturesGradient(uint32_t particleIdx, TFeaturesVec& featuresGrad, TFeaturesVec* featuresGradSum, uint32_t tileThreadIdx) {
        if constexpr (synchedThread) {
            // Perform warp reduction
#pragma unroll
            for (int mask = 1; mask < warpSize; mask *= 2) {
#pragma unroll
                for (int i = 0; i < ExtParams::FeaturesDim; ++i) {
                    featuresGrad[i] += __shfl_xor_sync(0xffffffff, featuresGrad[i], mask);
                }
            }

            // First thread in the warp performs the atomic add
            if ((tileThreadIdx & (warpSize - 1)) == 0) {
#pragma unroll
                for (int i = 0; i < ExtParams::FeaturesDim; i++) {
                    atomicAdd(&featuresGradSum[particleIdx][i], featuresGrad[i]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < ExtParams::FeaturesDim; ++i) {
                atomicAdd(&featuresGradSum[particleIdx][i], featuresGrad[i]);
            }
        }
    }

    template <bool synchedThread = true>
    __forceinline__ __device__ void processHitBwdUpdateDensityGradient(uint32_t particleIdx, DensityRawParameters& densityRawParameters, uint32_t tileThreadIdx) {
        if constexpr (synchedThread) {
            // Perform warp reduction
#pragma unroll
            for (int mask = 1; mask < warpSize; mask *= 2) {
                densityRawParameters.position.x += __shfl_xor_sync(0xffffffff, densityRawParameters.position.x, mask);
                densityRawParameters.position.y += __shfl_xor_sync(0xffffffff, densityRawParameters.position.y, mask);
                densityRawParameters.position.z += __shfl_xor_sync(0xffffffff, densityRawParameters.position.z, mask);
                densityRawParameters.density += __shfl_xor_sync(0xffffffff, densityRawParameters.density, mask);
                densityRawParameters.quaternion.x += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.x, mask);
                densityRawParameters.quaternion.y += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.y, mask);
                densityRawParameters.quaternion.z += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.z, mask);
                densityRawParameters.quaternion.w += __shfl_xor_sync(0xffffffff, densityRawParameters.quaternion.w, mask);
                densityRawParameters.scale.x += __shfl_xor_sync(0xffffffff, densityRawParameters.scale.x, mask);
                densityRawParameters.scale.y += __shfl_xor_sync(0xffffffff, densityRawParameters.scale.y, mask);
                densityRawParameters.scale.z += __shfl_xor_sync(0xffffffff, densityRawParameters.scale.z, mask);
            }

            // First thread in the warp performs the atomic add
            if ((tileThreadIdx & (warpSize - 1)) == 0) {
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.x, densityRawParameters.position.x);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.y, densityRawParameters.position.y);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.z, densityRawParameters.position.z);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].density, densityRawParameters.density);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.x, densityRawParameters.quaternion.x);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.y, densityRawParameters.quaternion.y);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.z, densityRawParameters.quaternion.z);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.w, densityRawParameters.quaternion.w);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.x, densityRawParameters.scale.x);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.y, densityRawParameters.scale.y);
                atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.z, densityRawParameters.scale.z);
            }
        } else {
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.x, densityRawParameters.position.x);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.y, densityRawParameters.position.y);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].position.z, densityRawParameters.position.z);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].density, densityRawParameters.density);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.x, densityRawParameters.quaternion.x);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.y, densityRawParameters.quaternion.y);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.z, densityRawParameters.quaternion.z);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].quaternion.w, densityRawParameters.quaternion.w);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.x, densityRawParameters.scale.x);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.y, densityRawParameters.scale.y);
            atomicAdd(&m_densityRawParameters.gradPtr[particleIdx].scale.z, densityRawParameters.scale.z);
        }
    }

private:
    ShRadiativeGaussianParticlesBuffer<DensityRawParameters, TDifferentiable>
        m_densityRawParameters;

    int m_featureActiveShDegree = 0;
    ShRadiativeGaussianParticlesBuffer<float3, TDifferentiable> m_featureRawParameters;
};
