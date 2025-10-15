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

#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

struct HitParticle {
    static constexpr float InvalidHitT = -1.0f;
    int idx                            = -1;
    float hitT                         = InvalidHitT;
    float alpha                        = 0.0f;
};

template <int K>
struct HitParticleKBuffer {
    __device__ HitParticleKBuffer() {
        m_numHits = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle();
        }
    }

    // insert a new hit into the kbuffer.
    // if the buffer is full overwrite the closest entry
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        if (isFull) {
            m_kbuffer[0].hitT = HitParticle::InvalidHitT;
        } else {
            m_numHits++;
        }
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
        }
    }

    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    inline __device__ uint32_t numHits() const {
        return m_numHits;
    }

    inline __device__ bool full() const {
        return m_numHits == K;
    }

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];
    }

private:
    HitParticle m_kbuffer[K];
    uint32_t m_numHits;
};

template <>
struct HitParticleKBuffer<0> {
    constexpr inline __device__ void insert(HitParticle& hitParticle) const {}
    constexpr inline __device__ HitParticle operator[](int) const { return HitParticle(); }
    constexpr inline __device__ uint32_t numHits() const { return 0; }
    constexpr inline __device__ bool full() const { return true; }
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const { return hitParticle; }
};

template <typename Particles, typename Params, bool Backward = false>
struct GUTKBufferRenderer : Params {

    using DensityParameters    = typename Particles::DensityParameters;
    using DensityRawParameters = typename Particles::DensityRawParameters;
    using TFeaturesVec         = typename Particles::TFeaturesVec;

    using TRayPayload         = RayPayload<Particles::FeaturesDim>;
    using TRayPayloadBackward = RayPayloadBackward<Particles::FeaturesDim>;

    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };

    struct PrefetchedRawParticleData {
        uint32_t idx;
        TFeaturesVec features;
        DensityRawParameters densityParameters;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles,
        const TFeaturesVec* __restrict__ particleFeatures,
        TFeaturesVec* __restrict__ particleFeaturesGradient) {

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;
            if constexpr (Params::PerRayParticleFeatures) {
                particles.featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                              hitParticle.alpha,
                                                              hitAlphaGrad,
                                                              hitParticle.idx,
                                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                              ray.featuresBackward,
                                                              ray.featuresGradient);
            } else {
                TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                particles.featuresIntegrateBwd(hitParticle.alpha,
                                               hitAlphaGrad,
                                               particleFeatures[hitParticle.idx],
                                               particleFeaturesGradientVec,
                                               ray.featuresBackward,
                                               ray.featuresGradient);
#pragma unroll
                for (int i = 0; i < Particles::FeaturesDim; ++i) {
                    atomicAdd(&(particleFeaturesGradient[hitParticle.idx][i]), particleFeaturesGradientVec[i]);
                }
            }

            particles.densityProcessHitBwdToBuffer<false>(ray.origin,
                                                          ray.direction,
                                                          hitParticle.idx,
                                                          hitParticle.alpha,
                                                          hitAlphaGrad,
                                                          ray.transmittanceBackward,
                                                          ray.transmittanceGradient,
                                                          hitParticle.hitT,
                                                          ray.hitTBackward,
                                                          ray.hitTGradient);

            ray.transmittance *= (1.0 - hitParticle.alpha);

        } else {
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,
                                              ray.transmittance,
                                              hitParticle.hitT,
                                              ray.hitT);

            particles.featureIntegrateFwd(hitWeight,
                                          Params::PerRayParticleFeatures ? particles.featuresFromBuffer(hitParticle.idx, ray.direction) : tcnn::max(particleFeatures[hitParticle.idx], 0.f),
                                          ray.features);

            if (hitWeight > 0.0f) ray.countHit();
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <typename TRay>
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,
                                       const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/,
                                       const float* __restrict__ /*particlesGlobalDepthPtr*/,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                       float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                       threedgut::MemoryHandles parametersGradient                        = {}) {

        using namespace threedgut;

        const uint32_t tileIdx                       = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices   = sortedTileRangeIndicesPtr[tileIdx];
        uint32_t tileNumParticlesToProcess           = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess        = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);
        const TFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        } else {
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        }
    }

    template <typename TRay>
    static inline __device__ void evalKBuffer(TRay& ray,
                                              Particles& particles,
                                              const tcnn::uvec2& tileParticleRangeIndices,
                                              uint32_t tileNumBlocksToProcess,
                                              uint32_t tileNumParticlesToProcess,
                                              const uint32_t tileThreadIdx,
                                              const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                              const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                              TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        using namespace threedgut;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleKBuffer<Params::KHitBufferSize> hitParticleKBuffer;

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;
                if (particles.densityHit(ray.origin,
                                         ray.direction,
                                         particleData.densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT) &&
                    (hitParticle.hitT > ray.tMinMax.x) &&
                    (hitParticle.hitT < ray.tMinMax.y)) {

                    if (hitParticleKBuffer.full()) {
                        processHitParticle(ray,
                                           hitParticleKBuffer.closestHit(hitParticle),
                                           particles,
                                           particleFeaturesBuffer,
                                           particleFeaturesGradientBuffer);
                    }
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                   hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer);
            }
        }
    }

    // Fine-grained balanced forward rendering: Gaussian-wise parallelism with warp-level optimization
    template <typename TRay>
    static inline __device__ void evalForwardNoKBufferBalanced(
        const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       const float* __restrict__ particlesGlobalDepthPtr,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       const tcnn::uvec2& tile,
                                       const tcnn::uvec2& tileGrid,
                                       const int laneId,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr     = nullptr,
                                       tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr = nullptr,
                                       float* __restrict__ particlesGlobalDepthGradPtr                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr        = nullptr,
                                       threedgut::MemoryHandles parametersGradient                    = {}) {

        using namespace threedgut;

        // Get tile data: each warp processes particles from a single 16x16 tile
        const uint32_t tileIdx = tile.y * tileGrid.x + tile.x;
        const tcnn::uvec2 tileParticleRangeIndices = sortedTileRangeIndicesPtr[tileIdx];

        uint32_t tileNumParticlesToProcess = tileParticleRangeIndices.y - tileParticleRangeIndices.x;

        // Setup feature buffers based on rendering mode
        const TFeaturesVec* particleFeaturesBuffer = 
            Params::PerRayParticleFeatures ? nullptr : 
            reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = 
            (Params::PerRayParticleFeatures || !Backward) ? nullptr : 
            reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        // Initialize particle system
        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        static_assert(Params::KHitBufferSize == 0, "evalForwardNoKBufferBalanced only supports K=0 (no hit buffer). Use evalKBuffer for K>0 cases.");

        // Warp-aligned processing: round up to multiple of WarpSize to avoid divergence
        constexpr uint32_t WarpSize = GUTParameters::Tiling::WarpSize;  // 32 threads per warp
        uint32_t alignedParticleCount = ((tileNumParticlesToProcess + WarpSize - 1) / WarpSize) * WarpSize;

        // Main loop: Gaussian-wise parallelism - WarpSize threads process Gaussians, single ray
        for (uint32_t j = laneId; j < alignedParticleCount; j += WarpSize) {
            if (!ray.isAlive()) break;

            float hitAlpha = 0.0f;
            float hitT = 0.0f;
            TFeaturesVec hitFeatures = TFeaturesVec::zero();
            bool validHit = false;

            // Step 1: Each thread tests one Gaussian intersection
            if (j < tileNumParticlesToProcess) {
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + j;
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];

                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    auto densityParams = particles.fetchDensityParameters(particleIdx);

                    if (particles.densityHit(ray.origin,
                                           ray.direction,
                                           densityParams,
                                           hitAlpha,
                                           hitT) &&
                        (hitT > ray.tMinMax.x) &&
                        (hitT < ray.tMinMax.y)) {

                        validHit = true;

                        // Get Gaussian features
                        if constexpr (Params::PerRayParticleFeatures) {
                            hitFeatures = particles.featuresFromBuffer(particleIdx, ray.direction);
                        } else {
                            hitFeatures = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                        }
                    }
                }
            }

            // Skip if no hits in this warp batch
            constexpr uint32_t WarpMask = GUTParameters::Tiling::WarpMask;  // 0xFFFFFFFF for full warp
            if (__all_sync(WarpMask, !validHit)) continue;

            // Step 2: Compute per-thread transmittance contribution
            float localTransmittance = validHit ? (1.0f - hitAlpha) : 1.0f;

            // Step 3: Warp-level prefix scan for cumulative transmittance
            for (uint32_t offset = 1; offset < WarpSize; offset <<= 1) {
                float n = __shfl_up_sync(WarpMask, localTransmittance, offset);
                if (laneId >= offset) {
                    localTransmittance *= n;
                }
            }

            // Get overall batch transmittance impact
            float batchTransmittance = __shfl_sync(WarpMask, localTransmittance, WarpSize - 1);
            float newTransmittance = ray.transmittance * batchTransmittance;

            // Step 4: Early termination detection - find exact termination point
            unsigned int earlyTerminationMask = __ballot_sync(WarpMask, 
                validHit && (ray.transmittance * localTransmittance) < Particles::MinTransmittanceThreshold);

            bool shouldTerminate = false;
            int terminationLane = -1;

            if (earlyTerminationMask) {
                terminationLane = __ffs(earlyTerminationMask) - 1; // Find first terminating lane
                shouldTerminate = true;
                ray.kill();
            }

            // Step 5: Warp reduction for feature accumulation
            TFeaturesVec accumulatedFeatures = TFeaturesVec::zero();
            float accumulatedHitT = 0.0f;
            uint32_t accumulatedHitCount = 0;

            // Only accumulate contributions before (and including) termination point
            bool shouldContribute = validHit && (!shouldTerminate || laneId <= terminationLane);

            if (shouldContribute) {
                // Use precomputed prefix transmittance, excluding current particle
                float prefixTransmittance = (laneId > 0) ? 
                    (localTransmittance / (1.0f - hitAlpha)) : 1.0f;
                float particleTransmittance = ray.transmittance * prefixTransmittance;
                float hitWeight = hitAlpha * particleTransmittance;

                // Compute weighted contributions
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    accumulatedFeatures[featIdx] = hitFeatures[featIdx] * hitWeight;
                }
                accumulatedHitT = hitT * hitWeight;
                accumulatedHitCount = (hitWeight > 0.0f) ? 1 : 0;
            }

            // Step 6: Warp-level reduction (tree-based sum)
            for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                for (uint32_t offset = WarpSize / 2; offset > 0; offset >>= 1) {
                    accumulatedFeatures[featIdx] += __shfl_down_sync(WarpMask, accumulatedFeatures[featIdx], offset);
                }
            }

            for (uint32_t offset = WarpSize / 2; offset > 0; offset >>= 1) {
                accumulatedHitT += __shfl_down_sync(WarpMask, accumulatedHitT, offset);
                accumulatedHitCount += __shfl_down_sync(WarpMask, accumulatedHitCount, offset);
            }

            // Step 7: Only lane 0 updates ray state (avoid race conditions)
            if (laneId == 0) {
                for (int featIdx = 0; featIdx < Particles::FeaturesDim; ++featIdx) {
                    ray.features[featIdx] += accumulatedFeatures[featIdx];
                }
                ray.hitT += accumulatedHitT;
                ray.countHit(accumulatedHitCount);
            }

            // Step 8: Update ray transmittance
            ray.transmittance = newTransmittance;

            // Break on early termination
            if (shouldTerminate) {
                break;
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;
        __shared__ PrefetchedRawParticleData prefetchedRawParticlesData[GUTParameters::Tiling::BlockSize];

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                    if constexpr (Params::PerRayParticleFeatures) {
                        prefetchedRawParticlesData[tileThreadIdx].features = TFeaturesVec::zero();
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].features = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                    }
                    prefetchedRawParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                    break;
                }

                const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();
                    break;
                }

                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;
                densityRawParametersGrad.position   = make_float3(0.0f);
                densityRawParametersGrad.quaternion = make_float4(0.0f);
                densityRawParametersGrad.scale      = make_float3(0.0f);

                TFeaturesVec featuresGrad = TFeaturesVec::zero();

                if (ray.isAlive()) {
                    particles.processHitBwd<Params::PerRayParticleFeatures>(
                        ray.origin,
                        ray.direction,
                        particleData.idx,
                        particleData.densityParameters,
                        &densityRawParametersGrad,
                        particleData.features,
                        &featuresGrad,
                        ray.transmittance,
                        ray.transmittanceBackward,
                        ray.transmittanceGradient,
                        ray.features,
                        ray.featuresBackward,
                        ray.featuresGradient,
                        ray.hitT,
                        ray.hitTBackward,
                        ray.hitTGradient);
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                if constexpr (!Params::PerRayParticleFeatures) {
                    particles.processHitBwdUpdateFeaturesGradient(particleData.idx, featuresGrad,
                                                                  particleFeaturesGradientBuffer, tileThreadIdx);
                }
                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};
