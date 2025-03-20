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

#include <3dgut/threedgut.cuh>

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/sensors/sensors.h>

#include <tiny-cuda-nn/common_host.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <limits>

using namespace tcnn;

namespace {

using namespace threedgut;

constexpr int featuresDim() {
    return model_ExternalParams::FeaturesDim;
}

// identify tiles start/end indices in the sorted tile/depth keys buffer
__global__ void computeSortedTileRangeIndices(
    int numKeys,
    const uint64_t* __restrict__ sortedTileDepthKeys,
    uvec2* __restrict__ tileRangeIndices) {

    const int keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (keyIdx >= numKeys) {
        return;
    }

    const uint32_t tileIdx = sortedTileDepthKeys[keyIdx] >> 32;
    const bool validTile   = tileIdx != GUTParameters::Tiling::InvalidTileIdx;
    if (keyIdx == 0) {
        if (validTile) {
            tileRangeIndices[tileIdx].x = keyIdx;
        }
    } else {
        const uint32_t prevKeyTileIdx = sortedTileDepthKeys[keyIdx - 1] >> 32;
        if (prevKeyTileIdx != tileIdx) {
            if (prevKeyTileIdx != GUTParameters::Tiling::InvalidTileIdx) {
                tileRangeIndices[prevKeyTileIdx].y = keyIdx;
            }
            if (validTile) {
                tileRangeIndices[tileIdx].x = keyIdx;
            }
        }
    }
    if (validTile && (keyIdx == numKeys - 1)) {
        tileRangeIndices[tileIdx].y = numKeys;
    }
}

// TODO : review this
inline uint32_t higherMsb(uint32_t n) {
    uint32_t msb  = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb) {
            msb += step;
        } else {
            msb -= step;
        }
    }
    if (n >> msb) {
        msb++;
    }
    return msb;
}

} // namespace

// TODO : per-stream n context cache
struct GUTRenderer::GutRenderForwardContext {
    cudaStream_t cudaStream;

    GutRenderForwardContext(cudaStream_t iCudaStream)
        : cudaStream(iCudaStream) {
    }

    ~GutRenderForwardContext() {
        const uint64_t processQueueHandle = reinterpret_cast<uint64_t>(cudaStream);
        Logger logger;
        unsortedTileDepthKeys.clear(processQueueHandle, logger);
        sortedTileDepthKeys.clear(processQueueHandle, logger);
        sortedTileRangeIndices.clear(processQueueHandle, logger);
        unsortedTileParticleIdx.clear(processQueueHandle, logger);
        sortedTileParticleIdx.clear(processQueueHandle, logger);
        sortingWorkingBuffer.clear(processQueueHandle, logger);
        particlesTilesCount.clear(processQueueHandle, logger);
        particlesTilesOffset.clear(processQueueHandle, logger);
        particlesProjectedPosition.clear(processQueueHandle, logger);
        particlesProjectedConicOpacity.clear(processQueueHandle, logger);
        particlesProjectedExtent.clear(processQueueHandle, logger);
        particlesGlobalDepth.clear(processQueueHandle, logger);
        particlesPrecomputedFeatures.clear(processQueueHandle, logger);
        particlesProjectedPositionGradient.clear(processQueueHandle, logger);
        particlesProjectedConicOpacityGradient.clear(processQueueHandle, logger);
        particlesGlobalDepthGradient.clear(processQueueHandle, logger);
        particlesPrecomputedFeaturesGradient.clear(processQueueHandle, logger);
        scanningWorkingBuffer.clear(processQueueHandle, logger);
    }

    CudaBuffer unsortedTileDepthKeys;
    CudaBuffer sortedTileDepthKeys;
    CudaBuffer sortedTileRangeIndices;
    CudaBuffer unsortedTileParticleIdx;
    CudaBuffer sortedTileParticleIdx;
    CudaBuffer sortingWorkingBuffer;

    Status updateTileSortingBuffers(const uvec2& tileGrid, int numKeys, cudaStream_t stream, const Logger& logger) {
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(stream);
        const bool uptodate        = unsortedTileDepthKeys.size() >= sizeof(uint64_t) * numKeys;
        if (!uptodate) {
            CHECK_STATUS_RETURN(unsortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(sortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(unsortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(sortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, queueHandle, logger));
            size_t sortingWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceRadixSort::SortPairs(
                    nullptr,
                    sortingWorkingBufferSize,
                    static_cast<const uint64_t*>(unsortedTileDepthKeys.data()),
                    static_cast<uint64_t*>(sortedTileDepthKeys.data()),
                    static_cast<const uint32_t*>(unsortedTileParticleIdx.data()),
                    static_cast<uint32_t*>(sortedTileParticleIdx.data()),
                    numKeys,
                    0, 32 + higherMsb(tileGrid.x * tileGrid.y),
                    stream),
                logger);
            CHECK_STATUS_RETURN(sortingWorkingBuffer.resize(sortingWorkingBufferSize, queueHandle, logger));
        }
        if (numKeys) {
            CHECK_STATUS_RETURN(sortedTileRangeIndices.resize(sizeof(uvec2) * tileGrid.x * tileGrid.y, queueHandle, logger));
            CUDA_CHECK_RETURN(cudaMemsetAsync(sortedTileRangeIndices.data(), 0, tileGrid.x * tileGrid.y * sizeof(uvec2), stream), logger);
        }
        return Status();
    }

    CudaBuffer particlesTilesCount;                            ///< number of intersected tiles per particles uint32_t [Nx1]
    CudaBuffer particlesTilesOffset;                           ///< cumulative sum of particle tiles count uint32_t [Nx1]
    CudaBuffer particlesProjectedPosition;                     ///< projected particles center Nx2
    CudaBuffer particlesProjectedConicOpacity;                 ///< projected particles conic and opacity Nx4
    CudaBuffer particlesProjectedExtent;                       ///< projected particles extent Nx2
    CudaBuffer particlesGlobalDepth;                           ///< particles global depth
    CudaBuffer particlesPrecomputedFeatures;                   ///< precomputed particle features float [NxFeaturesDim]
    mutable CudaBuffer particlesProjectedPositionGradient;     ///< projected particles center Nx2
    mutable CudaBuffer particlesProjectedConicOpacityGradient; ///< projected particles conic and opacity Nx4
    mutable CudaBuffer particlesGlobalDepthGradient;           ///< particles global depth
    mutable CudaBuffer particlesPrecomputedFeaturesGradient;   ///< precomputed particle features float [NxFeaturesDim]
    CudaBuffer scanningWorkingBuffer;                          ///< working buffer to compute the cumulative sum of particles/tiles intersections number

    inline Status updateParticlesWorkingBuffers(int numParticles, cudaStream_t cudaStream, const Logger& logger) {
        const bool uptodate = particlesTilesCount.size() >= numParticles * sizeof(uint32_t);
        if (!uptodate) {
            const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);
            CHECK_STATUS_RETURN(particlesTilesCount.resize(numParticles * sizeof(uint32_t), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesTilesOffset.resize(numParticles * sizeof(uint32_t), queueHandle, logger));
            size_t scanningWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceScan::InclusiveSum(
                    nullptr,
                    scanningWorkingBufferSize,
                    static_cast<const uint32_t*>(particlesTilesCount.data()),
                    static_cast<uint32_t*>(particlesTilesOffset.data()),
                    numParticles,
                    cudaStream),
                logger);
            CHECK_STATUS_RETURN(scanningWorkingBuffer.resize(scanningWorkingBufferSize, queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedPosition.resize(numParticles * sizeof(vec2), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedConicOpacity.resize(numParticles * sizeof(vec4), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedExtent.resize(numParticles * sizeof(vec2), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesGlobalDepth.resize(numParticles * sizeof(float), queueHandle, logger));
        }

        return Status();
    }

    inline Status updateParticlesProjectionGradientBuffers(int numParticles, cudaStream_t cudaStream, const Logger& logger) const {
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);

        CHECK_STATUS_RETURN(particlesProjectedPositionGradient.enlarge(numParticles * sizeof(vec2), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedPositionGradient.data(), 0, numParticles * sizeof(vec2), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesProjectedConicOpacityGradient.resize(numParticles * sizeof(vec4), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedConicOpacityGradient.data(), 0, numParticles * sizeof(vec4), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesGlobalDepthGradient.resize(numParticles * sizeof(float), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesGlobalDepthGradient.data(), 0, numParticles * sizeof(float), cudaStream), logger);

        return Status();
    }

    inline Status
    updateParticlesFeaturesBuffer(int featuresSize, cudaStream_t cudaStream, const Logger& logger) {
        CHECK_STATUS_RETURN(particlesPrecomputedFeatures.enlarge(featuresSize * sizeof(float), reinterpret_cast<uint64_t>(cudaStream), logger));
        return Status();
    }

    inline Status updateParticlesFeaturesGradientBuffer(uint32_t featuresSize, cudaStream_t cudaStream, const Logger& logger) const {
        const size_t newSize = featuresSize * sizeof(float);
        CHECK_STATUS_RETURN(particlesPrecomputedFeaturesGradient.enlarge(newSize, reinterpret_cast<uint64_t>(cudaStream), logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesPrecomputedFeaturesGradient.data(), 0, newSize, cudaStream), logger);
        return Status();
    }
};

threedgut::GUTRenderer::GUTRenderer(const nlohmann::json& config, const Logger& logger)
    : m_logger(logger) {
}

threedgut::GUTRenderer::~GUTRenderer() {
}

threedgut::Status threedgut::GUTRenderer::renderForward(const RenderParameters& params,
                                                        const vec3* wordlRayOriginCudaPtr,
                                                        const vec3* worldRayDirectionCudaPtr,
                                                        float* worldHitDistanceCudaPtr,
                                                        vec4* radianceDensityCudaPtr,
                                                        Parameters& parameters,
                                                        int cudaDeviceIndex,
                                                        cudaStream_t cudaStream) {

    if (!m_forwardContext) {
        m_forwardContext = std::make_unique<GutRenderForwardContext>(cudaStream);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render");

    const uvec2 tileGrid{
        div_round_up<uint32_t>(params.resolution.x, GUTParameters::Tiling::BlockX),
        div_round_up<uint32_t>(params.resolution.y, GUTParameters::Tiling::BlockY),
    };
    const uint32_t numParticles = parameters.values.numParticles;

    // NOTE: to properly support rolling shutter camera, we need to interpolate sensor pose in the kernel
    const TSensorPose sensorPose    = interpolatedSensorPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f); // Transform from world to sensor space
    const TSensorPose sensorPoseInv = sensorPoseInverse(sensorPose);                                                          // Transform from sensor to world space

    CHECK_STATUS_RETURN(m_forwardContext->updateParticlesWorkingBuffers(numParticles, cudaStream, m_logger));
    if (!/*m_settings.perRayFeatures*/TGUTRendererParams::PerRayParticleFeatures) {
        CHECK_STATUS_RETURN(m_forwardContext->updateParticlesFeaturesBuffer(numParticles * featuresDim(), cudaStream, m_logger));
    }

    {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::project"};
        ::projectOnTiles<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            // NOTE : the sensor world position is an approximated position for preprocessing gaussian colors
            /*sensorWorldPosition=*/sensorPoseInverse(sensorPose).slice<0, 3>(),
            // NOTE : this sensor to world transform is used to estimate the Z-depth of the particles
            sensorPoseToMat(sensorPose) /** tcnn::mat4(params.objectToWorldTransform)*/,
            params.sensorState,
            (uint32_t*)m_forwardContext->particlesTilesCount.data(),
            (tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (tcnn::vec2*)m_forwardContext->particlesProjectedExtent.data(),
            (float*)m_forwardContext->particlesGlobalDepth.data(),
            (float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    deviceLaunchesLogger.push("render::prepare-expand");

    // inplace cumulative sum over list of number of intersected tiles per particles
    size_t scanningWorkingBufferSize = m_forwardContext->scanningWorkingBuffer.size();
    // TODO : check if using not inplace version has perf benefits
    CUDA_CHECK_RETURN(
        cub::DeviceScan::InclusiveSum(
            m_forwardContext->scanningWorkingBuffer.data(),
            scanningWorkingBufferSize,
            static_cast<const uint32_t*>(m_forwardContext->particlesTilesCount.data()),
            static_cast<uint32_t*>(m_forwardContext->particlesTilesOffset.data()),
            numParticles,
            cudaStream),
        m_logger);

    // fetch total number of particle/tile intersections to launch and resize the sorting buffers
    uint32_t numParticleTileIntersections;
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(&numParticleTileIntersections,
                        static_cast<uint32_t*>(m_forwardContext->particlesTilesOffset.data()) + numParticles - 1,
                        sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        cudaStream),
        m_logger);
    cudaStreamSynchronize(cudaStream);

    if (numParticleTileIntersections == 0) {
        return Status();
    }

    // sorting buffers allocation
    CHECK_STATUS_RETURN(
        m_forwardContext->updateTileSortingBuffers(tileGrid, numParticleTileIntersections, cudaStream, m_logger));

    deviceLaunchesLogger.pop("render::prepare-expand");

    {
        const auto expandProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::expand"};
        ::expandTileProjections<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            params.sensorState,
            (const uint32_t*)m_forwardContext->particlesTilesOffset.data(),
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const tcnn::vec2*)m_forwardContext->particlesProjectedExtent.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            parameters.m_dptrParametersBuffer,
            (uint64_t*)m_forwardContext->unsortedTileDepthKeys.data(),
            (uint32_t*)m_forwardContext->unsortedTileParticleIdx.data());
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    deviceLaunchesLogger.push("render::sort");

    // Sort complete list of (duplicated) Gaussian indices by keys
    size_t sortingWorkingBufferSize = m_forwardContext->sortingWorkingBuffer.size();
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
                          m_forwardContext->sortingWorkingBuffer.data(),
                          sortingWorkingBufferSize,
                          static_cast<const uint64_t*>(m_forwardContext->unsortedTileDepthKeys.data()),
                          static_cast<uint64_t*>(m_forwardContext->sortedTileDepthKeys.data()),
                          static_cast<const uint32_t*>(m_forwardContext->unsortedTileParticleIdx.data()),
                          static_cast<uint32_t*>(m_forwardContext->sortedTileParticleIdx.data()),
                          numParticleTileIntersections,
                          0, 32 + higherMsb(tileGrid.x * tileGrid.y), cudaStream),
                      m_logger);

    // Compute the tile range indices in the sorted keys
    linear_kernel(
        computeSortedTileRangeIndices, /*shmem=*/0, cudaStream,
        numParticleTileIntersections,
        static_cast<const uint64_t*>(m_forwardContext->sortedTileDepthKeys.data()),
        static_cast<uvec2*>(m_forwardContext->sortedTileRangeIndices.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);

    deviceLaunchesLogger.pop("render::sort");

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::render"};
        ::render<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
            params,
            (const tcnn::uvec2*)m_forwardContext->sortedTileRangeIndices.data(),
            (const uint32_t*)m_forwardContext->sortedTileParticleIdx.data(),
            (const tcnn::vec3*)wordlRayOriginCudaPtr,
            (const tcnn::vec3*)worldRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (float*)worldHitDistanceCudaPtr,
            (tcnn::vec4*)radianceDensityCudaPtr,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    return Status();
}

threedgut::Status threedgut::GUTRenderer::renderBackward(const RenderParameters& params,
                                                         const vec3* wordlRayOriginCudaPtr,
                                                         const vec3* worldRayDirectionCudaPtr,
                                                         const float* worldHitDistanceCudaPtr,
                                                         const float* worldHitDistanceGradientCudaPtr,
                                                         const vec4* radianceDensityCudaPtr,
                                                         const vec4* radianceDensityGradientCudaPtr,
                                                         vec3* wordlRayOriginGradientCudaPtr,
                                                         vec3* worldRayDirectionGradientCudaPtr,
                                                         Parameters& parameters,
                                                         int cudaDeviceIndex,
                                                         cudaStream_t cudaStream) {

    if (!m_forwardContext || (m_forwardContext->cudaStream != cudaStream)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] cannot render backward, invalid forward context on device %d, cudaStream = %p.",
                     cudaDeviceIndex, cudaStream);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render-backward");

    const uvec2 tileGrid{
        div_round_up<uint32_t>(params.resolution.x, GUTParameters::Tiling::BlockX),
        div_round_up<uint32_t>(params.resolution.y, GUTParameters::Tiling::BlockY),
    };
    const uint32_t numParticles = parameters.values.numParticles;

    // NOTE: to properly support rolling shutter camera, we need to interpolate sensor pose in the kernel
    const TSensorPose sensorPose    = interpolatedSensorPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f); // Transform from world to sensor space
    const TSensorPose sensorPoseInv = sensorPoseInverse(sensorPose);                                                          // Transform from sensor to world space

    if (numParticles == 0) {
        LOG_ERROR(m_logger, "[GUTRenderer] number of particles is 0, cannot render backward.");
    }

    if (!/*m_settings.perRayFeatures*/TGUTRendererParams::PerRayParticleFeatures) {
        CHECK_STATUS_RETURN(
            m_forwardContext->updateParticlesFeaturesGradientBuffer(numParticles * featuresDim(), cudaStream, m_logger));
    }

    if (/*m_settings.renderMode == Settings::Splat*/TGUTProjectorParams::BackwardProjection) {
        CHECK_STATUS_RETURN(
            m_forwardContext->updateParticlesProjectionGradientBuffers(numParticles, cudaStream, m_logger));
    }

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward::render"};
        ::renderBackward<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
            params,
            (const tcnn::uvec2*)m_forwardContext->sortedTileRangeIndices.data(),
            (const uint32_t*)m_forwardContext->sortedTileParticleIdx.data(),
            (const tcnn::vec3*)wordlRayOriginCudaPtr,
            (const tcnn::vec3*)worldRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const float*)worldHitDistanceCudaPtr,
            (const float*)worldHitDistanceGradientCudaPtr,
            (const tcnn::vec4*)radianceDensityCudaPtr,
            (const tcnn::vec4*)radianceDensityGradientCudaPtr,
            (tcnn::vec3*)wordlRayOriginGradientCudaPtr,
            (tcnn::vec3*)worldRayDirectionGradientCudaPtr,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer,
            (tcnn::vec2*)m_forwardContext->particlesProjectedPositionGradient.data(),
            (tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacityGradient.data(),
            (float*)m_forwardContext->particlesGlobalDepthGradient.data(),
            (float*)m_forwardContext->particlesPrecomputedFeaturesGradient.data(),
            parameters.m_dptrGradientsBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    if (!/*m_settings.perRayFeatures*/TGUTRendererParams::PerRayParticleFeatures) {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward::project"};
        ::projectBackward<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            // NOTE : the sensor world position is an approximated position for preprocessing gaussian colors
            /*sensorWorldPosition=*/sensorPoseInverse(sensorPose).slice<0, 3>(),
            // NOTE : this sensor to world transform is used to estimate the Z-depth of the particles
            sensorPoseToMat(sensorPose),
            (const uint32_t*)m_forwardContext->particlesTilesCount.data(),
            parameters.m_dptrParametersBuffer,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPositionGradient.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacityGradient.data(),
            (const float*)m_forwardContext->particlesGlobalDepthGradient.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeaturesGradient.data(),
            parameters.m_dptrGradientsBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    return Status();
}
