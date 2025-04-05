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

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgut/splatRaster.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/core/ScalarType.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>

//------------------------------------------------------------------------------
// CUDA / OPTIX macros
//------------------------------------------------------------------------------

#define CUDA_CHECK_LAST(logger) \
    CUDA_CHECK(logger, cudaGetLastError())

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

using namespace threedgut;

namespace {

static void THREEDGUT_LOGGER_CB logCallback(uint8_t level, const char* msg, void* data) {
    std::ostream& stream = (level > LoggerParameters::Error) ? std::cout : std::cerr;
    stream << "[3dgut][" << LoggerParameters::levelToString(level) << "] ::: "
           << msg << std::flush << std::endl;
}

} // namespace

//------------------------------------------------------------------------------
// SplatRaster
//------------------------------------------------------------------------------
inline void* voidDataPtr(torch::Tensor& tensor) {
    if (tensor.size(0) == 0) {
        return nullptr;
    }
    switch (tensor.scalar_type()) {
    case torch::kFloat32:
        return tensor.contiguous().data_ptr<float>();
    case torch::kHalf:
        return tensor.contiguous().data_ptr<torch::Half>();
    case torch::kInt32:
        return tensor.contiguous().data_ptr<int32_t>();
    case torch::kInt64:
        return tensor.contiguous().data_ptr<int64_t>();
    // case torch::kUInt32:
    //     return tensor.contiguous().data_ptr<uint32_t>();
    // case torch::kUInt64:
    //     return tensor.contiguous().data_ptr<uint64_t>();
    default:
        throw std::runtime_error{"[3dgut] Unknown precision torch->void: " + std::string(c10::toString(tensor.scalar_type()))};
    }
}

using DeviceQueueHandle = uint64_t;

threedgut::TSensorPose poseInverse(const TSensorPose& pose) {
    static_assert(sizeof(TSensorPose) == sizeof(threedgut::TSensorPose));
    const threedgut::TSensorPose iPose = threedgut::sensorPoseInverse({pose.elems[0], pose.elems[1], pose.elems[2], pose.elems[3], pose.elems[4], pose.elems[5], pose.elems[6]});
    return {iPose[0], iPose[1], iPose[2], iPose[3], iPose[4], iPose[5], iPose[6]};
}

TSensorState toSensorState(threedgut::TTimestamp startTs, torch::Tensor sensorsStartPose, threedgut::TTimestamp endTs, torch::Tensor sensorsEndPose) {
    const torch::Tensor sensorsStartPoseHost = sensorsStartPose.cpu();
    const torch::Tensor sensorsEndPoseHost   = sensorsEndPose.cpu();

    auto startInvPosePtr = reinterpret_cast<const threedgut::TSensorPose*>(sensorsStartPoseHost.data_ptr<float>());
    auto endInvPosePtr   = reinterpret_cast<const threedgut::TSensorPose*>(sensorsEndPoseHost.data_ptr<float>());

    return threedgut::TSensorState{startTs, /*poseInverse*/ (*startInvPosePtr), endTs, /*poseInverse*/ (*endInvPosePtr)};
}

struct SplatRaster::CudaTimer {
    const char* _tag;
    cudaStream_t _stream;
    cudaEvent_t _start = 0, _stop = 0;
    bool _valid = false, _stopped = false;

    CudaTimer(const char* tag, cudaStream_t stream)
        : _tag(tag), _stream(stream) {
        _valid = cudaEventCreate(&_start) == cudaSuccess;
        _valid = _valid && (cudaEventCreate(&_stop) == cudaSuccess);
        if (_valid) {
            cudaEventRecord(_start, _stream);
        }
    }
    ~CudaTimer() {
        if (_stop) {
            cudaEventDestroy(_stop);
        }
        if (_start) {
            cudaEventDestroy(_start);
        }
    }

    std::string tag() const {
        return _tag;
    }

    void stop() {
        if (_valid && !_stopped) {
            cudaEventRecord(_stop, _stream);
            _stopped = true;
        }
    }

    float collect() {
        float milliseconds = 1e09f;
        if (_valid) {
            stop();
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&milliseconds, _start, _stop);
        }
        return milliseconds;
    }
};

SplatRaster::SplatRaster(const nlohmann::json& config)
    : m_logLevel(static_cast<uint8_t>(LoggerParameters::Debug))
    , m_logger(LoggerParameters{m_logLevel, logCallback, nullptr, nullptr, nullptr})
    , m_renderer(std::make_unique<GUTRenderer>(config, m_logger)) {

    const auto& renderConfig = config["render"];
    m_enableKernelTimings    = renderConfig.value("enable_kernel_timings", false);

    m_parameters.valuesBuffer.resize(sizeof(m_parameters.values), 0, m_logger);
    m_parameters.parametersBuffer.resize(sizeof(m_parameters.parameters), 0, m_logger);
    m_parameters.gradientsBuffer.resize(sizeof(m_parameters.gradients), 0, m_logger);

    m_parameters.parameters.dptrValuesBuffer = m_parameters.valuesBuffer.data();
    m_parameters.m_dptrParametersBuffer      = (uint64_t*)m_parameters.parametersBuffer.data();
    m_parameters.m_dptrGradientsBuffer       = (uint64_t*)m_parameters.gradientsBuffer.data();
}

SplatRaster::~SplatRaster(void) {
}

std::tuple<torch::Tensor, torch::Tensor>
SplatRaster::trace(uint32_t frameNumber, int numActiveFeatures,
                   torch::Tensor particleDensity,
                   torch::Tensor particleRadiance,
                   torch::Tensor rayOrigin,
                   torch::Tensor rayDirection,
                   torch::Tensor rayTimestamp,
                   TSensorModel sensorModel,
                   TTimestamp startTimestamp,
                   TTimestamp endTimestamp,
                   torch::Tensor sensorsStartPose,
                   torch::Tensor sensorsEndPose) {

    const int cudaDeviceIndex = rayOrigin.get_device();
    cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

    const int width  = rayOrigin.size(2);
    const int height = rayOrigin.size(1);

    const uint32_t numParticles = particleDensity.size(0);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor rayRadianceDensity = torch::zeros({height, width, 4}, opts);
    torch::Tensor rayHitDistance     = torch::ones({height, width, 1}, opts).multiply(1e06f);

    m_parameters.values.numParticles               = numParticles;
    m_parameters.values.radianceSphDegree          = numActiveFeatures;
    m_parameters.parameters.dptrDensityParameters  = voidDataPtr(particleDensity);
    m_parameters.parameters.dptrRadianceParameters = voidDataPtr(particleRadiance);
    m_parameters.valuesBuffer.setFromHost(&m_parameters.values, sizeof(m_parameters.values), reinterpret_cast<uint64_t>(cudaStream), m_logger);
    m_parameters.parametersBuffer.setFromHost(&m_parameters.parameters, sizeof(m_parameters.parameters), reinterpret_cast<uint64_t>(cudaStream), m_logger);

    std::shared_ptr<CudaTimer> timer;
    if (m_enableKernelTimings) {
        timer = std::make_shared<CudaTimer>("forward_render", cudaStream);
        m_timers.emplace_back(timer);
        // keep only the last m_maxNumTimers timers
        if (m_timers.size() > m_maxNumTimers) {
            m_timers.pop_front();
        }
    }

    threedgut::RenderParameters renderParameters;
    renderParameters.id          = frameNumber;
    renderParameters.resolution  = tcnn::ivec2{width, height};
    renderParameters.sensorModel = sensorModel;
    renderParameters.sensorState = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
    renderParameters.objectAABB  = threedgut::BoundingBox{tcnn::vec3{-1e06f, -1e06f, -1e06f}, tcnn::vec3{1e06f, 1e06f, 1e06f}};

    ErrorCode status = m_renderer->renderForward(
        renderParameters,
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayOrigin)),
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayDirection)),
        reinterpret_cast<float*>(voidDataPtr(rayHitDistance)),
        reinterpret_cast<tcnn::vec4*>(voidDataPtr(rayRadianceDensity)),
        m_parameters,
        cudaDeviceIndex,
        cudaStream);

    CUDA_CHECK_LAST(m_logger);

    if (timer) {
        timer->stop();
    }

    return std::tuple<torch::Tensor, torch::Tensor>(rayRadianceDensity, rayHitDistance);
}

std::tuple<torch::Tensor, torch::Tensor>
SplatRaster::traceBwd(uint32_t frameNumber, int numActiveFeatures,
                      torch::Tensor particleDensity,
                      torch::Tensor particleRadiance,
                      torch::Tensor rayOrigin,
                      torch::Tensor rayDirection,
                      torch::Tensor rayTimestamp,
                      TSensorModel sensorModel,
                      TTimestamp startTimestamp,
                      TTimestamp endTimestamp,
                      torch::Tensor sensorsStartPose,
                      torch::Tensor sensorsEndPose,
                      // Gradients
                      torch::Tensor rayRadianceDensity,
                      torch::Tensor rayRadianceDensityGradient,
                      torch::Tensor rayHitDistance,
                      torch::Tensor rayHitDistanceGradient) {

    const int cudaDeviceIndex = rayOrigin.get_device();
    cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

    const int width  = rayOrigin.size(2);
    const int height = rayOrigin.size(1);

    const uint32_t numParticles = particleDensity.size(0);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor particleDensityGradient  = torch::zeros({particleDensity.size(0), particleDensity.size(1)}, opts);
    torch::Tensor particleRadianceGradient = torch::zeros({particleRadiance.size(0), particleRadiance.size(1)}, opts);

    const bool rayBackpropagation = false;

    torch::Tensor rayOriginGradient;
    torch::Tensor rayDirectionGradient;
    if (rayBackpropagation) {
        rayOriginGradient    = torch::zeros({rayOrigin.size(0), rayOrigin.size(1), rayOrigin.size(2), 3}, opts);
        rayDirectionGradient = torch::zeros({rayDirection.size(0), rayDirection.size(1), rayDirection.size(2), 3}, opts);
    }

    m_parameters.values.numParticles               = numParticles;
    m_parameters.values.radianceSphDegree          = numActiveFeatures;
    m_parameters.parameters.dptrDensityParameters  = voidDataPtr(particleDensity);
    m_parameters.parameters.dptrRadianceParameters = voidDataPtr(particleRadiance);
    m_parameters.gradients.dptrDensityGradients    = voidDataPtr(particleDensityGradient);
    m_parameters.gradients.dptrRadianceGradients   = voidDataPtr(particleRadianceGradient);
    m_parameters.valuesBuffer.setFromHost(&m_parameters.values, sizeof(m_parameters.values), reinterpret_cast<uint64_t>(cudaStream), m_logger);
    m_parameters.parametersBuffer.setFromHost(&m_parameters.parameters, sizeof(m_parameters.parameters), reinterpret_cast<uint64_t>(cudaStream), m_logger);
    m_parameters.gradientsBuffer.setFromHost(&m_parameters.gradients, sizeof(m_parameters.gradients), reinterpret_cast<uint64_t>(cudaStream), m_logger);

    std::shared_ptr<CudaTimer> timer;
    if (m_enableKernelTimings) {
        timer = std::make_shared<CudaTimer>("backward_render", cudaStream);
        m_timers.emplace_back(timer);
        // keep only the last m_maxNumTimers timers
        if (m_timers.size() > m_maxNumTimers) {
            m_timers.pop_front();
        }
    }

    threedgut::RenderParameters renderParameters;
    renderParameters.id          = frameNumber;
    renderParameters.resolution  = tcnn::ivec2{width, height};
    renderParameters.sensorModel = sensorModel;
    renderParameters.sensorState = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
    renderParameters.objectAABB  = threedgut::BoundingBox{tcnn::vec3{-1e06f, -1e06f, -1e06f}, tcnn::vec3{1e06f, 1e06f, 1e06f}};

    ErrorCode status = m_renderer->renderBackward(
        renderParameters,
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayOrigin)),
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayDirection)),
        reinterpret_cast<float*>(voidDataPtr(rayHitDistance)),
        reinterpret_cast<float*>(voidDataPtr(rayHitDistanceGradient)),
        reinterpret_cast<tcnn::vec4*>(voidDataPtr(rayRadianceDensity)),
        reinterpret_cast<tcnn::vec4*>(voidDataPtr(rayRadianceDensityGradient)),
        rayBackpropagation ? reinterpret_cast<tcnn::vec3*>(voidDataPtr(rayOriginGradient)) : nullptr,
        rayBackpropagation ? reinterpret_cast<tcnn::vec3*>(voidDataPtr(rayDirectionGradient)) : nullptr,
        m_parameters, cudaDeviceIndex, cudaStream);

    CUDA_CHECK_LAST(m_logger);

    if (m_enableKernelTimings) {
        timer->stop();
    }

    return std::tuple<torch::Tensor, torch::Tensor>(particleDensityGradient, particleRadianceGradient);
}

std::map<std::string, float>
SplatRaster::collectTimes() {

    if (!m_timers.empty()) {

        std::map<std::string, float> timings;
        std::map<std::string, int> counts;

        for (auto& timer : m_timers) {
            const auto tag = timer->tag();

            if (timings.count(tag) == 0) {
                timings[tag] = timer->collect();
                counts[tag]  = 1;
            }

            else {
                timings[tag] += timer->collect();
                counts[tag] += 1;
            }
        }

        for (auto& [tag, time] : timings) {
            m_timings[tag] = time / counts[tag];
        }

        m_timers.clear();
    }

    return m_timings;
}
