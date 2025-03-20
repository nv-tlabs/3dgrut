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

#include <3dgut/utils/status.h>

#include <cuda_runtime.h>

#define CUDA_SUCCEEDED(result) ((result) == cudaSuccess)
#define CUDA_FAILED(result) ((result) != cudaSuccess)

#define CUDA_CHECK(logger, result)                                                                                     \
    {                                                                                                                  \
        cudaError_t _result = (result);                                                                                \
        if (CUDA_FAILED(_result)) {                                                                                    \
            LOG_ERROR(                                                                                                 \
                logger, "CUDA error %d: %s - %s)", (_result), cudaGetErrorName(_result), cudaGetErrorString(_result)); \
        }                                                                                                              \
    }

#define CUDA_CHECK_RETURN(result, logger)                                                   \
    do {                                                                                    \
        cudaError_t _result = result;                                                       \
        if (CUDA_FAILED(_result)) {                                                         \
            _SET_ERROR(logger, ErrorCode::Runtime, #result " failed: %d: %s - %s at %s:%d", \
                       (_result), cudaGetErrorName(_result), cudaGetErrorString(_result),   \
                       __FILE__, __LINE__);                                                 \
            return ___status;                                                               \
        }                                                                                   \
    } while (0)

#define CUDA_CHECK_STREAM_RETURN(stream, logger)                             \
    do {                                                                     \
        if (logger.level() >= LoggerParameters::DebugSyncDevice) {           \
            cudaStreamSynchronize(stream);                                   \
            std::cout << "<<< " << __FILE__ << "@" << __LINE__ << std::endl; \
        }                                                                    \
        CUDA_CHECK_RETURN(cudaGetLastError(), logger);                       \
    } while (0)

/// @brief : a scoped cuda device guard
class CudaCheckDeviceGuard {
    int _prevDeviceIndex = -1;
    bool _check          = true;

public:
    CudaCheckDeviceGuard(int deviceIndex) {
        int currDeviceIndex;
        _check = CUDA_SUCCEEDED(cudaGetDevice(&currDeviceIndex));
        if (_check && (deviceIndex != currDeviceIndex)) {
            _prevDeviceIndex = currDeviceIndex;
            _check           = CUDA_SUCCEEDED(cudaSetDevice(deviceIndex));
        }
    }
    ~CudaCheckDeviceGuard() {
        if (_check && _prevDeviceIndex != -1) {
            cudaSetDevice(_prevDeviceIndex);
        }
    }
    inline bool check() const {
        return _check;
    }
};
