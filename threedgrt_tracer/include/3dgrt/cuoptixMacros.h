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

//------------------------------------------------------------------------------
// CUDA / OPTIX macros
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = call;                                     \
        if (error != cudaSuccess) {                                   \
            std::stringstream ss;                                     \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << cudaGetErrorString(error)                           \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";         \
        }                                                             \
    } while (0)

#define CUDA_CHECK_LAST() \
    CUDA_CHECK(cudaGetLastError())

#define OPTIX_CHECK(call)                                              \
    do {                                                               \
        OptixResult res = call;                                        \
        if (res != OPTIX_SUCCESS) {                                    \
            std::stringstream ss;                                      \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":" \
               << __LINE__ << ")\n";                                   \
        }                                                              \
    } while (0)

#define OPTIX_CHECK_LOG(call)                                                                   \
    do {                                                                                        \
        OptixResult res                  = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                                          \
        sizeof_log                       = sizeof(log); /* reset sizeof_log for future calls */ \
        if (res != OPTIX_SUCCESS) {                                                             \
            std::stringstream ss;                                                               \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"                          \
               << __LINE__ << ")\nLog:\n"                                                       \
               << log                                                                           \
               << (sizeof_log_returned > sizeof(log) ? "<TRUNCATED>" : "")                      \
               << "\n";                                                                         \
        }                                                                                       \
    } while (0)

#define NVRTC_CHECK_ERROR(func)                                                                           \
    do {                                                                                                  \
        nvrtcResult code = func;                                                                          \
        if (code != NVRTC_SUCCESS)                                                                        \
            throw std::runtime_error("ERROR: " __FILE__ "(): " + std::string(nvrtcGetErrorString(code))); \
    } while (0)

// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS                                                                                                                 \
    "-std=c++14", "-arch", "compute_75", "-use_fast_math", "-lineinfo", "--extra-device-vectorization", "-default-device", "-rdc", "true", \
        "-D__OPTIX__"
