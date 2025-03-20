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

#include <3dgut/utils/logger.h>

namespace threedgut {

enum class ErrorCode {
    None,                  ///< No error == success
    InvalidResourcesError, ///< A resource is not valid
    BadInput,              ///< An argument has an unexpected value
    OutOfMemory,           ///< Out of memory (allocation) error
    NotImplemented,        ///< Calling of function that is not implemented
    Runtime,               ///< Generic runtime error
    Num                    ///< Number of valid error codes
};

#define THREEDGUT_SUCCESS(errorCode) (static_cast<threedgut::ErrorCode>(errorCode) == threedgut::ErrorCode::None)
#define THREEDGUT_FAILED(errorCode) (static_cast<threedgut::ErrorCode>(errorCode) != threedgut::ErrorCode::None)

class Status {
public:
    inline static ErrorCode getLastError() {
        ErrorCode lastError = _lastError;
        _lastError          = ErrorCode::None;
        return lastError;
    };

    inline static const char* message(ErrorCode err) {
        return _errorMessage[static_cast<uint16_t>(err)];
    }

    Status(ErrorCode error = ErrorCode::None, bool setLast = true)
        : _error(error) {
        if ((error != ErrorCode::None) && setLast) {
            _lastError = error;
        }
    }

    inline bool success() const { return _error == ErrorCode::None; };

    inline operator bool() const { return success(); }

    inline operator ErrorCode() const { return _error; }

private:
    ErrorCode _error;

private:
    static constexpr char const* _errorMessage[static_cast<uint16_t>(ErrorCode::Num)] = {
        "Success",
        "Invalid resources",
        "Invalid inputs",
        "Out of memory",
        "Missing implementation"};
    static ErrorCode _lastError;
};

#define CHECK_STATUS_RETURN(status)   \
    do {                              \
        const auto __status = status; \
        if (!__status) {              \
            return __status;          \
        }                             \
    } while (0)

#define _SET_ERROR(logger, error, fmt, ...)  \
    static_assert(error != ErrorCode::None); \
    LOG_ERROR(logger, fmt, ##__VA_ARGS__);   \
    [[maybe_unused]] auto ___status = Status(error);

#define SET_ERROR(logger, error, fmt, ...)             \
    do {                                               \
        _SET_ERROR(logger, error, fmt, ##__VA_ARGS__); \
    } while (0)

#define RETURN_ERROR(logger, error, fmt, ...)          \
    do {                                               \
        _SET_ERROR(logger, error, fmt, ##__VA_ARGS__); \
        return ___status;                              \
    } while (0)

} // namespace threedgut
