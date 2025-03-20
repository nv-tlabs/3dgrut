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

#include <cstdint>
#include <cstring>

#include <iostream>
#include <vector>


#ifndef THREEDGUT_LOGGER_CB
#ifdef _WIN32
#define THREEDGUT_LOGGER_CB __cdecl
#else
#define THREEDGUT_LOGGER_CB
#endif
#endif

namespace threedgut {

// ----------------------------------------------------------------------------- 
// 
//

struct LoggerParameters {

    enum : uint8_t {
        Fatal,
        Error,
        Warning,
        Info,
        Debug,
        DebugSyncDevice,
        NumLevels
    };

    typedef void(THREEDGUT_LOGGER_CB* Callback)(uint8_t level, const char* msg, void* data);
    typedef void(THREEDGUT_LOGGER_CB* DeviceLaunchCallback)(bool start,
                                                        const char* tag,
                                                        int deviceIndex,
                                                        uint64_t deviceQueue,
                                                        void* data);

    uint8_t maximumLevel                     = Error;
    Callback callback                        = nullptr;
    void* callbackData                       = nullptr;
    DeviceLaunchCallback deviceLauncCallback = nullptr;
    void* deviceLaunchCallbackData           = nullptr;

    static const char* levelToString(uint8_t level) {
        switch (level) {
        case Fatal:
            return "FATAL";
        case Error:
            return "ERROR";
        case Warning:
            return "WARNING";
        case Info:
            return "INFO";
        case Debug:
            return "DEBUG";
        case DebugSyncDevice:
            return "DEBUG-SYNC-DEVICE";
        }
        return "UNDEFINED";
    }
};

// ----------------------------------------------------------------------------- 
// 
//

class Logger {
public:
    static void THREEDGUT_LOGGER_CB defaultLogCallback(uint8_t level, const char* msg, void* data) {
        std::ostream& stream = (level > LoggerParameters::Error) ? std::cout : std::cerr;
        stream << "[3dgut][" << LoggerParameters::levelToString(level) << "] ::: " << msg << std::flush << std::endl;
    }

    Logger(const LoggerParameters& params)
        : m_maxLogLevel(params.maximumLevel), m_logFn(params.callback), m_logFnData(params.callbackData), m_logDeviceLaunchFn(params.deviceLauncCallback), m_logDeviceLaunchFnData(params.deviceLaunchCallbackData) {
    }
    Logger(uint8_t maxLogLevel                                            = LoggerParameters::Error,
           LoggerParameters::Callback logFn                               = defaultLogCallback,
           void* logFnData                                                = nullptr,
           LoggerParameters::DeviceLaunchCallback logDeviceLaunchCallback = nullptr,
           void* logDeviceLaunchCallbackData                              = nullptr)
        : m_maxLogLevel(maxLogLevel), m_logFn(logFn), m_logFnData(logFnData), m_logDeviceLaunchFn(logDeviceLaunchCallback), m_logDeviceLaunchFnData(logDeviceLaunchCallbackData) {
    }
    virtual ~Logger() = default;

    virtual inline void log(uint8_t level, const char* message) const {
        if (level <= m_maxLogLevel) {
            m_logFn(level, message, m_logFnData);
        }
    }

    virtual inline void logDeviceLaunch(bool start, const char* tag, int deviceIndex, uint64_t deviceQueue) const {
        if (m_logDeviceLaunchFn) {
            m_logDeviceLaunchFn(start, tag, deviceIndex, deviceQueue, m_logDeviceLaunchFnData);
        }
    }

    inline uint8_t level() const {
        return m_maxLogLevel;
    }

    inline bool deviceLaunchEnabled() const {
        return m_logDeviceLaunchFn;
    }

private:
    uint8_t m_maxLogLevel;

    LoggerParameters::Callback m_logFn;
    void* m_logFnData;
    LoggerParameters::DeviceLaunchCallback m_logDeviceLaunchFn;
    void* m_logDeviceLaunchFnData;
};

#define LOG_FMT(logger, level, fmt, ...)                 \
    do {                                                 \
        constexpr uint16_t maxMsgLength = 1024;          \
        char msg[maxMsgLength];                          \
        snprintf(msg, maxMsgLength, fmt, ##__VA_ARGS__); \
        logger.log(level, msg);                          \
    } while (0)

#define LOG_DEBUG(logger, fmt, ...) LOG_FMT(logger, threedgut::LoggerParameters::Debug, fmt, ##__VA_ARGS__)
#define LOG_INFO(logger, fmt, ...) LOG_FMT(logger, threedgut::LoggerParameters::Info, fmt, ##__VA_ARGS__)
#define LOG_WARN(logger, fmt, ...) LOG_FMT(logger, threedgut::LoggerParameters::Warning, fmt, ##__VA_ARGS__)
#define LOG_ERROR(logger, fmt, ...) LOG_FMT(logger, threedgut::LoggerParameters::Error, fmt, ##__VA_ARGS__)
#define LOG_FATAL(logger, fmt, ...) LOG_FMT(logger, threedgut::LoggerParameters::Fatal, fmt, ##__VA_ARGS__)

#define PROFILE_DEVICE_START(logger, tag, deviceIndex, deviceQueue) \
    logger.logDeviceLaunch(true, tag, deviceIndex, deviceQueue)
#define PROFILE_DEVICE_END(logger, tag, deviceIndex, deviceQueue) \
    logger.logDeviceLaunch(false, tag, deviceIndex, deviceQueue)

// ----------------------------------------------------------------------------- 
// 
//

class DeviceLaunchesLogger {
private:
    const Logger& m_logger;
    const int m_deviceIndex;
    const uint64_t m_deviceQueue;
    std::vector<const char*> m_profilingTags;

    inline bool enabled() const { return m_logger.deviceLaunchEnabled(); }

public:
    DeviceLaunchesLogger(const Logger& logger, int deviceIndex, uint64_t deviceQueue)
        : m_logger(logger), m_deviceIndex(deviceIndex), m_deviceQueue(deviceQueue) {
        if (enabled()) {
            m_profilingTags.reserve(8);
        }
    }
    ~DeviceLaunchesLogger() {
        for (size_t i = m_profilingTags.size(); i > 0; --i) {
            PROFILE_DEVICE_END(m_logger, m_profilingTags[i - 1], m_deviceIndex, m_deviceQueue);
        }
    }

    inline void push(const char* tag) {
        if (enabled()) {
            m_profilingTags.push_back(tag);
            PROFILE_DEVICE_START(m_logger, tag, m_deviceIndex, m_deviceQueue);
        }
    };

    inline void pop() {
        if (enabled()) {
            if (m_profilingTags.empty()) {
                LOG_ERROR(m_logger, "DeviceLaunchesLogger : cannot pop : no pushed launch");
            } else {
                PROFILE_DEVICE_END(m_logger, m_profilingTags.back(), m_deviceIndex, m_deviceQueue);
                m_profilingTags.pop_back();
            }
        }
    };

    inline void pop(const char* tag) {
        if (enabled()) {
            PROFILE_DEVICE_END(m_logger, tag, m_deviceIndex, m_deviceQueue);
            if (strcmp(m_profilingTags.back(), tag)) {
                LOG_ERROR(m_logger, "LogProfiler wrong tag : %s / %s.", m_profilingTags.back(), tag);
            } else {
                m_profilingTags.pop_back();
            }
        }
    };

    class ScopePush final {
        DeviceLaunchesLogger& m_logger;

    public:
        ScopePush(DeviceLaunchesLogger& logger, const char* tag)
            : m_logger(logger) {
            m_logger.push(tag);
        }
        ~ScopePush() {
            m_logger.pop();
        }
    };
};

} // namespace threedgut
