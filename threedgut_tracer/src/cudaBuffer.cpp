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

#include <3dgut/utils/cuda/cudaBuffer.h>
#include <3dgut/utils/cuda/cudaMemoryAllocator.h>
#include <3dgut/utils/status.h>

threedgut::ErrorCode threedgut::Status::_lastError;

threedgut::CudaBuffer::~CudaBuffer() {
    if (m_owner && (m_size > 0)) {
        CudaMemoryAllocator::get().free(m_data, m_size);
    }
}

size_t threedgut::CudaBuffer::size() const {
    return m_size;
}

const void* threedgut::CudaBuffer::data() const {
    return m_data;
}

void* threedgut::CudaBuffer::data() {
    return m_data;
}

uint64_t threedgut::CudaBuffer::handle() const {
    return reinterpret_cast<uint64_t>(m_data);
}

threedgut::Status threedgut::CudaBuffer::resize(size_t size, uint64_t processQueueHandle, const Logger& logger) {
    if (size != m_size) {
        CHECK_STATUS_RETURN(clear(processQueueHandle, logger));
        if (size > 0) {
            CHECK_STATUS_RETURN(CudaMemoryAllocator::get().allocateAsync(m_data, size, reinterpret_cast<cudaStream_t>(processQueueHandle), logger));
        }
        m_size = size;
    }
    return Status();
}

threedgut::Status threedgut::CudaBuffer::enlarge(size_t size, uint64_t processQueueHandle, const Logger& logger) {
    if (size <= m_size) {
        return Status();
    }
    return resize(size, processQueueHandle, logger);
}

threedgut::Status threedgut::CudaBuffer::setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger) {

    Status status = detach(false, processQueueHandle, logger);
    if (!status) {
        return status;
    }

    status = resize(size, processQueueHandle, logger);
    if (status) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(m_data, hostMemory, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(processQueueHandle)), logger);
    }
    return status;
}

threedgut::Status threedgut::CudaBuffer::setFromDevice(const void* deviceMemory, size_t size, bool attach, uint64_t processQueueHandle, const Logger& logger) {

    Status status = detach(false, processQueueHandle, logger);
    if (!status) {
        return status;
    }

    if (attach) {
        status = clear(processQueueHandle, logger);
        if (status) {
            m_size  = size;
            m_data  = const_cast<void*>(deviceMemory);
            m_owner = false;
        }
    } else {
        status = resize(size, processQueueHandle, logger);
        if (status) {
            CUDA_CHECK_RETURN(cudaMemcpyAsync(m_data, deviceMemory, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(processQueueHandle)), logger);
        }
    }

    return status;
}

bool threedgut::CudaBuffer::attached() const {
    return !m_owner;
}

threedgut::Status threedgut::CudaBuffer::detach(bool copy, uint64_t processQueueHandle, const Logger& logger) {
    if (!attached()) {
        return Status();
    }
    return copy ? setFromDevice(m_data, m_size, false, processQueueHandle, logger) : clear(processQueueHandle, logger);
}

threedgut::Status threedgut::CudaBuffer::clear(uint64_t processQueueHandle, const Logger& logger) {
    if (m_owner && (m_size > 0)) {
        CHECK_STATUS_RETURN(CudaMemoryAllocator::get().freeAsync(m_data, m_size, reinterpret_cast<cudaStream_t>(processQueueHandle), logger));
    }
    m_size  = 0;
    m_data  = nullptr;
    m_owner = true;

    return Status();
}