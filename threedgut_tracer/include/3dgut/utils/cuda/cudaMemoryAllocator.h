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

#include <3dgut/utils/cuda/cudaCommon.h>
#include <3dgut/utils/status.h>

namespace threedgut {

struct CudaMemoryAllocator final {

    CudaMemoryAllocator()                           = default;
    CudaMemoryAllocator(CudaMemoryAllocator const&) = delete;
    void operator=(CudaMemoryAllocator const&)      = delete;

public:
    ~CudaMemoryAllocator() = default;

    static inline CudaMemoryAllocator& get() {
        static CudaMemoryAllocator instance;
        return instance;
    }

    static inline size_t deviceUsedMemory(int deviceIndex) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return total - free;
    }

    inline Status allocateAsync(void*& ptr, size_t size, cudaStream_t stream, const Logger& logger) {
        CUDA_CHECK_RETURN(cudaMallocAsync(&ptr, size, stream), logger);
        m_totalAllocatedSize += size;
        return Status();
    }

    inline Status freeAsync(void* ptr, size_t size, cudaStream_t stream, const Logger& logger) {
        CUDA_CHECK_RETURN(cudaFreeAsync(ptr, stream), logger);
        if (m_totalAllocatedSize < size) {
            RETURN_ERROR(logger,
                         ErrorCode::BadInput,
                         "CudaMemoryAllocator : freeing more memory than allocated [%lu/%lu]",
                         static_cast<uint64_t>(size), static_cast<uint64_t>(m_totalAllocatedSize));
        }
        m_totalAllocatedSize -= size;
        return Status();
    }

    inline void free(void* ptr, size_t size) {
        cudaFree(ptr);
        m_totalAllocatedSize -= size;
    }

    inline size_t currentlyAllocated() const {
        return m_totalAllocatedSize;
    }

private:
    size_t m_totalAllocatedSize = 0;
};

} // namespace threedgut