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

class CudaBuffer {
public:
    ~CudaBuffer();

    size_t size() const;
    
    const void* data() const;
    
    void* data();

    uint64_t handle() const;

    Status resize(size_t size, uint64_t processQueueHandle, const Logger& logger);

    Status enlarge(size_t size, uint64_t processQueueHandle, const Logger& logger);

    Status setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger);

    Status setFromDevice(const void* deviceMemory, size_t size, bool attach, uint64_t processQueueHandle, const Logger& logger);

    bool attached() const;

    Status detach(bool copy, uint64_t processQueueHandle, const Logger& logger);

    Status clear(uint64_t processQueueHandle, const Logger& logger);

private:
    size_t m_size = 0;
    void* m_data  = nullptr;
    bool m_owner  = true;
};

} // namespace threedgut