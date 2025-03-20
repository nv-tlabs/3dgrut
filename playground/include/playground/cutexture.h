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

template <typename T, int C=1>
class CudaTexture2DFloatObject final {
    cudaArray_t _array = 0;
    cudaTextureObject_t _tex = 0;

    void release() {
        if (_tex) {
            cudaDestroyTextureObject(_tex);
            _tex = 0;
        }
        if (_array) {
            cudaFreeArray(_array);
            _array = 0;
        }
    };

public:
    CudaTexture2DFloatObject() = default;
    CudaTexture2DFloatObject(const T* hostData, const int height, const int width) {
        static_assert((C>0) && (C<5), "CudaTexture2DFloatObject");
        reset(hostData, height, width);
    }
    ~CudaTexture2DFloatObject() { release(); }

    void reset(const T* hostData, const int height, const int width) {
        release();
        constexpr int nBytes = sizeof(T) * 8;
        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(nBytes, (C>1 ? nBytes : 0),  (C>2 ? nBytes : 0),  (C>3 ? nBytes : 0), cudaChannelFormatKindFloat);
        cudaMallocArray(&_array, &channelDesc, width, height);

        cudaMemcpy2DToArray(
            _array, 0, 0, hostData, width * sizeof(T) * C, width * sizeof(T) * C, height, cudaMemcpyDeviceToDevice
        );

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = _array;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&_tex, &resDesc, &texDesc, NULL);
    }
    inline const cudaTextureObject_t& tex() const { return _tex; }
    inline const bool isTexInitialized() const { return _tex != 0; }
};

using CudaTexture2DFloat2Object = CudaTexture2DFloatObject<float, 2>;
using CudaTexture2DFloat4Object = CudaTexture2DFloatObject<float, 4>;
using CudaTexture2DFloat1Object = CudaTexture2DFloatObject<float, 1>;
