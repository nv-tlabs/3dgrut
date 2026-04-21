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

#include <3dgrt/tensorAccessor.h>

//------------------------------------------------------------------------------
// Torch access
//------------------------------------------------------------------------------

template <typename scalar_t>
inline scalar_t* getPtr(torch::Tensor tensor) {
    if (tensor.dtype() == torch::kInt32) {
        return reinterpret_cast<scalar_t*>(tensor.contiguous().data_ptr<int>());
    } else if (tensor.dtype() == torch::kFloat32) {
        return reinterpret_cast<scalar_t*>(tensor.contiguous().data_ptr<float>());
    } else if (tensor.dtype() == torch::kHalf) {
        return reinterpret_cast<scalar_t*>(tensor.contiguous().data_ptr<at::Half>());
    } else {
        throw std::runtime_error("getPtr(tensor) received a tensor of unsupported type");
    }
}

// Note: `reinterpret_cast` is used on the raw `tensor.data_ptr()` so that this
// template instantiates for element types without a PyTorch scalar_type entry
// (e.g. `__half`). Dtype agreement is the caller's responsibility.
template <class T, int N, template <typename U> class PtrTraits = DefaultPtrTraits>
PackedTensorAccessor32<T, N> packed_accessor32(torch::Tensor tensor) {
    return PackedTensorAccessor32<T, N, PtrTraits>(reinterpret_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr()),
                                                   tensor.sizes().data(), tensor.strides().data());
}