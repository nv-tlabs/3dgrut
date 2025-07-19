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
//
//
// Selective Adam implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/Adam.cpp),
// which is based on the original implementation https://github.com/humansensinglab/taming-3dgs that uderlines the work
//
// Taming 3DGS: High-Quality Radiance Fields with Limited Resources by
// Saswat Subhajyoti Mallick*, Rahul Goel*, Bernhard Kerbl, Francisco Vicente Carrasco, Markus Steinberger and Fernando De La Torre
//
// If you use this code in your research, please cite the above works.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <torch/extension.h>

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_CUDA(x) \
    {                            \
        CHECK_CUDA(x);           \
        CHECK_CONTIGUOUS(x);     \
    }

namespace threedgrut
{
    void selective_adam_update_launch(
        at::Tensor &param,            // [N, ...]
        const at::Tensor &param_grad, // [N, ...]
        at::Tensor &exp_avg,          // [N, ...]
        at::Tensor &exp_avg_sq,       // [N, ...]
        const at::Tensor &visibility, // [N]
        const float lr,
        const float b1,
        const float b2,
        const float eps);

    void selective_adam_update(
        torch::Tensor &param,            // [N, ...]
        const torch::Tensor &param_grad, // [N, ...]
        torch::Tensor &exp_avg,          // [N, ...]
        torch::Tensor &exp_avg_sq,       // [N, ...]
        const torch::Tensor &visibility, // [N]
        const float lr,
        const float b1,
        const float b2,
        const float eps)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(param));

        CHECK_CONTIGUOUS_CUDA(param);
        CHECK_CONTIGUOUS_CUDA(param_grad);
        CHECK_CONTIGUOUS_CUDA(exp_avg);
        CHECK_CONTIGUOUS_CUDA(exp_avg_sq);
        CHECK_CONTIGUOUS_CUDA(visibility);

        TORCH_CHECK(visibility.dim() == 1, "'visibility' should be 1D tensor");
        TORCH_CHECK(visibility.size(0) == param.size(0), "'visibility' first dimension should have the same batch size as 'param'");

        selective_adam_update_launch(param, param_grad, exp_avg, exp_avg_sq, visibility, lr, b1, b2, eps);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("selective_adam_update", &threedgrut::selective_adam_update);
}
