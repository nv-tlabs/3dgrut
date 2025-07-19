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
// Selective Adam implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/AdamCUDA.cu),
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

#include <ATen/Dispatch.h> // AT_DISPATCH_XXX
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace threedgrut
{

    // step on a grid of size (N, M)
    // N is always number of gaussians

    template <typename scalar_t>
    __global__ void selective_adam_update_kernel(
        scalar_t *__restrict__ param,
        const scalar_t *__restrict__ param_grad,
        scalar_t *__restrict__ exp_avg,
        scalar_t *__restrict__ exp_avg_sq,
        const bool *__restrict__ visibility,
        const float lr,
        const float b1,
        const float b2,
        const float eps,
        const uint32_t N,
        const uint32_t M)
    {
        auto p_idx = cg::this_grid().thread_rank();
        const uint32_t g_idx = p_idx / M;

        if (g_idx >= N)
            return;

        if (!visibility[g_idx])
            return;

        const float register_param_grad = param_grad[p_idx];

        float register_exp_avg = exp_avg[p_idx];
        float register_exp_avg_sq = exp_avg_sq[p_idx];
        register_exp_avg = b1 * register_exp_avg + (1.0f - b1) * register_param_grad;
        register_exp_avg_sq = b2 * register_exp_avg_sq + (1.0f - b2) * register_param_grad * register_param_grad;

        const float step = -lr * register_exp_avg / (sqrt(register_exp_avg_sq) + eps);

        param[p_idx] += step;
        exp_avg[p_idx] = register_exp_avg;
        exp_avg_sq[p_idx] = register_exp_avg_sq;
    }

    void selective_adam_update_launch(
        at::Tensor &param,            // [N, ...]
        const at::Tensor &param_grad, // [N, ...]
        at::Tensor &exp_avg,          // [N, ...]
        at::Tensor &exp_avg_sq,       // [N, ...]
        const at::Tensor &visibility, // [N]
        const float lr,
        const float b1,
        const float b2,
        const float eps)
    {
        const uint32_t N = param.size(0);
        const uint32_t M = param.numel() / N;
        const uint32_t n_elements = N * M;

        // skip the kernel launch if there are no elements
        if (n_elements == 0)
            return;

        // clang-format off
        AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adam_kernel", [&]() { 
            selective_adam_update_kernel<scalar_t>
                <<<(n_elements + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    param.data_ptr<scalar_t>(),
                    param_grad.data_ptr<scalar_t>(),
                    exp_avg.data_ptr<scalar_t>(),
                    exp_avg_sq.data_ptr<scalar_t>(),
                    visibility.data_ptr<bool>(),
                    lr, b1, b2, eps, N, M
                );
        });
        // clang-format on
    }

}
