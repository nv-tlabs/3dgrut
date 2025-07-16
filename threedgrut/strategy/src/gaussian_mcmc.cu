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
// MCMC implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/mcmc.py), 
// which is based on the original implementation https://github.com/ubc-vision/3dgs-mcmc that uderlines the work
//
// 3D Gaussian Splatting as Markov Chain Monte Carlo by 
// Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi and Kwang Moo Yie
//
// If you use this code in your research, please cite the above works.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "gaussian_mcmc.h"
#include <ATen/cuda/CUDAContext.h>

__global__ void compute_relocation_kernel(
    const int N,
    const float* __restrict__ opacities,
    const float* __restrict__ scales,
    const int* __restrict__ ratios,
    const float* __restrict__ binoms,
    const int n_max,
    float* __restrict__ new_opacities,
    float* __restrict__ new_scales) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    const int n_idx       = ratios[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    const float new_opaticty  = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx);
    new_opacities[idx] = new_opaticty;

    // compute new scale
    #pragma unroll
    for (int i = 1; i <= n_idx; ++i) {
        #pragma unroll
        for (int k = 0; k <= (i - 1); ++k) {
            const float bin_coeff = binoms[(i - 1) * n_max + k];
            const float term      = (pow(-1.0f, k) / sqrt(static_cast<float>(k + 1))) *
                         pow(new_opaticty, k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
       
    const float coeff = opacities[idx] / denom_sum;
    #pragma unroll 3
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
}

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor_cu(
    const torch::Tensor opacities,
    const torch::Tensor scales,
    const torch::Tensor ratios,
    const torch::Tensor binoms,
    const int max_num_gaussians) {

    const int32_t num_gaussians = opacities.size(0);
    TORCH_CHECK(scales.size(0) == num_gaussians, "scales size mismatch");
    TORCH_CHECK(ratios.size(0) == num_gaussians, "ratios size mismatch");

    torch::Tensor new_opacities = torch::empty_like(opacities);
    torch::Tensor new_scales    = torch::empty_like(scales);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256, blocks = (num_gaussians + threads - 1) / threads;

    if (num_gaussians) {
        compute_relocation_kernel<<<blocks, threads, 0, stream>>>(
            num_gaussians,
            opacities.data_ptr<float>(),
            scales.data_ptr<float>(),
            ratios.data_ptr<int>(),
            binoms.data_ptr<float>(),
            max_num_gaussians,
            new_opacities.data_ptr<float>(),
            new_scales.data_ptr<float>());
    }

    return std::make_tuple(new_opacities, new_scales);
}