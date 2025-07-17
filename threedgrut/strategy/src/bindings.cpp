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
// MCMC implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/Relocation.cpp), 
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
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
    
std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
    const torch::Tensor opacities,
    const torch::Tensor scales,
    const torch::Tensor ratios,
    const torch::Tensor binoms,
    const int n_max) {
    CHECK_INPUT(opacities);
    CHECK_INPUT(scales);
    CHECK_INPUT(ratios);
    CHECK_INPUT(binoms);

    return compute_relocation_tensor_cu(opacities, scales, ratios, binoms, n_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_relocation_tensor", &compute_relocation_tensor);
}