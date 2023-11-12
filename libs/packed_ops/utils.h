// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor packed_weighted_sum_cu(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor rays_a
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_two_packs_sorted_aligned_fw_cu(
    const torch::Tensor vals_a, 
    const torch::Tensor pack_infos_a,
    const torch::Tensor vals_b,
    const torch::Tensor pack_infos_b
);

torch::Tensor packed_cumsum_cu(
    const torch::Tensor data,
    const torch::Tensor pack_infos, 
    bool exclusive,
    bool reverse
);

torch::Tensor packed_cumprod_cu(
    const torch::Tensor data,
    const torch::Tensor pack_infos, 
    bool exclusive,
    bool reverse
);

torch::Tensor packed_add_cu(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
);

torch::Tensor packed_sub_cu(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
);

torch::Tensor packed_mul_cu(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
);

torch::Tensor packed_div_cu(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
);

std::tuple<torch::Tensor, torch::Tensor> packed_invert_cdf_cu(
    const torch::Tensor bins, 
    const torch::Tensor cdfs, 
    const torch::Tensor u_vals,
    const torch::Tensor pack_infos
);

torch::Tensor packed_sum_cu(
    const torch::Tensor data,
    const torch::Tensor pack_infos
);
