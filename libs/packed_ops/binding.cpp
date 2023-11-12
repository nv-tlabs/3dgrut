// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#include "utils.h"
#include <vector>

torch::Tensor packed_weighted_sum(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor rays_a
){
    CHECK_INPUT(data);
    CHECK_INPUT(weights);
    CHECK_INPUT(rays_a);

    return packed_weighted_sum_cu(data, weights, rays_a);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  merge_two_packs_sorted_aligned_fw(
    const torch::Tensor vals_a, 
    const torch::Tensor pack_infos_a,
    const torch::Tensor vals_b,
    const torch::Tensor pack_infos_b
){
    CHECK_INPUT(vals_a);
    CHECK_INPUT(pack_infos_a);
    CHECK_INPUT(vals_b);
    CHECK_INPUT(pack_infos_b);

    return merge_two_packs_sorted_aligned_fw_cu(vals_a, pack_infos_a, vals_b, pack_infos_b);
}

torch::Tensor packed_cumsum(
    const torch::Tensor data,
    const torch::Tensor pack_infos, 
    bool exclusive,
    bool reverse
){
    CHECK_INPUT(data);
    CHECK_INPUT(pack_infos);

    return packed_cumsum_cu(data, pack_infos, exclusive, reverse);
}

torch::Tensor packed_cumprod(
    const torch::Tensor data,
    const torch::Tensor pack_infos, 
    bool exclusive,
    bool reverse
){
    CHECK_INPUT(data);
    CHECK_INPUT(pack_infos);

    return packed_cumprod_cu(data, pack_infos, exclusive, reverse);
}

torch::Tensor packed_add(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
){
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_infos);

    return packed_add_cu(data, other, pack_infos);
}

torch::Tensor packed_sub(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
){
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_infos);

    return packed_sub_cu(data, other, pack_infos);
}

torch::Tensor packed_mul(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
){
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_infos);

    return packed_mul_cu(data, other, pack_infos);
}

torch::Tensor packed_div(
    const torch::Tensor data, 
    const torch::Tensor other,
    const torch::Tensor pack_infos
){
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_infos);

    return packed_div_cu(data, other, pack_infos);
}

std::tuple<torch::Tensor, torch::Tensor> packed_invert_cdf(
    const torch::Tensor bins, 
    const torch::Tensor cdfs, 
    const torch::Tensor u_vals,
    const torch::Tensor pack_infos
){
    CHECK_INPUT(bins);
    CHECK_INPUT(cdfs);
    CHECK_INPUT(u_vals);
    CHECK_INPUT(pack_infos);

    return packed_invert_cdf_cu(bins, cdfs, u_vals, pack_infos);
}

torch::Tensor packed_sum(
    torch::Tensor data,
    torch::Tensor pack_infos
){   
    CHECK_INPUT(data);
    CHECK_INPUT(pack_infos);

    return packed_sum_cu(data, pack_infos);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("merge_two_packs_sorted_aligned_fw", &merge_two_packs_sorted_aligned_fw);
    m.def("packed_cumsum", &packed_cumsum);     
    m.def("packed_cumprod", &packed_cumprod);
    m.def("packed_add", &packed_add);
    m.def("packed_sub", &packed_sub);
    m.def("packed_mul", &packed_mul);
    m.def("packed_div", &packed_div);
    m.def("packed_sum", &packed_sum);
    m.def("packed_weighted_sum", &packed_weighted_sum);
    m.def("packed_invert_cdf", &packed_invert_cdf);
}
