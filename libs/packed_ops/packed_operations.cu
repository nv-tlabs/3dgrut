// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#include "utils.h"

template <typename scalar_t>
inline __device__ int32_t binary_search_unsafe(scalar_t val, const scalar_t* data, int32_t length) {
    // Borrowed from intant-ngp
    // Returns "right" bound index of the found interval.
    // (None or data[return-1]) <= val < data[return]
    // Allows val less than the minimum data.
    // Disallows val larger than the maximum data. (will return wrong index)
    if (length == 0) {
		return 0;
	}
	int32_t it;
	int32_t count, step;
	count = length;

	int32_t first = 0;
	while (count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if (data[it] < val) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}
	return first;
}

template <typename scalar_t>
inline __device__ int32_t binary_search(scalar_t val, const scalar_t* data, int32_t length) {
	if (length == 0) {
		return 0;
	}
	int32_t first = binary_search_unsafe<scalar_t>(val, data, length);
	return std::min(first, length-1);
}

template <typename scalar_t>
__global__ void merge_two_packs_sorted_aligned_fw_kernel(
    const int32_t num_packs, 
    const int32_t num_feats_a, 
    const scalar_t* __restrict__ vals_a, 
    const int32_t* __restrict__ pack_infos_a, 
    const int32_t num_feats_b, 
    const scalar_t* __restrict__ vals_b, 
    const int32_t* __restrict__ pack_infos_b, 
    const int32_t* __restrict__ pack_infos_merged,
    int32_t* __restrict__ pidx_a,
    int32_t* __restrict__ pidx_b
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    const int32_t begin_a = pack_infos_a[tidx * 2];
    const int32_t length_a = pack_infos_a[tidx * 2 + 1];

    const int32_t begin_b = pack_infos_b[tidx * 2];
    const int32_t length_b = pack_infos_b[tidx * 2 + 1];

    const int32_t begin_out = pack_infos_merged[tidx * 2];

    vals_a += begin_a;
    pidx_a += begin_a;

    vals_b += begin_b;
    pidx_b += begin_b;
    

    // If tere are no samples in a skip this step and directly fill in the pidx_b
    if (length_a > 0){
        int32_t last_i = 0;
        for (int32_t j=0; j < length_b; ++j) {
            int32_t i = binary_search_unsafe<scalar_t>(vals_b[j], vals_a+last_i, length_a-last_i) + last_i;
            pidx_b[j] = i;
            if (i < length_a) pidx_a[i]++;
            last_i = i;
        }

        // From i count to `pidx_a` offset
        pidx_a[0] += begin_out;
        for (int32_t i=1; i < length_a; ++i) {
            pidx_a[i] += pidx_a[i-1] + 1;
        }
    
    }

    // From i to `pidx_b` offset
    int32_t acc = 1;
    int32_t last_i = -1;
    for (int32_t j=0; j < length_b; ++j) {
        int32_t i = pidx_b[j];
        pidx_b[j] = ((i==last_i) ? (++acc) : (acc=0)) + ((i==0) ? begin_out : (pidx_a[i-1]+1));
        last_i = i;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_two_packs_sorted_aligned_fw_cu(
    torch::Tensor vals_a, 
    torch::Tensor pack_infos_a,
    torch::Tensor vals_b,
    torch::Tensor pack_infos_b
) {

    const int32_t num_packs = pack_infos_a.size(0);
    const int32_t num_feats_a = vals_a.size(0);
    const int32_t num_feats_b = vals_b.size(0);

    torch::Tensor n_per_pack = pack_infos_a.select(1, 1) + pack_infos_b.select(1, 1);
    torch::Tensor cumsum = n_per_pack.cumsum(0, torch::kInt);
    torch::Tensor pack_infos = torch::stack({cumsum-n_per_pack, n_per_pack}, 1);

    torch::Tensor pidx_a = torch::zeros({ num_feats_a }, pack_infos_a.options());
    torch::Tensor pidx_b = torch::zeros({ num_feats_b }, pack_infos_b.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(vals_a.scalar_type(), "merge_two_packs_sorted_aligned_fw_cu", ([&] {
        merge_two_packs_sorted_aligned_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, 
            num_feats_a, 
            vals_a.data_ptr<scalar_t>(), 
            pack_infos_a.data_ptr<int32_t>(), 
            num_feats_b, 
            vals_b.data_ptr<scalar_t>(), 
            pack_infos_b.data_ptr<int32_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            pidx_a.data_ptr<int32_t>(), 
            pidx_b.data_ptr<int32_t>() 
        );
    }));

    return {pidx_a, pidx_b, pack_infos};
}


template<typename scalar_t>
__global__ void packed_cumsum_kernel(
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int32_t* __restrict__ pack_infos,  
    const int32_t offset,
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    if (offset == 0) {
        for (int32_t j=0; j<feat_dim; ++j){
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j){
        for (int32_t i=begin+1; i<end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i-offset) * feat_dim + j] + feats_out[(i-1) * feat_dim + j];
        }
    }  
}


template<typename scalar_t>
__global__ void packed_cumsum_reverse_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int32_t* __restrict__ pack_infos,
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    if (offset == 0) {
        for (int32_t j=0; j<feat_dim; ++j){
            feats_out[(end-1) * feat_dim + j] = feats_in[(end-1) * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j){
        for (int32_t i=end-2; i>=(int32_t)begin; --i) {
            feats_out[i * feat_dim + j] = feats_in[(i+offset) * feat_dim + j] + feats_out[(i+1) * feat_dim + j];
        }
    }  
}

torch::Tensor packed_cumsum_cu(
    torch::Tensor data,
    torch::Tensor pack_infos, 
    bool exclusive,
    bool reverse
) {

    int32_t num_feats = data.size(0);
    int32_t num_packs = pack_infos.size(0);
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());
    int32_t offset = exclusive ? 1 : 0;

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    if (reverse){
        AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_cumsum_cu", ([&] {
            packed_cumsum_reverse_kernel<scalar_t><<<blocks, threads>>>(
                num_packs, 
                num_feats, 
                feat_dim, 
                data.data_ptr<scalar_t>(), 
                pack_infos.data_ptr<int32_t>(), 
                offset, 
                feats_out.data_ptr<scalar_t>()
            );
        }));
    } else {
        AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_cumsum_cu", ([&] {
            packed_cumsum_kernel<scalar_t><<<blocks, threads>>>(
                num_packs, 
                num_feats, 
                feat_dim, 
                data.data_ptr<scalar_t>(), 
                pack_infos.data_ptr<int32_t>(), 
                offset, 
                feats_out.data_ptr<scalar_t>()
            );
        }));
    }

    return feats_out;
}


// Modified from https://github.com/NVIDIAGameWorks/kaolin
template<typename scalar_t>
__global__ void packed_cumprod_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int32_t* __restrict__ pack_infos,
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    
    if (begin == end) { 
      // NSHARP CHANGED: can't touch any of the arrays below if size == 0
      return;
    }

    if (offset == 0) {
        for (int32_t j=0; j<feat_dim; ++j){
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
    }

    if (offset == 1) {
        for (int32_t j=0; j<feat_dim; ++j){
            feats_out[begin * feat_dim + j] = 1.0;
        }
    }

    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j){
        for (int32_t i=begin+1; i<end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i-offset) * feat_dim + j] * feats_out[(i-1) * feat_dim + j];
        }
    }
}

template<typename scalar_t>
__global__ void packed_cumprod_reverse_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int32_t* __restrict__ pack_infos,
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];

    if (begin == end) return; // can't touch any of the arrays below if size == 0

    if (offset == 0) {
        for (int32_t j=0; j<feat_dim; ++j){
            feats_out[(end-1) * feat_dim + j] = feats_in[(end-1) * feat_dim + j];
        }
    }

    if (offset == 1) {
        for (int32_t j=0; j<feat_dim; ++j){
            feats_out[(end-1) * feat_dim + j] = 1.0;
        }
    }

    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j){
        for (int32_t i=end-2; i>=(int32_t)begin; --i) {
            feats_out[i * feat_dim + j] = feats_in[(i+offset) * feat_dim + j] * feats_out[(i+1) * feat_dim + j];
        }
    }  
}

torch::Tensor packed_cumprod_cu(
    torch::Tensor data, 
    torch::Tensor pack_infos, 
    bool exclusive,
    bool reverse
) {
    int32_t num_feats = data.size(0);
    int32_t num_packs = pack_infos.size(0);
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());
    int32_t offset = exclusive ? 1 : 0;

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    if (reverse) {
        AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_cumprod_cu", ([&] {
            packed_cumprod_reverse_kernel<scalar_t><<<blocks, threads>>>(
                num_packs, 
                num_feats, 
                feat_dim, 
                data.data_ptr<scalar_t>(), 
                pack_infos.data_ptr<int32_t>(),
                offset, 
                feats_out.data_ptr<scalar_t>()
            );
        }));
    } else {
        AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_cumprod_cu", ([&] {
            packed_cumprod_kernel<scalar_t><<<blocks, threads>>>(
                num_packs, 
                num_feats, 
                feat_dim, 
                data.data_ptr<scalar_t>(), 
                pack_infos.data_ptr<int32_t>(), 
                offset, 
                feats_out.data_ptr<scalar_t>()
            );
        }));
    }

    return feats_out;
}



template<typename scalar_t>
__global__ void packed_add_fw_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats, 
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in,
    const scalar_t* __restrict__ other_in,
    const int32_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j) {
        for (int32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] + other_in[j];
        }
    }
}

torch::Tensor packed_add_cu(
    torch::Tensor data, 
    torch::Tensor other,
    torch::Tensor pack_infos
) {

	torch::TensorArg data_arg(data, "data", 1);
	torch::TensorArg other_arg(other, "other", 2);
    torch::checkSameDim(__func__, data_arg, other_arg);

    int32_t num_packs = pack_infos.size(0);
    int32_t num_feats = data.size(0);
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor data_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_add_cu", ([&] {
        packed_add_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, num_feats, feat_dim, 
            data.data_ptr<scalar_t>(), 
            other.data_ptr<scalar_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            data_out.data_ptr<scalar_t>()
        );
    }));

    return data_out;
}

template<typename scalar_t>
__global__ void packed_sub_fw_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats, 
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in,
    const scalar_t* __restrict__ other_in,
    const int32_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j) {
        for (int32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] - other_in[j];
        }
    }
}

torch::Tensor packed_sub_cu(
    torch::Tensor data, 
    torch::Tensor other,
    torch::Tensor pack_infos
) {

	torch::TensorArg data_arg(data, "data", 1);
	torch::TensorArg other_arg(other, "other", 2);
    torch::checkSameDim(__func__, data_arg, other_arg);

    int32_t num_packs = pack_infos.size(0);
    int32_t num_feats = data.size(0);
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor data_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_sub_cu", ([&] {
        packed_sub_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, num_feats, feat_dim, 
            data.data_ptr<scalar_t>(), 
            other.data_ptr<scalar_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            data_out.data_ptr<scalar_t>()
        );
    }));

    return data_out;
}

template<typename scalar_t>
__global__ void packed_mul_fw_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats, 
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int32_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j) {
        for (int32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] * other_in[j];
        }
    }

}

torch::Tensor packed_mul_cu(
    torch::Tensor data, 
    torch::Tensor other,
    torch::Tensor pack_infos
) {

	torch::TensorArg data_arg(data, "data", 1);
	torch::TensorArg other_arg(other, "other", 2);
    torch::checkSameDim(__func__, data_arg, other_arg);

    int32_t num_packs = pack_infos.size(0);
    int32_t num_feats = data.size(0);
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor data_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_mul_cu", ([&] {
        packed_mul_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, num_feats, feat_dim, 
            data.data_ptr<scalar_t>(), 
            other.data_ptr<scalar_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            data_out.data_ptr<scalar_t>()
        );
    }));

    return data_out;
}


template<typename scalar_t>
__global__ void packed_div_fw_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats, 
    const int32_t feat_dim, 
    const scalar_t* __restrict__ feats_in,
    const scalar_t* __restrict__ other_in, 
    const int32_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (int32_t j=0; j<feat_dim; ++j) {
        for (int32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] / other_in[j];
        }
    }
}

torch::Tensor packed_div_cu(
    torch::Tensor data, 
    torch::Tensor other,
    torch::Tensor pack_infos
) {

	torch::TensorArg data_arg(data, "data", 1);
	torch::TensorArg other_arg(other, "other", 2);
    torch::checkSameDim(__func__, data_arg, other_arg);

    int32_t num_packs = pack_infos.size(0);
    int32_t num_feats = data.size(0);
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor data_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_div_cu", ([&] {
        packed_div_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, num_feats, feat_dim, 
            data.data_ptr<scalar_t>(), 
            other.data_ptr<scalar_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            data_out.data_ptr<scalar_t>()
        );
    }));

    return data_out;
}

template <typename scalar_t>
__global__ void packed_invert_cdf_fw_kernel(
    const int32_t num_packs,
    const int32_t num_feats,
    const scalar_t* __restrict__ bins,
    const scalar_t* __restrict__ cdfs,
    const int32_t* __restrict__ pack_infos,
    const scalar_t* __restrict__ u_vals,
    int32_t num_to_sample,
    scalar_t* __restrict__ samples,
    int32_t* __restrict__ bin_idx
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    const scalar_t eps = 1.0e-5f;

    const int32_t begin = pack_infos[tidx * 2];
    const int32_t length = pack_infos[tidx * 2 + 1];

    int32_t out_begin = tidx * num_to_sample;

    bins += begin;
    cdfs += begin;
    u_vals += out_begin;
    bin_idx += out_begin;
    samples += out_begin;
    for (int32_t i=0; i < num_to_sample; ++i) {
        scalar_t u = u_vals[i];
        int32_t pos = binary_search(u, cdfs, length); // `packed_searchsorted`

        bin_idx[i] = pos;
        if (pos == 0) {
            samples[i] = bins[0];
        } else {
            int32_t pos_prev = pos - 1;
            scalar_t pmf = cdfs[pos] - cdfs[pos_prev];
            samples[i] = pmf < eps ? bins[pos_prev] : (bins[pos_prev] + ((u - cdfs[pos_prev]) / pmf) * (bins[pos] - bins[pos_prev]));
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> packed_invert_cdf_cu(
    torch::Tensor bins, 
    torch::Tensor cdfs, 
    torch::Tensor u_vals,
    torch::Tensor pack_infos
) {

    const int32_t num_feats = bins.size(0);
    const int32_t num_packs = pack_infos.size(0);
    const int32_t num_to_sample = u_vals.size(1);

    // `bin_idx` should always of the same size as u_vals;
    torch::Tensor bin_idx = torch::full_like(u_vals, -1, u_vals.options().dtype(torch::kInt));
    torch::Tensor t_samples = torch::zeros_like(u_vals, u_vals.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bins.scalar_type(), "packed_invert_cdf_cu", ([&] {
        packed_invert_cdf_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, num_feats, 
            bins.data_ptr<scalar_t>(), 
            cdfs.data_ptr<scalar_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            u_vals.data_ptr<scalar_t>(), 
            num_to_sample, 
            t_samples.data_ptr<scalar_t>(), 
            bin_idx.data_ptr<int32_t>()
        );
    }));

    return {t_samples, bin_idx};
}

template<typename scalar_t>
__global__ void packed_sum_fw_kernel(
    const int32_t num_packs,
    const int32_t num_feats, 
    const int32_t feat_dim, 
    const scalar_t* __restrict__ data, 
    const int32_t* __restrict__ pack_infos, 
    scalar_t* __restrict__ feats_out
) {

    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];

    for (int32_t i=begin; i < end; ++i) {
        for (int32_t j=0; j<feat_dim; ++j) {
            feats_out[tidx * feat_dim + j] += data[i * feat_dim + j];
        }
    }
}

torch::Tensor packed_sum_cu(
    torch::Tensor data,
    torch::Tensor pack_infos
) {
    int32_t num_packs = pack_infos.size(0);
    int32_t num_feats = data.size(0);
    
    int32_t feat_dim = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_packs}, data.options()) : torch::zeros({num_packs, feat_dim}, data.options());

    const int threads = 256, blocks = (num_packs+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data.scalar_type(), "packed_sum_cu", ([&] {
        packed_sum_fw_kernel<scalar_t><<<blocks, threads>>>(
            num_packs, 
            num_feats, 
            feat_dim, 
            data.data_ptr<scalar_t>(), 
            pack_infos.data_ptr<int32_t>(), 
            feats_out.data_ptr<scalar_t>()
        );
    }));

    return feats_out;
}


template <typename scalar_t>
__global__ void packed_weigthed_sum_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_infos,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> accumulated_data
){
    const int32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= pack_infos.size(0)) return;

    const int32_t start_idx = pack_infos[tidx][0];
    const int32_t N_samples = pack_infos[tidx][1];
    int32_t samples = 0;

    while (samples < N_samples) {
        const int32_t s = start_idx + samples;
        for (int32_t i=0; i < data.size(1); i++) {
            accumulated_data[tidx][i] += weights[s] * data[s][i];
        }
        samples++;
    }
}

torch::Tensor packed_weighted_sum_cu(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_infos
)
{
    const int32_t N_packs = pack_infos.size(0), N = data.size(0);
    auto accumulated_data = torch::zeros({N_packs, data.size(1)}, data.options());

    const int32_t threads = 512, blocks = (N_packs+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.type(), "packed_weighted_sum_cu",
    ([&] {
        packed_weigthed_sum_kernel<scalar_t><<<blocks, threads>>>(
            data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            pack_infos.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            accumulated_data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return accumulated_data;
}
