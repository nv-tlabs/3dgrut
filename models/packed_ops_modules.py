# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import torch

from torch.cuda.amp.autocast_mode import custom_fwd, custom_bwd
from torch.autograd.function import once_differentiable

try:
    from libs.packed_ops import packed_ops  # type: ignore
except ImportError:
    import packed_ops  # type: ignore

class _TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))




class PackedWeightedSum(torch.autograd.Function):
    """
    Computes the weighted sum of a packed tensor representation

    Inputs:
        data: (N, M) tensor of data to be summed
        weights: (N,) weights for each sample in the data (each row)
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        accumulated_data: (M) weighted sum of the input data along each column
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, data, weights, pack_infos):
        accumulated_data = packed_ops.packed_weighted_sum(data, weights, pack_infos)

        ctx.save_for_backward(data, weights, pack_infos)

        return accumulated_data

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, dL_daccumulated_data):
        data, weights, pack_infos = ctx.saved_tensors
        grad_feats = dL_daccumulated_data.repeat_interleave(pack_infos[..., 1], dim=0)

        dL_dweights = (data * grad_feats).sum(1)
        dL_ddata = weights[:, None].repeat(1, data.shape[1]) * grad_feats

        return dL_ddata, dL_dweights, None


def packed_weighted_sum(
    data: list[torch.Tensor],
    weights: torch.Tensor,
    pack_infos: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Computes a weighted sum of all data points along the 1st dimension
    """

    assert len(set([d.shape[0] for d in data])) == 1, "All elements need to have the same amount of data points"
    assert (
        data[0].shape[0] == weights.shape[0]
    ), "The number of data points needs to be the same as the number of weights"
    assert torch.all(torch.tensor([len(d.shape) == 2 for d in data])), "All tensors need to be 2d"

    # Get the lenghts of datapoints
    chunks = torch.cumsum(torch.tensor([d.shape[1] for d in data]), 0)[:-1]

    data_cat = torch.cat(data, dim=1)

    if data_cat.requires_grad or weights.requires_grad:
        accumulated_data = PackedWeightedSum.apply(data_cat, weights, pack_infos)
    else:
        accumulated_data = packed_ops.packed_weighted_sum(data_cat, weights, pack_infos)

    return torch.tensor_split(accumulated_data, chunks, dim=1)


class PackedCumsum(torch.autograd.Function):
    """
    Compute the cumulative sum of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative sum will be computed
        reverse: bool if true a reverse (back to front) cumsum will be computed

    Outputs:
        cumsum: (N, m) cumulative sum of the input packed tensor
    """

    @staticmethod
    def forward(ctx, data, pack_infos, exclusive, reverse):
        cumsum = packed_ops.packed_cumsum(data, pack_infos, exclusive, reverse)

        ctx.save_for_backward(pack_infos)
        ctx.flags = (exclusive, reverse)

        return cumsum

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dcumsum):
        (pack_infos,) = ctx.saved_tensors
        exclusive, reverse = ctx.flags
        dL_ddata = packed_ops.packed_cumsum(dL_dcumsum, pack_infos, exclusive, not reverse)

        return dL_ddata, None, None, None


def packed_cumsum(
    data: torch.Tensor, pack_infos: torch.Tensor, exclusive: bool = False, reverse: bool = False
) -> torch.Tensor:
    """
    Compute the cumulative sum of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_infos: (n_packs, 2) start_idx, N_samples
                meaning each entry corresponds to the a single pack,
                whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative sum will be computed
        reverse: bool if true a reverse (back to front) cumsum will be computed

    Outputs:
        _ : (N, m) cumulative sum of the input packed tensor
    """

    if data.requires_grad:
        return PackedCumsum.apply(data.contiguous(), pack_infos, exclusive, reverse)  # type: ignore
    else:
        return packed_ops.packed_cumsum(data.contiguous(), pack_infos, exclusive, reverse)


class PackedCumprod(torch.autograd.Function):
    """
    Compute the cumulative product of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_infos: (n_packs, 2) start_idx, N_samples
                meaning each entry corresponds to the a single pack,
                whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative product will be computed
        reverse: bool if true a reverse (back to front) cumulative product will be computed

    Outputs:
        cumprod: (N, m) cumulative product of the input packed tensor
    """

    @staticmethod
    def forward(ctx, data, pack_infos, exclusive, reverse):
        cumprod = packed_ops.packed_cumprod(data, pack_infos, exclusive, reverse)

        ctx.save_for_backward(pack_infos, cumprod, data)
        ctx.flags = (exclusive, reverse)

        return cumprod

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dcumprod):
        # Gradient computation taken from tensorflow:
        # https://github.com/tensorflow/tensorflow/blob/51fd9c024c4544ba1ef60862ec3f55b6e3ae79b1/tensorflow/python/ops/math_grad.py#L891

        pack_infos, cumprod, data = ctx.saved_tensors
        exclusive, reverse = ctx.flags
        out = packed_ops.packed_cumsum(dL_dcumprod * cumprod, pack_infos, exclusive, not reverse)
        dL_ddata = out / data
        dL_ddata[dL_ddata.isnan()] = 0

        return dL_ddata, None, None, None


def packed_cumprod(
    data: torch.Tensor, pack_infos: torch.Tensor, exclusive: bool = False, reverse: bool = False
) -> torch.Tensor:
    """
    Compute the cumulative product of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_infos: (n_packs, 2) start_idx, N_samples
                meaning each entry corresponds to the a single pack,
                whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative product will be computed
        reverse: bool if true a reverse (back to front) cumulative product will be computed

    Outputs:
        cumprod: (N, m) cumulative product of the input packed tensor
    """

    if data.requires_grad:
        return PackedCumprod.apply(data.contiguous(), pack_infos, exclusive, reverse)  # type: ignore
    else:
        return packed_ops.packed_cumprod(data.contiguous(), pack_infos, exclusive, reverse)


class PackedAdd(torch.autograd.Function):
    """Calculate pack-wise addition: feats + other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise addition results
    """

    @staticmethod
    def forward(ctx, data, other, pack_infos):
        ctx.save_for_backward(pack_infos)
        return packed_ops.packed_add(data, other, pack_infos)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_add):
        pack_infos = ctx.saved_tensors[0]

        dL_ddata = None
        dL_dother = None
        if ctx.needs_input_grad[0]:
            dL_ddata = dL_dpacked_add
        if ctx.needs_input_grad[1]:
            dL_dother = packed_ops.packed_sum(dL_dpacked_add, pack_infos)

        return dL_ddata, dL_dother, None


def packed_add(data: torch.Tensor, other: torch.Tensor, pack_infos: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise addition: feats + other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise addition results
    """
    if data.requires_grad or other.requires_grad:
        return PackedAdd.apply(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore
    else:
        return packed_ops.packed_add(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore


class PackedSub(torch.autograd.Function):
    """Calculate pack-wise subtraction: feats - other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """

    @staticmethod
    def forward(ctx, data, other, pack_infos):
        ctx.save_for_backward(pack_infos)
        return packed_ops.packed_sub(data, other, pack_infos)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_sub):
        pack_infos = ctx.saved_tensors[0]

        dL_ddata = None
        dL_dother = None
        if ctx.needs_input_grad[0]:
            dL_ddata = dL_dpacked_sub
        if ctx.needs_input_grad[1]:
            dL_dother = -packed_ops.packed_sum(dL_dpacked_sub, pack_infos)

        return dL_ddata, dL_dother, None


def packed_sub(data: torch.Tensor, other: torch.Tensor, pack_infos: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise subtraction: feats - other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """
    if data.requires_grad or other.requires_grad:
        return PackedSub.apply(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore
    else:
        return packed_ops.packed_sub(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore


class PackedMul(torch.autograd.Function):
    """Calculate pack-wise multiplication: feats * other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise multiplication results
    """

    @staticmethod
    def forward(ctx, data, other, pack_infos):
        ctx.save_for_backward(data, other, pack_infos)
        return packed_ops.packed_mul(data, other, pack_infos)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_mul):
        data, other, pack_infos = ctx.saved_tensors

        dL_ddata = None
        dL_dother = None
        if ctx.needs_input_grad[0]:
            dL_ddata = packed_ops.packed_mul(dL_dpacked_mul, other, pack_infos)

        if ctx.needs_input_grad[1]:
            dL_dother = packed_ops.packed_sum(dL_dpacked_mul * data, pack_infos)

        return dL_ddata, dL_dother, None


def packed_mul(data: torch.Tensor, other: torch.Tensor, pack_infos: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise multiplication: feats * other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise multiplication results
    """
    if data.requires_grad or other.requires_grad:
        return PackedMul.apply(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore
    else:
        return packed_ops.packed_mul(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore


class PackedDiv(torch.autograd.Function):
    """Calculate pack-wise division: feats / other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """

    @staticmethod
    def forward(ctx, data, other, pack_infos):
        ctx.save_for_backward(data, other, pack_infos)
        return packed_ops.packed_div(data, other, pack_infos)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_div):
        data, other, pack_infos = ctx.saved_tensors

        dL_ddata = None
        dL_dother = None
        if ctx.needs_input_grad[0]:
            dL_ddata = packed_ops.packed_div(dL_dpacked_div, other, pack_infos)
        if ctx.needs_input_grad[1]:
            grad_other = packed_ops.packed_div(-dL_dpacked_div * data, other * other, pack_infos)
            dL_dother = packed_ops.packed_sum(grad_other, pack_infos)

        return dL_ddata, dL_dother, None


def packed_div(data: torch.Tensor, other: torch.Tensor, pack_infos: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise division: feats / other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """
    if data.requires_grad or other.requires_grad:
        return PackedDiv.apply(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore
    else:
        return packed_ops.packed_div(data.contiguous(), other.contiguous(), pack_infos)  # type: ignore


class PackedSum(torch.autograd.Function):
    """
    Computes the sum of a packed tensor along the zero dimension

    Inputs:
        data: (N, m) data for which we want to compute the sum
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        _ : (N_rays, m) sum of the input packed tensor
    """

    @staticmethod
    def forward(ctx, data, pack_infos):
        ctx.save_for_backward(pack_infos)
        return packed_ops.packed_sum(data, pack_infos)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dsum):
        pack_infos = ctx.saved_tensors[0]
        dL_ddata = dL_dsum.repeat_interleave(pack_infos[..., 1], dim=0)

        return dL_ddata, None


def packed_sum(feats: torch.Tensor, pack_infos: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum of a packed tensor along the zero dimension

    Inputs:
        data: (N, m) data for which we want to compute the sum
        pack_infos: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        _ : (N_rays, m) sum of the input packed tensor
    """

    if feats.requires_grad:
        return PackedSum.apply(feats.contiguous(), pack_infos)
    else:
        return packed_ops.packed_sum(feats.contiguous(), pack_infos)
