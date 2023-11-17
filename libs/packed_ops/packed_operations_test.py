# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import unittest

import torch

# C++ / CUDA libs
try:
    from libs.packed_ops import packed_ops  # type: ignore
except ImportError:
    import packed_ops  # type: ignore

from utils.tests import CommonTestCase, NonDeterministicTestCase
from models.packed_ops_modules import (
    packed_cumsum,
    packed_cumprod,
    packed_add,
    packed_sub,
    packed_mul,
    packed_div,
    packed_sum,
    PackedWeightedSum,
)


class TestMergeTwoPackedSortedArrays(CommonTestCase):
    def setUp(self):
        self.n_rays = 1000
        self.a_samples = torch.randint(0, 100, (self.n_rays,))
        self.b_samples = torch.randint(0, 100, (self.n_rays,))

        self.packed_a = torch.cat(
            [torch.cumsum(self.a_samples, 0)[:, None].roll(1, 0), self.a_samples[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_b = torch.cat(
            [torch.cumsum(self.b_samples, 0)[:, None].roll(1, 0), self.b_samples[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a[0, 0] = 0
        self.packed_b[0, 0] = 0

        self.data_a = []
        self.data_b = []

        for i in range(self.n_rays):
            self.data_a.append(
                torch.sort(torch.rand(int(self.packed_a[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
            self.data_b.append(
                torch.sort(torch.rand(int(self.packed_b[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
        self.data_a = torch.cat(self.data_a)
        self.data_b = torch.cat(self.data_b)

        # Data with guaranteed zero entries (no sample for that ray)
        self.n_rays_zero = 3
        self.a_samples_zero = torch.cat(
            [torch.randint(0, 20, (self.n_rays_zero - 2,)), torch.zeros((1,)), 20 * torch.ones((1,))]
        )
        self.b_samples_zero = torch.randint(0, 20, (3,))

        self.packed_a_zero = torch.cat(
            [torch.cumsum(self.a_samples_zero, 0)[:, None].roll(1, 0), self.a_samples_zero[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_b_zero = torch.cat(
            [torch.cumsum(self.b_samples_zero, 0)[:, None].roll(1, 0), self.b_samples_zero[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a_zero[0, 0] = 0
        self.packed_b_zero[0, 0] = 0

        self.data_a_zero = []
        self.data_b_zero = []
        for i in range(self.n_rays_zero):
            self.data_a_zero.append(
                torch.sort(torch.rand(int(self.packed_a_zero[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
            self.data_b_zero.append(
                torch.sort(torch.rand(int(self.packed_b_zero[i, 1].item())).to(device="cuda", dtype=torch.float)).values
            )
        self.data_a_zero = torch.cat(self.data_a_zero)
        self.data_b_zero = torch.cat(self.data_b_zero)

    def test_compare_to_torch_sort_merge(self):
        assert isinstance(self.data_a, torch.Tensor)
        assert isinstance(self.data_b, torch.Tensor)

        torch_sort = []
        for i in range(self.n_rays):
            torch_sort.append(
                torch.sort(
                    torch.cat(
                        [
                            self.data_a[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]],
                            self.data_b[self.packed_b[i, 0] : self.packed_b[i, 0] + self.packed_b[i, 1]],
                        ]
                    )
                ).values
            )
        torch_sort = torch.cat(torch_sort)

        # Merge using the packed_ops's merge packed tensors
        p_idx_a, p_idx_b, _ = packed_ops.merge_two_packs_sorted_aligned_fw(
            self.data_a, self.packed_a, self.data_b, self.packed_b
        )
        merged_packed_ops = torch.empty(self.data_a.shape[0] + self.data_b.shape[0]).cuda()

        merged_packed_ops[p_idx_a] = self.data_a
        merged_packed_ops[p_idx_b] = self.data_b

        self._compareTensor(merged_packed_ops.cpu(), torch_sort.cpu())

    def test_compare_to_torch_sort_merge_with_zero_entries(self):
        assert isinstance(self.data_a_zero, torch.Tensor)
        assert isinstance(self.data_b_zero, torch.Tensor)

        torch_sort = []
        for i in range(self.n_rays_zero):
            torch_sort.append(
                torch.sort(
                    torch.cat(
                        [
                            self.data_a_zero[
                                self.packed_a_zero[i, 0] : self.packed_a_zero[i, 0] + self.packed_a_zero[i, 1]
                            ],
                            self.data_b_zero[
                                self.packed_b_zero[i, 0] : self.packed_b_zero[i, 0] + self.packed_b_zero[i, 1]
                            ],
                        ]
                    )
                ).values
            )
        torch_sort = torch.cat(torch_sort)

        # Merge using the packed_ops's merge packed tensors
        p_idx_a, p_idx_b, _ = packed_ops.merge_two_packs_sorted_aligned_fw(
            self.data_a_zero, self.packed_a_zero, self.data_b_zero, self.packed_b_zero
        )
        merged_packed_ops = torch.empty(self.data_a_zero.shape[0] + self.data_b_zero.shape[0]).cuda()
        merged_packed_ops[p_idx_a] = self.data_a_zero
        merged_packed_ops[p_idx_b] = self.data_b_zero

        self._compareTensor(merged_packed_ops.cpu(), torch_sort.cpu())


class TestPackedArithmeticOperations(NonDeterministicTestCase):
    def setUp(self):
        self.n_rays = 1000
        self.n_features = 300
        self.a_samples = torch.randint(0, 100, (self.n_rays,))

        self.packed_a = torch.cat(
            [torch.cumsum(self.a_samples, 0)[:, None].roll(1, 0), self.a_samples[:, None]], dim=1
        ).to(device="cuda", dtype=torch.int32)
        self.packed_a[0, 0] = 0

        self.data_a = []
        for i in range(self.n_rays):
            self.data_a.append(
                torch.rand(int(self.packed_a[i, 1].item()), requires_grad=True).to(device="cuda", dtype=torch.float)
            )

        self.data_a = torch.cat(self.data_a)
        self.data_b = torch.rand((int(self.packed_a[-1].sum().item()), self.n_features), requires_grad=True).to(
            device="cuda", dtype=torch.float
        )
        self.weights_b = torch.rand((self.data_b.shape[0],), requires_grad=True).cuda()

    def test_packed_cumsum(self):
        assert isinstance(self.data_a, torch.Tensor)

        torch_cumsum = []
        torch_cumsum_exclusive = []
        for i in range(self.n_rays):
            torch_cumsum.append(
                torch.cumsum(self.data_a[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]], dim=0)
            )

            tmp_cumsum = torch.cumsum(
                self.data_a[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]], dim=0
            ).roll(1, 0)
            # If there are some elements, set the first after the roll-over to 0.0
            if tmp_cumsum.shape[0] > 0:
                tmp_cumsum[0] = 0.0
            torch_cumsum_exclusive.append(tmp_cumsum)

        torch_cumsum = torch.cat(torch_cumsum)
        torch_cumsum_exclusive = torch.cat(torch_cumsum_exclusive)

        ours_cumsum = packed_cumsum(self.data_a, self.packed_a, False, False)
        ours_cumsum_exclusive = packed_cumsum(self.data_a, self.packed_a, True, False)

        self._compareTensor(
            torch_cumsum.detach().cpu(), ours_cumsum.detach().cpu(), absolute_decimal=5, relative_decimal=5
        )
        self._compareTensor(
            torch_cumsum_exclusive.detach().cpu(),
            ours_cumsum_exclusive.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
        )

        # Generate the GT cumulative sum: compute the loss and compare the gradients
        gt_cumsum = torch.rand_like(torch_cumsum, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_cumsum - gt_cumsum))
        loss_ours = torch.mean(torch.square(ours_cumsum - gt_cumsum))

        grad_torch_cumsum = torch.autograd.grad(loss_torch, (self.data_a), grad_outputs=torch.ones_like(loss_torch))[0]
        grad_ours_cumsum = torch.autograd.grad(loss_ours, (self.data_a), grad_outputs=torch.ones_like(loss_ours))[0]
        self._compareTensor(
            grad_torch_cumsum.detach().cpu(),
            grad_ours_cumsum.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=-1,
            ratio_of_permitted_failures=0.04,
        )

        grad_torch_cumsum_exclusive = torch.autograd.grad(
            torch_cumsum_exclusive, (self.data_a), grad_outputs=torch.ones_like(torch_cumsum_exclusive)
        )[0]
        grad_ours_cumsum_exclusive = torch.autograd.grad(
            ours_cumsum_exclusive, (self.data_a), grad_outputs=torch.ones_like(ours_cumsum_exclusive)
        )[0]
        self._compareTensor(
            grad_torch_cumsum_exclusive.detach().cpu(),
            grad_ours_cumsum_exclusive.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=-1,
            ratio_of_permitted_failures=0.04,
        )

    def test_packed_cumprod(self):
        assert isinstance(self.data_a, torch.Tensor)

        torch_cumprod = []
        torch_cumprod_exclusive = []
        for i in range(self.n_rays):
            torch_cumprod.append(
                torch.cumprod(self.data_a[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]], dim=0)
            )

            tmp_cumprod = torch.cumprod(
                self.data_a[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]], dim=0
            ).roll(1, 0)
            # If there are some elements, set the first after the roll-over to 1.0 in accordance with Tensorflow
            if tmp_cumprod.shape[0] > 0:
                tmp_cumprod[0] = 1.0
            torch_cumprod_exclusive.append(tmp_cumprod)

        torch_cumprod = torch.cat(torch_cumprod)
        torch_cumprod_exclusive = torch.cat(torch_cumprod_exclusive)

        ours_cumprod = packed_cumprod(self.data_a, self.packed_a, False, False)
        ours_cumprod_exclusive = packed_cumprod(self.data_a, self.packed_a, True, False)

        self._compareTensor(
            torch_cumprod.detach().cpu(), ours_cumprod.detach().cpu(), absolute_decimal=5, relative_decimal=5
        )
        self._compareTensor(
            torch_cumprod_exclusive.detach().cpu(),
            ours_cumprod_exclusive.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
        )

        # Generate the GT cumulative product: compute the loss and compare the gradients
        gt_cumprod = torch.rand_like(torch_cumprod, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_cumprod - gt_cumprod))
        loss_ours = torch.mean(torch.square(ours_cumprod - gt_cumprod))

        grad_torch_cumprod = torch.autograd.grad(loss_torch, (self.data_a), grad_outputs=torch.ones_like(loss_torch))[0]
        grad_ours_cumprod = torch.autograd.grad(loss_ours, (self.data_a), grad_outputs=torch.ones_like(loss_ours))[0]

        self._compareTensor(
            grad_torch_cumprod.detach().cpu(),
            grad_ours_cumprod.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=-1,
            ratio_of_permitted_failures=0.04,
        )

        grad_torch_cumprod_exclusive = torch.autograd.grad(
            torch_cumprod_exclusive, (self.data_a), grad_outputs=torch.ones_like(torch_cumprod_exclusive)
        )[0]
        grad_ours_cumprod_exclusive = torch.autograd.grad(
            ours_cumprod_exclusive, (self.data_a), grad_outputs=torch.ones_like(ours_cumprod_exclusive)
        )[0]
        self._compareTensor(
            grad_torch_cumprod_exclusive.detach().cpu(),
            grad_ours_cumprod_exclusive.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=-1,
            ratio_of_permitted_failures=0.04,
        )

    def test_packed_sum(self):
        assert isinstance(self.data_b, torch.Tensor)

        torch_sum = []
        for i in range(self.n_rays):
            torch_sum.append(
                torch.sum(self.data_b[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]], dim=0)
            )

        torch_sum = torch.stack(torch_sum, dim=0)
        ours_sum = packed_sum(self.data_b, self.packed_a)

        self._compareTensor(torch_sum.detach().cpu(), ours_sum.detach().cpu(), absolute_decimal=5, relative_decimal=5)

        # Generate the GT sum: compute the loss and compare the gradients
        gt_sum = torch.rand_like(torch_sum, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_sum - gt_sum))
        loss_ours = torch.mean(torch.square(ours_sum - gt_sum))

        grad_torch_sum = torch.autograd.grad(loss_torch, (self.data_b), grad_outputs=torch.ones_like(loss_torch))[0]
        grad_ours_sum = torch.autograd.grad(loss_ours, (self.data_b), grad_outputs=torch.ones_like(loss_ours))[0]
        self._compareTensor(
            grad_torch_sum.detach().cpu(),
            grad_ours_sum.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )

    def test_packed_add(self):
        assert isinstance(self.data_b, torch.Tensor)

        other = (
            torch.rand((self.n_rays, self.data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float)
            - 0.5
        )

        torch_add = []
        for i in range(self.n_rays):
            torch_add.append(
                self.data_b[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]] + other[i][None, :]
            )

        torch_add = torch.cat(torch_add, dim=0)
        ours_add = packed_add(self.data_b, other, self.packed_a)
        self._compareTensor(torch_add.detach().cpu(), ours_add.detach().cpu(), absolute_decimal=5, relative_decimal=5)

        # Generate the GT addition result: compute the loss and compare the gradients
        gt_add = torch.rand_like(torch_add, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_add - gt_add))
        loss_ours = torch.mean(torch.square(ours_add - gt_add))

        grad_torch_add_a, grad_torch_add_other = torch.autograd.grad(
            loss_torch, (self.data_b, other), grad_outputs=torch.ones_like(loss_torch)
        )
        grad_ours_add_a, grad_ours_add_other = torch.autograd.grad(
            loss_ours, (self.data_b, other), grad_outputs=torch.ones_like(loss_ours)
        )
        self._compareTensor(
            grad_torch_add_a.detach().cpu(),
            grad_ours_add_a.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )

        self._compareTensor(
            grad_torch_add_other.detach().cpu(),
            grad_ours_add_other.detach().cpu(),
            absolute_decimal=0,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

    def test_packed_sub(self):
        assert isinstance(self.data_b, torch.Tensor)

        other = (
            torch.rand((self.n_rays, self.data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float)
            - 0.5
        )

        torch_sub = []
        for i in range(self.n_rays):
            torch_sub.append(
                self.data_b[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]] - other[i][None, :]
            )

        torch_sub = torch.cat(torch_sub, dim=0)
        ours_sub = packed_sub(self.data_b, other, self.packed_a)
        self._compareTensor(torch_sub.detach().cpu(), ours_sub.detach().cpu(), absolute_decimal=5, relative_decimal=5)

        # Generate the GT div: compute the loss and compare the gradients
        gt_sub = torch.rand_like(torch_sub, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_sub - gt_sub))
        loss_ours = torch.mean(torch.square(ours_sub - gt_sub))

        grad_torch_sub_a, grad_torch_sub_other = torch.autograd.grad(
            loss_torch, (self.data_b, other), grad_outputs=torch.ones_like(loss_torch)
        )
        grad_ours_sub_a, grad_ours_sub_other = torch.autograd.grad(
            loss_ours, (self.data_b, other), grad_outputs=torch.ones_like(loss_ours)
        )
        self._compareTensor(
            grad_torch_sub_a.detach().cpu(),
            grad_ours_sub_a.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )

        self._compareTensor(
            grad_torch_sub_other.detach().cpu(),
            grad_ours_sub_other.detach().cpu(),
            absolute_decimal=0,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

    def test_packed_mul(self):
        assert isinstance(self.data_b, torch.Tensor)

        other = (
            torch.rand((self.n_rays, self.data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float)
            - 0.5
        )

        torch_mul = []
        for i in range(self.n_rays):
            torch_mul.append(
                self.data_b[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]] * other[i][None, :]
            )

        torch_mul = torch.cat(torch_mul, dim=0)
        ours_mul = packed_mul(self.data_b, other, self.packed_a)
        self._compareTensor(torch_mul.detach().cpu(), ours_mul.detach().cpu(), absolute_decimal=5, relative_decimal=5)

        # Generate the GT div: compute the loss and compare the gradients
        gt_mul = torch.rand_like(torch_mul, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_mul - gt_mul))
        loss_ours = torch.mean(torch.square(ours_mul - gt_mul))

        grad_torch_mul_a, grad_torch_mul_other = torch.autograd.grad(
            loss_torch, (self.data_b, other), grad_outputs=torch.ones_like(loss_torch)
        )
        grad_ours_mul_a, grad_ours_mul_other = torch.autograd.grad(
            loss_ours, (self.data_b, other), grad_outputs=torch.ones_like(loss_ours)
        )
        self._compareTensor(
            grad_torch_mul_a.detach().cpu(),
            grad_ours_mul_a.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )

        self._compareTensor(
            grad_torch_mul_other.detach().cpu(),
            grad_ours_mul_other.detach().cpu(),
            absolute_decimal=0,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

    def test_packed_div(self):
        assert isinstance(self.data_b, torch.Tensor)

        other = (
            torch.rand((self.n_rays, self.data_b.shape[1]), requires_grad=True).to(device="cuda", dtype=torch.float)
            - 0.5
        )

        torch_div = []
        for i in range(self.n_rays):
            torch_div.append(
                self.data_b[self.packed_a[i, 0] : self.packed_a[i, 0] + self.packed_a[i, 1]] / other[i][None, :]
            )

        torch_div = torch.cat(torch_div, dim=0)
        ours_div = packed_div(self.data_b, other, self.packed_a)
        self._compareTensor(torch_div.detach().cpu(), ours_div.detach().cpu(), absolute_decimal=5, relative_decimal=5)

        # Generate the GT div: compute the loss and compare the gradients
        gt_div = torch.rand_like(torch_div, requires_grad=False)
        loss_torch = torch.mean(torch.square(torch_div - gt_div))
        loss_ours = torch.mean(torch.square(ours_div - gt_div))

        grad_torch_div_a, grad_torch_div_other = torch.autograd.grad(
            loss_torch, (self.data_b, other), grad_outputs=torch.ones_like(loss_torch)
        )
        grad_ours_div_a, grad_ours_div_other = torch.autograd.grad(
            loss_ours, (self.data_b, other), grad_outputs=torch.ones_like(loss_ours)
        )
        self._compareTensor(
            grad_torch_div_a.detach().cpu(),
            grad_ours_div_a.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )

        self._compareTensor(
            grad_torch_div_other.detach().cpu(),
            grad_ours_div_other.detach().cpu(),
            absolute_decimal=0,
            relative_decimal=5,
            ratio_of_permitted_failures=0.04,
        )  # Do not compare the absolute value here as the difference can be very large for degenerate cases

    def test_inverted_cdf(self):

        n_rays = 1000
        n_samples = 128
        samples = torch.ones(n_rays) * n_samples

        bins = torch.sort(torch.rand((n_rays, n_samples)), dim=1).values.to(device="cuda")

        packed_a = torch.cat([torch.cumsum(samples, 0)[:, None].roll(1, 0), samples[:, None]], dim=1).to(
            device="cuda", dtype=torch.int32
        )
        packed_a[0, 0] = 0

        # TODO: Below is the Zian's code for reference (check if Zian if ok to keep the sample size for cdf (removing one sample and normalizing))
        weights = (torch.rand((n_rays, n_samples)) + 1e-5).to(device="cuda")
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1).roll(1, 0)
        cdf[:, 0] = 0.0
        cdf /= cdf[:, -1:]

        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device).contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        # Check if our cuda implementation returns the same samples
        samples_ours, bin_idx_ours = packed_ops.packed_invert_cdf(bins, cdf, u, packed_a)

        self._compareTensor(
            samples.detach().cpu(),
            samples_ours.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=5,
            ratio_of_permitted_failures=0.01,
        )

        self._compareTensor(
            inds.detach().cpu(),
            bin_idx_ours.detach().cpu(),
            absolute_decimal=6,
            relative_decimal=6,
            ratio_of_permitted_failures=0.01,
        )

    def test_compare_weighted_sum_to_torch(self):
        assert isinstance(self.data_b, torch.Tensor)
        assert isinstance(self.weights_b, torch.Tensor)

        # Reference torch implementation
        weighted_sum_torch = []
        for ray in self.packed_a:
            weights = self.weights_b[ray[0] : ray[0] + ray[1]].reshape(-1, 1)
            data = self.data_b[ray[0] : ray[0] + ray[1], :]
            weighted_sum_torch.append(torch.sum(weights * data, dim=0))
        weighted_sum_torch = torch.vstack(weighted_sum_torch)

        # Compare the weights
        weighted_sum_ours = PackedWeightedSum.apply(self.data_b, self.weights_b, self.packed_a)
        self._compareTensor(
            weighted_sum_torch.detach().cpu(), weighted_sum_ours.detach().cpu(), absolute_decimal=5, relative_decimal=5
        )

        # Generate the GT weighted sum: compute the loss and compare the gradients
        gt_weighted_sum = torch.rand_like(weighted_sum_torch, requires_grad=False)
        loss_ours = torch.mean(torch.square(weighted_sum_ours - gt_weighted_sum))
        loss_torch = torch.mean(torch.square(weighted_sum_torch - gt_weighted_sum))

        grad_torch = torch.autograd.grad(
            loss_torch, (self.data_b, self.weights_b), grad_outputs=torch.ones_like(loss_torch)
        )
        grad_ours = torch.autograd.grad(
            loss_ours, (self.data_b, self.weights_b), grad_outputs=torch.ones_like(loss_ours)
        )
        for a, b in zip(grad_ours, grad_torch):
            self._compareTensor(
                a.detach().cpu(), b.detach().cpu(), absolute_decimal=4, ratio_of_permitted_failures=0.04
            )


if __name__ == "__main__":
    unittest.main()
