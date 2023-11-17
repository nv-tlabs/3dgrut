# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import unittest

import torch
import numpy as np


class CommonTestCase(unittest.TestCase):
    def _compareTensor(
        self,
        a_in: np.ndarray | torch.Tensor,
        b_in: np.ndarray | torch.Tensor,
        decimal=6,
    ):
        np.testing.assert_array_almost_equal(
            a_in.cpu().numpy() if isinstance(a_in, torch.Tensor) else a_in,
            b_in.cpu().numpy() if isinstance(b_in, torch.Tensor) else b_in,
            decimal=decimal,
        )


class NonDeterministicTestCase(unittest.TestCase):
    # In non-deterministic test cases we additionally allow for some failure cases due
    # to the non-deterministic nature of the test
    def _compareTensor(
        self,
        a_in: np.ndarray | torch.Tensor,
        b_in: np.ndarray | torch.Tensor,
        absolute_decimal=6,
        relative_decimal=6,
        ratio_of_permitted_failures=0.02,
    ):
        a = a_in.cpu().numpy() if isinstance(a_in, torch.Tensor) else a_in
        b = b_in.cpu().numpy() if isinstance(b_in, torch.Tensor) else b_in

        self.assertEqual(a.shape, b.shape)
        assert isinstance(absolute_decimal, int), "Absolute decimal precision needs to be an integer value"
        assert isinstance(relative_decimal, int), "Relative decimal precision needs to be an integer value"

        a = np.expand_dims(a.flatten(), axis=1)
        b = np.expand_dims(b.flatten(), axis=1)

        # Check if nans exist and enforce they are at the same locations. Then replace them for further checks
        self.assertTrue(np.all(np.isnan(a) == np.isnan(b)))
        a = np.nan_to_num(a)
        b = np.nan_to_num(b)

        n_elements = len(a)
        absolute_diff = np.abs(a - b)
        if absolute_decimal > 0:
            n_above_max_abs = np.where(absolute_diff > 1.5 * 10 ** (-absolute_decimal))[0].shape[0]

            if n_above_max_abs / n_elements > ratio_of_permitted_failures:
                max_abs_diff = np.max(absolute_diff)
                raise AssertionError(
                    f"More than {ratio_of_permitted_failures}% of cases failed the absolute difference check, with the largest being {max_abs_diff}"
                )

        if relative_decimal > 0:
            is_zero = b == 0
            relative_diff = absolute_diff[~is_zero] / np.abs(b[~is_zero])
            n_above_max_rel = np.where(relative_diff > 1.5 * 10 ** (-relative_decimal))[0].shape[0]

            if n_above_max_rel / np.sum(~is_zero) > ratio_of_permitted_failures:
                max_rel_diff = np.max(relative_diff)
                raise AssertionError(
                    f"More than {ratio_of_permitted_failures*100}% of cases failed the relative difference check, with the largest being {max_rel_diff}"
                )
