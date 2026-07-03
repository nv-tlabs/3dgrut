# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from threedgrut.model.geometry import apply_points_transform


def test_apply_points_transform_preserves_dtype_and_device() -> None:
    points = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], dtype=torch.float32)
    transform = np.array(
        [
            [0.0, -2.0, 0.0, 4.0],
            [2.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transformed = apply_points_transform(points, transform)

    expected = torch.tensor([[0.0, 1.0, 9.0], [4.0, -3.0, 7.0]], dtype=torch.float32)
    torch.testing.assert_close(transformed, expected)
    assert transformed.dtype == points.dtype
    assert transformed.device == points.device


def test_apply_points_transform_none_returns_original_tensor() -> None:
    points = torch.zeros((2, 3))
    assert apply_points_transform(points, None) is points


@pytest.mark.parametrize(
    "points,transform,match",
    [
        (torch.zeros((2, 4)), torch.eye(4), r"points must have shape \(N, 3\)"),
        (torch.zeros((2, 3)), torch.eye(3), r"transform must have shape \(4, 4\)"),
        (torch.zeros((2, 3)), torch.full((4, 4), torch.inf), "finite"),
    ],
)
def test_apply_points_transform_rejects_invalid_inputs(points, transform, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        apply_points_transform(points, transform)
