# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("ncore")

from threedgrut.datasets.colmap_gsplat import (
    align_principal_axes,
    normalize_world_space,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _cameras_at(centers: np.ndarray) -> np.ndarray:
    cameras = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
    cameras[:, :3, 3] = centers
    return cameras


def test_similarity_from_cameras_matches_focus_centering_and_median_scale() -> None:
    cameras = _cameras_at(
        np.array(
            [
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
    )

    transform = similarity_from_cameras(cameras)

    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] /= np.sqrt(2.0)
    np.testing.assert_allclose(transform, expected, atol=1e-12)


def test_principal_axis_alignment_centers_and_diagonalizes_points() -> None:
    points = np.array(
        [
            [-4.0, -1.0, 0.0],
            [-2.0, 0.0, 0.5],
            [0.0, 3.0, 1.0],
            [1.0, -3.0, 1.5],
            [3.0, 0.5, 2.0],
            [5.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )

    transform = align_principal_axes(points)
    aligned = transform_points(transform, points)
    covariance = np.cov(aligned, rowvar=False)

    median_point = np.median(points, axis=0, keepdims=True)
    np.testing.assert_allclose(transform_points(transform, median_point), np.zeros((1, 3)), atol=1e-12)
    np.testing.assert_allclose(covariance, np.diag(np.diag(covariance)), atol=1e-12)
    assert np.linalg.det(transform[:3, :3]) > 0
    assert np.all(np.diff(np.diag(covariance)) <= 0)


def test_world_normalization_returns_one_transform_for_cameras_and_points() -> None:
    cameras = _cameras_at(
        np.array(
            [
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
    )
    points = np.array(
        [
            [-3.0, -1.0, -0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.5],
            [1.0, -2.0, 1.0],
            [2.0, 0.5, 1.5],
            [4.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )

    normalized_cameras, normalized_points, transform = normalize_world_space(cameras, points)

    np.testing.assert_allclose(normalized_cameras, transform_cameras(transform, cameras), atol=1e-12)
    np.testing.assert_allclose(normalized_points, transform_points(transform, points), atol=1e-12)
