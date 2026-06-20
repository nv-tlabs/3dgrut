# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for canonical-frame estimation and the attribute bake."""

import numpy as np

from threedgrut.export.transforms import estimate_pca_frame, resolve_frame_transform


def _rotation(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_pca_frame_recovers_axes_and_centers():
    rng = np.random.default_rng(0)
    # Anisotropic cloud: variance x > z > y (y is the thin/up axis).
    local = rng.standard_normal((20000, 3)) * np.array([5.0, 0.3, 2.0])
    R_true = _rotation(1)
    c_true = np.array([3.0, -2.0, 7.0])
    world = local @ R_true.T + c_true

    T = estimate_pca_frame(world, up_axis="y", origin="centroid")
    canon = world @ T[:3, :3].T + T[:3, 3]
    # Robust (trimmed) centroid at origin — small residual from trimmed tails is expected.
    assert np.allclose(canon.mean(axis=0), 0.0, atol=5e-2)
    # Variance ordering in canonical frame: x largest, y (up) smallest.
    var = canon.var(axis=0)
    assert var[0] > var[2] > var[1]


def test_pca_frame_robust_to_floaters():
    rng = np.random.default_rng(2)
    local = rng.standard_normal((20000, 3)) * np.array([4.0, 0.2, 1.5])
    R_true = _rotation(3)
    world = local @ R_true.T
    clean = estimate_pca_frame(world, up_axis="y", origin="centroid")
    # Inject a few far floaters with tiny opacity weight.
    floaters = rng.standard_normal((50, 3)) * 500.0
    world2 = np.concatenate([world, floaters], axis=0)
    weights = np.concatenate([np.ones(len(world)), np.full(len(floaters), 1e-4)])
    robust = estimate_pca_frame(world2, weights, up_axis="y", origin="centroid")
    # Rotation rows align (up to sign) despite the floaters.
    for i in range(3):
        assert abs(abs(float(clean[i, :3] @ robust[i, :3])) - 1.0) < 1e-2


def test_pca_frame_plane_origin_sits_at_robust_min():
    rng = np.random.default_rng(4)
    # Flat slab thick in x,z, thin in y, sitting above y=0.
    local = rng.standard_normal((20000, 3)) * np.array([5.0, 0.2, 5.0]) + np.array([0.0, 10.0, 0.0])
    T = estimate_pca_frame(local, up_axis="y", origin="plane", up_min_percentile=2.0)
    canon = local @ T[:3, :3].T + T[:3, 3]
    # Up origin near the low percentile (~0), not at the centroid height.
    assert np.percentile(canon[:, 1], 2.0) == np.float64(np.percentile(canon[:, 1], 2.0))
    assert abs(np.percentile(canon[:, 1], 2.0)) < 1e-2


def test_apply_frame_round_trip_transitive():
    """Baking T then T^{-1} returns the original attributes (positions, quats, SH)."""
    from threedgrut.export.accessor import GaussianAttributes
    from threedgrut.export.partition import apply_frame_to_attributes

    rng = np.random.default_rng(5)
    n, deg = 64, 3
    m = (deg + 1) ** 2 - 1
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    attrs = GaussianAttributes(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        rotations=quats,
        scales=np.abs(rng.standard_normal((n, 3))).astype(np.float32),
        densities=rng.random((n, 1)).astype(np.float32),
        albedo=rng.standard_normal((n, 3)).astype(np.float32),
        specular=rng.standard_normal((n, m * 3)).astype(np.float32),
    )
    R = _rotation(6)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([1.0, -2.0, 3.0])
    Tinv = np.linalg.inv(T)

    baked = apply_frame_to_attributes(attrs, T, deg)
    back = apply_frame_to_attributes(baked, Tinv, deg)

    assert np.allclose(back.positions, attrs.positions, atol=1e-4)
    assert np.allclose(back.specular, attrs.specular, atol=1e-4)
    # Quaternions equal up to sign.
    dots = np.abs(np.sum(back.rotations * attrs.rotations, axis=1))
    assert np.allclose(dots, 1.0, atol=1e-4)


def test_resolve_frame_modes():
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((1000, 3)) * np.array([3.0, 0.2, 1.0])
    assert np.allclose(resolve_frame_transform("none"), np.eye(4))
    T = resolve_frame_transform("pca", fields=[(pts, None)], up_axis="y", origin="centroid")
    assert T.shape == (4, 4)
