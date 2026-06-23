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

"""Tests for SH coefficient rotation (equivariance, round-trip transitivity, homomorphism)."""

import numpy as np
import torch

from threedgrut.export.sh_rotation import band_rotation_matrices, eval_sh, num_sh_coefficients, rotate_specular


def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rotate_full(coeffs: torch.Tensor, R: torch.Tensor, deg: int) -> torch.Tensor:
    """Rotate full coeffs [N, K, 3]: DC untouched, specular bands rotated."""
    n = coeffs.shape[0]
    m = num_sh_coefficients(deg) - 1
    albedo = coeffs[:, 0, :]
    specular = coeffs[:, 1:, :].reshape(n, m * 3)
    rotated = rotate_specular(specular, R, deg).reshape(n, m, 3)
    out = coeffs.clone()
    out[:, 0, :] = albedo
    out[:, 1:, :] = rotated
    return out


def test_sh_rotation_equivariance_held_out_dirs():
    """f'(d) == f(Rᵀ d) on directions NOT used to build the rotation matrices."""
    for deg in (1, 2, 3):
        R = _random_rotation(7 + deg)
        n = 8
        k = num_sh_coefficients(deg)
        g = torch.Generator().manual_seed(100 + deg)
        coeffs = torch.randn(n, k, 3, generator=g, dtype=torch.float64)
        rotated = _rotate_full(coeffs, R, deg)

        d = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=torch.float64), dim=1)
        lhs = eval_sh(rotated, d, deg)  # f'(d)
        rhs = eval_sh(coeffs, d @ R, deg)  # f(Rᵀ d)
        assert torch.allclose(lhs, rhs, atol=1e-9), f"degree {deg} equivariance failed"


def test_sh_rotation_round_trip_identity():
    """Rotating by R then Rᵀ returns the original coefficients (transitivity)."""
    deg = 3
    R = _random_rotation(42)
    n, m = 5, num_sh_coefficients(deg) - 1
    g = torch.Generator().manual_seed(3)
    specular = torch.randn(n, m * 3, generator=g, dtype=torch.float64)
    there = rotate_specular(specular, R, deg)
    back = rotate_specular(there, R.transpose(0, 1), deg)
    assert torch.allclose(back, specular, atol=1e-9)


def test_sh_rotation_homomorphism():
    """D(R1 R2) == D(R1) D(R2) per band (group representation)."""
    R1 = _random_rotation(11)
    R2 = _random_rotation(22)
    a = band_rotation_matrices(R1 @ R2, 3)
    b = band_rotation_matrices(R1, 3)
    c = band_rotation_matrices(R2, 3)
    for l in (1, 2, 3):
        assert torch.allclose(a[l], b[l] @ c[l], atol=1e-9), f"band {l} homomorphism failed"


def test_sh_rotation_identity_is_noop():
    deg = 3
    n, m = 4, num_sh_coefficients(deg) - 1
    g = torch.Generator().manual_seed(5)
    specular = torch.randn(n, m * 3, generator=g, dtype=torch.float64)
    out = rotate_specular(specular, torch.eye(3, dtype=torch.float64), deg)
    assert torch.allclose(out, specular, atol=1e-9)
