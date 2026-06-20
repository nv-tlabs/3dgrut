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

"""
Rotation of spherical-harmonics radiance coefficients.

When a Gaussian field is rigidly re-framed by a rotation R and the transform is *baked* into
the data (as for PLY, which has no scene-graph transform), the view-dependent SH coefficients
must be rotated too, otherwise specular/view-dependent colour points the wrong way.

The per-band rotation matrices are derived numerically from this codebase's own real-SH basis
(the standard 3DGS constants below) via a sample-and-solve: for band l we require

    f'(d) = f(Rᵀ d)   for every direction d,

i.e. ``B(d) · c' = B(Rᵀ d) · c``. Stacking 2l+1 sample directions gives ``c' = B⁻¹ B_rot c``.
Deriving the matrices from the same basis the renderer evaluates makes the rotation correct for
this convention by construction (validated by held-out-direction equivariance in the tests).
"""

from typing import Dict

import numpy as np
import torch

# Standard 3DGS real spherical-harmonics constants (match threedgrut/utils/render.py:C0..C2).
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def num_sh_coefficients(degree: int) -> int:
    """Total number of SH coefficients (including DC) for a given degree."""
    return (degree + 1) ** 2


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate the 3DGS real-SH basis at ``dirs``.

    Args:
        degree: Max SH degree (0..3).
        dirs: ``[P, 3]`` direction vectors (need not be normalized; callers pass unit vectors).

    Returns:
        ``[P, (degree+1)^2]`` basis values, ordered DC, band 1 (3), band 2 (5), band 3 (7).
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = [torch.full_like(x, _C0)]
    if degree >= 1:
        cols += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        cols += [
            _C2[0] * xy,
            _C2[1] * yz,
            _C2[2] * (2.0 * zz - xx - yy),
            _C2[3] * xz,
            _C2[4] * (xx - yy),
        ]
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        cols += [
            _C3[0] * y * (3.0 * xx - yy),
            _C3[1] * x * y * z,
            _C3[2] * y * (4.0 * zz - xx - yy),
            _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            _C3[4] * x * (4.0 * zz - xx - yy),
            _C3[5] * z * (xx - yy),
            _C3[6] * x * (xx - 3.0 * yy),
        ]
    return torch.stack(cols, dim=-1)


def eval_sh(coeffs: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate SH (with DC) at directions. ``coeffs`` is ``[N, (deg+1)^2, C]``, ``dirs`` ``[N, 3]``.

    Returns ``[N, C]``. Used by tests to check rotation equivariance against the renderer basis.
    """
    basis = sh_basis(degree, dirs)  # [N, K]
    return torch.einsum("nk,nkc->nc", basis, coeffs)


# Deterministic, well-conditioned sample directions per band (enough to invert each band block).
def _band_sample_dirs(num: int, device, dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(20260620 + num)
    dirs = torch.randn(num, 3, generator=g).to(device=device, dtype=dtype)
    return torch.nn.functional.normalize(dirs, dim=1)


def band_rotation_matrices(R: torch.Tensor, max_degree: int) -> Dict[int, torch.Tensor]:
    """Per-band SH rotation matrices for a 3x3 rotation ``R`` (degrees 1..max_degree).

    Band l's coefficients ``c_l`` map to ``D_l @ c_l`` where ``f'(d) = f(Rᵀ d)``.
    """
    R = R.to(dtype=torch.float64)
    out: Dict[int, torch.Tensor] = {}
    for l in range(1, max_degree + 1):
        n = 2 * l + 1
        dirs = _band_sample_dirs(n, R.device, torch.float64)
        # Columns of the full basis belonging to band l.
        start = l * l  # offset of band l within the (deg+1)^2 layout
        basis = sh_basis(l, dirs)[:, start : start + n]  # [n, n] = Y_l(d_i)
        basis_rot = sh_basis(l, dirs @ R)[:, start : start + n]  # Y_l(Rᵀ d_i)   (row d @ R == Rᵀ d)
        # B @ D = B_rot  ->  D = solve(B, B_rot)
        out[l] = torch.linalg.solve(basis, basis_rot)
    return out


def rotate_specular(specular: torch.Tensor, R: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Rotate higher-order SH coefficients (bands 1..deg) by ``R``.

    ``specular`` is ``[N, M*3]`` feature-major (M = (deg+1)^2 - 1), matching the model/PLY layout.
    The DC term (band 0) lives in ``albedo`` and is rotation-invariant, so it is not touched here.
    """
    n = specular.shape[0]
    m = (max_degree + 1) ** 2 - 1
    if specular.shape[1] != m * 3 or m == 0:
        return specular
    src = specular.reshape(n, m, 3).to(dtype=torch.float64)
    out = torch.empty_like(src)  # fresh output: never mutate the caller's tensor
    blocks = band_rotation_matrices(R, max_degree)
    offset = 0
    for l in range(1, max_degree + 1):
        width = 2 * l + 1
        band = src[:, offset : offset + width, :]  # [N, 2l+1, 3]
        out[:, offset : offset + width, :] = torch.einsum("ij,njc->nic", blocks[l], band)
        offset += width
    return out.reshape(n, m * 3).to(dtype=specular.dtype)
