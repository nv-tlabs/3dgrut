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

"""One-shot bake of a fixed PPISP transform into Gaussian SH coefficients.

The bake is the per-pixel PPISP transform with:
    * vignetting disabled (vignetting_params zeroed),
    * exposure = mean exposure across the chosen camera's training frames,
    * color params = mean color params across the chosen camera's training frames,
    * CRF = the chosen camera's CRF.

Always applied to each Gaussian's DC base color (via ``features_albedo``).
When ``higher_order=True``, the per-Gaussian 3x3 Jacobian of the same transform
(evaluated at each Gaussian's DC color) is also applied to every higher-order
SH coefficient triple in ``features_specular`` (linearization around the DC).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from ppisp import PPISP, ppisp_apply

from threedgrut.export.usd.writers.ppisp_writer import build_camera_frame_mapping
from threedgrut.utils.render import RGB2SH, SH2RGB


def get_camera_frame_indices(train_dataset, camera_id: int) -> List[int]:
    """Return the global training-frame indices belonging to ``camera_id``."""
    camera_names, camera_frames = build_camera_frame_mapping(train_dataset)
    assert 0 <= camera_id < len(camera_names), (
        f"camera_id must be in [0, {len(camera_names) - 1}], got {camera_id}."
    )
    frame_indices = camera_frames[camera_names[camera_id]]
    assert frame_indices, f"No training frames found for camera_id={camera_id}."
    return frame_indices


def compute_camera_means(
    ppisp: PPISP,
    train_dataset,
    camera_id: int,
) -> Tuple[float, torch.Tensor]:
    """Mean exposure offset and color params over a camera's training frames.

    Returns:
        (exposure_mean, color_mean) where ``color_mean`` has shape ``[8]`` and
        lives on the same device as ``ppisp.color_params``.
    """
    frame_indices = get_camera_frame_indices(train_dataset, camera_id)
    indices = torch.tensor(frame_indices, dtype=torch.long, device=ppisp.exposure_params.device)
    exposure_mean = float(ppisp.exposure_params[indices].mean().item())
    color_mean = ppisp.color_params[indices].mean(dim=0).detach()
    return exposure_mean, color_mean


def _bake_dc_through_ppisp(
    dc_rgb_linear: torch.Tensor,
    ppisp: PPISP,
    camera_id: int,
    exposure_mean: float,
    color_mean: torch.Tensor,
) -> torch.Tensor:
    """Apply PPISP (no vignetting) per Gaussian to the DC RGB color.

    Uses the official PPISP CUDA forward as the source of truth, with single-row
    exposure/color tensors so ``frame_idx=0`` selects the supplied means.
    Vignetting is disabled by passing zero vignetting params (falloff = 1).
    """
    device = dc_rgb_linear.device
    dtype = dc_rgb_linear.dtype
    n = dc_rgb_linear.shape[0]

    exposure_params = torch.tensor([exposure_mean], device=device, dtype=dtype)
    color_params = color_mean.to(device=device, dtype=dtype).unsqueeze(0)
    vignetting_params = torch.zeros_like(ppisp.vignetting_params, device=device, dtype=dtype)
    pixel_coords = torch.zeros(n, 2, device=device, dtype=dtype)

    return ppisp_apply(
        exposure_params=exposure_params,
        vignetting_params=vignetting_params,
        color_params=color_params,
        crf_params=ppisp.crf_params,
        rgb_in=dc_rgb_linear.contiguous(),
        pixel_coords=pixel_coords,
        resolution_w=1,
        resolution_h=1,
        camera_idx=camera_id,
        frame_idx=0,
    )


def _bake_dc_with_jacobian_through_ppisp(
    dc_rgb_linear: torch.Tensor,
    ppisp: PPISP,
    camera_id: int,
    exposure_mean: float,
    color_mean: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run PPISP forward and extract the per-Gaussian 3x3 Jacobian of the output RGB
    w.r.t. the input RGB via three one-hot output cotangents through the existing
    PPISP CUDA backward.

    Returns:
        (rgb_out, jacobian) where rgb_out has shape ``[N, 3]`` and jacobian has
        shape ``[N, 3, 3]`` with ``jacobian[g, k, j] = d rgb_out[g, k] / d rgb_in[g, j]``.
    """
    rgb_in = dc_rgb_linear.detach().clone().requires_grad_(True)
    rgb_out = _bake_dc_through_ppisp(
        dc_rgb_linear=rgb_in,
        ppisp=ppisp,
        camera_id=camera_id,
        exposure_mean=exposure_mean,
        color_mean=color_mean,
    )

    n = rgb_in.shape[0]
    jacobian = torch.empty(n, 3, 3, device=rgb_in.device, dtype=rgb_in.dtype)
    for k in range(3):
        grad_out = torch.zeros_like(rgb_out)
        grad_out[:, k] = 1.0
        (grads,) = torch.autograd.grad(
            outputs=rgb_out,
            inputs=rgb_in,
            grad_outputs=grad_out,
            retain_graph=(k < 2),
        )
        jacobian[:, k, :] = grads

    return rgb_out.detach(), jacobian.detach()


def _apply_jacobian_to_specular(features_specular: torch.nn.Parameter, jacobian: torch.Tensor) -> None:
    """In-place linearization of higher-order SH coefficients by ``jacobian``.

    ``features_specular`` is laid out as ``[N, K * 3]`` with the per-coefficient
    RGB triple contiguous (``[k0_r, k0_g, k0_b, k1_r, k1_g, k1_b, ...]``);
    each triple is multiplied by the per-Gaussian 3x3 Jacobian.
    """
    n, total = features_specular.shape
    assert total % 3 == 0, f"features_specular last-dim ({total}) must be divisible by 3."
    k_sh = total // 3
    specular_rgb = features_specular.view(n, k_sh, 3)
    transformed = torch.einsum("nij,nkj->nki", jacobian, specular_rgb)
    specular_rgb.copy_(transformed)


def simple_bake(
    model,
    ppisp: PPISP,
    train_dataset,
    camera_id: int,
    higher_order: bool = False,
) -> Tuple[float, torch.Tensor]:
    """Mutate ``model.features_albedo`` (and optionally ``model.features_specular``)
    to encode the mean PPISP transform for ``camera_id``.

    Returns:
        (exposure_mean, color_mean) used in the bake, for logging/inspection.
    """
    exposure_mean, color_mean = compute_camera_means(ppisp, train_dataset, camera_id)

    if higher_order:
        with torch.enable_grad():
            dc_rgb_linear = SH2RGB(model.features_albedo).detach()
            dc_rgb_baked, jacobian = _bake_dc_with_jacobian_through_ppisp(
                dc_rgb_linear=dc_rgb_linear,
                ppisp=ppisp,
                camera_id=camera_id,
                exposure_mean=exposure_mean,
                color_mean=color_mean,
            )
        with torch.no_grad():
            model.features_albedo.copy_(RGB2SH(dc_rgb_baked))
            _apply_jacobian_to_specular(model.features_specular, jacobian)
    else:
        with torch.no_grad():
            dc_rgb_linear = SH2RGB(model.features_albedo.detach())
            dc_rgb_baked = _bake_dc_through_ppisp(
                dc_rgb_linear=dc_rgb_linear,
                ppisp=ppisp,
                camera_id=camera_id,
                exposure_mean=exposure_mean,
                color_mean=color_mean,
            )
            model.features_albedo.copy_(RGB2SH(dc_rgb_baked))

    return exposure_mean, color_mean
