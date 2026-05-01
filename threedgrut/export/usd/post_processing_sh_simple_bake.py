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

"""One-shot bake of a fixed PPISP transform into Gaussian SH coefficients."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from ppisp import PPISP, ppisp_apply

from threedgrut.utils.post_processing_linear_to_srgb import srgb_to_linear
from threedgrut.utils.render import RGB2SH, SH2RGB

logger = logging.getLogger(__name__)

# A handful of Gaussians sit at PPISP-saturation extremes where the chain
# rule through pow() blows up (cond(J) > 1e8, |J|_F > 1e4). Their rotated
# specular norms grow by 4+ orders of magnitude and dominate Adam's
# variance estimate, stalling the fit. Past p99 of |J|_F (~3.4 on bonsai),
# rotations are unreliable; ``5.0`` keeps a small safety margin.
JACOBIAN_FRO_NORM_CLIP = 5.0


def get_fixed_frame_params(
    ppisp: PPISP,
    frame_id: int,
) -> Tuple[float, torch.Tensor]:
    """Return exposure offset and color params for one fixed PPISP frame."""
    num_frames = int(ppisp.exposure_params.shape[0])
    if frame_id < 0 or frame_id >= num_frames:
        raise ValueError(f"frame_id must be in [0, {num_frames - 1}], got {frame_id}.")
    exposure = float(ppisp.exposure_params[frame_id].item())
    color = ppisp.color_params[frame_id].detach()
    return exposure, color


def _bake_dc_through_ppisp(
    dc_rgb_linear: torch.Tensor,
    ppisp: PPISP,
    camera_id: int,
    exposure: float,
    color: torch.Tensor,
) -> torch.Tensor:
    """Apply PPISP with no vignetting to each Gaussian DC RGB color."""
    device = dc_rgb_linear.device
    dtype = dc_rgb_linear.dtype
    num_gaussians = dc_rgb_linear.shape[0]

    exposure_params = torch.tensor([exposure], device=device, dtype=dtype)
    color_params = color.to(device=device, dtype=dtype).unsqueeze(0)
    vignetting_params = torch.zeros_like(ppisp.vignetting_params, device=device, dtype=dtype)
    pixel_coords = torch.zeros(num_gaussians, 2, device=device, dtype=dtype)

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
    exposure: float,
    color: torch.Tensor,
    apply_srgb_to_linear: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run PPISP forward and extract per-Gaussian RGB Jacobians.

    When ``apply_srgb_to_linear`` is True, both the returned RGB and the
    Jacobian correspond to ``srgb_to_linear(PPISP(X))`` so the higher-order
    SH rotation stays in the same color space as the DC bake.
    """
    rgb_in = dc_rgb_linear.detach().clone().requires_grad_(True)
    rgb_ppisp = _bake_dc_through_ppisp(
        dc_rgb_linear=rgb_in,
        ppisp=ppisp,
        camera_id=camera_id,
        exposure=exposure,
        color=color,
    )
    rgb_out = srgb_to_linear(rgb_ppisp) if apply_srgb_to_linear else rgb_ppisp

    num_gaussians = rgb_in.shape[0]
    jacobian = torch.empty(num_gaussians, 3, 3, device=rgb_in.device, dtype=rgb_in.dtype)
    for channel in range(3):
        grad_out = torch.zeros_like(rgb_out)
        grad_out[:, channel] = 1.0
        (grads,) = torch.autograd.grad(
            outputs=rgb_out,
            inputs=rgb_in,
            grad_outputs=grad_out,
            retain_graph=(channel < 2),
        )
        jacobian[:, channel, :] = grads

    return rgb_out.detach(), jacobian.detach()


def _apply_jacobian_to_specular(features_specular: torch.nn.Parameter, jacobian: torch.Tensor) -> None:
    """In-place linearization of higher-order SH coefficients by ``jacobian``.

    Gaussians whose Jacobian is non-finite or has Frobenius norm above
    :data:`JACOBIAN_FRO_NORM_CLIP` keep their trained specular (i.e. J is
    replaced by the identity for them) -- avoids polluting Adam's variance
    estimate with rare PPISP-saturation outliers.
    """
    num_gaussians, total = features_specular.shape
    if total % 3 != 0:
        raise ValueError(f"features_specular last-dim ({total}) must be divisible by 3.")
    num_sh_coeffs = total // 3
    specular_rgb = features_specular.view(num_gaussians, num_sh_coeffs, 3)

    j_fro = torch.linalg.norm(jacobian, ord="fro", dim=(1, 2))
    safe = torch.isfinite(j_fro) & (j_fro <= JACOBIAN_FRO_NORM_CLIP)
    eye = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype).expand_as(jacobian)
    jacobian_safe = torch.where(safe[:, None, None], jacobian, eye)
    n_clipped = int((~safe).sum().item())
    if n_clipped > 0:
        logger.info(
            "Jacobian rotation clipped on %d/%d gaussians (|J|_F > %.1f or non-finite); "
            "their trained features_specular preserved.",
            n_clipped, num_gaussians, JACOBIAN_FRO_NORM_CLIP,
        )

    transformed = torch.einsum("nij,nkj->nki", jacobian_safe, specular_rgb)
    specular_rgb.copy_(transformed)


def simple_bake(
    model,
    ppisp: PPISP,
    camera_id: int,
    frame_id: int,
    higher_order: bool = False,
    apply_srgb_to_linear: bool = False,
) -> Tuple[float, torch.Tensor]:
    """Mutate SH coefficients with one fixed PPISP camera/frame transform.

    PPISP outputs display-referred values (its CRF folds in gamma-like
    encoding). Storing those directly in linear SH coefficients leaves the
    asset double-encoded: downstream consumers that themselves apply a
    linear→sRGB step (``threedgrut/utils/post_processing_linear_to_srgb``,
    Kit's tonemap, etc.) will gamma-correct on top of an already-encoded
    image. ``apply_srgb_to_linear=True`` runs an inverse sRGB on the PPISP
    output before ``RGB2SH`` so the SH coefficients land in linear scene-
    referred space and a downstream ``linear_to_srgb`` undoes the
    transformation cleanly.
    """
    exposure, color = get_fixed_frame_params(ppisp, frame_id)

    def _maybe_srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
        return srgb_to_linear(rgb) if apply_srgb_to_linear else rgb

    if higher_order:
        with torch.enable_grad():
            dc_rgb_linear = SH2RGB(model.features_albedo).detach()
            dc_rgb_baked, jacobian = _bake_dc_with_jacobian_through_ppisp(
                dc_rgb_linear=dc_rgb_linear,
                ppisp=ppisp,
                camera_id=camera_id,
                exposure=exposure,
                color=color,
                apply_srgb_to_linear=apply_srgb_to_linear,
            )
        with torch.no_grad():
            # dc_rgb_baked already includes srgb_to_linear when requested,
            # so RGB2SH gets the right color space directly.
            model.features_albedo.copy_(RGB2SH(dc_rgb_baked))
            _apply_jacobian_to_specular(model.features_specular, jacobian)
    else:
        with torch.no_grad():
            dc_rgb_linear = SH2RGB(model.features_albedo.detach())
            dc_rgb_baked = _bake_dc_through_ppisp(
                dc_rgb_linear=dc_rgb_linear,
                ppisp=ppisp,
                camera_id=camera_id,
                exposure=exposure,
                color=color,
            )
            model.features_albedo.copy_(RGB2SH(_maybe_srgb_to_linear(dc_rgb_baked)))

    return exposure, color
