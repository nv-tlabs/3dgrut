# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""Linear-to-sRGB post-processing for training and inference.

This module implements ``post_processing.method: "linear-to-srgb"`` (see ``configs/base_gs.yaml``).
The trainer applies it to ``pred_rgb`` after the forward render and **before** photometric loss,
so use it when **ground-truth images are sRGB / display-referred** and the **renderer output is
linear scene-referred RGB** (typical for splatting).

Integration:

- **Training:** ``Trainer3DGRUT.init_post_processing`` builds :class:`LinearToSrgbPostProcessing`
  when ``conf.post_processing.method == "linear-to-srgb"``. No optimizers; regularization term is
  always zero (:meth:`get_regularization_loss`).
- **Inference:** ``Renderer.from_checkpoint`` restores the module from the checkpoint when the
  saved config uses the same method.

The forward signature matches ``threedgrut.utils.render.apply_post_processing``; unused arguments are ignored.

The piecewise rule matches ``thirdparty/tiny-cuda-nn/scripts/common.py`` ``linear_to_srgb``
(NumPy); this file uses the same math in PyTorch (no NumPy dependency on that script at runtime).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear RGB to sRGB nonlinear light (IEC 61966-2-1 style piecewise).

    Same branch structure as ``linear_to_srgb`` in ``thirdparty/tiny-cuda-nn/scripts/common.py``:

    .. code-block:: python

        np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

    with ``limit = 0.0031308``. Linear values above ``1`` can yield encoded values above ``1`` (HDR).

    Args:
        x: Linear RGB tensor (any shape).

    Returns:
        Encoded values, same shape / dtype / device as ``x``.
    """
    limit = 0.0031308
    positive_x = torch.clamp(x, min=1e-08)
    return torch.where(
        x > limit,
        1.055 * torch.pow(positive_x, 1.0 / 2.4) - 0.055,
        12.92 * x,
    )


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`linear_to_srgb`: sRGB encoded values back to linear.

    Piecewise IEC 61966-2-1 with break point at ``0.04045``:

    .. code-block:: python

        np.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    Round-trips :func:`linear_to_srgb` to fp32 epsilon for ``x`` in [0, 1];
    HDR values (``x > 1``) are passed through the upper branch identically
    to the encode side.

    Args:
        x: sRGB-encoded tensor (any shape).

    Returns:
        Linear values, same shape / dtype / device as ``x``.
    """
    limit = 0.04045
    positive_x = torch.clamp(x + 0.055, min=1e-08)
    return torch.where(
        x < limit,
        x / 12.92,
        torch.pow(positive_x / 1.055, 2.4),
    )


class LinearToSrgbPostProcessing(nn.Module):
    """``nn.Module`` wrapper so linear-to-sRGB can plug into the shared post-processing path.

    ``forward`` receives flattened RGB ``[N, 3]`` from ``apply_post_processing`` plus PPISP-style
    metadata (pixel coordinates, resolution, camera / frame indices, exposure). Only
    ``pred_rgb_flat`` is used; other arguments exist for API compatibility with PPISP.

    There are **no learnable parameters**. Checkpoints still store an (empty) ``state_dict`` for
    this module when training with this method.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_reg_loss_zero", torch.tensor(0.0))

    def forward(
        self,
        pred_rgb_flat: torch.Tensor,
        pixel_coords_flat: torch.Tensor,
        resolution=None,
        camera_idx=None,
        frame_idx=None,
        exposure_prior=None,
    ) -> torch.Tensor:
        """Encode ``pred_rgb_flat`` with :func:`linear_to_srgb`.

        Args:
            pred_rgb_flat: ``[H*W, 3]`` linear RGB (contiguous, batch size 1 upstream).
            pixel_coords_flat: Unused (PPISP contract).
            resolution: Unused.
            camera_idx: Unused.
            frame_idx: Unused.
            exposure_prior: Unused.

        Returns:
            Same shape as ``pred_rgb_flat`` (piecewise IEC-style encode; see :func:`linear_to_srgb`).
        """
        del pixel_coords_flat, resolution, camera_idx, frame_idx, exposure_prior
        return linear_to_srgb(pred_rgb_flat)

    def get_regularization_loss(self) -> torch.Tensor:
        """Scalar zero on the module device; required by the trainer alongside PPISP."""
        return self._reg_loss_zero
