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

"""Focused tests for PPISP SH bake tensor helpers."""

import pytest
import torch

from threedgrut.export.usd.post_processing_sh_bake import (
    estimate_achromatic_vignetting,
    scale_sh_output,
)
from threedgrut.utils.render import C0


class _DummySHModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features_albedo = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.10, -0.20, 0.30],
                    [0.40, 0.05, -0.15],
                ],
                dtype=torch.float64,
            )
        )
        self.features_specular = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                    [0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
                ],
                dtype=torch.float64,
            )
        )


class _DummyPPISP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vignetting_params = torch.zeros((1, 3, 5), dtype=torch.float64)


def test_scale_sh_output_compensates_dc_offset() -> None:
    model = _DummySHModel()
    scale = 2.25

    dc_rgb_before = model.features_albedo.detach() * C0 + 0.5
    specular_before = model.features_specular.detach().clone()

    scale_sh_output(model, scale)

    torch.testing.assert_close(model.features_specular, specular_before * scale)
    torch.testing.assert_close(model.features_albedo * C0 + 0.5, dc_rgb_before * scale)


def test_estimate_achromatic_vignetting_uses_max_resolution_for_portrait_images() -> None:
    ppisp = _DummyPPISP()
    ppisp.vignetting_params[:, :, 2] = -0.5
    pixel_coords = torch.tensor(
        [
            [10.0, 40.0],
            [20.0, 20.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    resolution = (20, 40)

    actual = estimate_achromatic_vignetting(ppisp, 0, pixel_coords, resolution)

    width, height = resolution
    max_res = float(max(width, height))
    uv = torch.stack(
        [
            (pixel_coords[:, 0] - float(width) * 0.5) / max_res,
            (pixel_coords[:, 1] - float(height) * 0.5) / max_res,
        ],
        dim=-1,
    )
    r2 = torch.sum(uv * uv, dim=-1, keepdim=True)
    expected = 1.0 - 0.5 * r2

    torch.testing.assert_close(actual, expected)


def test_apply_jacobian_to_specular_clips_nonfinite_and_large_jacobians() -> None:
    pytest.importorskip("ppisp")
    from threedgrut.export.usd.post_processing_sh_simple_bake import (
        JACOBIAN_FRO_NORM_CLIP,
        _apply_jacobian_to_specular,
    )

    features_specular = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            ],
            dtype=torch.float64,
        ),
        requires_grad=False,
    )
    original = features_specular.detach().clone()
    jacobian = torch.stack(
        [
            torch.diag(torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64)),
            torch.eye(3, dtype=torch.float64) * (JACOBIAN_FRO_NORM_CLIP + 1.0),
            torch.full((3, 3), float("nan"), dtype=torch.float64),
        ]
    )

    _apply_jacobian_to_specular(features_specular, jacobian)

    expected = original.clone()
    expected[0] = torch.tensor([0.5, 3.0, 6.0, 2.0, 7.5, 12.0], dtype=torch.float64)
    torch.testing.assert_close(features_specular, expected)
