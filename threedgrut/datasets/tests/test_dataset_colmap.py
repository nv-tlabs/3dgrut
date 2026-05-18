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

import numpy as np
import pytest

pytest.importorskip("ncore")

from threedgrut.datasets.dataset_colmap import _opencv_pinhole_intrinsics_from_colmap


@pytest.mark.parametrize(
    "model,params,expected_focal,expected_principal,expected_radial,expected_tangential",
    [
        (
            "SIMPLE_RADIAL",
            [100.0, 40.0, 45.0, 0.1],
            [50.0, 50.0],
            [20.0, 22.5],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            "RADIAL",
            [100.0, 40.0, 45.0, 0.1, 0.2],
            [50.0, 50.0],
            [20.0, 22.5],
            [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0],
        ),
        (
            "OPENCV",
            [100.0, 110.0, 40.0, 45.0, 0.1, 0.2, 0.01, 0.02],
            [50.0, 55.0],
            [20.0, 22.5],
            [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02],
        ),
        (
            "FULL_OPENCV",
            [100.0, 110.0, 40.0, 45.0, 0.1, 0.2, 0.01, 0.02, 0.3, 0.4, 0.5, 0.6],
            [50.0, 55.0],
            [20.0, 22.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.01, 0.02],
        ),
    ],
)
def test_opencv_pinhole_intrinsics_from_colmap(
    model,
    params,
    expected_focal,
    expected_principal,
    expected_radial,
    expected_tangential,
):
    focal, principal, radial, tangential, thin_prism = _opencv_pinhole_intrinsics_from_colmap(
        model, np.asarray(params), scaling_factor=2
    )

    np.testing.assert_allclose(focal, expected_focal)
    np.testing.assert_allclose(principal, expected_principal)
    np.testing.assert_allclose(radial, expected_radial)
    np.testing.assert_allclose(tangential, expected_tangential)
    np.testing.assert_allclose(thin_prism, np.zeros((4,), dtype=np.float32))


def test_opencv_pinhole_intrinsics_rejects_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported distorted pinhole camera model"):
        _opencv_pinhole_intrinsics_from_colmap("FOV", np.zeros((5,)), scaling_factor=1)
