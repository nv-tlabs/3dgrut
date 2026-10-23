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

"""Tests for the camera USD writer: intrinsics are authored with the omni:lensdistortion schemas."""

import numpy as np
import pytest
from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from pxr import Gf, Usd

from threedgrut.datasets.datasetNcore import NCoreDataset
from threedgrut.export.usd.exporter import _extract_camera_params_from_dataset
from threedgrut.export.usd.writers.camera import export_cameras_to_usd

_LEGACY_ATTRIBUTES = ("cameraProjectionType", "fthetaWidth", "fthetaPolyA", "openCVFx", "p0", "s0", "fthetaMaxFov")


def _pinhole() -> OpenCVPinholeCameraModelParameters:
    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([1920, 1280], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([935.5, 635.25], dtype=np.float32),
        focal_length=np.array([2059.0, 2060.0], dtype=np.float32),
        radial_coeffs=np.array([0.04, -0.34, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        tangential_coeffs=np.array([0.0018, -0.00055], dtype=np.float32),
        thin_prism_coeffs=np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
    )


def _fisheye() -> OpenCVFisheyeCameraModelParameters:
    return OpenCVFisheyeCameraModelParameters(
        resolution=np.array([3840, 2160], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([1928.0, 1083.5], dtype=np.float32),
        focal_length=np.array([1913.5, 1914.0], dtype=np.float32),
        radial_coeffs=np.array([-0.03, -0.005, -0.0008, 0.001], dtype=np.float32),
        max_angle=float(np.deg2rad(70)),
    )


@pytest.fixture
def export_single_camera():
    stages = []  # keep the stages alive, prims are invalidated when their stage is destroyed

    def export(params) -> Usd.Prim:
        stage = Usd.Stage.CreateInMemory()
        stages.append(stage)
        paths = export_cameras_to_usd(
            stage,
            poses=np.eye(4, dtype=np.float64)[None],
            camera_names=["cam"],
            frame_to_camera=[0],
            camera_params=[params],
        )
        return stage.GetPrimAtPath(paths["cam"])

    return export


def _api_schemas(prim: Usd.Prim) -> list:
    # the lens distortion schemas are not registered with the USD runtime, so read the raw metadata
    api_schemas = prim.GetMetadata("apiSchemas")
    return list(api_schemas.GetAddedOrExplicitItems()) if api_schemas else []


def _get(prim: Usd.Prim, name: str):
    return prim.GetAttribute(f"omni:lensdistortion:{name}").Get()


def test_pinhole_uses_lens_distortion_schema(export_single_camera):
    prim = export_single_camera(_pinhole())
    assert _api_schemas(prim) == ["OmniLensDistortionOpenCvPinholeAPI"]
    assert _get(prim, "model") == "opencvPinhole"
    assert _get(prim, "opencvPinhole:imageSize") == Gf.Vec2i(1920, 1280)
    assert _get(prim, "opencvPinhole:fx") == pytest.approx(2059.0)
    assert _get(prim, "opencvPinhole:fy") == pytest.approx(2060.0)
    # the principal point is exported as is, in the same corner-based convention as NCore
    assert _get(prim, "opencvPinhole:cx") == pytest.approx(935.5)
    assert _get(prim, "opencvPinhole:cy") == pytest.approx(635.25)
    assert [_get(prim, f"opencvPinhole:k{i}") for i in range(1, 7)] == pytest.approx([0.04, -0.34, 0.1, 0.2, 0.3, 0.4])
    assert [_get(prim, f"opencvPinhole:p{i}") for i in (1, 2)] == pytest.approx([0.0018, -0.00055])
    assert [_get(prim, f"opencvPinhole:s{i}") for i in range(1, 5)] == pytest.approx([0.01, 0.02, 0.03, 0.04])


def test_pinhole_standard_camera_attributes_match_the_projection(export_single_camera):
    from pxr import UsdGeom

    params = _pinhole()
    prim = export_single_camera(params)
    width, height = (int(v) for v in params.resolution)
    fx, fy = (float(v) for v in params.focal_length)
    cx, cy = (float(v) for v in params.principal_point)
    projection = np.array(UsdGeom.Camera(prim).GetCamera(0).frustum.ComputeProjectionMatrix())
    for x, y, z in ((0.3, -0.2, -2.0), (-0.5, 0.4, -3.0), (0.7, 0.5, -4.0)):
        clip = np.array([x, y, z, 1.0]) @ projection
        ndc = clip[:2] / clip[3]
        assert (ndc[0] + 1) / 2 * width == pytest.approx(fx * x / -z + cx, abs=1e-2)
        assert (1 - ndc[1]) / 2 * height == pytest.approx(cy - fy * y / -z, abs=1e-2)


def test_fisheye_uses_lens_distortion_schema(export_single_camera):
    prim = export_single_camera(_fisheye())
    assert _api_schemas(prim) == ["OmniLensDistortionOpenCvFisheyeAPI"]
    assert _get(prim, "model") == "opencvFisheye"
    assert _get(prim, "opencvFisheye:imageSize") == Gf.Vec2i(3840, 2160)
    assert _get(prim, "opencvFisheye:fx") == pytest.approx(1913.5)
    assert _get(prim, "opencvFisheye:cy") == pytest.approx(1083.5)
    assert [_get(prim, f"opencvFisheye:k{i}") for i in range(1, 5)] == pytest.approx([-0.03, -0.005, -0.0008, 0.001])


@pytest.mark.parametrize("make_params", [_pinhole, _fisheye], ids=["pinhole", "fisheye"])
@pytest.mark.parametrize("size", [(1920, 1080), (1921, 1081), (1921, 1080)], ids=["even", "odd", "odd_width"])
def test_opencv_principal_point_is_exported_as_is_for_even_and_odd_image_sizes(export_single_camera, make_params, size):
    params = make_params()
    params.resolution = np.array(size, dtype=np.uint64)
    prim = export_single_camera(params)
    model = "opencvPinhole" if isinstance(params, OpenCVPinholeCameraModelParameters) else "opencvFisheye"
    cx, cy = (float(v) for v in params.principal_point)
    assert _get(prim, f"{model}:cx") == pytest.approx(cx)
    assert _get(prim, f"{model}:cy") == pytest.approx(cy)


@pytest.mark.parametrize("params", [_pinhole(), _fisheye()], ids=["pinhole", "fisheye"])
def test_legacy_attributes_are_not_authored(export_single_camera, params):
    prim = export_single_camera(params)
    for name in _LEGACY_ATTRIBUTES:
        assert not prim.HasAttribute(name), name


def _ftheta(poly=(0.0, 0.00245, 1e-9, 0.0, 0.0, 0.0), reference=None) -> FThetaCameraModelParameters:
    return FThetaCameraModelParameters(
        resolution=np.array([1936, 1216], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([970.5, 600.25], dtype=np.float32),
        reference_poly=reference or FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array(poly, dtype=np.float32),
        angle_to_pixeldist_poly=np.array([0.0, 408.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        max_angle=float(np.deg2rad(100)),
    )


def test_ftheta_uses_lens_distortion_schema(export_single_camera):
    prim = export_single_camera(_ftheta())
    assert _api_schemas(prim) == ["OmniLensDistortionFthetaAPI"]
    assert _get(prim, "model") == "ftheta"
    assert _get(prim, "ftheta:nominalWidth") == pytest.approx(1936.0)
    assert _get(prim, "ftheta:nominalHeight") == pytest.approx(1216.0)
    assert _get(prim, "ftheta:opticalCenter") == Gf.Vec2f(971.0, 600.75)
    assert _get(prim, "ftheta:maxFov") == pytest.approx(200.0, abs=1e-3)
    assert [_get(prim, f"ftheta:k{i}") for i in range(3)] == pytest.approx([0.0, 0.00245, 1e-9])


def test_ftheta_k5_is_authored_only_when_not_zero(export_single_camera):
    zero_k5 = export_single_camera(_ftheta())
    assert [zero_k5.HasAttribute(f"omni:lensdistortion:ftheta:k{i}") for i in range(6)] == [True] * 5 + [False]
    nonzero_k5 = export_single_camera(_ftheta(poly=(0.0, 0.00245, 0.0, 0.0, 0.0, 1e-15)))
    assert nonzero_k5.HasAttribute("omni:lensdistortion:ftheta:k5")


def test_ftheta_angle_to_pixeldist_reference_polynomial_warns(export_single_camera, caplog):
    reference = FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST
    export_single_camera(_ftheta(reference=reference))
    assert "reference polynomial" in caplog.text


class _FakeCameraModel:
    def __init__(self, params):
        self._params = params

    def get_parameters(self):
        return self._params


class _FakeUniqueSensorId:
    def __init__(self, id):
        self.id = id


class _FakeNCoreDataset(NCoreDataset):
    """Camera-blocked frames: 2 frames of camera_a (pinhole), then 3 frames of camera_b (FTheta).

    Camera names are unique sensor ids ("<camera>@<sequence>"), the camera models are keyed by logical camera id.
    """

    def __init__(self, camera_b_params=None):  # does not load any sequence
        self.sequence_camera_models = {
            "seq": {
                "camera_a": _FakeCameraModel(_pinhole()),
                "camera_b": _FakeCameraModel(camera_b_params or _ftheta()),
            }
        }
        self.sequence_camera_unique_ids = {
            "seq": {"camera_a": _FakeUniqueSensorId("camera_a@seq"), "camera_b": _FakeUniqueSensorId("camera_b@seq")}
        }

    def __len__(self):
        return 5

    def get_camera_names(self):
        return ["camera_a@seq", "camera_b@seq"]

    def get_n_frames_per_camera(self, unique_sensors=True):
        return np.array([2, 3], dtype=np.int32)


def test_ncore_dataset_camera_params_are_extracted_per_frame():
    params = _extract_camera_params_from_dataset(_FakeNCoreDataset())
    assert [type(p).__name__ for p in params] == ["OpenCVPinholeCameraModelParameters"] * 2 + [
        "FThetaCameraModelParameters"
    ] * 3


def test_ncore_dataset_unsupported_camera_model_yields_none():
    params = _extract_camera_params_from_dataset(_FakeNCoreDataset(camera_b_params=object()))
    assert params[2:] == [None] * 3


def test_unsupported_model_falls_back_to_default_focal_length(export_single_camera):
    prim = export_single_camera(object())
    assert _api_schemas(prim) == []
    assert prim.GetAttribute("focalLength").Get() == pytest.approx(24.0)
