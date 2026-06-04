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
Export/Import tests with mock ExportableModel.

Tests the full export pipeline: ExportableModel -> Exporter -> File -> Importer -> verify data.
"""

import sys
import tempfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Sdf, Usd

from threedgrut.export.base import ExportableModel
from threedgrut.export.formats import PLYExporter
from threedgrut.export.importers import PLYImporter, USDImporter
from threedgrut.export.usd.exporter import USDExporter


def _assert_default_camera_render_product(
    stage: Usd.Stage,
    camera_name: str = "camera_0000",
    resolution: tuple[int, int] = (640, 480),
) -> None:
    product_path = f"/Render/{camera_name}"
    render_var_path = f"{product_path}/LdrColor"
    product = stage.GetPrimAtPath(product_path)
    assert product.IsValid()
    assert product.GetTypeName() == "RenderProduct"
    assert product.GetRelationship("camera").GetTargets() == [Sdf.Path(f"/World/Cameras/{camera_name}")]
    assert product.GetRelationship("orderedVars").GetTargets() == [Sdf.Path(render_var_path)]
    assert tuple(product.GetAttribute("resolution").Get()) == resolution

    render_var = stage.GetPrimAtPath(render_var_path)
    assert render_var.IsValid()
    assert render_var.GetTypeName() == "RenderVar"
    assert render_var.GetAttribute("sourceName").Get() == "LdrColor"
    assert not stage.GetPrimAtPath(f"{product_path}/HdrColor").IsValid()


class MockGaussianModel(ExportableModel):
    """Mock ExportableModel with known test data for verification."""

    def __init__(
        self,
        num_gaussians: int = 10,
        sh_degree: int = 3,
        device: str = "cpu",
        seed: int = 42,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self.device = device

        # Create known test data
        # Positions: simple grid pattern for easy verification
        self._positions = torch.tensor(
            [[i * 0.5, j * 0.5, 0.0] for i in range(num_gaussians) for j in range(1)],
            dtype=torch.float32,
            device=device,
        )[:num_gaussians]

        # Rotations: identity quaternions (w=1, x=y=z=0)
        self._rotations = torch.zeros((num_gaussians, 4), dtype=torch.float32, device=device)
        self._rotations[:, 0] = 1.0  # w component

        # Scales: pre-activation (log space), known values
        self._scales_preact = torch.full(
            (num_gaussians, 3), -2.0, dtype=torch.float32, device=device
        )  # exp(-2) ≈ 0.135

        # Densities: pre-activation (logit space), known values
        self._densities_preact = torch.full(
            (num_gaussians, 1), 2.0, dtype=torch.float32, device=device
        )  # sigmoid(2) ≈ 0.88

        # Albedo (DC term): known RGB values
        self._albedo = torch.tensor([[0.5, 0.3, 0.2]] * num_gaussians, dtype=torch.float32, device=device)

        # Specular (higher-order SH): zeros for simplicity
        num_specular_coeffs = (sh_degree + 1) ** 2 - 1
        self._specular = torch.zeros((num_gaussians, num_specular_coeffs * 3), dtype=torch.float32, device=device)
        self.features_albedo = self._albedo
        self.features_specular = self._specular

    def get_positions(self) -> torch.Tensor:
        return self._positions

    def get_max_n_features(self) -> int:
        return self.sh_degree

    def get_n_active_features(self) -> int:
        return self.sh_degree

    def get_scale(self, preactivation: bool = False) -> torch.Tensor:
        if preactivation:
            return self._scales_preact
        return torch.exp(self._scales_preact)

    def get_rotation(self, preactivation: bool = False) -> torch.Tensor:
        if preactivation:
            return self._rotations
        return torch.nn.functional.normalize(self._rotations, dim=1)

    def get_density(self, preactivation: bool = False) -> torch.Tensor:
        if preactivation:
            return self._densities_preact
        return torch.sigmoid(self._densities_preact)

    def get_features_albedo(self) -> torch.Tensor:
        return self._albedo

    def get_features_specular(self) -> torch.Tensor:
        return self._specular


class MockCameraDataset:
    """Minimal dataset exposing camera poses for USD camera export tests."""

    image_w = 640
    image_h = 480
    intrinsics = [500.0, 500.0, 320.0, 240.0]

    def __len__(self) -> int:
        return 2

    def get_poses(self) -> np.ndarray:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(self), axis=0)
        poses[1, 0, 3] = 1.0
        return poses

    def get_camera_names(self):
        return ["camera_0000"]

    def get_camera_idx(self, frame_idx: int) -> int:
        return 0


class MockMultiCameraDataset(MockCameraDataset):
    """Minimal multi-camera dataset with interleaved physical camera frames."""

    def __len__(self) -> int:
        return 6

    def get_poses(self) -> np.ndarray:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(self), axis=0)
        poses[:, 0, 3] = np.arange(len(self), dtype=np.float64)
        return poses

    def get_camera_names(self):
        return ["camera_left", "camera_right"]

    def get_camera_idx(self, frame_idx: int) -> int:
        return frame_idx % 2


class MockCameraDatasetNoIntrinsics(MockCameraDataset):
    """Camera dataset with poses but no native image resolution metadata."""

    intrinsics = None


def _install_fake_ppisp_module(monkeypatch):
    class PPISP(torch.nn.Module):
        __module__ = "ppisp"

        def __init__(self) -> None:
            super().__init__()
            self.num_cameras = 1
            self.config = SimpleNamespace(use_controller=False)
            self.controllers = torch.nn.ModuleList()
            self.exposure_params = torch.tensor([0.1, -0.2], dtype=torch.float32)
            self.color_params = torch.zeros((2, 8), dtype=torch.float32)
            self.vignetting_params = torch.zeros((1, 3, 5), dtype=torch.float32)
            self.crf_params = torch.zeros((1, 3, 4), dtype=torch.float32)

    ppisp_module = types.ModuleType("ppisp")
    ppisp_module.PPISP = PPISP
    monkeypatch.setitem(sys.modules, "ppisp", ppisp_module)
    return PPISP


class TestPLYExportImport:
    """Test PLY export from ExportableModel and import back."""

    def test_ply_export_import_positions(self):
        """Test that positions are preserved through PLY export/import."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"

            # Export
            exporter = PLYExporter()
            exporter.export(model, ply_path)

            # Import
            importer = PLYImporter(max_sh_degree=3)
            attrs, caps = importer.load(ply_path)

            # Verify positions
            expected_positions = model.get_positions().cpu().numpy()
            np.testing.assert_allclose(
                attrs.positions,
                expected_positions,
                rtol=1e-5,
                atol=1e-6,
                err_msg="Positions mismatch after PLY export/import",
            )

    def test_ply_export_import_scales(self):
        """Test that scales (pre-activation) are preserved."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"

            exporter = PLYExporter()
            exporter.export(model, ply_path)

            importer = PLYImporter(max_sh_degree=3)
            attrs, caps = importer.load(ply_path)

            # PLY stores pre-activation values
            expected_scales = model.get_scale(preactivation=True).cpu().numpy()
            np.testing.assert_allclose(
                attrs.scales, expected_scales, rtol=1e-5, atol=1e-6, err_msg="Scales mismatch after PLY export/import"
            )

    def test_ply_export_import_densities(self):
        """Test that densities (pre-activation) are preserved."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"

            exporter = PLYExporter()
            exporter.export(model, ply_path)

            importer = PLYImporter(max_sh_degree=3)
            attrs, caps = importer.load(ply_path)

            expected_densities = model.get_density(preactivation=True).cpu().numpy()
            np.testing.assert_allclose(
                attrs.densities,
                expected_densities,
                rtol=1e-5,
                atol=1e-6,
                err_msg="Densities mismatch after PLY export/import",
            )

    def test_ply_export_import_rotations(self):
        """Test that rotations are preserved."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"

            exporter = PLYExporter()
            exporter.export(model, ply_path)

            importer = PLYImporter(max_sh_degree=3)
            attrs, caps = importer.load(ply_path)

            expected_rotations = model.get_rotation(preactivation=True).cpu().numpy()
            np.testing.assert_allclose(
                attrs.rotations,
                expected_rotations,
                rtol=1e-5,
                atol=1e-6,
                err_msg="Rotations mismatch after PLY export/import",
            )

    def test_ply_export_import_albedo(self):
        """Test that albedo (SH DC term) is preserved."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"

            exporter = PLYExporter()
            exporter.export(model, ply_path)

            importer = PLYImporter(max_sh_degree=3)
            attrs, caps = importer.load(ply_path)

            expected_albedo = model.get_features_albedo().cpu().numpy()
            np.testing.assert_allclose(
                attrs.albedo, expected_albedo, rtol=1e-5, atol=1e-6, err_msg="Albedo mismatch after PLY export/import"
            )


class TestUSDExportImport:
    """Test USD LightField export from ExportableModel and import back."""

    def test_usd_export_import_positions(self):
        """Test that positions are preserved through USD export/import."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"

            # Export (USD LightField uses post-activation values)
            exporter = USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            )
            exporter.export(model, usd_path)

            # Import
            importer = USDImporter()
            attrs, caps = importer.load(usd_path)

            # Verify positions
            expected_positions = model.get_positions().cpu().numpy()
            np.testing.assert_allclose(
                attrs.positions,
                expected_positions,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Positions mismatch after USD export/import",
            )

    def test_usd_export_import_scales_post_activation(self):
        """Test that scales (post-activation) are preserved in USD."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"

            exporter = USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            )
            exporter.export(model, usd_path)

            importer = USDImporter()
            attrs, caps = importer.load(usd_path)

            # USD LightField stores post-activation (actual) scales
            expected_scales = model.get_scale(preactivation=False).cpu().numpy()
            np.testing.assert_allclose(
                attrs.scales, expected_scales, rtol=1e-4, atol=1e-5, err_msg="Scales mismatch after USD export/import"
            )

    def test_usd_export_import_densities_post_activation(self):
        """Test that densities (post-activation/opacity) are preserved in USD."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"

            exporter = USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            )
            exporter.export(model, usd_path)

            importer = USDImporter()
            attrs, caps = importer.load(usd_path)

            # USD LightField stores post-activation (opacity in [0,1])
            expected_densities = model.get_density(preactivation=False).cpu().numpy()
            np.testing.assert_allclose(
                attrs.densities,
                expected_densities,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Densities mismatch after USD export/import",
            )

    def test_usd_export_import_rotations(self):
        """Test that rotations are preserved in USD."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"

            exporter = USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            )
            exporter.export(model, usd_path)

            importer = USDImporter()
            attrs, caps = importer.load(usd_path)

            # Rotations should be normalized quaternions
            expected_rotations = model.get_rotation(preactivation=False).cpu().numpy()
            np.testing.assert_allclose(
                attrs.rotations,
                expected_rotations,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Rotations mismatch after USD export/import",
            )

    def test_usd_export_import_albedo(self):
        """Test that albedo (SH DC term) is preserved in USD."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"

            exporter = USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            )
            exporter.export(model, usd_path)

            importer = USDImporter()
            attrs, caps = importer.load(usd_path)

            expected_albedo = model.get_features_albedo().cpu().numpy()
            np.testing.assert_allclose(
                attrs.albedo, expected_albedo, rtol=1e-4, atol=1e-5, err_msg="Albedo mismatch after USD export/import"
            )


class TestExportImportConsistency:
    """Test consistency between different export formats."""

    def test_ply_usd_positions_match(self):
        """Test that positions match between PLY and USD exports."""
        model = MockGaussianModel(num_gaussians=10, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"
            usd_path = Path(tmpdir) / "test.usdz"

            # Export both formats
            PLYExporter().export(model, ply_path)
            USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(model, usd_path)

            # Import both
            ply_attrs, _ = PLYImporter(max_sh_degree=3).load(ply_path)
            usd_attrs, _ = USDImporter().load(usd_path)

            # Positions should match exactly
            np.testing.assert_allclose(
                ply_attrs.positions,
                usd_attrs.positions,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Positions differ between PLY and USD exports",
            )

    def test_usd_export_passes_usd_validation(self):
        """Exported USD stage passes OpenUSD stage validators (run inside USDExporter.export)."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(model, usd_path)


def _find_prim_with_color_space_api(stage: Usd.Stage):
    """Return the first prim that has ColorSpaceAPI applied (the Gaussian particle prim)."""
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(Usd.ColorSpaceAPI):
            return prim
    return None


class TestUSDExportColorSpace:
    """Test that USD export applies ColorSpaceAPI with correct color space name."""

    def test_usd_export_color_space_default_srgb(self):
        """Export with linear_srgb=False (default) sets color space to srgb_rec709_display."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
                linear_srgb=False,
            ).export(model, usd_path)
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            prim = _find_prim_with_color_space_api(stage)
            assert prim is not None, "No prim with ColorSpaceAPI found"
            api = Usd.ColorSpaceAPI(prim)
            attr = api.GetColorSpaceNameAttr()
            assert attr, "ColorSpaceName attribute missing"
            assert attr.Get() == "srgb_rec709_display"

    def test_usd_export_color_space_linear_srgb(self):
        """Export with linear_srgb=True sets color space to lin_rec709_scene."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
                linear_srgb=True,
            ).export(model, usd_path)
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            prim = _find_prim_with_color_space_api(stage)
            assert prim is not None, "No prim with ColorSpaceAPI found"
            api = Usd.ColorSpaceAPI(prim)
            attr = api.GetColorSpaceNameAttr()
            assert attr, "ColorSpaceName attribute missing"
            assert attr.Get() == "lin_rec709_scene"

    def test_usd_export_color_space_from_config(self):
        """Export via from_config with linear_srgb in config sets correct color space."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        conf = SimpleNamespace(export_usd=SimpleNamespace(linear_srgb=True, export_cameras=False))
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            exporter = USDExporter.from_config(conf)
            exporter.export(model, usd_path)
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            prim = _find_prim_with_color_space_api(stage)
            assert prim is not None, "No prim with ColorSpaceAPI found"
            api = Usd.ColorSpaceAPI(prim)
            assert api.GetColorSpaceNameAttr().Get() == "lin_rec709_scene"

    def test_usdz_export_camera_is_composed_from_root_stage(self):
        """USDZ camera prims are authored where the package root composes them."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        dataset = MockCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            USDExporter(
                half_precision=False,
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(model, usd_path, dataset=dataset, validation_dataset=MockCameraDataset())
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            assert stage.GetPrimAtPath("/World/Cameras/camera_0000").IsValid()
            assert stage.GetPrimAtPath("/World/Cameras/camera_0000_val").IsValid()
            assert not stage.GetPrimAtPath("/World/gaussians/Cameras/camera_0000").IsValid()
            assert stage.GetStartTimeCode() == 0.0
            assert stage.GetEndTimeCode() == 1.0
            _assert_default_camera_render_product(stage)
            _assert_default_camera_render_product(stage, "camera_0000_val")
            render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings == {"rtx:post:tonemap:op": 2}

    @pytest.mark.parametrize("suffix", [".usda", ".usdz"])
    def test_usd_export_authors_nre_borel_render_settings_without_runtime_ppisp(self, suffix: str):
        """ParticleField exports match nre-borel default render settings without PPISP."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        dataset = MockCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / f"test{suffix}"
            USDExporter(
                half_precision=False,
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(model, usd_path, dataset=dataset, validate_usd=False)

            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings == {"rtx:post:tonemap:op": 2}

    def test_usd_export_with_native_ppisp_disables_gaussian_skip_tonemapping(self, monkeypatch):
        """Runtime PPISP consumes HDR Gaussian output, so Kit must not skip tonemapping."""
        from threedgrut.export.usd.exporter import (
            MODE_POST_PROCESSING_EXPORT_OMNI_NATIVE,
        )

        PPISP = _install_fake_ppisp_module(monkeypatch)
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        dataset = MockCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            USDExporter(
                half_precision=False,
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
                post_processing_export_mode=MODE_POST_PROCESSING_EXPORT_OMNI_NATIVE,
            ).export(model, usd_path, dataset=dataset, post_processing=PPISP(), validate_usd=False)

            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings["rtx:post:tonemap:op"] == 2
            assert render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"] is False

    def test_usd_export_requires_dataset_when_cameras_enabled(self):
        """Camera-enabled exports must not silently produce camera-less USD."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usda"
            with pytest.raises(ValueError, match="export_cameras=True requires a dataset"):
                USDExporter(
                    half_precision=False,
                    export_cameras=True,
                    export_background=False,
                    apply_normalizing_transform=False,
                ).export(model, usd_path, validate_usd=False)

    def test_usd_export_requires_render_product_resolution(self):
        """Camera-enabled exports must author RenderProducts, not only cameras."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usda"
            with pytest.raises(ValueError, match="no RenderProducts"):
                USDExporter(
                    half_precision=False,
                    export_cameras=True,
                    export_background=False,
                    apply_normalizing_transform=False,
                ).export(
                    model,
                    usd_path,
                    dataset=MockCameraDatasetNoIntrinsics(),
                    validate_usd=False,
                )

    def test_multi_camera_time_codes_are_global_dataset_indices(self):
        """Multi-camera exports use GLOBAL dataset frame indices as USD time codes.

        ``MockMultiCameraDataset`` interleaves left/right via ``frame_idx % 2``
        across 6 frames, so left owns global indices ``[0, 2, 4]`` and right
        owns ``[1, 3, 5]``. Authoring those exact indices as time samples
        keeps the OVRTX-vs-PyTorch comparator's basename match working without
        any sidecar.
        """
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        dataset = MockMultiCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usda"
            USDExporter(
                half_precision=False,
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(model, usd_path, dataset=dataset, validate_usd=False)

            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            assert stage.GetStartTimeCode() == 0.0
            assert stage.GetEndTimeCode() == 5.0

            left_transform = stage.GetPrimAtPath("/World/Cameras/camera_left").GetAttribute("xformOp:transform")
            right_transform = stage.GetPrimAtPath("/World/Cameras/camera_right").GetAttribute("xformOp:transform")
            assert left_transform.GetTimeSamples() == [0.0, 2.0, 4.0]
            assert right_transform.GetTimeSamples() == [1.0, 3.0, 5.0]


class TestUSDSampleExports:
    """Sample USD exports that exercise representative exporter options."""

    @pytest.mark.parametrize("suffix", [".usda", ".usdz"])
    def test_sample_standard_export_with_cameras_and_timing(self, suffix: str):
        """Standard export writes openable stages for both layer and package outputs."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        dataset = MockCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / f"sample{suffix}"
            USDExporter(
                half_precision=False,
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
                frames_per_second=24.0,
                radiance_scale=1.25,
            ).export(model, usd_path, dataset=dataset, validate_usd=False)

            assert usd_path.exists()
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            assert stage.GetTimeCodesPerSecond() == 24.0
            assert stage.GetPrimAtPath("/World/Cameras/camera_0000").IsValid()
            _assert_default_camera_render_product(stage)
            assert _find_prim_with_color_space_api(stage) is not None


class TestUSDExportSortingModeHint:
    """Test ParticleField sortingModeHint authoring."""

    def test_usd_export_sorting_mode_hint_ray_hit_distance(self):
        """Export can author the usd-core 26.5 rayHitDistance sorting hint."""
        model = MockGaussianModel(num_gaussians=5, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
                sorting_mode_hint="rayHitDistance",
            ).export(model, usd_path)
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            prim = _find_prim_with_color_space_api(stage)
            assert prim is not None, "No Gaussian particle prim found"
            assert prim.GetAttribute("sortingModeHint").Get() == "rayHitDistance"

    def test_usd_export_sorting_mode_hint_rejects_unknown_token(self):
        """Unsupported sorting hints fail before authoring invalid USD."""
        with pytest.raises(ValueError, match="Unsupported ParticleField sortingModeHint"):
            USDExporter(sorting_mode_hint="frontToBack")


class TestNuRecExport:
    """Smoke tests for the NuRec exporter's camera and RenderProduct authoring.

    NuRec USDZs are composed of ``default.usda`` (the package root) which
    references ``gauss.usda`` at ``/World/gauss``. Cameras authored on
    ``gauss.usda`` therefore appear under ``/World/gauss/Cameras/...`` in
    the composed view, while the ``/Render`` scope sits outside ``/World``
    and is only reachable by opening the gauss layer directly.
    """

    @staticmethod
    def _open_gauss_layer(usdz_path: Path, tmp_path: Path) -> Usd.Stage:
        """Extract gauss.usda from a NuRec USDZ and open it as a standalone stage."""
        import zipfile

        with zipfile.ZipFile(usdz_path) as zf:
            zf.extract("gauss.usda", path=tmp_path)
        stage = Usd.Stage.Open(str(tmp_path / "gauss.usda"))
        assert stage, f"Failed to open gauss.usda inside {usdz_path}"
        return stage

    def test_nurec_export_writes_camera_and_default_render_product(self):
        """Without PPISP, NuRec USDZ ships per-camera xform + /Render LdrColor product."""
        from threedgrut.export.usd.nurec.exporter import NuRecExporter

        model = MockGaussianModel(num_gaussians=8, sh_degree=3)
        dataset = MockCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            usd_path = tmp_path / "test.usdz"
            NuRecExporter(
                export_cameras=True,
                export_post_processing=False,
            ).export(model, usd_path, dataset=dataset)

            assert usd_path.exists()

            composed = Usd.Stage.Open(str(usd_path))
            assert composed.GetPrimAtPath("/World/gauss/Cameras/camera_0000").IsValid()
            assert "renderSettings" in composed.GetRootLayer().customLayerData
            render_settings = composed.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings["rtx:post:registeredCompositing:invertToneMap"] is True
            assert render_settings["rtx:post:registeredCompositing:invertColorCorrection"] is True
            assert "rtx:rtpt:gaussian:skipTonemapping:enabled" not in render_settings

            gauss = self._open_gauss_layer(usd_path, tmp_path)
            assert gauss.GetPrimAtPath("/World/Cameras/camera_0000").IsValid()
            _assert_default_camera_render_product(gauss)

    def test_nurec_export_with_native_ppisp_authors_root_spg_and_ppisp_render_settings(self, monkeypatch):
        """Native NuRec PPISP export exposes the SPG graph and disables registered-compositing inversions."""
        from threedgrut.export.usd.exporter import (
            MODE_POST_PROCESSING_EXPORT_OMNI_NATIVE,
        )
        from threedgrut.export.usd.nurec.exporter import NuRecExporter

        PPISP = _install_fake_ppisp_module(monkeypatch)
        model = MockGaussianModel(num_gaussians=8, sh_degree=3)
        dataset = MockCameraDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            NuRecExporter(
                export_cameras=True,
                export_post_processing=True,
                post_processing_export_mode=MODE_POST_PROCESSING_EXPORT_OMNI_NATIVE,
            ).export(model, usd_path, dataset=dataset, post_processing=PPISP())

            composed = Usd.Stage.Open(str(usd_path))
            assert composed.GetPrimAtPath("/World/gauss/Cameras/camera_0000").IsValid()
            ppisp_camera = composed.GetPrimAtPath("/World/gauss/Cameras/camera_0000_ppisp")
            assert ppisp_camera.IsValid()
            assert ppisp_camera.GetAttribute("ppisp:responsivity").IsValid()
            assert composed.GetPrimAtPath("/Render/camera_0000/PPISP").IsValid()
            render_settings = composed.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings["rtx:post:registeredCompositing:invertToneMap"] is False
            assert render_settings["rtx:post:registeredCompositing:invertColorCorrection"] is False
            assert render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"] is False

            product = composed.GetPrimAtPath("/Render/camera_0000")
            assert product.GetRelationship("camera").GetTargets() == [
                Sdf.Path("/World/gauss/Cameras/camera_0000_ppisp")
            ]
            ldr = composed.GetPrimAtPath("/Render/camera_0000/LdrColor")
            assert ldr.GetAttribute("omni:rtx:aov").GetConnections() == [
                Sdf.Path("/Render/camera_0000/PPISP.outputs:PPISPColor")
            ]
            with zipfile.ZipFile(usd_path) as zf:
                names = set(zf.namelist())
            assert {"ppisp_usd_spg.usda", "ppisp_usd_spg.cu", "ppisp_usd_spg.cu.lua"} <= names

    def test_nurec_export_writes_validation_camera_with_val_suffix(self):
        """validation_dataset surfaces a separate ``<name>_val`` camera and product."""
        from threedgrut.export.usd.nurec.exporter import NuRecExporter

        model = MockGaussianModel(num_gaussians=8, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            usd_path = tmp_path / "test.usdz"
            NuRecExporter(
                export_cameras=True,
                export_post_processing=False,
            ).export(
                model,
                usd_path,
                dataset=MockCameraDataset(),
                validation_dataset=MockCameraDataset(),
            )

            composed = Usd.Stage.Open(str(usd_path))
            assert composed.GetPrimAtPath("/World/gauss/Cameras/camera_0000").IsValid()
            assert composed.GetPrimAtPath("/World/gauss/Cameras/camera_0000_val").IsValid()

            gauss = self._open_gauss_layer(usd_path, tmp_path)
            _assert_default_camera_render_product(gauss)
            _assert_default_camera_render_product(gauss, "camera_0000_val")

    def test_nurec_export_requires_dataset_when_cameras_enabled(self):
        """NuRec camera-enabled exports must not silently omit camera data."""
        from threedgrut.export.usd.nurec.exporter import NuRecExporter

        model = MockGaussianModel(num_gaussians=8, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            with pytest.raises(ValueError, match="export_cameras=True requires a dataset"):
                NuRecExporter(
                    export_cameras=True,
                    export_post_processing=False,
                ).export(model, usd_path)

    def test_nurec_export_multi_camera_uses_global_dataset_indices_as_time_codes(self):
        """Multi-camera NuRec exports keep the OVRTX-vs-PyTorch basename match."""
        from threedgrut.export.usd.nurec.exporter import NuRecExporter

        model = MockGaussianModel(num_gaussians=8, sh_degree=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            usd_path = tmp_path / "test.usdz"
            NuRecExporter(
                export_cameras=True,
                export_post_processing=False,
            ).export(model, usd_path, dataset=MockMultiCameraDataset())

            gauss = self._open_gauss_layer(usd_path, tmp_path)
            left_transform = gauss.GetPrimAtPath("/World/Cameras/camera_left").GetAttribute("xformOp:transform")
            right_transform = gauss.GetPrimAtPath("/World/Cameras/camera_right").GetAttribute("xformOp:transform")
            assert left_transform.GetTimeSamples() == [0.0, 2.0, 4.0]
            assert right_transform.GetTimeSamples() == [1.0, 3.0, 5.0]

    def test_nurec_export_rejects_unsupported_render_method(self):
        """The exporter refuses anything other than 3dgut / 3dgrt up front."""
        from threedgrut.export.usd.nurec.exporter import (
            NuRecExporter,
            _get_default_nurec_conf,
        )

        model = MockGaussianModel(num_gaussians=4, sh_degree=3)
        conf = _get_default_nurec_conf()
        conf.render.method = "inria"
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"
            with pytest.raises(ValueError, match="render.method to be '3dgut' or '3dgrt'"):
                NuRecExporter().export(model, usd_path, conf=conf)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
