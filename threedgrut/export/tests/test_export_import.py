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

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Usd

from threedgrut.export.base import ExportableModel
from threedgrut.export.formats import PLYExporter
from threedgrut.export.importers import PLYImporter, USDImporter
from threedgrut.export.usd.exporter import USDExporter


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
        conf = SimpleNamespace(export_usd=SimpleNamespace(linear_srgb=True))
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
            ).export(model, usd_path, dataset=dataset)
            stage = Usd.Stage.Open(str(usd_path))
            assert stage
            assert stage.GetPrimAtPath("/World/Cameras/camera_0000").IsValid()
            assert not stage.GetPrimAtPath("/World/gaussians/Cameras/camera_0000").IsValid()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
