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
Round-trip tests for transcode functionality.

Tests that data can be converted between formats and back with acceptable precision.
Run with: pytest threedgrut/export/tests/test_transcode_roundtrip.py -v
"""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.adapter import AttributesExportAdapter
from threedgrut.export.formats import PLYExporter
from threedgrut.export.importers import PLYImporter, USDImporter
from threedgrut.export.scripts.transcode import detect_input_format, transcode, transcode_files
from threedgrut.export.transforms import usd_matrix_to_numpy
from threedgrut.export.usd.exporter import USDExporter


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


def _write_fake_cuda_spg_sidecars(root: Path) -> None:
    (root / "ppisp_fake_spg.usda").write_text(
        """#usda 1.0
(
    defaultPrim = "PPISP"
)

def Shader "PPISP"
{
    uniform token info:implementationSource = "sourceAsset"
    uniform asset info:spg:sourceAsset = @./ppisp_fake_spg.cu@
    uniform token info:spg:sourceAsset:subIdentifier = "fakeProcess"
}
""",
        encoding="utf-8",
    )
    (root / "ppisp_fake_spg.cu").write_text("// fake CUDA SPG source\n", encoding="utf-8")
    (root / "ppisp_fake_spg.cu.lua").write_text("-- fake CUDA SPG launcher\n", encoding="utf-8")


def _add_fake_cuda_spg_to_render_product(stage: Usd.Stage) -> None:
    shader = stage.DefinePrim("/Render/camera_0000/PPISP", "Shader")
    shader.GetReferences().AddReference("ppisp_fake_spg.usda")
    shader.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set("sourceAsset")
    shader.CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
        Sdf.AssetPath("ppisp_fake_spg.cu")
    )
    shader.CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        "fakeProcess"
    )


def _find_particlefield_prim(stage: Usd.Stage) -> Usd.Prim:
    for prim in stage.Traverse():
        if prim.IsA(UsdVol.ParticleField):
            return prim
    raise AssertionError("No ParticleField prim found")


def _find_nurec_volume_prim(stage: Usd.Stage) -> Usd.Prim:
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Volume":
            continue
        attr = prim.GetAttribute("omni:nurec:isNuRecVolume")
        if attr.IsValid() and attr.Get():
            return prim
    raise AssertionError("No NuRec volume prim found")


def _set_source_gaussian_root_transform(stage: Usd.Stage) -> Gf.Matrix4d:
    gaussian_prim = _find_particlefield_prim(stage)
    root = gaussian_prim.GetParent()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslate(Gf.Vec3d(1.25, -2.5, 0.75))
    UsdGeom.Xformable(root).AddTransformOp(opSuffix="testPose").Set(matrix)
    return UsdGeom.Xformable(gaussian_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _assert_usd_matrices_close(actual: Gf.Matrix4d, expected: Gf.Matrix4d) -> None:
    assert np.allclose(usd_matrix_to_numpy(actual), usd_matrix_to_numpy(expected), atol=1e-6)


def _xform_op_order(prim: Usd.Prim) -> list[str]:
    return [str(token) for token in prim.GetAttribute("xformOpOrder").Get()]


def create_test_attributes(num_gaussians: int = 100, sh_degree: int = 3) -> GaussianAttributes:
    """Create synthetic test Gaussian attributes.

    Args:
        num_gaussians: Number of Gaussians to generate
        sh_degree: SH degree for specular coefficients

    Returns:
        GaussianAttributes with random but valid data
    """
    np.random.seed(42)  # Reproducible tests

    # Positions: random in unit cube
    positions = np.random.randn(num_gaussians, 3).astype(np.float32) * 2.0

    # Rotations: random unit quaternions (wxyz)
    rotations = np.random.randn(num_gaussians, 4).astype(np.float32)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    # Scales: random positive values (log scale for pre-activation)
    scales = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5 - 2.0  # log scale around 0.1

    # Densities: random values (logit for pre-activation)
    densities = np.random.randn(num_gaussians, 1).astype(np.float32) * 2.0  # logit values

    # Albedo: SH DC term
    albedo = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5

    # Specular: higher-order SH coefficients
    num_specular = (sh_degree + 1) ** 2 - 1
    specular = np.random.randn(num_gaussians, num_specular * 3).astype(np.float32) * 0.1

    return GaussianAttributes(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        albedo=albedo,
        specular=specular,
    )


def create_test_capabilities(num_gaussians: int = 100, sh_degree: int = 3) -> ModelCapabilities:
    """Create test model capabilities."""
    return ModelCapabilities(
        has_spherical_harmonics=True,
        sh_degree=sh_degree,
        num_gaussians=num_gaussians,
        is_surfel=False,
        density_activation="sigmoid",
        scale_activation="exp",
    )


def compare_attributes(
    original: GaussianAttributes,
    loaded: GaussianAttributes,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict:
    """Compare two GaussianAttributes and return differences.

    Args:
        original: Original attributes
        loaded: Loaded attributes after round-trip
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dict with comparison results
    """
    results = {}

    # Compare each attribute
    for attr_name in ["positions", "rotations", "scales", "densities", "albedo", "specular"]:
        orig_arr = getattr(original, attr_name)
        load_arr = getattr(loaded, attr_name)

        # Check shapes match
        if orig_arr.shape != load_arr.shape:
            results[attr_name] = {
                "match": False,
                "error": f"Shape mismatch: {orig_arr.shape} vs {load_arr.shape}",
            }
            continue

        # Check values match within tolerance
        is_close = np.allclose(orig_arr, load_arr, rtol=rtol, atol=atol)
        max_diff = np.max(np.abs(orig_arr - load_arr))
        mean_diff = np.mean(np.abs(orig_arr - load_arr))

        results[attr_name] = {
            "match": is_close,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
        }

    return results


class TestPLYRoundTrip:
    """Test PLY format round-trip."""

    def test_ply_export_import_roundtrip(self):
        """Test PLY export and import preserves data."""
        attrs = create_test_attributes(100, sh_degree=3)
        caps = create_test_capabilities(100, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "test.ply"

            # Export (PLY expects pre-activation)
            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            exporter = PLYExporter()
            exporter.export(adapter, ply_path)

            # Import
            importer = PLYImporter(max_sh_degree=3)
            loaded_attrs, loaded_caps = importer.load(ply_path)

            # Compare
            results = compare_attributes(attrs, loaded_attrs, rtol=1e-5, atol=1e-6)

            for attr_name, result in results.items():
                assert result["match"], f"PLY round-trip failed for {attr_name}: {result}"


class TestUSDLightFieldRoundTrip:
    """Test USD LightField format round-trip."""

    def test_usd_lightfield_export_import_roundtrip(self):
        """Test USD LightField export and import preserves data."""
        np.random.seed(42)
        num_gaussians = 100
        sh_degree = 3

        positions = np.random.randn(num_gaussians, 3).astype(np.float32) * 2.0
        rotations = np.random.randn(num_gaussians, 4).astype(np.float32)
        rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
        scales = np.exp(np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5 - 2.0)
        densities = 1.0 / (1.0 + np.exp(-np.random.randn(num_gaussians, 1).astype(np.float32) * 2.0))
        albedo = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5
        num_specular = (sh_degree + 1) ** 2 - 1
        specular = np.random.randn(num_gaussians, num_specular * 3).astype(np.float32) * 0.1

        attrs = GaussianAttributes(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            albedo=albedo,
            specular=specular,
        )
        caps = create_test_capabilities(num_gaussians, sh_degree)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test.usdz"

            # Export (LightField expects post-activation)
            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=False)
            exporter = USDExporter(
                half_precision=False,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            )
            exporter.export(adapter, usd_path)

            # Import
            importer = USDImporter()
            loaded_attrs, loaded_caps = importer.load(usd_path)

            # Compare
            results = compare_attributes(attrs, loaded_attrs, rtol=1e-4, atol=1e-5)

            for attr_name, result in results.items():
                assert result["match"], f"USD LightField round-trip failed for {attr_name}: {result}"


class TestCrossFormatTranscode:
    """Test transcoding between different formats."""

    def test_transcode_api_ply_to_lightfield(self):
        """Test the transcode entrypoint converts PLY to USD LightField."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "input.ply"
            usd_path = Path(tmpdir) / "output.usdz"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            PLYExporter().export(adapter, ply_path)

            transcode(
                input_path=ply_path,
                output_path=usd_path,
                output_format="lightfield",
                render_order_hint="rayHitDistance",
                validate_usd=False,
            )

            assert usd_path.exists()
            assert detect_input_format(usd_path) == "lightfield"

            loaded_attrs, _ = USDImporter().load(usd_path)
            expected_attrs = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            expected_post = GaussianAttributes(
                positions=expected_attrs.get_positions().cpu().numpy(),
                rotations=expected_attrs.get_rotation(preactivation=False).cpu().numpy(),
                scales=expected_attrs.get_scale(preactivation=False).cpu().numpy(),
                densities=expected_attrs.get_density(preactivation=False).cpu().numpy(),
                albedo=expected_attrs.get_features_albedo().cpu().numpy(),
                specular=expected_attrs.get_features_specular().cpu().numpy(),
            )
            results = compare_attributes(expected_post, loaded_attrs, rtol=1e-4, atol=1e-5)
            for attr_name, result in results.items():
                assert result["match"], f"PLY→USD transcode failed for {attr_name}: {result}"

    def test_transcode_api_ply_to_nurec_without_dataset(self):
        """PLY→NuRec transcode does not require regenerated camera data."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ply_path = tmp_path / "input.ply"
            output_path = tmp_path / "output.usdz"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            PLYExporter().export(adapter, ply_path)

            transcode(
                input_path=ply_path,
                output_path=output_path,
                output_format="nurec",
                validate_usd=False,
            )

            assert output_path.exists()
            assert detect_input_format(output_path) == "nurec"
            stage = Usd.Stage.Open(str(output_path))
            render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings["rtx:post:registeredCompositing:invertToneMap"] is True
            assert render_settings["rtx:post:registeredCompositing:invertColorCorrection"] is True
            assert "rtx:rtpt:gaussian:skipTonemapping:enabled" not in render_settings
            with zipfile.ZipFile(output_path) as zf:
                names = set(zf.namelist())
            assert {"default.usda", "gauss.usda", "output.nurec"} <= names

    def test_transcode_api_lightfield_to_ply_roundtrip(self):
        """Test the transcode entrypoint converts USD LightField back to PLY."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "input.usdz"
            ply_path = Path(tmpdir) / "output.ply"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, usd_path, validate_usd=False)

            transcode(
                input_path=usd_path,
                output_path=ply_path,
                output_format="ply",
                max_sh_degree=3,
                validate_usd=False,
            )

            assert ply_path.exists()
            assert detect_input_format(ply_path) == "ply"

            loaded_attrs, _ = PLYImporter(max_sh_degree=3).load(ply_path)
            results = compare_attributes(attrs, loaded_attrs, rtol=1e-2, atol=1e-3)
            for attr_name, result in results.items():
                assert result["match"], f"USD→PLY transcode failed for {attr_name}: {result}"

    def test_transcode_api_usd_to_usd_copies_render_products(self):
        """USD→USD transcode preserves source /Render RenderProducts."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_path = tmp_path / "output.usda"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, dataset=MockCameraDataset(), validate_usd=False)

            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format="lightfield",
                copy_cameras_source=(input_path, tmp_path),
                validate_usd=False,
            )

            stage = Usd.Stage.Open(str(output_path))
            assert stage
            assert stage.GetStartTimeCode() == 0.0
            assert stage.GetEndTimeCode() == 1.0
            product = stage.GetPrimAtPath("/Render/camera_0000")
            assert product.IsValid()
            assert product.GetTypeName() == "RenderProduct"
            assert product.GetRelationship("camera").GetTargets() == [Sdf.Path("/World/Cameras/camera_0000")]
            assert tuple(product.GetAttribute("resolution").Get()) == (640, 480)
            assert product.GetRelationship("orderedVars").GetTargets() == [Sdf.Path("/Render/camera_0000/LdrColor")]

    def test_transcode_api_usd_to_usd_preserves_source_gaussian_pose(self):
        """USD→USD transcode preserves the source Gaussian local-to-world pose."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_path = tmp_path / "output.usda"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, validate_usd=False)

            stage = Usd.Stage.Open(str(input_path))
            assert stage
            expected_matrix = _set_source_gaussian_root_transform(stage)
            stage.GetRootLayer().Export(str(input_path))

            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format="lightfield",
                copy_cameras_source=(input_path, tmp_path),
                validate_usd=False,
            )

            output_stage = Usd.Stage.Open(str(output_path))
            assert output_stage
            output_prim = _find_particlefield_prim(output_stage)
            output_matrix = UsdGeom.Xformable(output_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            _assert_usd_matrices_close(output_matrix, expected_matrix)

    def test_transcode_api_usd_to_usdz_copies_cuda_spg_sidecars(self):
        """USD→USDZ transcode preserves copied /Render CUDA SPG sidecars."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_path = tmp_path / "output.usdz"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, dataset=MockCameraDataset(), validate_usd=False)

            stage = Usd.Stage.Open(str(input_path))
            assert stage
            _add_fake_cuda_spg_to_render_product(stage)
            stage.GetRootLayer().Export(str(input_path))
            _write_fake_cuda_spg_sidecars(tmp_path)

            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format="lightfield",
                copy_cameras_source=(input_path, tmp_path),
                validate_usd=False,
            )

            stage = Usd.Stage.Open(str(output_path))
            assert stage
            assert stage.GetStartTimeCode() == 0.0
            assert stage.GetEndTimeCode() == 1.0
            assert stage.GetPrimAtPath("/Render/camera_0000/PPISP").IsValid()
            with zipfile.ZipFile(output_path) as zf:
                names = set(zf.namelist())
            assert {"ppisp_fake_spg.usda", "ppisp_fake_spg.cu", "ppisp_fake_spg.cu.lua"} <= names

    def test_transcode_api_usd_to_nurec_copies_render_products_and_cuda_spg_sidecars(self):
        """USD→NuRec transcode preserves source /Render and CUDA SPG sidecars."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_path = tmp_path / "output.usdz"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, dataset=MockCameraDataset(), validate_usd=False)

            stage = Usd.Stage.Open(str(input_path))
            assert stage
            _add_fake_cuda_spg_to_render_product(stage)
            stage.GetRootLayer().Export(str(input_path))
            _write_fake_cuda_spg_sidecars(tmp_path)

            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format="nurec",
                copy_cameras_source=(input_path, tmp_path),
                validate_usd=False,
            )

            stage = Usd.Stage.Open(str(output_path))
            assert stage
            assert stage.GetStartTimeCode() == 0.0
            assert stage.GetEndTimeCode() == 1.0
            assert not stage.GetPrimAtPath("/World/gaussians").IsValid()
            assert stage.GetPrimAtPath("/Render/camera_0000").IsValid()
            assert stage.GetPrimAtPath("/Render/camera_0000/PPISP").IsValid()
            render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
            assert render_settings["rtx:post:registeredCompositing:invertToneMap"] is False
            assert render_settings["rtx:post:registeredCompositing:invertColorCorrection"] is False
            assert render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"] is False
            with zipfile.ZipFile(output_path) as zf:
                names = set(zf.namelist())
                gauss_usda = zf.read("gauss.usda").decode("utf-8")
            assert {"ppisp_fake_spg.usda", "ppisp_fake_spg.cu", "ppisp_fake_spg.cu.lua"} <= names
            assert "renderSettings" in gauss_usda
            assert "rtx:post:registeredCompositing:invertToneMap" in gauss_usda
            assert "rtx:post:registeredCompositing:invertColorCorrection" in gauss_usda
            assert "rtx:rtpt:gaussian:skipTonemapping:enabled" in gauss_usda

    def test_transcode_api_usd_to_nurec_preserves_source_gaussian_pose(self):
        """USD→NuRec transcode preserves the source Gaussian local-to-world pose."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_path = tmp_path / "output.usdz"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, validate_usd=False)

            stage = Usd.Stage.Open(str(input_path))
            assert stage
            expected_matrix = _set_source_gaussian_root_transform(stage)
            stage.GetRootLayer().Export(str(input_path))

            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format="nurec",
                copy_cameras_source=(input_path, tmp_path),
                validate_usd=False,
            )

            output_stage = Usd.Stage.Open(str(output_path))
            assert output_stage
            output_prim = _find_nurec_volume_prim(output_stage)
            output_matrix = UsdGeom.Xformable(output_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            _assert_usd_matrices_close(output_matrix, expected_matrix)

    def test_transcode_coordinate_transform_order_matches_nurec(self):
        """USD→USD and USD→NuRec compose source pose and coordinate transform identically."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_lightfield_path = tmp_path / "output_lightfield.usda"
            output_nurec_path = tmp_path / "output_nurec.usdz"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, validate_usd=False)

            stage = Usd.Stage.Open(str(input_path))
            assert stage
            _set_source_gaussian_root_transform(stage)
            stage.GetRootLayer().Export(str(input_path))

            transcode(
                input_path=input_path,
                output_path=output_lightfield_path,
                output_format="lightfield",
                apply_coordinate_transform=True,
                validate_usd=False,
            )
            transcode(
                input_path=input_path,
                output_path=output_nurec_path,
                output_format="nurec",
                apply_coordinate_transform=True,
                validate_usd=False,
            )

            lightfield_stage = Usd.Stage.Open(str(output_lightfield_path))
            assert lightfield_stage
            lightfield_prim = _find_particlefield_prim(lightfield_stage)
            lightfield_root = lightfield_prim.GetParent()

            nurec_stage = Usd.Stage.Open(str(output_nurec_path))
            assert nurec_stage
            nurec_prim = _find_nurec_volume_prim(nurec_stage)

            assert _xform_op_order(lightfield_root) == ["xformOp:transform:sourcePose", "xformOp:transform"]
            assert _xform_op_order(nurec_prim) == ["xformOp:transform:sourcePose", "xformOp:transform"]

            lightfield_matrix = UsdGeom.Xformable(lightfield_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            nurec_matrix = UsdGeom.Xformable(nurec_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            _assert_usd_matrices_close(lightfield_matrix, nurec_matrix)

    def test_ply_to_usd_to_ply(self):
        """Test PLY → USD LightField → PLY transcode chain."""
        attrs = create_test_attributes(100, sh_degree=3)
        caps = create_test_capabilities(100, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path1 = Path(tmpdir) / "original.ply"
            usd_path = Path(tmpdir) / "intermediate.usdz"
            ply_path2 = Path(tmpdir) / "final.ply"

            # Step 1: Export original PLY (pre-activation)
            adapter1 = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            PLYExporter().export(adapter1, ply_path1)

            # Step 2: Import PLY
            ply_importer = PLYImporter(max_sh_degree=3)
            attrs2, caps2 = ply_importer.load(ply_path1)

            # Step 3: Export to USD LightField (needs post-activation, adapter handles conversion)
            adapter2 = AttributesExportAdapter(attrs2, caps2, is_preactivation=True)
            USDExporter(
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter2, usd_path)

            # Step 4: Import USD
            usd_importer = USDImporter()
            attrs3, caps3 = usd_importer.load(usd_path)

            # Step 5: Export back to PLY (needs pre-activation, adapter handles conversion)
            adapter3 = AttributesExportAdapter(attrs3, caps3, is_preactivation=False)
            PLYExporter().export(adapter3, ply_path2)

            # Step 6: Import final PLY
            attrs4, caps4 = ply_importer.load(ply_path2)

            # Compare original with final (relaxed tolerance due to activation conversions)
            results = compare_attributes(attrs, attrs4, rtol=1e-2, atol=1e-3)

            for attr_name, result in results.items():
                assert result["match"], f"PLY→USD→PLY transcode failed for {attr_name}: {result}"

    def test_transcode_split_large_gaussians_subdivides(self):
        """transcode --split-large-gaussians runs the exporter's split path on oversized prims."""
        attrs = create_test_attributes(300, sh_degree=3)
        caps = create_test_capabilities(300, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ply_path = tmp_path / "in.ply"
            out_path = tmp_path / "out.usda"

            PLYExporter().export(AttributesExportAdapter(attrs, caps, is_preactivation=True), ply_path)
            transcode(
                ply_path,
                out_path,
                "lightfield",
                max_per_volume=64,
                split_large_gaussians=True,
                validate_usd=False,
            )

            stage = Usd.Stage.Open(str(out_path))
            fields = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
            assert len(fields) > 1  # oversized prim subdivided
            importer = USDImporter()
            total = sum(a.num_gaussians for a, _ in importer.load_fields(out_path))
            # Splitting can only add Gaussians, never drop them.
            assert total >= 300

    def test_transcode_usd_to_ply_low_sh_degree(self):
        """USD→PLY of a sub-degree-3 source must not crash on the specular reshape."""
        attrs = create_test_attributes(16, sh_degree=2)
        caps = create_test_capabilities(16, sh_degree=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            usd_path = tmp_path / "in.usda"
            ply_path = tmp_path / "out.ply"

            USDExporter(
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(AttributesExportAdapter(attrs, caps, is_preactivation=True), usd_path, validate_usd=False)

            transcode(usd_path, ply_path, "ply", validate_usd=False)

            assert ply_path.exists()
            sub, _ = PLYImporter(max_sh_degree=3).load(ply_path)
            assert sub.num_gaussians == 16

    def test_transcode_usd_partition_keeps_prims_and_copies_render_products(self):
        """USD→USD with --max-gaussians-per-volume subdivides the prim and still copies /Render."""
        attrs = create_test_attributes(32, sh_degree=3)
        caps = create_test_capabilities(32, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.usda"
            output_path = tmp_path / "output.usda"

            adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
            USDExporter(
                export_cameras=True,
                export_background=False,
                apply_normalizing_transform=False,
            ).export(adapter, input_path, dataset=MockCameraDataset(), validate_usd=False)

            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format="lightfield",
                copy_cameras_source=(input_path, tmp_path),
                validate_usd=False,
                max_per_volume=8,
            )

            stage = Usd.Stage.Open(str(output_path))
            assert stage
            fields = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
            assert len(fields) > 1  # the 32-Gaussian prim was subdivided
            # Source cameras / render products copied as-is.
            product = stage.GetPrimAtPath("/Render/camera_0000")
            assert product.IsValid()
            assert product.GetTypeName() == "RenderProduct"
            assert stage.GetPrimAtPath("/World/Cameras/camera_0000").IsValid()

    def test_transcode_multiple_ply_to_multiprim_usd(self):
        """Several PLY inputs combine into one USD with one ParticleField prim per input."""
        counts = [20, 35, 15]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ply_paths = []
            for i, n in enumerate(counts):
                a = create_test_attributes(n, sh_degree=3)
                c = create_test_capabilities(n, sh_degree=3)
                p = tmp_path / f"in_{i}.ply"
                PLYExporter().export(AttributesExportAdapter(a, c, is_preactivation=True), p)
                ply_paths.append(p)

            out = tmp_path / "combined.usda"
            transcode_files(ply_paths, out, "lightfield", validate_usd=False)

            stage = Usd.Stage.Open(str(out))
            fields = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
            assert len(fields) == len(counts)

            importer = USDImporter()
            fields = importer.load_fields(out)
            assert len(fields) == len(counts)
            assert sum(a.num_gaussians for a, _ in fields) == sum(counts)

    def test_transcode_multiple_ply_to_multivolume_nurec(self):
        """Several PLY inputs combine into one NuRec USDZ with one UsdVol.Volume per input."""
        counts = [20, 30]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ply_paths = []
            for i, n in enumerate(counts):
                a = create_test_attributes(n, sh_degree=3)
                c = create_test_capabilities(n, sh_degree=3)
                p = tmp_path / f"in_{i}.ply"
                PLYExporter().export(AttributesExportAdapter(a, c, is_preactivation=True), p)
                ply_paths.append(p)

            out = tmp_path / "combined.usdz"
            transcode_files(ply_paths, out, "nurec", validate_usd=False)

            stage = Usd.Stage.Open(str(out))
            volumes = [
                p
                for p in stage.Traverse()
                if p.GetTypeName() == "Volume"
                and p.GetAttribute("omni:nurec:isNuRecVolume").IsValid()
                and p.GetAttribute("omni:nurec:isNuRecVolume").Get()
            ]
            assert len(volumes) == len(counts)

    def test_transcode_multiple_ply_to_usd_subdivides_only_oversized(self):
        """Per-prim partitioning: small inputs are kept intact, oversized ones are subdivided."""
        small, large = 10, 300
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ply_paths = []
            for i, n in enumerate((small, large)):
                a = create_test_attributes(n, sh_degree=3)
                c = create_test_capabilities(n, sh_degree=3)
                p = tmp_path / f"in_{i}.ply"
                PLYExporter().export(AttributesExportAdapter(a, c, is_preactivation=True), p)
                ply_paths.append(p)

            out = tmp_path / "combined.usda"
            transcode_files(ply_paths, out, "lightfield", validate_usd=False, max_per_volume=64)

            importer = USDImporter()
            sizes = sorted(a.num_gaussians for a, _ in importer.load_fields(out))
            assert sum(sizes) == small + large
            assert len(sizes) >= 3  # small (1 prim) + large (>=2 prims)
            assert min(sizes) == small  # small input preserved as a single prim
            assert max(sizes) <= 64  # every prim respects the budget


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
