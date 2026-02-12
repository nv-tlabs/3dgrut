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
from pathlib import Path

import numpy as np
import pytest

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.adapter import AttributesExportAdapter
from threedgrut.export.importers import PLYImporter, USDImporter
from threedgrut.export.formats import PLYExporter
from threedgrut.export.usd.exporter import USDExporter


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
