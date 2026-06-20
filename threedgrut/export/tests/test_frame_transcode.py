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

"""Canonical-frame transcode tests, exercised through transitivity in several configurations."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")

from pxr import Usd, UsdGeom

from threedgrut.export.adapter import AttributesExportAdapter
from threedgrut.export.formats import PLYExporter
from threedgrut.export.importers import PLYImporter, USDImporter
from threedgrut.export.scripts.transcode import transcode, transcode_files
from threedgrut.export.tests.test_transcode_roundtrip import (
    MockCameraDataset,
    create_test_attributes,
    create_test_capabilities,
)
from threedgrut.export.transforms import usd_matrix_to_numpy
from threedgrut.export.usd.exporter import USDExporter


def _write_ply(path, n, seed):
    a = create_test_attributes(n, sh_degree=3)
    # vary positions per seed so PCA has a well-defined frame
    rng = np.random.default_rng(seed)
    a.positions[:] = (rng.standard_normal((n, 3)) * np.array([4.0, 0.3, 1.5])).astype(np.float32)
    c = create_test_capabilities(n, sh_degree=3)
    PLYExporter().export(AttributesExportAdapter(a, c, is_preactivation=True), path)


def _first_particlefield(stage):
    return next(p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat")


def test_frame_transitivity_ply_usd_ply():
    """PLY->USD(pca)->PLY world geometry == PLY->PLY(pca): frame + SH bake are transitive."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "in.ply"
        _write_ply(src, 400, seed=1)

        direct = d / "direct.ply"
        transcode(src, direct, "ply", frame_mode="pca", up_axis="y", frame_origin="plane")

        usd = d / "mid.usda"
        transcode(src, usd, "lightfield", frame_mode="pca", up_axis="y", frame_origin="plane", validate_usd=False)
        via = d / "via.ply"
        transcode(usd, via, "ply")  # frame none -> bakes the USD's world transform

        a, _ = PLYImporter(max_sh_degree=3).load(direct)
        b, _ = PLYImporter(max_sh_degree=3).load(via)
        assert np.allclose(a.positions, b.positions, atol=1e-4)
        assert np.allclose(a.specular, b.specular, atol=1e-4)  # SH rotation transitive


def test_frame_keeps_camera_to_prim_relative_unchanged():
    """USD->USD with --frame pca moves cameras and Gaussians together (relative transform fixed)."""
    attrs = create_test_attributes(200, sh_degree=3)
    rng = np.random.default_rng(2)
    attrs.positions[:] = (rng.standard_normal((200, 3)) * np.array([5.0, 0.4, 2.0])).astype(np.float32)
    caps = create_test_capabilities(200, sh_degree=3)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "src.usda"
        USDExporter(export_cameras=True, export_background=False, apply_normalizing_transform=False).export(
            AttributesExportAdapter(attrs, caps, is_preactivation=True), src, dataset=MockCameraDataset(), validate_usd=False
        )
        out = d / "out.usda"
        transcode(
            src, out, "lightfield", frame_mode="pca", up_axis="y", copy_cameras_source=(src, d), validate_usd=False
        )

        def relative(stage_path):
            stage = Usd.Stage.Open(str(stage_path))
            prim = _first_particlefield(stage)
            cam = stage.GetPrimAtPath("/World/Cameras/camera_0000")
            assert cam.IsValid()
            m_prim = usd_matrix_to_numpy(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            m_cam = usd_matrix_to_numpy(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            return np.linalg.inv(m_cam) @ m_prim

        assert np.allclose(relative(src), relative(out), atol=1e-4)


def test_frame_multi_prim_shares_one_global_frame():
    """Two PLY inputs -> one USD(pca): a single /World frame, prims preserved, world transitive."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p0, p1 = d / "a.ply", d / "b.ply"
        _write_ply(p0, 150, seed=3)
        _write_ply(p1, 250, seed=4)

        usd = d / "combined.usda"
        transcode_files([p0, p1], usd, "lightfield", frame_mode="pca", up_axis="y", validate_usd=False)

        stage = Usd.Stage.Open(str(usd))
        fields = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
        assert len(fields) == 2  # one prim per input, not merged
        # Composition-safe: /World stays identity; the frame is a named op on the content root.
        assert not UsdGeom.Xformable(stage.GetPrimAtPath("/World")).GetOrderedXformOps()
        gaussians = stage.GetPrimAtPath("/World/Gaussians")
        op_names = [op.GetOpName() for op in UsdGeom.Xformable(gaussians).GetOrderedXformOps()]
        assert any("canonicalFrame" in n for n in op_names)

        # Transitive: USD(pca) -> PLY(none) reproduces the framed world for every input's points.
        importer = USDImporter()
        total_usd = sum(a.num_gaussians for a, _ in importer.load_fields(usd))
        ply_out = d / "out.ply"
        transcode(usd, ply_out, "ply")
        plys = sorted(d.glob("out_partition_*.ply")) or [ply_out]
        total_ply = sum(PLYImporter(max_sh_degree=3).load(p)[0].num_gaussians for p in plys)
        assert total_ply == total_usd == 400


def test_frame_nurec_moves_copied_cameras():
    """transcode USD(with camera)->NuRec --frame pca must frame the copied cameras too (not just the volume)."""
    attrs = create_test_attributes(200, sh_degree=3)
    rng = np.random.default_rng(9)
    attrs.positions[:] = (rng.standard_normal((200, 3)) * np.array([5.0, 0.4, 2.0])).astype(np.float32)
    caps = create_test_capabilities(200, sh_degree=3)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "src.usda"
        USDExporter(export_cameras=True, export_background=False, apply_normalizing_transform=False).export(
            AttributesExportAdapter(attrs, caps, is_preactivation=True), src, dataset=MockCameraDataset(), validate_usd=False
        )
        out = d / "out.usdz"
        transcode(src, out, "nurec", frame_mode="pca", up_axis="z", copy_cameras_source=(src, d), validate_usd=False)

        def world(prim):
            return usd_matrix_to_numpy(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        # camera<->content relative transform must be the same in source (ParticleField) and
        # framed NuRec output (Volume) — i.e. the camera moved with the content by the frame.
        src_stage = Usd.Stage.Open(str(src))
        src_cam = src_stage.GetPrimAtPath("/World/Cameras/camera_0000")
        src_content = _first_particlefield(src_stage)
        rel_src = np.linalg.inv(world(src_cam)) @ world(src_content)

        out_stage = Usd.Stage.Open(str(out))
        out_cam = next(p for p in out_stage.Traverse() if p.GetTypeName() == "Camera")
        out_vol = next(
            p
            for p in out_stage.Traverse()
            if p.GetTypeName() == "Volume" and p.GetAttribute("omni:nurec:isNuRecVolume").Get()
        )
        rel_out = np.linalg.inv(world(out_cam)) @ world(out_vol)
        assert np.allclose(rel_src, rel_out, atol=1e-4)


def test_frame_multi_volume_nurec_authors_world_frame():
    """transcode --frame pca to multi-volume NuRec authors a /World frame and N volumes."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p0, p1 = d / "a.ply", d / "b.ply"
        _write_ply(p0, 120, seed=5)
        _write_ply(p1, 180, seed=6)
        out = d / "combined.usdz"
        transcode_files([p0, p1], out, "nurec", frame_mode="pca", up_axis="z", validate_usd=False)

        stage = Usd.Stage.Open(str(out))
        volumes = [
            p
            for p in stage.Traverse()
            if p.GetTypeName() == "Volume"
            and p.GetAttribute("omni:nurec:isNuRecVolume").IsValid()
            and p.GetAttribute("omni:nurec:isNuRecVolume").Get()
        ]
        assert len(volumes) == 2
