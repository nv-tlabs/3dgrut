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

"""Tests for spatial volume partitioning and partitioned export."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from threedgrut.export.base import ExportableModel
from threedgrut.export.partition import (
    gaussian_covariances,
    gaussian_extents,
    kdtree_partition,
    partition_scene,
    so3_to_quaternion_wxyz,
    split_large_gaussians,
)


class RandomGaussianModel(ExportableModel):
    """Minimal ExportableModel with random-but-valid Gaussians for partition tests."""

    def __init__(self, num_gaussians: int, sh_degree: int = 3, seed: int = 0, scale: float = 0.05):
        g = torch.Generator().manual_seed(seed)
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self._positions = torch.rand((num_gaussians, 3), generator=g) * 10.0
        quats = torch.randn((num_gaussians, 4), generator=g)
        self._rotations = torch.nn.functional.normalize(quats, dim=1)
        self._scales_pre = torch.full((num_gaussians, 3), float(np.log(scale)))
        self._density_pre = torch.full((num_gaussians, 1), 2.0)
        self._albedo = torch.rand((num_gaussians, 3), generator=g)
        num_specular = ((sh_degree + 1) ** 2 - 1) * 3
        self._specular = torch.zeros((num_gaussians, num_specular))

    def get_positions(self):
        return self._positions

    def get_max_n_features(self):
        return self.sh_degree

    def get_n_active_features(self):
        return self.sh_degree

    def get_scale(self, preactivation: bool = False):
        return self._scales_pre if preactivation else torch.exp(self._scales_pre)

    def get_rotation(self, preactivation: bool = False):
        return self._rotations

    def get_density(self, preactivation: bool = False):
        return self._density_pre if preactivation else torch.sigmoid(self._density_pre)

    def get_features_albedo(self):
        return self._albedo

    def get_features_specular(self):
        return self._specular


# ---------------------------------------------------------------------------
# KD-tree partitioning
# ---------------------------------------------------------------------------


def test_single_partition_when_under_cap():
    model = RandomGaussianModel(num_gaussians=50)
    result = partition_scene(model, max_per_volume=100)
    assert result.num_partitions == 1
    assert result.is_partitioned is False
    assert torch.all(result.labels == 0)


def test_kdtree_caps_covers_and_balances():
    positions = torch.rand((1000, 3)) * 10.0
    max_per_volume = 100
    labels, num_partitions = kdtree_partition(positions, max_per_volume)

    counts = torch.bincount(labels, minlength=num_partitions)
    # Cap respected.
    assert int(counts.max()) <= max_per_volume
    # Every Gaussian assigned exactly once across all partitions.
    assert int(counts.sum()) == 1000
    assert set(labels.tolist()) == set(range(num_partitions))
    # Balanced: median-split keeps counts within a factor of two.
    assert int(counts.min()) > max_per_volume // 2 - 1


def test_kdtree_device_budget(monkeypatch):
    """KD-tree device is chosen from the free GPU-memory budget, falling back to CPU."""
    import threedgrut.export.partition as P

    cpu = torch.device("cpu")

    # No CUDA -> always CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert P._resolve_kdtree_device(10**9, cpu).type == "cpu"

    # CUDA available with plenty of free memory and a small point count -> CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (40 * 1024**3, 48 * 1024**3))
    assert P._resolve_kdtree_device(1_000_000, cpu).type == "cuda"

    # Same free memory but a point count whose KD-tree budget exceeds it -> CPU fallback.
    huge = int(40 * 1024**3 * 0.8 / P._KDTREE_BYTES_PER_POINT) + 1
    assert P._resolve_kdtree_device(huge, cpu).type == "cpu"


def test_partition_scene_respects_cap():
    model = RandomGaussianModel(num_gaussians=1000, seed=3)
    result = partition_scene(model, max_per_volume=128)
    assert result.is_partitioned is True
    counts = torch.bincount(result.labels, minlength=result.num_partitions)
    assert int(counts.max()) <= 128
    assert int(counts.sum()) == 1000
    # full_attributes order matches labels.
    attrs = result.full_attributes(preactivation=False)
    assert attrs.num_gaussians == 1000


def test_partition_in_normalized_frame_groups_all_within_cap():
    """Partitioning in the principal-axis frame still caps, covers, and reorders nothing."""
    model = RandomGaussianModel(num_gaussians=1000, seed=3)
    result = partition_scene(model, max_per_volume=128, normalized_frame=True)
    assert result.is_partitioned is True
    counts = torch.bincount(result.labels, minlength=result.num_partitions)
    assert int(counts.max()) <= 128
    assert int(counts.sum()) == 1000
    # Grouping only: the exported geometry is the original, unrotated positions.
    attrs = result.full_attributes(preactivation=False)
    assert attrs.num_gaussians == 1000
    assert np.allclose(attrs.positions, model.get_positions().cpu().numpy(), atol=1e-5)


def test_rotate_to_principal_axes_robust_to_transparent_floater():
    """The principal axis follows the opaque bulk, not a far transparent floater."""
    from threedgrut.export.partition import _rotate_to_principal_axes

    torch.manual_seed(0)
    bulk = torch.randn(2000, 3) * torch.tensor([5.0, 0.5, 0.5])  # elongated along X
    floater = torch.tensor([[0.0, 0.0, 1000.0]])  # far, transparent
    pts = torch.cat([bulk, floater], dim=0)
    weights = torch.cat([torch.ones(2000), torch.zeros(1)])

    rotated = _rotate_to_principal_axes(pts, weights)
    # eigh is ascending, so the largest-variance axis is the last coordinate. It must track the
    # bulk's long (X) axis, i.e. the floater along Z neither dominates nor flips the basis.
    bulk_spreads = rotated[:2000].std(dim=0)
    assert int(bulk_spreads.argmax()) == 2


# ---------------------------------------------------------------------------
# Oversized-Gaussian splitting
# ---------------------------------------------------------------------------


def test_so3_quaternion_roundtrip():
    quats = torch.nn.functional.normalize(torch.randn(32, 4, dtype=torch.float64), dim=1)
    from threedgrut.utils.misc import quaternion_to_so3

    R = quaternion_to_so3(quats)
    recovered = so3_to_quaternion_wxyz(R)
    R2 = quaternion_to_so3(recovered)
    assert torch.allclose(R, R2, atol=1e-6)


def test_split_preserves_moments_and_shrinks_footprint():
    # One large anisotropic Gaussian at a known location.
    n_sigma = 3.0
    post = {
        "positions": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        "scales": torch.tensor([[2.0, 0.4, 0.3]], dtype=torch.float64),
        "rotations": torch.nn.functional.normalize(torch.tensor([[0.7, 0.2, 0.1, 0.3]], dtype=torch.float64), dim=1),
        "densities": torch.tensor([[0.6]], dtype=torch.float64),
        "albedo": torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float64),
        "specular": torch.zeros((1, 45), dtype=torch.float64),
    }
    parent_mu = post["positions"][0]
    parent_cov = gaussian_covariances(post["scales"], post["rotations"])[0]
    parent_extent = gaussian_extents(post["scales"], post["rotations"], n_sigma=n_sigma).max()

    out, added = split_large_gaussians(post, target_size=0.5, n_sigma=n_sigma, max_splits=1)
    assert added == 1  # one parent -> two children

    # Equal-weight mixture moments equal the parent's (moment-preserving split).
    mus = out["positions"]
    covs = gaussian_covariances(out["scales"], out["rotations"])
    w = 1.0 / mus.shape[0]
    mix_mean = (w * mus).sum(dim=0)
    mix_cov = (w * (covs + torch.einsum("ni,nj->nij", mus, mus))).sum(dim=0) - torch.outer(mix_mean, mix_mean)

    assert torch.allclose(mix_mean, parent_mu, atol=1e-6)
    assert torch.allclose(mix_cov, parent_cov, atol=1e-6)

    # Children footprints are smaller than the parent's principal footprint.
    child_extent = gaussian_extents(out["scales"], out["rotations"], n_sigma=n_sigma).max()
    assert float(child_extent) < float(parent_extent)


def test_partition_with_split_increases_count():
    model = RandomGaussianModel(num_gaussians=800, seed=5, scale=0.5)
    base = partition_scene(model, max_per_volume=128)
    split = partition_scene(model, max_per_volume=128, split=True, split_target_size=0.5, max_splits=2)
    assert split.metrics["total_exported"] >= base.metrics["total_exported"]
    assert split.metrics["num_split_added"] >= 0


# ---------------------------------------------------------------------------
# PLY round-trip
# ---------------------------------------------------------------------------


def test_ply_partition_roundtrip():
    from threedgrut.export.formats import export_partitions
    from threedgrut.export.importers import PLYImporter

    model = RandomGaussianModel(num_gaussians=500, seed=7)
    result = partition_scene(model, max_per_volume=120)
    assert result.num_partitions > 1

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "scene.ply"
        written = export_partitions(result, out)
        assert len(written) == result.num_partitions
        # No bare scene.ply when partitioned; suffixed files instead.
        assert not out.exists()
        assert (Path(d) / "scene_partitions.json").exists()

        total = 0
        importer = PLYImporter(max_sh_degree=model.get_max_n_features())
        for p in written:
            attrs, _caps = importer.load(p)
            total += attrs.num_gaussians
        assert total == 500


def test_ply_single_partition_no_suffix():
    from threedgrut.export.formats import export_partitions

    model = RandomGaussianModel(num_gaussians=50, seed=1)
    result = partition_scene(model, max_per_volume=100)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "scene.ply"
        written = export_partitions(result, out)
        assert written == [out]
        assert out.exists()
        assert not (Path(d) / "scene_partitions.json").exists()


# ---------------------------------------------------------------------------
# USD round-trip
# ---------------------------------------------------------------------------


def _export_partitioned_usd(model, result, out: Path, *, validate_usd: bool) -> None:
    """Export a PartitionResult to USD via USDExporter's ``partition`` kwarg (geometry only)."""
    from threedgrut.export.usd.exporter import USDExporter

    USDExporter(
        export_cameras=False,
        export_background=False,
        apply_normalizing_transform=False,
    ).export(model, out, partition=result, validate_usd=validate_usd)


def test_usd_partition_prims():
    pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")
    from pxr import Usd

    model = RandomGaussianModel(num_gaussians=500, seed=11)
    result = partition_scene(model, max_per_volume=120)
    assert result.num_partitions > 1

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "scene.usda"
        _export_partitioned_usd(model, result, out, validate_usd=True)
        stage = Usd.Stage.Open(str(out))
        particle_fields = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
        assert len(particle_fields) == result.num_partitions


def test_usd_single_partition_default_path():
    pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")
    from pxr import Usd

    model = RandomGaussianModel(num_gaussians=40, seed=13)
    result = partition_scene(model, max_per_volume=100)
    assert not result.is_partitioned

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "scene.usda"
        _export_partitioned_usd(model, result, out, validate_usd=True)
        stage = Usd.Stage.Open(str(out))
        # Regular single-prim layout: no Partition_* prims.
        assert stage.GetPrimAtPath("/World/Gaussians/gaussians").IsValid()
        assert not any(p.GetName().startswith("Partition_") for p in stage.Traverse())


# ---------------------------------------------------------------------------
# Transcode integration (multi-prim USD <-> partitioned outputs)
# ---------------------------------------------------------------------------


def _write_partitioned_usd(path: Path, num_gaussians: int, max_per_volume: int, seed: int):
    """Helper: author a multi-ParticleField USD and return (model, num_partitions)."""
    model = RandomGaussianModel(num_gaussians=num_gaussians, seed=seed)
    result = partition_scene(model, max_per_volume=max_per_volume)
    assert result.num_partitions > 1
    _export_partitioned_usd(model, result, path, validate_usd=False)
    return model, result.num_partitions


def test_usd_importer_loads_prims_as_separate_fields():
    pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")
    from threedgrut.export.importers import USDImporter

    with tempfile.TemporaryDirectory() as d:
        usd_path = Path(d) / "multi.usda"
        _model, num_partitions = _write_partitioned_usd(usd_path, num_gaussians=500, max_per_volume=120, seed=17)

        importer = USDImporter()
        fields = importer.load_fields(usd_path)
        # One field per ParticleField prim; never concatenated.
        assert len(fields) == num_partitions
        assert sum(attrs.num_gaussians for attrs, _ in fields) == 500
        # load() returns the first field for the single-field contract.
        first_attrs, _ = importer.load(usd_path)
        assert first_attrs.num_gaussians == fields[0][0].num_gaussians


def test_transcode_multiprim_usd_to_multi_ply():
    pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")
    from threedgrut.export.importers import PLYImporter
    from threedgrut.export.scripts.transcode import transcode

    with tempfile.TemporaryDirectory() as d:
        usd_path = Path(d) / "multi.usda"
        _model, num_partitions = _write_partitioned_usd(usd_path, num_gaussians=500, max_per_volume=120, seed=19)

        out_ply = Path(d) / "out.ply"
        transcode(usd_path, out_ply, output_format="ply")

        # One PLY per source ParticleField prim, plus a manifest; no bare out.ply.
        ply_files = sorted((Path(d)).glob("out_partition_*.ply"))
        assert len(ply_files) == num_partitions
        assert not out_ply.exists()
        assert (Path(d) / "out_partitions.json").exists()

        total = 0
        importer = PLYImporter(max_sh_degree=3)
        for p in ply_files:
            sub, _ = importer.load(p)
            total += sub.num_gaussians
        assert total == 500


def test_transcode_usd_to_usd_repartitions_into_more_prims():
    pytest.importorskip("pxr", reason="usd-core (pxr) is only available on linux x86_64")
    from pxr import Usd

    from threedgrut.export.scripts.transcode import transcode
    from threedgrut.export.usd.exporter import USDExporter

    with tempfile.TemporaryDirectory() as d:
        # A regular single-prim USD as the source.
        src = Path(d) / "single.usda"
        model = RandomGaussianModel(num_gaussians=600, seed=23)
        USDExporter(
            export_cameras=False,
            export_background=False,
            apply_normalizing_transform=False,
            export_post_processing=False,
        ).export(model, src, dataset=None, conf=None, validate_usd=False)
        src_stage = Usd.Stage.Open(str(src))
        src_fields = [p for p in src_stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
        assert len(src_fields) == 1

        # Re-partition during transcode according to the max Gaussian budget.
        out = Path(d) / "repartitioned.usda"
        transcode(src, out, output_format="lightfield", max_per_volume=128, validate_usd=True)

        out_stage = Usd.Stage.Open(str(out))
        out_fields = [p for p in out_stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
        assert len(out_fields) > 1
