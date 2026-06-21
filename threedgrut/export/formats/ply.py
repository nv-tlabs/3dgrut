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

"""PLY format exporter for Gaussian splatting models."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from threedgrut.export.accessor import GaussianAttributes, GaussianExportAccessor
from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.utils.logger import logger

if TYPE_CHECKING:
    from threedgrut.export.partition import PartitionResult


class PLYExporter(ModelExporter):
    """Exporter for PLY format files.

    Implements export functionality for Gaussian models
    in the PLY file format.
    """

    @staticmethod
    def _construct_list_of_attributes(features_albedo, features_specular, scale, rotation):
        """Construct the list of PLY attribute names."""
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        # DC coefficients (albedo)
        for i in range(features_albedo.shape[1]):
            attrs.append(f"f_dc_{i}")
        # Higher-order SH coefficients (specular)
        for i in range(features_specular.shape[1]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(scale.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(rotation.shape[1]):
            attrs.append(f"rot_{i}")
        return attrs

    @classmethod
    def _write_ply(cls, attrs: GaussianAttributes, output_path: Path) -> None:
        """Serialize a single ``GaussianAttributes`` (pre-activation) to a PLY file."""
        num_gaussians = attrs.num_gaussians

        # Create normal vectors (placeholder, pointing up)
        mogt_nrm = np.repeat(np.array([[0, 0, 1]], dtype=np.float32), repeats=num_gaussians, axis=0)

        # Reshape specular coefficients for PLY format (channel-major layout)
        # From [N, M*3] to [N, M, 3] to [N, 3, M] to [N, M*3] (channel-major).
        # Derive M from the actual array width: importers pad specular to a fixed (degree-3)
        # width while reporting the true SH degree in caps, so deriving M from a degree value
        # would mismatch the stored width for sources below the padding degree.
        num_speculars = attrs.specular.shape[1] // 3
        mogt_specular = attrs.specular.reshape((num_gaussians, num_speculars, 3))
        mogt_specular = mogt_specular.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))

        # Build PLY dtype
        dtype_full = [
            (attribute, "f4")
            for attribute in cls._construct_list_of_attributes(
                attrs.albedo, mogt_specular, attrs.scales, attrs.rotations
            )
        ]

        # Create PLY element
        elements = np.empty(num_gaussians, dtype=dtype_full)
        attributes = np.concatenate(
            (attrs.positions, mogt_nrm, attrs.albedo, mogt_specular, attrs.densities, attrs.scales, attrs.rotations),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(str(output_path))

    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, **kwargs) -> None:
        """Export the model to a PLY file.

        Args:
            model: The model to export (must implement ExportableModel)
            output_path: Path where the PLY file will be saved
            dataset: Optional dataset (not used for PLY export)
            conf: Optional configuration (not used for PLY export)
            **kwargs: Additional parameters (not used for PLY export)
        """
        logger.info(f"exporting ply file to {output_path}...")

        # Use accessor to get attributes
        accessor = GaussianExportAccessor(model, conf)
        attrs = accessor.get_attributes(preactivation=True)
        self._write_ply(attrs, Path(output_path))


@torch.no_grad()
def export_partitions(results, output_path: Path) -> List[Path]:
    """Write partitioned source(s) to PLY point clouds.

    ``results`` is one or more :class:`PartitionResult` (one per source). With a single output
    partition, writes ``<stem>.ply`` (identical to a regular PLY export). With multiple, writes one
    ``<stem>_partition_NNN.ply`` per partition (numbered across all sources) plus a
    ``<stem>_partitions.json`` manifest. Returns the list of written PLY paths.
    """
    if not isinstance(results, (list, tuple)):
        results = [results]
    output_path = Path(output_path)
    stem_path = output_path.with_suffix("")
    total = sum(r.num_partitions for r in results)

    if total <= 1:
        out = stem_path.with_suffix(".ply")
        logger.info(f"exporting ply file to {out}...")
        PLYExporter._write_ply(results[0].full_attributes(preactivation=True), out)
        return [out]

    written: List[Path] = []
    manifest = {"num_partitions": total, "partitions": []}
    width = max(3, len(str(total - 1)))
    running = 0
    for result in results:
        for _pid, sub in result.iter_partitions(preactivation=True):
            out = stem_path.parent / f"{stem_path.name}_partition_{running:0{width}d}.ply"
            mins = sub.positions.min(axis=0).tolist() if sub.num_gaussians else None
            maxs = sub.positions.max(axis=0).tolist() if sub.num_gaussians else None
            extent = f"min={[round(v, 4) for v in mins]} max={[round(v, 4) for v in maxs]}" if mins else "empty"
            logger.info(f"exporting partition {running} ({sub.num_gaussians} gaussians, extent {extent}) to {out}...")
            PLYExporter._write_ply(sub, out)
            written.append(out)
            manifest["partitions"].append(
                {
                    "id": running,
                    "num_gaussians": int(sub.num_gaussians),
                    "file": out.name,
                    "aabb_min": mins,
                    "aabb_max": maxs,
                }
            )
            running += 1

    manifest_path = stem_path.parent / f"{stem_path.name}_partitions.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"wrote PLY partition manifest to {manifest_path}")
    return written
