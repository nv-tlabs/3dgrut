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
USD exporter for spatially partitioned Gaussian scenes.

Thin wrapper over :class:`~threedgrut.export.usd.exporter.USDExporter` that authors one
``ParticleField3DGaussianSplat`` prim per partition (``/World/Gaussians/Partition_NNN``) by
passing the :class:`PartitionResult` through the ``partition`` kwarg. A single partition yields
the regular single-prim layout. Delegating keeps camera / RenderProduct / source-prim copy and
packaging behavior identical to the standard exporter.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from threedgrut.export.base import ExportableModel
from threedgrut.export.usd.particle_field_hints import (
    DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
    normalize_particle_field_sorting_mode_hint,
)

if TYPE_CHECKING:
    from threedgrut.export.partition import PartitionResult

logger = logging.getLogger(__name__)


def _resolved_output_path(output_path: Path) -> Path:
    """Match USDExporter packaging: non-USD suffixes become .usdz."""
    suffix = output_path.suffix.lower()
    if suffix in (".usd", ".usda", ".usdc"):
        return output_path
    if suffix == ".usdz":
        return output_path
    return output_path.with_suffix(".usdz")


class VolumePartitionUSDExporter:
    """Export a :class:`PartitionResult` to USD with one ParticleField prim per partition."""

    def __init__(
        self,
        half_geometry: bool = False,
        half_features: bool = False,
        sorting_mode_hint: str = DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
    ):
        self.half_geometry = half_geometry
        self.half_features = half_features
        self.sorting_mode_hint = normalize_particle_field_sorting_mode_hint(sorting_mode_hint)

    @torch.no_grad()
    def export(
        self,
        model: ExportableModel,
        result: "PartitionResult",
        output_path: Path,
        conf=None,
        validate_usd: bool = True,
        **kwargs,
    ) -> Path:
        """Write the partitioned scene to ``output_path`` (``.usdz``/``.usda``/``.usd``).

        Extra kwargs (e.g. ``copy_cameras_source``, ``copy_source_skip_subtrees``,
        ``source_gaussian_transform``, ``apply_coordinate_transform``) are forwarded to
        :class:`USDExporter` so non-Gaussian prims (cameras, RenderProducts, …) are copied as-is.
        """
        from threedgrut.export.usd.exporter import USDExporter

        output_path = Path(output_path)
        USDExporter(
            half_geometry=self.half_geometry,
            half_features=self.half_features,
            export_cameras=False,
            export_background=False,
            apply_normalizing_transform=False,
            sorting_mode_hint=self.sorting_mode_hint,
            export_post_processing=False,
        ).export(
            model,
            output_path,
            dataset=None,
            conf=conf,
            validate_usd=validate_usd,
            partition=result,
            **kwargs,
        )
        return _resolved_output_path(output_path)
