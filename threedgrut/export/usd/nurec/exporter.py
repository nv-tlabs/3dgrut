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
NuRec Exporter for Omniverse-compatible USDZ files.

This exporter produces USDZ files in the NuRec format for rendering in
Omniverse Kit and Isaac Sim. For standard OpenUSD export, use USDExporter.
"""

import gzip
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import msgpack
import numpy as np
import torch

from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.export.transforms import estimate_normalizing_transform
from threedgrut.export.usd.nurec.serializer import (
    serialize_nurec_usd,
    serialize_usd_default_layer,
    write_to_usdz,
)
from threedgrut.export.usd.nurec.templates import NamedSerialized, fill_3dgut_template
from threedgrut.utils.logger import logger


def _get_default_nurec_conf() -> SimpleNamespace:
    """Minimal config for NuRec export when no training config is available (e.g. transcode from PLY)."""
    conf = SimpleNamespace()
    conf.model = SimpleNamespace(density_activation="sigmoid", scale_activation="exp")
    conf.render = SimpleNamespace(
        method="3dgut",
        particle_kernel_degree=2,
        particle_kernel_density_clamping=True,
        particle_kernel_min_response=0.0113,
        particle_radiance_sph_degree=3,
        min_transmittance=0.0001,
        splat=SimpleNamespace(
            global_z_order=True,
            n_rolling_shutter_iterations=5,
            ut_alpha=1.0,
            ut_beta=2.0,
            ut_kappa=0.0,
            ut_require_all_sigma_points_valid=False,
            ut_in_image_margin_factor=0.1,
            rect_bounding=True,
            tight_opacity_bounding=True,
            tile_based_culling=True,
            k_buffer_size=0,
        ),
    )
    conf.export_usd = SimpleNamespace(apply_normalizing_transform=True)
    return conf


class NuRecExporter(ModelExporter):
    """Exporter for NuRec/Omniverse USDZ format files.

    Implements export functionality for Gaussian models in the NuRec USDZ format,
    which allows for rendering in Omniverse Kit and Isaac Sim.

    This is the legacy format. For standard OpenUSD export with ParticleField3DGaussianSplat
    schema, use USDExporter instead.
    """

    @torch.no_grad()
    def export(
        self, model: ExportableModel, output_path: Path, dataset=None, conf: Dict[str, Any] = None, **kwargs
    ) -> None:
        """Export the model to a NuRec USDZ file.

        Args:
            model: The model to export (must implement ExportableModel)
            output_path: Path where the USDZ file will be saved
            dataset: Optional dataset to get camera poses for upright transform
            conf: Configuration parameters for the renderer
            **kwargs: Additional parameters for export
        """
        logger.info(f"exporting nurec usdz file to {output_path}...")

        if conf is None:
            conf = _get_default_nurec_conf()
        if conf.render.method not in ["3dgut", "3dgrt"]:
            raise ValueError(
                f"NuRec export requires render.method to be '3dgut' or '3dgrt', got '{conf.render.method}'"
            )

        # Use accessor to get model data
        accessor = GaussianExportAccessor(model, conf)
        attrs = accessor.get_attributes(preactivation=True)
        caps = accessor.get_capabilities()

        # Apply normalizing transform if enabled and dataset is provided
        normalizing_transform = np.eye(4)
        export_usd = getattr(conf, "export_usd", None)
        apply_transform = getattr(export_usd if export_usd is not None else conf, "apply_normalizing_transform", True)
        if apply_transform and dataset is not None:
            try:
                poses = dataset.get_poses()
                normalizing_transform = estimate_normalizing_transform(poses)
                logger.info("Applying normalizing transform to NuRec export")
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to apply normalizing transform: {e}")
                normalizing_transform = np.eye(4)

        # Set up common parameters
        template_params = {
            "positions": attrs.positions,
            "rotations": attrs.rotations,
            "scales": attrs.scales,
            "densities": attrs.densities,
            "features_albedo": attrs.albedo,
            "features_specular": attrs.specular,
            "n_active_features": caps.sh_degree,
            "density_kernel_degree": conf.render.particle_kernel_degree,
            # Common renderer configuration parameters
            "density_activation": conf.model.density_activation,
            "scale_activation": conf.model.scale_activation,
            "rotation_activation": "normalize",  # Always normalize for rotations
            "density_kernel_density_clamping": conf.render.particle_kernel_density_clamping,
            "density_kernel_min_response": conf.render.particle_kernel_min_response,
            "radiance_sph_degree": conf.render.particle_radiance_sph_degree,
            "transmittance_threshold": conf.render.min_transmittance,
        }

        if conf.render.method == "3dgut":
            # 3DGUT-specific splatting parameters
            template_params.update(
                {
                    "global_z_order": conf.render.splat.global_z_order,
                    "n_rolling_shutter_iterations": conf.render.splat.n_rolling_shutter_iterations,
                    "ut_alpha": conf.render.splat.ut_alpha,
                    "ut_beta": conf.render.splat.ut_beta,
                    "ut_kappa": conf.render.splat.ut_kappa,
                    "ut_require_all_sigma_points": conf.render.splat.ut_require_all_sigma_points_valid,
                    "image_margin_factor": conf.render.splat.ut_in_image_margin_factor,
                    "rect_bounding": conf.render.splat.rect_bounding,
                    "tight_opacity_bounding": conf.render.splat.tight_opacity_bounding,
                    "tile_based_culling": conf.render.splat.tile_based_culling,
                    "k_buffer_size": conf.render.splat.k_buffer_size,
                }
            )
        else:
            # For 3DGRT renderer, fall back to default splatting parameters
            logger.warning("Using 3DGUT NuRec template for 3DGRT data, may see slight loss of quality.")

        template = fill_3dgut_template(**template_params)

        # Compress the data
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
            packed = msgpack.packb(template)
            f.write(packed)

        model_file = NamedSerialized(filename=output_path.stem + ".nurec", serialized=buffer.getvalue())

        apply_coordinate_transform = kwargs.get("apply_coordinate_transform", False)

        # Create USD representations
        gauss_usd = serialize_nurec_usd(
            model_file,
            attrs.positions,
            normalizing_transform,
            apply_coordinate_transform=apply_coordinate_transform,
        )
        default_usd = serialize_usd_default_layer(gauss_usd)

        # Write the final USDZ file
        write_to_usdz(output_path, model_file, gauss_usd, default_usd)
