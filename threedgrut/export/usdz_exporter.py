# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gzip
import io
from pathlib import Path
from typing import Any, Dict

import msgpack
import numpy as np
import torch

from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.export.nurec_templates import NamedSerialized, fill_3dgut_template
from threedgrut.export.usd_util import serialize_nurec_usd, serialize_usd_default_layer, write_to_usdz
from threedgrut.export.normalizing_transform import estimate_normalizing_transform
from threedgrut.utils.logger import logger


class USDZExporter(ModelExporter):
    """Exporter for USDZ format files.

    Implements export functionality for Gaussian models in the USDZ file format, 
    which allows for rendering in Omniverse Kit and Isaac Sim.
    """

    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path,
               dataset=None, conf: Dict[str, Any] = None, **kwargs) -> None:
        """Export the model to a USDZ file.

        Args:
            model: The model to export (must implement ExportableModel)
            output_path: Path where the USDZ file will be saved
            dataset: Optional dataset to get camera poses for upright transform
            conf: Configuration parameters for the renderer
            **kwargs: Additional parameters for export
        """
        logger.info(f"exporting usdz file to {output_path}...")

        if not conf.render.method in ["3dgut", "3dgrt"]:
            raise ValueError(
                f"Not supported for USDZ export: {conf.render.method}")

        # Get model data
        positions = model.get_positions().detach().cpu().numpy()
        rotations = model.get_rotation(
            preactivation=True).detach().cpu().numpy()
        scales = model.get_scale(preactivation=True).detach().cpu().numpy()
        densities = model.get_density(
            preactivation=True).detach().cpu().numpy()
        features_albedo = model.get_features_albedo().detach().cpu().numpy()
        features_specular = model.get_features_specular().detach().cpu().numpy()
        n_active_features = model.get_n_active_features()

        # Apply normalizing transform if enabled and dataset is provided
        normalizing_transform = np.eye(4)
        if (conf.export_usdz.apply_normalizing_transform and
                dataset is not None):
            try:
                poses = dataset.get_poses()
                normalizing_transform = estimate_normalizing_transform(poses)
                logger.info("Applying normalizing transform to USDZ export")
            except Exception as e:
                logger.warning(f"Failed to apply normalizing transform: {e}")
                normalizing_transform = np.eye(4)

        # Set up common parameters
        template_params = {
            "positions": positions,
            "rotations": rotations,
            "scales": scales,
            "densities": densities,
            "features_albedo": features_albedo,
            "features_specular": features_specular,
            "n_active_features": n_active_features,
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
            template_params.update({
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
                "k_buffer_size": conf.render.splat.k_buffer_size
            })
        else:
            # For 3DGRT renderer, fall back to default splatting parameters
            logger.warning(
                "Using 3DGUT NuRec template for 3DGRT data, may see slight loss of quality.")

        template = fill_3dgut_template(**template_params)

        # Compress the data
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
            packed = msgpack.packb(template)
            f.write(packed)

        model_file = NamedSerialized(
            filename=output_path.stem + ".nurec",
            serialized=buffer.getvalue()
        )

        # Create USD representations
        gauss_usd = serialize_nurec_usd(
            model_file, positions, normalizing_transform)
        default_usd = serialize_usd_default_layer(gauss_usd)

        # Write the final USDZ file
        write_to_usdz(output_path, model_file, gauss_usd, default_usd)
