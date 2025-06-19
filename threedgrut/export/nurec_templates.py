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

import zipfile
from typing import Dict, Any, Union
from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class NamedSerialized:
    """
    Class to store serialized data with a filename.
    """
    filename: str
    serialized: Union[str, bytes]

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        """
        Save the serialized data to a zip file.

        Args:
            zip_file: Zip file to save the data to
        """
        zip_file.writestr(self.filename, self.serialized)


def _fill_state_dict_tensors(
    template: Dict[str, Any],
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    dtype=np.float16
) -> None:
    """
    Helper function to fill the state dict tensors in a template.

    Args:
        template: Template dictionary to fill
        positions: Gaussian positions (N, 3)
        rotations: Gaussian rotations (N, 4)
        scales: Gaussian scales (N, 3)
        densities: Gaussian densities (N, 1)
        features_albedo: Gaussian albedo features (N, 3)
        features_specular: Gaussian specular features (N, M)
        n_active_features: Active SH degree
        dtype: Data type to convert to (default: np.float16)
    """
    # Convert data to specified format for efficiency
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.positions"] = positions.astype(
        dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.rotations"] = rotations.astype(
        dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.scales"] = scales.astype(
        dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.densities"] = densities.astype(
        dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(
        dtype).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(
        dtype).tobytes()

    # Create empty extra_signal tensor
    extra_signal = np.zeros((positions.shape[0], 0), dtype=dtype)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()

    # Store n_active_features as binary data (64-bit integer)
    n_active_features_binary = np.array(
        [n_active_features], dtype=np.int64).tobytes()
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.n_active_features"] = n_active_features_binary

    # Store shapes
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.positions.shape"] = list(
        positions.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.rotations.shape"] = list(
        rotations.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.scales.shape"] = list(
        scales.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.densities.shape"] = list(
        densities.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_albedo.shape"] = list(
        features_albedo.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.features_specular.shape"] = list(
        features_specular.shape)
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.extra_signal.shape"] = list(
        extra_signal.shape)
    # Empty array for scalar value
    template["nre_data"]["state_dict"][".gaussians_nodes.gaussians.n_active_features.shape"] = []


def fill_3dgut_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    density_activation: str = "sigmoid",
    scale_activation: str = "exp",
    rotation_activation: str = "normalize",
    density_kernel_degree: int = 2,
    density_kernel_density_clamping: bool = False,
    density_kernel_min_response: float = 0.0,
    radiance_sph_degree: int = 3,
    transmittance_threshold: float = 0.001,
    global_z_order: bool = False,
    n_rolling_shutter_iterations: int = 5,
    ut_alpha: float = 1.0,
    ut_beta: float = 2.0,
    ut_kappa: float = 0.0,
    ut_require_all_sigma_points: bool = False,
    image_margin_factor: float = 0.1,
    rect_bounding: bool = True,
    tight_opacity_bounding: bool = True,
    tile_based_culling: bool = True,
    k_buffer_size: int = 0
) -> Dict[str, Any]:
    """
    Create and fill the 3DGUT JSON template with gaussian data.

    Args:
        positions: Gaussian positions (N, 3)
        rotations: Gaussian rotations (N, 4)
        scales: Gaussian scales (N, 3)
        densities: Gaussian densities (N, 1)
        features_albedo: Gaussian albedo features (N, 3)
        features_specular: Gaussian specular features (N, M)
        n_active_features: Active SH degree

        Render parameters interfaced between 3DGRUT and NuRec:

        density_kernel_degree: Kernel degree for density computation
        density_activation: Activation function for density
        scale_activation: Activation function for scale
        rotation_activation: Activation function for rotation
        density_kernel_density_clamping: Whether to clamp density kernel
        density_kernel_min_response: Minimum response for density kernel
        radiance_sph_degree: SH degree for radiance
        transmittance_threshold: Threshold for transmittance (min_transmittance in 3DGRUT)

        3DGUT-specific splatting parameters:

        global_z_order: Whether to use global z-order
        n_rolling_shutter_iterations: Number of rolling shutter iterations
        ut_alpha: Alpha parameter for unscented transform
        ut_beta: Beta parameter for unscented transform
        ut_kappa: Kappa parameter for unscented transform
        ut_require_all_sigma_points: Whether to require all sigma points
        image_margin_factor: Image margin factor (ut_in_image_margin_factor in 3DGRUT)
        rect_bounding: Whether to use rectangular bounding
        tight_opacity_bounding: Whether to use tight opacity bounding
        tile_based_culling: Whether to use tile-based culling
        k_buffer_size: Size of the k-buffer

    Returns:
        Dictionary with the filled 3DGUT template
    """
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": density_activation,
                        "scale_activation": scale_activation,
                        "rotation_activation": rotation_activation,
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,  # TODO: Does this have an equivalent in 3DGRUT?
                            "density_kernel_degree": density_kernel_degree,
                            "density_kernel_density_clamping": density_kernel_density_clamping,
                            "density_kernel_min_response": density_kernel_min_response,
                            "radiance_sph_degree": radiance_sph_degree,
                        },
                        "transmittance_threshold": transmittance_threshold,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": global_z_order,
                    "projection": {
                        "n_rolling_shutter_iterations": n_rolling_shutter_iterations,
                        "ut_dim": 3,  # TODO: Does this have an equivalent in 3DGRUT?
                        "ut_alpha": ut_alpha,
                        "ut_beta": ut_beta,
                        "ut_kappa": ut_kappa,
                        "ut_require_all_sigma_points": ut_require_all_sigma_points,
                        "image_margin_factor": image_margin_factor,
                        "min_projected_ray_radius": 0.5477225575051661
                    },
                    "culling": {
                        "rect_bounding": rect_bounding,
                        "tight_opacity_bounding": tight_opacity_bounding,
                        "tile_based": tile_based_culling,
                        "near_clip_distance": 0.2,  # TODO: Does this have an equivalent in 3DGRUT?
                        # TODO: Does this have an equivalent in 3DGRUT?
                        "far_clip_distance": 3.402823466e+38
                    },
                    "render": {
                        "mode": "kbuffer",
                        "k_buffer_size": k_buffer_size
                    }
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {
                    "name": "skip-appearance",
                    "embedding_dim": 0,
                    "device": "cuda"
                },
                "background": {
                    "name": "skip-background",
                    "device": "cuda",
                    "composite_in_linear_space": False
                }
            },
            "state_dict": {
                "._extra_state": {
                    "obj_track_ids": {
                        "gaussians": []
                    }
                },
                ".gaussians_nodes.gaussians.positions": None,
                ".gaussians_nodes.gaussians.rotations": None,
                ".gaussians_nodes.gaussians.scales": None,
                ".gaussians_nodes.gaussians.densities": None,
                ".gaussians_nodes.gaussians.extra_signal": None,
                ".gaussians_nodes.gaussians.features_albedo": None,
                ".gaussians_nodes.gaussians.features_specular": None,
                ".gaussians_nodes.gaussians.n_active_features": None,
                # Shapes
                ".gaussians_nodes.gaussians.positions.shape": None,
                ".gaussians_nodes.gaussians.rotations.shape": None,
                ".gaussians_nodes.gaussians.scales.shape": None,
                ".gaussians_nodes.gaussians.densities.shape": None,
                ".gaussians_nodes.gaussians.extra_signal.shape": None,
                ".gaussians_nodes.gaussians.features_albedo.shape": None,
                ".gaussians_nodes.gaussians.features_specular.shape": None,
                ".gaussians_nodes.gaussians.n_active_features.shape": None
            }
        }
    }

    # Fill in the state dict tensors
    _fill_state_dict_tensors(
        template, positions, rotations, scales, densities,
        features_albedo, features_specular, n_active_features
    )

    return template
