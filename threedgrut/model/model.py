# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
from plyfile import PlyData, PlyElement

import threedgrt_tracer
import threedgrut.model.background as background
import threedgut_tracer
from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.utils import read_colmap_points3D_text, read_next_bytes
from threedgrut.export import PLYExporter
from threedgrut.export.base import ExportableModel
from threedgrut.model.geometry import k_nearest_neighbors, nearest_neighbor_dist_cpuKD
from threedgrut.optimizers import SelectiveAdam
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    get_activation_function,
    get_scheduler,
    quaternion_to_so3,
    sh_degree_to_num_features,
    sh_degree_to_specular_dim,
    to_np,
    to_torch,
)
from threedgrut.model.features import Features
from threedgrut.utils.render import RGB2SH

class MixtureOfGaussians(torch.nn.Module, ExportableModel):
    """ """

    @property
    def num_gaussians(self):
        return self.positions.shape[0]

    def feature_fields(self) -> list[str]:
        """Returns a list of feature field names - subclasses can override"""
        if self.feature_type == Features.Type.SH:
            return [
                "features_albedo",
                "features_specular",
            ]
        elif self.feature_type == Features.Type.NHT:
            return ["features"]
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_max_n_features(self) -> int:
        return self.max_n_features

    def get_n_active_features(self) -> int:
        return self.n_active_features

    def get_features_albedo(self) -> torch.Tensor:
        if self.feature_type == Features.Type.SH:
            return self.features_albedo
        else:
            raise AttributeError(f"features_albedo not available in feature_type='{self.feature_type.name.lower()}' mode")

    def get_features_specular(self) -> torch.Tensor:
        if self.feature_type == Features.Type.SH:
            return self.features_specular
        else:
            raise AttributeError(f"features_specular not available in feature_type='{self.feature_type.name.lower()}' mode")

    def get_features(self):
        if self.feature_type == Features.Type.SH:
            return torch.cat((self.features_albedo, self.features_specular), dim=1)
        elif self.feature_type == Features.Type.NHT:
            return self.features  # [N, K]
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def get_scale(self, preactivation=False):
        if preactivation:
            return self.scale
        else:
            return self.scale_activation(self.scale)

    def get_rotation(self, preactivation=False):
        if preactivation:
            return self.rotation
        else:
            return self.rotation_activation(self.rotation)

    def get_density(self, preactivation=False):
        if preactivation:
            return self.density
        else:
            return self.density_activation(self.density)

    def get_covariance(self) -> torch.Tensor:
        scales = self.get_scale()

        S = torch.zeros((self.num_gaussians, 3, 3), dtype=scales.dtype, device=self.device)
        R = quaternion_to_so3(self.get_rotation())

        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]

        return R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)

    def get_model_parameters(self) -> dict:
        assert self.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"

        model_params = {
            "positions": self.positions,
            "rotation": self.rotation,
            "scale": self.scale,
            "density": self.density,
            "background": self.background.state_dict(),
            # Add other attributes that we need at restore
            "n_active_features": self.n_active_features,
            "max_n_features": self.max_n_features,
            "progressive_training": self.progressive_training,
            "scene_extent": self.scene_extent,
            # Add optimizer state dict
            "optimizer": self.optimizer.state_dict(),
            "config": self.conf,
            # Feature type and dimensions
            "feature_type": self.feature_type.name.lower(),  # Store as string for serialization
            "particle_feature_dim": self.particle_feature_dim,
            "ray_feature_dim": self.ray_feature_dim,
        }

        if self.progressive_training:
            model_params["feature_dim_increase_interval"] = self.feature_dim_increase_interval
            model_params["feature_dim_increase_step"] = self.feature_dim_increase_step

        if self.feature_type == Features.Type.SH:
            model_params["features_albedo"] = self.features_albedo
            model_params["features_specular"] = self.features_specular
        elif self.feature_type == Features.Type.NHT:
            model_params["features"] = self.features

        return model_params

    def __init__(self, conf, scene_extent=None):
        super().__init__()

        # Store config early - needed for feature type detection
        self.conf = conf
        self.scene_extent = scene_extent

        sh_degree = conf.model.progressive_training.max_n_features
        render_sph_degree = conf.render.particle_radiance_sph_degree
        if sh_degree > render_sph_degree:
            logger.warning(
                f"model.progressive_training.max_n_features ({sh_degree}) is greater than "
                f"render.particle_radiance_sph_degree ({render_sph_degree}). "
                f"Clamping max_n_features to {render_sph_degree}."
            )
            sh_degree = render_sph_degree
        specular_dim = sh_degree_to_specular_dim(sh_degree)
        self.positions = torch.nn.Parameter(
            torch.empty([0, 3])
        )  # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
        self.rotation = torch.nn.Parameter(
            torch.empty([0, 4])
        )  # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
        self.scale = torch.nn.Parameter(torch.empty([0, 3]))  # Anisotropic scale of each Gaussian [n_gaussians, 3]
        self.density = torch.nn.Parameter(torch.empty([0, 1]))  # Density of each Gaussian [n_gaussians, 1]

        # Feature type configuration - determine feature storage mode
        self.feature_type = Features.Type.from_string(self.conf.model.feature_type)

        primitive_type = (getattr(conf.render, "primitive_type", None) or "").lower()
        if self.feature_type == Features.Type.NHT and primitive_type == "trisurfel":
            raise ValueError(
                "Trisurfels are not supported in NHT mode. Use primitive_type 'instances' or 'icosahedron'."
            )

        if self.feature_type == Features.Type.SH:
            # Spherical harmonics mode: separate albedo and specular features
            self.features_albedo = torch.nn.Parameter(
                torch.empty([0, 3])
            )  # Feature vector of the 0th order SH coefficients [n_gaussians, 3]
            self.features_specular = torch.nn.Parameter(
                torch.empty([0, specular_dim])
            )  # Features of the higher order SH coefficients [n_gaussians, specular_dim]
            self.particle_feature_dim = 3 + specular_dim  # SH coeffs (input to tracer)
            self.ray_feature_dim = 3  # RGB output from tracer
        elif self.feature_type == Features.Type.NHT:
            # NHT: per-particle feature vector, decoder maps rendered features -> RGB
            feat = Features(conf)
            num_points = feat.num_interpolation_points
            nht_dim = int(conf.model.nht_features.dim)
            if nht_dim % num_points != 0:
                raise ValueError(
                    f"nht_features.dim={nht_dim} must be divisible by num_interpolation_points={num_points} "
                    f"(interpolation_type + primitive)"
                )
            self.nht_num_interpolation_points = num_points
            self.particle_feature_dim = feat.particle_feature_dim
            self.ray_feature_dim = feat.ray_feature_dim
            self.features = torch.nn.Parameter(
                torch.empty([0, self.particle_feature_dim])
            )  # NHT features [n_gaussians, particle_feature_dim]
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}. Must be 'sh' or 'nht'.")

        self.max_sh_degree = sh_degree

        self.positions_gradient_norm = None

        self.device = "cuda"
        self.optimizer = None
        self.density_activation = get_activation_function(self.conf.model.density_activation)
        self.density_activation_inv = get_activation_function(self.conf.model.density_activation, inverse=True)
        self.scale_activation = get_activation_function(self.conf.model.scale_activation)
        self.scale_activation_inv = get_activation_function(self.conf.model.scale_activation, inverse=True)
        self.rotation_activation = get_activation_function("normalize")  # The default value of the dim parameter is 1

        self.background = background.make(self.conf.model.background.name, self.conf.model.background)

        # Check if we would like to do progressive training
        self.n_active_features = min(self.conf.model.progressive_training.init_n_features, sh_degree)
        self.max_n_features = (
            sh_degree  # For SH, this is the SH degree (clamped if > render.particle_radiance_sph_degree)
        )
        self.progressive_training = False
        if self.n_active_features < self.max_n_features:
            self.feature_dim_increase_interval = self.conf.model.progressive_training.increase_frequency
            self.feature_dim_increase_step = self.conf.model.progressive_training.increase_step
            self.progressive_training = True

        if conf.render.method == "3dgrt":
            self.renderer = threedgrt_tracer.Tracer(conf)
        elif conf.render.method == "3dgut":
            self.renderer = threedgut_tracer.Tracer(conf)
        else:
            raise ValueError(f"Unknown rendering method: {conf.render.method}")

        # State of gradients of Gaussian parameters
        self._gaussians_frozen = False

    @torch.no_grad()
    def build_acc(self, rebuild=True):
        self.renderer.build_acc(self, rebuild)

    def freeze_gaussians(self) -> None:
        """Freeze all Gaussian parameters for PPISP controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        self.positions.requires_grad = False
        self.rotation.requires_grad = False
        self.scale.requires_grad = False
        self.density.requires_grad = False

        if self.feature_type == Features.Type.SH:
            self.features_albedo.requires_grad = False
            self.features_specular.requires_grad = False
        elif self.feature_type == Features.Type.NHT:
            self.features.requires_grad = False

        self._gaussians_frozen = True
        logger.info("❄️ [Distillation] Gaussian parameters frozen")

    def validate_fields(self):
        num_gaussians = self.num_gaussians
        assert self.positions.shape == (num_gaussians, 3)
        assert self.density.shape == (num_gaussians, 1)
        assert self.rotation.shape == (num_gaussians, 4)
        assert self.scale.shape == (num_gaussians, 3)

        if self.feature_type == Features.Type.SH:
            assert self.features_albedo.shape == (num_gaussians, 3)
            specular_sh_dims = sh_degree_to_specular_dim(self.max_n_features)
            assert self.features_specular.shape == (num_gaussians, specular_sh_dims)
        elif self.feature_type == Features.Type.NHT:
            assert self.features.shape == (num_gaussians, self.particle_feature_dim)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def init_from_colmap(self, root_path: str, observer_pts):
        # Special case for scannetpp dataset
        if self.conf.dataset.type == "scannetpp":
            points_file = os.path.join(root_path, "colmap", "points3D.txt")
            pts, rgb, _ = read_colmap_points3D_text(points_file)
            file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)

        else:
            points_file = os.path.join(root_path, "sparse/0", "points3D.bin")
            # also handle nonbinary points files
            if not os.path.isfile(points_file):
                points_file = os.path.join(root_path, "sparse/0", "points3D.txt")
                pts, rgb, _ = read_colmap_points3D_text(points_file)
                file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
                file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)
            else:
                with open(points_file, "rb") as file:
                    n_pts = read_next_bytes(file, 8, "Q")[0]
                    logger.info(f"Found {n_pts} colmap points")

                    file_pts = np.zeros((n_pts, 3), dtype=np.float32)
                    file_rgb = np.zeros((n_pts, 3), dtype=np.float32)

                    for i_pt in range(n_pts):
                        # read the points
                        pt_data = read_next_bytes(file, 43, "QdddBBBd")
                        file_pts[i_pt, :] = np.array(pt_data[1:4])
                        file_rgb[i_pt, :] = np.array(pt_data[4:7])
                        # NOTE: error stored in last element of file, currently not used

                        # skip the track data
                        t_len = read_next_bytes(file, num_bytes=8, format_char_sequence="Q")[0]
                        read_next_bytes(file, num_bytes=8 * t_len, format_char_sequence="ii" * t_len)

                file_pts = torch.tensor(file_pts, dtype=torch.float32, device=self.device)
                file_rgb = torch.tensor(file_rgb, dtype=torch.uint8, device=self.device)

        assert file_rgb.dtype == torch.uint8, "Expecting RGB values to be in [0, 255] range"
        self.default_initialize_from_points(
            file_pts,
            observer_pts,
            file_rgb,
            use_observer_pts=self.conf.initialization.use_observation_points,
        )

    def init_from_fused_point_cloud(self, pc_path: str, observer_pts):
        """
        Initialize gaussians from an fused point cloud PLY file.
        Similar to init_from_colmap but loads from a given PLY file instead of sparse/0/points3D.txt

        Args:
            pc_path: Path to the PLY point cloud file
            observer_pts: Observer points tensor for scale initialization
        """
        logger.info(f"Loading fused point cloud from {pc_path}...")

        # Read PLY file
        plydata = PlyData.read(pc_path)
        vertices = plydata["vertex"]

        # Extract XYZ coordinates
        xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

        # Extract RGB colors (check if they exist)
        if "red" in vertices and "green" in vertices and "blue" in vertices:
            rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)
        else:
            # If no colors, initialize with random colors
            logger.warning("No RGB data found in point cloud, using random colors")
            rgb = np.random.randint(0, 256, size=(len(vertices), 3), dtype=np.uint8)

        # Convert to torch tensors
        file_pts = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)

        logger.info(f"Loaded {len(file_pts)} points from accumulated point cloud")

        # Initialize using the same method as COLMAP
        assert file_rgb.dtype == torch.uint8, "Expecting RGB values to be in [0, 255] range"
        self.default_initialize_from_points(
            file_pts,
            observer_pts,
            file_rgb,
            use_observer_pts=self.conf.initialization.use_observation_points,
        )

    def init_from_pretrained_point_cloud(self, pc_path: str, set_optimizable_parameters: bool = True):
        if self.feature_type != Features.Type.SH:
            raise NotImplementedError(f"init_from_pretrained_point_cloud only supports feature_type='sh', got '{self.feature_type.name.lower()}'")
        data = PlyData.read(pc_path)
        num_gaussians = len(data["vertex"])
        self.positions = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.rotation = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (
                            data["vertex"]["rot_0"],
                            data["vertex"]["rot_1"],
                            data["vertex"]["rot_2"],
                            data["vertex"]["rot_3"],
                        ),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.scale = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (
                            data["vertex"]["scale_0"],
                            data["vertex"]["scale_1"],
                            data["vertex"]["scale_2"],
                        ),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.density = torch.nn.Parameter(
            to_torch(
                data["vertex"]["opacity"].astype(np.float32).reshape(num_gaussians, 1),
                device=self.device,
            )
        )
        self.features_albedo = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (
                            data["vertex"]["f_dc_0"],
                            data["vertex"]["f_dc_1"],
                            data["vertex"]["f_dc_2"],
                        ),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore

        feats_sph = to_torch(
            np.transpose(
                np.stack(
                    (
                        data["vertex"]["f_rest_0"],
                        data["vertex"]["f_rest_1"],
                        data["vertex"]["f_rest_2"],
                        data["vertex"]["f_rest_3"],
                        data["vertex"]["f_rest_4"],
                        data["vertex"]["f_rest_5"],
                        data["vertex"]["f_rest_6"],
                        data["vertex"]["f_rest_7"],
                        data["vertex"]["f_rest_8"],
                        data["vertex"]["f_rest_9"],
                        data["vertex"]["f_rest_10"],
                        data["vertex"]["f_rest_11"],
                        data["vertex"]["f_rest_12"],
                        data["vertex"]["f_rest_13"],
                        data["vertex"]["f_rest_14"],
                        data["vertex"]["f_rest_15"],
                        data["vertex"]["f_rest_16"],
                        data["vertex"]["f_rest_17"],
                        data["vertex"]["f_rest_18"],
                        data["vertex"]["f_rest_19"],
                        data["vertex"]["f_rest_20"],
                        data["vertex"]["f_rest_21"],
                        data["vertex"]["f_rest_22"],
                        data["vertex"]["f_rest_23"],
                        data["vertex"]["f_rest_24"],
                        data["vertex"]["f_rest_25"],
                        data["vertex"]["f_rest_26"],
                        data["vertex"]["f_rest_27"],
                        data["vertex"]["f_rest_28"],
                        data["vertex"]["f_rest_29"],
                        data["vertex"]["f_rest_30"],
                        data["vertex"]["f_rest_31"],
                        data["vertex"]["f_rest_32"],
                        data["vertex"]["f_rest_33"],
                        data["vertex"]["f_rest_34"],
                        data["vertex"]["f_rest_35"],
                        data["vertex"]["f_rest_36"],
                        data["vertex"]["f_rest_37"],
                        data["vertex"]["f_rest_38"],
                        data["vertex"]["f_rest_39"],
                        data["vertex"]["f_rest_40"],
                        data["vertex"]["f_rest_41"],
                        data["vertex"]["f_rest_42"],
                        data["vertex"]["f_rest_43"],
                        data["vertex"]["f_rest_44"],
                    ),
                    dtype=np.float32,
                )
            ),
            device=self.device,
        )

        # reinterpret from C-style to F-style layout
        feats_sph = feats_sph.reshape(num_gaussians, 3, -1).transpose(-1, -2).reshape(num_gaussians, -1)

        self.features_specular = torch.nn.Parameter(feats_sph)

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    @torch.no_grad()
    def init_from_random_point_cloud(
        self,
        num_gaussians: int = 100_000,
        dtype=torch.float32,
        set_optimizable_parameters: bool = True,
        xyz_max=1.5,
        xyz_min=-1.5,
    ):
        logger.info(f"Generating random point cloud ({num_gaussians})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz in [-1.5, 1.5] -> standard NeRF convention, people often scale with 0.33 to get it to [-0.5, 0.5]
        fused_point_cloud = (
            torch.rand((num_gaussians, 3), dtype=dtype, device=self.device) * (xyz_max - xyz_min) + xyz_min
        )
        # sh albedo in [0, 0.0039]
        fused_color = torch.rand((num_gaussians, 3), dtype=dtype, device=self.device) / 255.0

        # Initialize features based on feature_type
        if self.feature_type == Features.Type.SH:
            features_albedo = fused_color.contiguous()
            max_sh_degree = self.max_n_features
            num_specular_features = sh_degree_to_specular_dim(max_sh_degree)
            features_specular = torch.zeros(
                (num_gaussians, num_specular_features), dtype=dtype, device=self.device
            ).contiguous()
        elif self.feature_type == Features.Type.NHT:
            init_min = float(getattr(self.conf.model.nht_features, "init_min", -5.0))
            init_max = float(getattr(self.conf.model.nht_features, "init_max", 5.0))
            features = (
                torch.rand((num_gaussians, self.particle_feature_dim), dtype=dtype, device=self.device)
                * (init_max - init_min) + init_min
            )

        dist = torch.clamp_min(nearest_neighbor_dist_cpuKD(fused_point_cloud), 1e-3)
        scales = torch.log(dist * self.conf.model.default_scale_factor)[..., None].repeat(1, 3)

        rots = torch.rand((num_gaussians, 4), device=self.device)
        rots[:, 0] = 1

        opacities = self.density_activation_inv(
            self.conf.model.default_density * torch.ones((num_gaussians, 1), dtype=dtype, device=self.device)
        )

        self.positions = torch.nn.Parameter(fused_point_cloud)  # type: ignore
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))

        if self.feature_type == Features.Type.SH:
            self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
            self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))
        elif self.feature_type == Features.Type.NHT:
            self.features = torch.nn.Parameter(features.to(dtype=dtype, device=self.device))

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer=True):
        # Backward compatibility: detect legacy checkpoints without feature_type
        if "feature_type" not in checkpoint and "features_albedo" in checkpoint:
            logger.info("Loading legacy checkpoint - auto-detecting feature_type='sh'")
            checkpoint["feature_type"] = "sh"
            checkpoint["particle_feature_dim"] = checkpoint["features_albedo"].shape[1] + checkpoint["features_specular"].shape[1]
            checkpoint["ray_feature_dim"] = 3

        # Load features based on feature_type (convert string to enum)
        checkpoint_feature_type_str = checkpoint.get("feature_type", "sh")
        checkpoint_feature_type = Features.Type.from_string(checkpoint_feature_type_str)

        # NHT: 3DGUT is compiled with PARTICLE_FEATURE_DIM / RAY_FEATURE_DIM from current config.
        # Checkpoints must match those compile-time constants or CUDA will read past feature buffers.
        if checkpoint_feature_type == Features.Type.NHT:
            if "features" not in checkpoint:
                raise ValueError("NHT checkpoint missing 'features' tensor")
            feat = checkpoint["features"]
            ck_pf = int(feat.shape[1])
            ck_rf = checkpoint.get("ray_feature_dim")
            if ck_pf != self.particle_feature_dim:
                raise ValueError(
                    f"NHT checkpoint features width is {ck_pf} but this build expects "
                    f"particle_feature_dim={self.particle_feature_dim} from config "
                    f"(model.nht_features.dim / interpolation). The 3DGUT CUDA extension was compiled for "
                    f"the config value; use the same nht_features (and render.primitive_type) as the run "
                    f"that produced the checkpoint, or train from scratch."
                )
            if ck_rf is not None and int(ck_rf) != self.ray_feature_dim:
                raise ValueError(
                    f"NHT checkpoint ray_feature_dim={ck_rf} does not match config ray_feature_dim="
                    f"{self.ray_feature_dim}. Align model.nht_features.activation with the checkpoint run."
                )
            if "particle_feature_dim" in checkpoint and int(checkpoint["particle_feature_dim"]) != ck_pf:
                logger.warning(
                    f"Checkpoint particle_feature_dim={checkpoint['particle_feature_dim']} disagrees with "
                    f"features.shape[1]={ck_pf}; using tensor shape."
                )

        # Load basic parameters
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.scene_extent = checkpoint["scene_extent"]

        # Load feature dimensions. For NHT, keep config-derived dims (validated above vs checkpoint tensors);
        # stale metadata keys must not override after a successful shape check.
        if checkpoint_feature_type != Features.Type.NHT:
            if "particle_feature_dim" in checkpoint:
                self.particle_feature_dim = checkpoint["particle_feature_dim"]
            if "ray_feature_dim" in checkpoint:
                self.ray_feature_dim = checkpoint["ray_feature_dim"]

        if checkpoint_feature_type == Features.Type.SH:
            self.features_albedo = checkpoint["features_albedo"]
            self.features_specular = checkpoint["features_specular"]
        elif checkpoint_feature_type == Features.Type.NHT:
            self.features = checkpoint["features"]
            self.nht_num_interpolation_points = Features(self.conf).num_interpolation_points
        else:
            raise ValueError(f"Unknown feature_type in checkpoint: {checkpoint_feature_type}")

        if self.progressive_training:
            self.feature_dim_increase_interval = checkpoint["feature_dim_increase_interval"]
            self.feature_dim_increase_step = checkpoint["feature_dim_increase_step"]

        self.background.load_state_dict(checkpoint["background"])
        if setup_optimizer:
            self.set_optimizable_parameters()
            self.setup_optimizer(state_dict=checkpoint["optimizer"])
        self.validate_fields()

    def init_from_lidar(self, point_cloud, observer_pts):
        """
        Initialize from lidar point cloud.
        Observer points can be any set locations that observation came from.
        Camera centers, ray source points, etc. They are used to estimate initial scales.
        """
        logger.info(f"Initializing based on lidar point cloud ...")

        self.default_initialize_from_points(
            point_cloud.xyz_end.to(device=self.device),
            observer_pts,
            point_cloud.color,
            use_observer_pts=self.conf.initialization.use_observation_points,
        )

    def default_initialize_from_points(self, pts, observer_pts, colors=None, use_observer_pts=True):
        """
        Given an Nx3 array of points (and optionally Nx3 rgb colors),
        initialize default values for the other parameters of the model
        """

        dtype = torch.float32

        # Local generator for deterministic random initialization (does not affect global RNG)
        rng = torch.Generator(device=self.device).manual_seed(self.conf.seed_initialization)

        N = pts.shape[0]
        positions = pts

        # Random rotations
        rots = torch.rand((N, 4), dtype=dtype, device=self.device, generator=rng)

        if use_observer_pts:
            # NOTE: it seems we get different scales compared to the original 3DGS implementation
            # estimate scales based on distances to observers
            dist_to_observers = torch.clamp_min(nearest_neighbor_dist_cpuKD(pts, observer_pts), 1e-7)
            observation_scale = dist_to_observers * self.conf.initialization.observation_scale_factor
        else:
            # Initialize the GS size to be the average dist of the 3 nearest neighbors
            dist2_avg = (k_nearest_neighbors(pts, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
            observation_scale = torch.sqrt(dist2_avg)

        observation_scale = observation_scale * self.conf.model.default_scale_factor

        scales = self.scale_activation_inv(observation_scale)[:, None].repeat(1, 3)

        # set density as a constant
        opacities = self.density_activation_inv(
            torch.full(
                (N, 1),
                fill_value=self.conf.model.default_density,
                dtype=dtype,
                device=self.device,
            )
        )

        # set colors, random if they weren't given
        if colors is None:
            colors = torch.randint(0, 256, (N, 3), dtype=torch.uint8, device=self.device, generator=rng)

        # Initialize features based on feature_type
        if self.feature_type == Features.Type.SH:
            features_albedo = to_torch(RGB2SH(to_np(colors.float() / 255.0)), device=self.device)
            num_specular_dims = sh_degree_to_specular_dim(self.max_n_features)
            features_specular = torch.zeros((N, num_specular_dims))
        elif self.feature_type == Features.Type.NHT:
            init_min = float(getattr(self.conf.model.nht_features, "init_min", -5.0))
            init_max = float(getattr(self.conf.model.nht_features, "init_max", 5.0))
            features = (
                torch.rand((N, self.particle_feature_dim), dtype=dtype, device=self.device, generator=rng)
                * (init_max - init_min) + init_min
            )

        self.positions = torch.nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))

        if self.feature_type == Features.Type.SH:
            self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
            self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))
        elif self.feature_type == Features.Type.NHT:
            self.features = torch.nn.Parameter(features.to(dtype=dtype, device=self.device))

        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()

    def setup_optimizer(self, state_dict=None):
        params = []
        for name, args in self.conf.optimizer.params.items():
            # Skip parameters that don't exist (e.g., 'features' in SH mode or 'features_albedo' in learned mode)
            if not hasattr(self, name):
                logger.info(f"Skipping optimizer parameter '{name}' - not present in {self.feature_type.name.lower()} mode")
                continue

            module = getattr(self, name)

            # If the module is a torch.nn.Module, we can add all of its trainable parameters to the optimizer
            if isinstance(module, torch.nn.Module):
                module_parameters = filter(lambda p: p.requires_grad and len(p) > 0, module.parameters())
                n_params = sum([np.prod(p.size(), dtype=int) for p in module_parameters])

                if n_params > 0:
                    params.append({"params": module.parameters(), "name": name, **args})

            # If the module is a torch.nn.Parameter, we can add it to the optimizer
            elif isinstance(module, torch.nn.Parameter):
                if module.requires_grad:
                    params.append({"params": [module], "name": name, **args})

        if self.conf.optimizer.type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps)
            logger.info("🔆 Using Adam optimizer")
        elif self.conf.optimizer.type == "selective_adam":
            self.optimizer = SelectiveAdam(params, lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps)
            logger.info("🔆 Using Selective Adam optimizer")
        else:
            raise ValueError(f"Unknown optimizer type: {self.conf.optimizer.type}")

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "positions":
                param_group["lr"] *= self.scene_extent  # Multiply the position lr by the scene scale

        self.setup_scheduler()

        # When loading from the checkpoint also load the state dict
        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)

    def setup_scheduler(self):
        self.schedulers = {}
        for name, args in self.conf.scheduler.items():
            if not hasattr(self, name):
                continue
            attr = getattr(self, name)
            if not (hasattr(attr, "requires_grad") and attr.requires_grad):
                continue
            if args.type is None:
                continue
            if name == "positions":
                self.schedulers[name] = get_scheduler(args.type)(
                    lr_init=args.lr_init * self.scene_extent,
                    lr_final=args.lr_final * self.scene_extent,
                    max_steps=args.max_steps,
                )
            elif name == "features":
                lr_init = getattr(self.conf.optimizer.params.features, "lr", 0.07)
                decay_final = getattr(args, "decay_final", 0.001)
                lr_final = lr_init * decay_final
                self.schedulers[name] = get_scheduler(args.type)(
                    lr_init=lr_init, lr_final=lr_final, max_steps=args.max_steps
                )
            else:
                self.schedulers[name] = get_scheduler(args.type)(**args)

    def scheduler_step(self, step):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.schedulers:
                lr = self.schedulers[param_group["name"]](step)
                if lr is not None:
                    param_group["lr"] = lr

    def set_optimizable_parameters(self):
        if not self.conf.model.optimize_density:
            self.density.requires_grad = False
        if not self.conf.model.optimize_rotation:
            self.rotation.requires_grad = False
        if not self.conf.model.optimize_scale:
            self.scale.requires_grad = False
        if not self.conf.model.optimize_position:
            self.positions.requires_grad = False

        # Handle feature optimization based on feature_type
        if self.feature_type == Features.Type.SH:
            if not self.conf.model.optimize_features_albedo:
                self.features_albedo.requires_grad = False
            if not self.conf.model.optimize_features_specular:
                self.features_specular.requires_grad = False
        elif self.feature_type == Features.Type.NHT:
            # For learned features, check if optimize_features config exists
            if not self.conf.model.optimize_features:
                self.features.requires_grad = False

    def update_optimizable_parameters(self, optimizable_tensors: dict[str, torch.Tensor]):
        for name, value in optimizable_tensors.items():
            setattr(self, name, value)

    def increase_num_active_features(self) -> None:
        self.n_active_features = min(self.max_n_features, self.n_active_features + self.feature_dim_increase_step)

    def get_active_feature_mask(self) -> torch.Tensor:
        if self.feature_type == Features.Type.SH:
            current_sh_degree = self.n_active_features
            max_sh_degree = self.max_n_features
            active_features = sh_degree_to_num_features(current_sh_degree)
            num_features = sh_degree_to_num_features(max_sh_degree)
        else:
            active_features = self.n_active_features
            num_features = self.max_n_features
        mask = torch.zeros((1, num_features), device=self.device, dtype=self.get_features().dtype)
        mask[0, :active_features] = 1.0
        return mask

    def clamp_density(self):
        updated_densities = torch.clamp(self.get_density(), min=1e-4, max=1.0 - 1e-4)
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.density = optimizable_tensors["density"]

    def forward(
        self, batch: Batch, train=False, frame_id=0
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: a Batch structure containing the input data
            train: a boolean indicating whether the model is in training mode
            frame_id: an integer indicating the frame id (default is 0)
        Returns:
            A dictionary containing the output of the model
        """
        return self.renderer.render(self, batch, train, frame_id)

    def trace(self, rays_o, rays_d, T_to_world=None):
        """Traces the model with the given rays. This method is a convenience method for ray-traced inference mode.
        If T_to_world is None, the rays are assumed to be in world space.
        Otherwise, the rays are assumed to be in camera space.
        rays_ori: torch.Tensor  # [B, H, W, 3] ray origins in arbitrary space
        rays_dir: torch.Tensor  # [B, H, W, 3] ray directions in arbitrary space
        T_to_world: torch.Tensor  # [B, 4, 4] transformation matrix from the ray space to the world space
        """
        if T_to_world is None:
            T_to_world = torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None]
        inputs = Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d)
        return self.renderer.render(self, inputs)

    @torch.no_grad()
    def export_ply(self, mogt_path: str):
        exporter = PLYExporter()
        exporter.export(self, Path(mogt_path))

    @torch.no_grad()
    def init_from_ply(self, mogt_path: str, init_model=True):
        if self.feature_type != Features.Type.SH:
            raise NotImplementedError(f"init_from_ply only supports feature_type='sh', got '{self.feature_type.name.lower()}'")
        plydata = PlyData.read(mogt_path)

        mogt_pos = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        mogt_densities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        num_gaussians = mogt_pos.shape[0]
        mogt_albedo = np.zeros((num_gaussians, 3))
        mogt_albedo[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        mogt_albedo[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        mogt_albedo[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        num_speculars = (self.max_n_features + 1) ** 2 - 1
        expected_extra_f_count = 3 * num_speculars

        mogt_specular = np.zeros((num_gaussians, expected_extra_f_count))
        if len(extra_f_names) == expected_extra_f_count:
            # Full spherical harmonics data available
            for idx, attr_name in enumerate(extra_f_names):
                mogt_specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
            mogt_specular = mogt_specular.reshape((num_gaussians, 3, num_speculars))
            mogt_specular = mogt_specular.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))
        elif len(extra_f_names) == 0:
            # Only DC components available, create zero-filled higher-order harmonics
            logger.info(f"PLY file only contains DC components, initializing higher-order spherical harmonics to zero")
        else:
            # Partial data - this is unexpected
            raise ValueError(
                f"Unexpected number of f_rest_ properties: found {len(extra_f_names)}, expected {expected_extra_f_count} or 0"
            )

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        mogt_scales = np.zeros((num_gaussians, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            mogt_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        mogt_rotation = np.zeros((num_gaussians, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            mogt_rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.positions = torch.nn.Parameter(torch.tensor(mogt_pos, dtype=self.positions.dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(
            torch.tensor(mogt_albedo, dtype=self.features_albedo.dtype, device=self.device)
        )
        self.features_specular = torch.nn.Parameter(
            torch.tensor(mogt_specular, dtype=self.features_specular.dtype, device=self.device)
        )
        self.density = torch.nn.Parameter(torch.tensor(mogt_densities, dtype=self.density.dtype, device=self.device))
        self.scale = torch.nn.Parameter(torch.tensor(mogt_scales, dtype=self.scale.dtype, device=self.device))
        self.rotation = torch.nn.Parameter(torch.tensor(mogt_rotation, dtype=self.rotation.dtype, device=self.device))

        self.n_active_features = self.max_n_features

        if init_model:
            self.set_optimizable_parameters()
            self.setup_optimizer()
            self.validate_fields()

    def copy_fields(self, other, deepcopy=False):
        """Copies fields from other onto self"""
        if self.optimizer is not None:
            raise NotImplementedError(
                "Operations that create copies of the model during training " "are currently not supported."
            )

        if deepcopy:
            self.positions = torch.nn.Parameter(other.positions.clone())
            self.rotation = torch.nn.Parameter(other.rotation.clone())
            self.scale = torch.nn.Parameter(other.scale.clone())
            self.density = torch.nn.Parameter(other.density.clone())
            if other.feature_type == Features.Type.SH:
                self.features_albedo = torch.nn.Parameter(other.features_albedo.clone())
                self.features_specular = torch.nn.Parameter(other.features_specular.clone())
            elif other.feature_type == Features.Type.NHT:
                self.features = torch.nn.Parameter(other.features.clone())
        else:
            self.positions = torch.nn.Parameter(other.positions)
            self.rotation = torch.nn.Parameter(other.rotation)
            self.scale = torch.nn.Parameter(other.scale)
            self.density = torch.nn.Parameter(other.density)
            if other.feature_type == Features.Type.SH:
                self.features_albedo = torch.nn.Parameter(other.features_albedo)
                self.features_specular = torch.nn.Parameter(other.features_specular)
            elif other.feature_type == Features.Type.NHT:
                self.features = torch.nn.Parameter(other.features)
        self.max_sh_degree = other.max_sh_degree
        self.n_active_features = other.n_active_features
        self.scene_extent = other.scene_extent
        self.progressive_training = other.progressive_training
        self.feature_dim_increase_interval = other.feature_dim_increase_interval
        self.feature_dim_increase_step = other.feature_dim_increase_step
        self.background = other.background
        self.feature_type = other.feature_type
        self.particle_feature_dim = other.particle_feature_dim
        self.ray_feature_dim = other.ray_feature_dim
        if hasattr(other, "nht_num_interpolation_points"):
            self.nht_num_interpolation_points = other.nht_num_interpolation_points
        self.validate_fields()

    def clone(self):
        other = MixtureOfGaussians(conf=self.conf, scene_extent=self.scene_extent)
        other.copy_fields(self, deepcopy=True)
        return other

    def __getitem__(self, idx):
        sliced = MixtureOfGaussians(conf=self.conf, scene_extent=self.scene_extent)
        sliced.copy_fields(self, deepcopy=False)
        sliced.positions = torch.nn.Parameter(sliced.positions[idx])
        sliced.rotation = torch.nn.Parameter(sliced.rotation[idx])
        sliced.scale = torch.nn.Parameter(sliced.scale[idx])
        sliced.density = torch.nn.Parameter(sliced.density[idx])
        if self.feature_type == Features.Type.SH:
            sliced.features_albedo = torch.nn.Parameter(sliced.features_albedo[idx])
            sliced.features_specular = torch.nn.Parameter(sliced.features_specular[idx])
        elif self.feature_type == Features.Type.NHT:
            sliced.features = torch.nn.Parameter(sliced.features[idx])
        return sliced

    def __len__(self):
        return self.positions.shape[0] if self.positions is not None else 0
