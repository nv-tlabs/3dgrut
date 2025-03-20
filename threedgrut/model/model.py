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
import os
from typing import Any

import msgpack
import numpy as np
import torch
from plyfile import PlyData, PlyElement

import threedgrut.model.background as background
from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.utils import PointCloud, read_next_bytes, read_colmap_points3D_text
from threedgrut.model.geometry import nearest_neighbor_dist_cpuKD
import threedgrt_tracer, threedgut_tracer
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    get_activation_function, 
    get_scheduler,
    sh_degree_to_num_features,
    sh_degree_to_specular_dim,
    to_np, to_torch,
)
from threedgrut.utils.render import RGB2SH


class MixtureOfGaussians(torch.nn.Module):
    """ """

    @property
    def num_gaussians(self):
        return self.positions.shape[0]

    def feature_fields(self) -> list[str]:
        """Returns a list of feature field names - subclasses can override"""
        return [
            "features_albedo",
            "features_specular",
        ]

    def get_features(self):
        return torch.cat((self.features_albedo, self.features_specular), dim=1)

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
        }

        if self.progressive_training:
            model_params["feature_dim_increase_interval"] = self.feature_dim_increase_interval
            model_params["feature_dim_increase_step"] = self.feature_dim_increase_step

        if self.feature_type == "sh":
            model_params["features_albedo"] = self.features_albedo
            model_params["features_specular"] = self.features_specular

        return model_params

    def __init__(self, conf, scene_extent=None):
        super().__init__()

        sh_degree = conf.model.progressive_training.max_n_features
        specular_dim = sh_degree_to_specular_dim(sh_degree)
        self.positions = torch.nn.Parameter(
            torch.empty([0, 3])
        )  # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
        self.rotation = torch.nn.Parameter(
            torch.empty([0, 4])
        )  # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
        self.scale = torch.nn.Parameter(torch.empty([0, 3]))  # Anisotropic scale of each Gaussian [n_gaussians, 3]
        self.density = torch.nn.Parameter(torch.empty([0, 1]))  # Density of each Gaussian [n_gaussians, 1]
        self.features_albedo = torch.nn.Parameter(
            torch.empty([0, 3])
        )  # Feature vector of the 0th order SH coefficients [n_gaussians, 3] (We split it into two due to different learning rates)
        self.features_specular = torch.nn.Parameter(
            torch.empty([0, specular_dim])
        )  # Features of the higher order SH coefficients [n_gaussians, specular_dim]
        self.max_sh_degree = sh_degree

        self.conf = conf
        self.scene_extent = scene_extent
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
        self.feature_type = self.conf.model.progressive_training.feature_type
        self.n_active_features = self.conf.model.progressive_training.init_n_features
        self.max_n_features = self.conf.model.progressive_training.max_n_features  # For SH, this is the SH degree
        self.progressive_training = False
        if self.n_active_features < self.max_n_features:
            self.feature_dim_increase_interval = self.conf.model.progressive_training.increase_frequency
            self.feature_dim_increase_step = self.conf.model.progressive_training.increase_step
            self.progressive_training = True

        # Rendering method
        if conf.render.method == "3dgrt":
           self.renderer = threedgrt_tracer.Tracer(conf)
        elif conf.render.method == "3dgut":
            self.renderer = threedgut_tracer.Tracer(conf)
        else:
            raise ValueError(f"Unknown rendering method: {conf.render.method}")
    
    @torch.no_grad()
    def build_acc(self, rebuild=True):
        self.renderer.build_acc(self, rebuild)

    def validate_fields(self):
        num_gaussians = self.num_gaussians
        assert self.positions.shape == (num_gaussians, 3)
        assert self.density.shape == (num_gaussians, 1)
        assert self.rotation.shape == (num_gaussians, 4)
        assert self.scale.shape == (num_gaussians, 3)

        if self.feature_type == "sh":
            assert self.features_albedo.shape == (num_gaussians, 3)
            specular_sh_dims = sh_degree_to_specular_dim(self.max_n_features)
            assert self.features_specular.shape == (num_gaussians, specular_sh_dims)
        else:
            raise ValueError("Neural features not yet supported.")

    def init_from_colmap(self, root_path: str, observer_pts):
        # Special case for scannetpp dataset
        if self.conf.dataset.type == "scannetpp":
            points_file = os.path.join(root_path, "colmap", "points3D.txt")
            pts, rgb, _ = read_colmap_points3D_text(points_file)
            file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            file_rgb = torch.tensor(rgb, dtype=torch.float32, device=self.device)

        else:
            points_file = os.path.join(root_path, "sparse/0", "points3D.bin")
            if not os.path.isfile(points_file):
                raise ValueError(f"colmap points file {points_file} not found")

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

            file_rgb = file_rgb / 255.0

            file_pts = torch.tensor(file_pts, dtype=torch.float32, device=self.device)
            file_rgb = torch.tensor(file_rgb, dtype=torch.float32, device=self.device)

        self.default_initialize_from_points(file_pts, observer_pts, file_rgb)

    def init_from_pretrained_point_cloud(self, pc_path: str, set_optimizable_parameters: bool = True):
        data = PlyData.read(pc_path)
        num_gaussians = len(data["vertex"])
        self.positions = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack((data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]), dtype=np.float32)
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
                        (data["vertex"]["scale_0"], data["vertex"]["scale_1"], data["vertex"]["scale_2"]),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.density = torch.nn.Parameter(
            to_torch(data["vertex"]["opacity"].astype(np.float32).reshape(num_gaussians, 1), device=self.device)
        )
        self.features_albedo = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (data["vertex"]["f_dc_0"], data["vertex"]["f_dc_1"], data["vertex"]["f_dc_2"]), dtype=np.float32
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

        features_albedo = features_specular = None
        if self.feature_type == "sh":
            features_albedo = fused_color.contiguous()
            max_sh_degree = self.max_n_features
            num_specular_features = sh_degree_to_specular_dim(max_sh_degree)
            features_specular = torch.zeros(
                (num_gaussians, num_specular_features), dtype=dtype, device=self.device
            ).contiguous()

        dist = torch.clamp_min(nearest_neighbor_dist_cpuKD(fused_point_cloud), 1e-3)
        scales = torch.log(dist)[..., None].repeat(1, 3)
        rots = torch.rand((num_gaussians, 4), device=self.device)
        rots[:, 0] = 1

        opacities = self.density_activation_inv(
            self.conf.model.default_density * torch.ones((num_gaussians, 1), dtype=dtype, device=self.device)
        )

        self.positions = torch.nn.Parameter(fused_point_cloud)  # type: ignore
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
        self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer=True):
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.features_albedo = checkpoint["features_albedo"]
        self.features_specular = checkpoint["features_specular"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.scene_extent = checkpoint["scene_extent"]

        if self.progressive_training:
            self.feature_dim_increase_interval = checkpoint["feature_dim_increase_interval"]
            self.feature_dim_increase_step = checkpoint["feature_dim_increase_step"]

        self.background.load_state_dict(checkpoint["background"])
        if setup_optimizer:
            self.set_optimizable_parameters()
            self.setup_optimizer(state_dict=checkpoint["optimizer"])
        self.validate_fields()

    def init_from_lidar(self, point_cloud: PointCloud, observer_pts):
        """
        Observer points can be any set locations that observation came from. Camera centers, ray source points, etc. They are used to esimate initial scales.
        """

        logger.info(f"Initializing based on lidar point cloud ...")

        # only initialize by default from points for now
        self.default_initialize_from_points(point_cloud.xyz_end.to(device=self.device), observer_pts, point_cloud.color)

    def default_initialize_from_points(self, pts, observer_pts, colors=None):
        """
        Given an Nx3 array of points (and optionally Nx3 rgb colors),
        initialize default values for the other parameters of the model
        """

        dtype = torch.float32

        N = pts.shape[0]
        positions = pts

        # identity rotations
        rots = torch.zeros((N, 4), dtype=dtype, device=self.device)
        rots[:, 0] = 1.0  # they're quaternions

        # estimate scales based on distances to observers
        dist_to_observers = torch.clamp_min(nearest_neighbor_dist_cpuKD(pts, observer_pts), 1e-7)
        observation_scale = dist_to_observers * self.conf.initialization.observation_scale_factor
        scales = self.scale_activation_inv(observation_scale)[:, None].repeat(1, 3)

        # set density as a constant
        opacities = self.density_activation_inv(
            torch.full((N, 1), fill_value=self.conf.model.default_density, dtype=dtype, device=self.device)
        )

        # set colors, constant if they weren't given
        if colors is None:
            features_albedo = torch.rand((N, 3), dtype=dtype, device=self.device) / 255.0
        else:
            features_albedo = to_torch(RGB2SH(to_np(colors.float() / 255.0)), device=self.device)

        num_specular_dims = sh_degree_to_specular_dim(self.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims))

        self.positions = torch.nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
        self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()

    def setup_optimizer(self, state_dict=None):
        params = []
        for name, args in self.conf.optimizer.params.items():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                module_parameters = filter(lambda p: p.requires_grad and len(p) > 0, module.parameters())
                n_params = sum([np.prod(p.size(), dtype=int) for p in module_parameters])

                if n_params > 0:
                    params.append({"params": module.parameters(), "name": name, **args})

            elif isinstance(module, torch.nn.Parameter):
                if module.requires_grad:
                    params.append({"params": [module], "name": name, **args})

        self.optimizer = torch.optim.Adam(params, lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps)

        self.setup_scheduler()

        # When loading from the checkpoint also load the state dict
        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)

    def setup_scheduler(self):
        self.schedulers = {}
        for name, args in self.conf.scheduler.items():
            if args.type is not None and getattr(self, name).requires_grad:
                if name == "positions":
                    self.schedulers[name] = get_scheduler(args.type)(
                        lr_init=args.lr_init * self.scene_extent,
                        lr_final=args.lr_final * self.scene_extent,
                        max_steps=args.max_steps,
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
        if not self.conf.model.optimize_features_albedo:
            self.features_albedo.requires_grad = False
        if not self.conf.model.optimize_features_specular:
            self.features_specular.requires_grad = False
        if not self.conf.model.optimize_rotation:
            self.rotation.requires_grad = False
        if not self.conf.model.optimize_scale:
            self.scale.requires_grad = False
        if not self.conf.model.optimize_position:
            self.positions.requires_grad = False

    def update_optimizable_parameters(self, optimizable_tensors: dict[str, torch.Tensor]):
        for name, value in optimizable_tensors.items():
            setattr(self, name, value)

    def increase_num_active_features(self) -> None:
        self.n_active_features = min(self.max_n_features, self.n_active_features + self.feature_dim_increase_step)

    def get_active_feature_mask(self) -> torch.Tensor:
        if self.feature_type == "sh":
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

    def set_density(self, mask, density):
        updated_densities = self.density.clone()
        updated_densities[mask] = density
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.density = optimizable_tensors["density"]

    def clamp_density(self):
        updated_densities = torch.clamp(self.get_density(), min=1e-4, max=1.0 - 1e-4)
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.density = optimizable_tensors["density"]

    def forward(self, batch: Batch, train=False, frame_id=0) -> dict[str, torch.Tensor]:
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
        """ Traces the model with the given rays. This method is a convenience method for ray-traced inference mode.
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

    def export_ingp(self, mogt_path: str, force_half: bool):
        export_dtype = torch.float16 if force_half else self.positions.dtype
        logger.info(f"exporting mogt file to {mogt_path}...")
        mogt_config: dict[str, Any] = {}
        mogt_config["nre_data"] = {"version": "0.0.1", "model": "mogt"}
        mogt_config["precision"] = "half" if export_dtype == torch.float16 else "single"
        mogt_config["mog_num"] = self.num_gaussians
        mogt_config["mog_sph_degree"] = self.max_n_features
        mogt_config["mog_positions"] = (
            self.positions.flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_scales"] = (
            self.get_scale().flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_rotations"] = (
            self.get_rotation().flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_densities"] = (
            self.get_density().flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_features"] = (
            self.get_features().flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        with gzip.open(ingp_filepath := mogt_path, "wb") as f:
            packed = msgpack.packb(mogt_config)
            f.write(packed)

    @torch.no_grad()
    def init_from_ingp(self, ingp_path, init_model=True):
        with gzip.open(ingp_path, "rb") as f:
            mogt_config = msgpack.unpackb(f.read())
        mog_num = mogt_config["mog_num"]
        self.n_active_features = self.max_n_features = mogt_config["mog_sph_degree"]
        import_dtype = np.float16 if mogt_config["precision"] == "half" else np.float32
        positions = (
            torch.from_numpy(np.frombuffer(mogt_config["mog_positions"], dtype=import_dtype))
            .to(device=self.device)
            .reshape(mog_num, 3)
        )
        scales = (
            torch.from_numpy(np.frombuffer(mogt_config["mog_scales"], dtype=import_dtype))
            .to(device=self.device)
            .reshape(mog_num, 3)
        )
        densities = (
            torch.from_numpy(np.frombuffer(mogt_config["mog_densities"], dtype=import_dtype))
            .to(device=self.device)
            .reshape(mog_num, 1)
        )
        rotations = (
            torch.from_numpy(np.frombuffer(mogt_config["mog_rotations"], dtype=import_dtype))
            .to(device=self.device)
            .reshape(mog_num, 4)
        )
        n_features = sh_degree_to_specular_dim(self.max_n_features)
        features = (
            torch.from_numpy(np.frombuffer(mogt_config["mog_features"], dtype=import_dtype))
            .to(device=self.device)
            .reshape(mog_num, n_features + 3)
        )
        features_albedo, features_specular = torch.split(features, [3, n_features], dim=1)

        self.positions = torch.nn.Parameter(positions)
        self.rotation = torch.nn.Parameter(rotations)
        self.scale = torch.nn.Parameter(self.scale_activation_inv(scales))
        self.density = torch.nn.Parameter(self.density_activation_inv(densities))
        self.features_albedo = torch.nn.Parameter(features_albedo)
        self.features_specular = torch.nn.Parameter(features_specular)

        self.n_active_features = self.max_n_features

        if init_model:
            self.set_optimizable_parameters()
            self.setup_optimizer()
            self.validate_fields()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_albedo.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_specular.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    @torch.no_grad()
    def export_ply(self, mogt_path:str):
        logger.info(f"exporting ply file to {mogt_path}...")
        mogt_pos =  self.positions.detach().cpu().numpy()
        num_gaussians = mogt_pos.shape[0]
        mogt_nrm = np.repeat(np.array([[0,0,1]], dtype=np.float32),repeats=num_gaussians, axis=0)
        mogt_albedo = self.features_albedo.detach().cpu().numpy()
        num_speculars = (self.max_n_features + 1) ** 2 - 1
        mogt_specular = self.features_specular.detach().cpu().numpy().reshape((num_gaussians,num_speculars,3))
        mogt_specular = mogt_specular.transpose(0, 2, 1).reshape((num_gaussians,num_speculars*3))
        mogt_densities = self.density.detach().cpu().numpy()
        mogt_scales = self.scale.detach().cpu().numpy()
        mogt_rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(num_gaussians, dtype=dtype_full)
        attributes = np.concatenate((mogt_pos, mogt_nrm, mogt_albedo, mogt_specular, mogt_densities, mogt_scales, mogt_rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(mogt_path)

    @torch.no_grad()
    def init_from_ply(self, mogt_path:str, init_model=True):
        plydata = PlyData.read(mogt_path)

        mogt_pos = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        mogt_densities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        num_gaussians = mogt_pos.shape[0]
        mogt_albedo = np.zeros((num_gaussians, 3))
        mogt_albedo[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        mogt_albedo[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        mogt_albedo[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        num_speculars = (self.max_n_features + 1) ** 2 - 1
        assert len(extra_f_names)==3*num_speculars
        mogt_specular = np.zeros((num_gaussians, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            mogt_specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
        mogt_specular = mogt_specular.reshape((num_gaussians,3,num_speculars))
        mogt_specular = mogt_specular.transpose(0, 2, 1).reshape((num_gaussians,num_speculars*3))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        mogt_scales = np.zeros((num_gaussians, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            mogt_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        mogt_rotation = np.zeros((num_gaussians, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            mogt_rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.positions = torch.nn.Parameter(torch.tensor(mogt_pos, dtype=self.positions.dtype,device=self.device))
        self.features_albedo = torch.nn.Parameter(torch.tensor(mogt_albedo, dtype=self.features_albedo.dtype,device=self.device))
        self.features_specular = torch.nn.Parameter(torch.tensor(mogt_specular,dtype=self.features_specular.dtype,device=self.device))
        self.density = torch.nn.Parameter(torch.tensor(mogt_densities,dtype=self.density.dtype,device=self.device))
        self.scale = torch.nn.Parameter(torch.tensor(mogt_scales,dtype=self.scale.dtype,device=self.device))
        self.rotation = torch.nn.Parameter(torch.tensor(mogt_rotation,dtype=self.rotation.dtype,device=self.device))

        self.n_active_features = self.max_n_features
        
        if init_model:
            self.set_optimizable_parameters()
            self.setup_optimizer()
            self.validate_fields()