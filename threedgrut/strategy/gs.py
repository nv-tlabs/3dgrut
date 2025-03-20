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

from typing import Optional

import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import get_activation_function, quaternion_to_so3


class GSStrategy(BaseStrategy):
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        # Parameters related to densification, pruning and reset
        self.split_n_gaussians = self.conf.model.densify.split.n_gaussians
        self.relative_size_threshold = self.conf.model.densify.relative_size_threshold
        self.prune_density_threshold = self.conf.model.prune.density_threshold
        self.clone_grad_threshold = self.conf.model.densify.clone_grad_threshold
        self.split_grad_threshold = self.conf.model.densify.split_grad_threshold
        self.new_max_density = self.conf.model.reset_density.new_max_density

        # Accumulation of the norms of the positions gradients
        self.densify_grad_norm_accum = torch.empty([0, 1])
        self.densify_grad_norm_denom = torch.empty([0, 1])

    def get_strategy_parameters(self) -> dict:
        params = {}

        params["densify_grad_norm_accum"] = (self.densify_grad_norm_accum,)
        params["densify_grad_norm_denom"] = (self.densify_grad_norm_denom,)

        return params

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        if checkpoint is not None:
            self.densify_grad_norm_accum = checkpoint["densify_grad_norm_accum"][0].detach()
            self.densify_grad_norm_denom = checkpoint["densify_grad_norm_denom"][0].detach()
        else:
            num_gaussians = self.model.num_gaussians
            self.densify_grad_norm_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_grad_norm_denom = torch.zeros((num_gaussians, 1), dtype=torch.int, device=self.model.device)

    def pre_backward(
        self,
        step: int,
    ) -> None:
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def post_backward(self, step: int, scene_extent: float, train_dataset) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""
        scene_updated = False
        # Densify the Gaussians
        if (
            step > self.conf.model.densify.start_iteration
            and step < self.conf.model.densify.end_iteration
            and step % self.conf.model.densify.frequency == 0
        ):
            self.densify_gaussians(scene_extent=scene_extent)
            scene_updated = True

        # Prune the Gaussians based on their opacity
        if (
            step > self.conf.model.prune.start_iteration
            and step < self.conf.model.prune.end_iteration
            and step % self.conf.model.prune.frequency == 0
        ):
            self.prune_gaussians_opacity()
            scene_updated = True

        # Prune the Gaussians based on their scales
        if (
            step > self.conf.model.prune_scale.start_iteration
            and step < self.conf.model.prune_scale.end_iteration
            and step % self.conf.model.prune_scale.frequency == 0
        ):
            self.prune_gaussians_scale(train_dataset)
            scene_updated = True

        # Decay the density values
        if (
            step > self.conf.model.density_decay.start_iteration
            and step < self.conf.model.density_decay.end_iteration
            and step % self.conf.model.density_decay.frequency == 0
        ):
            self.decay_density()

        # Reset the Gaussian density
        if (
            step > self.conf.model.reset_density.start_iteration
            and step < self.conf.model.reset_density.end_iteration
            and step % self.conf.model.reset_density.frequency == 0
        ):
            self.reset_density()

        # SH: Every N its we increase the levels of SH up to a maximum degree
        if self.model.progressive_training and step > 0 and step % self.model.feature_dim_increase_interval == 0:
            self.model.increase_num_active_features()

        return scene_updated

    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(self, sensor_position):
        params_grad = self.model.positions.grad
        mask = (params_grad != 0).max(dim=1)[0]
        assert params_grad is not None
        distance_to_camera = (self.model.positions[mask] - sensor_position).norm(dim=1, keepdim=True)

        self.densify_grad_norm_accum[mask] += (
            torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2
        )
        self.densify_grad_norm_denom[mask] += 1

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, scene_extent):
        self.densify_params_grad(scene_extent)

    def densify_params_grad(self, scene_extent):
        assert self.model.optimizer is not None, "Optimizer need to be initialized before splitting and cloning the Gaussians"
        densify_grad_norm = self.densify_grad_norm_accum / self.densify_grad_norm_denom
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0

        self.clone_gaussians(densify_grad_norm.squeeze(), scene_extent)
        self.split_gaussians(densify_grad_norm.squeeze(), scene_extent)

        torch.cuda.empty_cache()

    @torch.cuda.nvtx.range("densify_postfix")
    def densify_postfix(self, add_gaussians):
        # Concatenate new tensors to the optimizer variables
        optimizable_tensors = self.concatenate_optimizer_tensors(add_gaussians)
        self.model.update_optimizable_parameters(optimizable_tensors)

        self.densify_grad_norm_accum = torch.zeros(
            (self.model.num_gaussians, 1), device=self.model.device, dtype=self.densify_grad_norm_accum.dtype
        )
        self.densify_grad_norm_denom = torch.zeros(
            (self.model.num_gaussians, 1), device=self.model.device, dtype=self.densify_grad_norm_denom.dtype
        )

    @torch.cuda.nvtx.range("split_gaussians")
    def split_gaussians(self, densify_grad_norm: torch.Tensor, scene_extent: float):
        n_init_points = self.model.num_gaussians

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the self.model.positions so only take the points up to size of the initial grad
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze()
        mask = torch.where(padded_grad >= self.split_grad_threshold, True, False)
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values > self.relative_size_threshold * scene_extent
        )

        stds = self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(self.split_n_gaussians, 1, 1)

        if self.conf.model.densify.share_density:
            self.model.set_density(mask, self.model.density[mask] / self.split_n_gaussians)

        add_gaussians = {
            "positions": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.model.positions[mask].repeat(self.split_n_gaussians, 1),
            "density": self.model.density[mask].repeat(self.split_n_gaussians, 1),
            "scale": get_activation_function(self.conf.model.scale_activation, inverse=True)(
                self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1) / (0.8 * self.split_n_gaussians)
            ),
            "rotation": self.model.rotation[mask].repeat(self.split_n_gaussians, 1),
        }
        for field_name in self.model.feature_fields():
            add_gaussians[field_name] = getattr(self.model, field_name)[mask].repeat(self.split_n_gaussians, 1)

        self.densify_postfix(add_gaussians)

        # stats
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Splitted {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        # Prune away the Gaussians that were originally slected
        valid = ~torch.cat((mask, torch.zeros(self.split_n_gaussians * mask.sum(), device="cuda", dtype=bool)))
        self.prune_gaussians(valid)

    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(self, densify_grad_norm: torch.Tensor, scene_extent: float):
        assert densify_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"
        # Extract points that satisfy the gradient condition
        mask = torch.where(densify_grad_norm >= self.clone_grad_threshold, True, False)

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values <= self.relative_size_threshold * scene_extent
        )

        if self.conf.model.densify.share_density:
            self.model.set_density(mask, self.model.density[mask] / 2)

        # stats
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        # Use the mask to dupicate these points
        add_gaussians = {
            "positions": self.model.positions[mask],
            "density": self.model.density[mask],
            "scale": self.model.scale[mask],
            "rotation": self.model.rotation[mask],
        }
        for field_name in self.model.feature_fields():
            add_gaussians[field_name] = getattr(self.model, field_name)[mask]

        self.densify_postfix(add_gaussians)

    def prune_gaussians_weight(self):
        # Prune the Gaussians based on their weight
        mask = self.model.rolling_weight_contrib[:, 0] >= self.conf.model.prune_weight.weight_threshold
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Weight-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.prune_gaussians(mask)

    def prune_gaussians_scale(self, dataset):
        cam_normals = torch.from_numpy(dataset.poses[:, :3, 2]).to(self.model.device)
        similarities = torch.matmul(self.model.positions, cam_normals.T)
        cam_dists = similarities.min(dim=1)[0].clamp(min=1e-8)
        ratio = self.model.get_scale().min(dim=1)[0] / cam_dists * dataset.intrinsic[0].max()

        # Prune the Gaussians based on their weight
        mask = ratio >= self.conf.model.prune_scale.threshold
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Scale-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.prune_gaussians(mask)

    def prune_gaussians_opacity(self):
        # Prune the Gaussians based on their opacity
        mask = self.model.get_density().squeeze() >= self.prune_density_threshold

        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Density-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.prune_gaussians(mask)

    @torch.cuda.nvtx.range("prune_gaussians")
    def prune_gaussians(self, valid_mask):
        # TODO: consider having a buffer of the contribution of Gaussians to the rendering -> this might avoid the need to reset opacity
        # TODO: we could also consider pruning away some of the large Gaussians?
        optimizable_tensors = self.prune_optimizer_tensors(valid_mask)
        self.model.update_optimizable_parameters(optimizable_tensors)

        self.densify_grad_norm_accum = self.densify_grad_norm_accum[valid_mask]
        self.densify_grad_norm_denom = self.densify_grad_norm_denom[valid_mask]

        torch.cuda.empty_cache()

    def replace_tensor_to_optimizer(self, tensor, name: str):
        assert self.model.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"
        optimizable_tensors = {}

        for group in self.model.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.model.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.model.optimizer.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                self.model.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_optimizer_tensors(self, mask):
        assert self.model.optimizer is not None, "Optimizer need to be initialized before concatenating the values"
        optimizable_tensors = {}
        for group in self.model.optimizer.param_groups:
            if group["name"] != "background":
                stored_state = self.model.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.model.optimizer.state[group["params"][0]]
                    group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.model.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def concatenate_optimizer_tensors(self, tensors_dict):
        assert self.model.optimizer is not None, "Optimizer need to be initialized before concatenating the values"

        optimizable_tensors = {}
        for group in self.model.optimizer.param_groups:
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.model.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                    )

                    del self.model.optimizer.state[group["params"][0]]
                    group["params"][0] = torch.nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(
                            group["params"][0].requires_grad
                        )
                    )
                    self.model.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = torch.nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(
                            group["params"][0].requires_grad
                        )
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def decay_density(self):
        decayed_densities = self.model.density_activation_inv(self.model.get_density() * self.conf.model.density_decay.gamma)
        optimizable_tensors = self.replace_tensor_to_optimizer(decayed_densities, "density")
        self.model.density = optimizable_tensors["density"]

    def reset_density(self):
        updated_densities = self.model.density_activation_inv(
            torch.min(self.model.get_density(), torch.ones_like(self.model.density) * self.new_max_density)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.model.density = optimizable_tensors["density"]