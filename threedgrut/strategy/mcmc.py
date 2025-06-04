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
#
# MCMC implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/mcmc.py), 
# which is based on the original implementation https://github.com/ubc-vision/3dgs-mcmc that uderlines the work
#
# 3D Gaussian Splatting as Markov Chain Monte Carlo by 
# Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi and Kwang Moo Yie
#
# If you use this code in your research, please cite the above works.

import math
from typing import Optional, Tuple

import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import _multinomial_sample, check_step_condition

_mcmc_plugin = None

def load_mcmc_plugin():
    global _mcmc_plugin
    if _mcmc_plugin is None:
        try:
            from . import lib_mcmc_cc as gaussian_mcmc
        except ImportError:
            from threedgrut.strategy.src.setup_mcmc import setup_mcmc
            setup_mcmc()
            import lib_mcmc_cc as gaussian_mcmc
        _mcmc_plugin = gaussian_mcmc

class MCMCStrategy(BaseStrategy):
    """Densification and prunning strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`

    MCMC Strategy interprets the training process of placing and optimizing Gaussians
    as a sampling process.

    Specifically, it periodically:
    - Moves "dead" Gaussians (low opacity) to the location of "live" Gaussians (high opacity).
    - Adds covariance dependent noise to the positions of the Gaussians.
    - Introduces new Gaussians sampled based on the opacity distribution.

    """

    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        load_mcmc_plugin()
        # Precompute the look up table for binomial coefficients (Eq 9 in the MCMC paper)
        self.binoms = torch.tensor(
            [
                [math.comb(n, k) if k <= n else 0 for k in range(config.strategy.binom_n_max)]
                for n in range(config.strategy.binom_n_max)
            ],
            dtype=torch.float32,
            device=self.model.device,
        )

    def post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        # Relocate dead gaussians to the alive areas
        if check_step_condition(step, self.conf.strategy.relocate.start_iteration, self.conf.strategy.relocate.end_iteration, self.conf.strategy.relocate.frequency):
            self.relocate_gaussians()

        # Add new Gaussians if the maximum number has not been reached
        if check_step_condition(step, self.conf.strategy.add.start_iteration, self.conf.strategy.add.end_iteration, self.conf.strategy.add.frequency):
            self.add_new_gaussians()

        # Perturb the positions of the Gaussians
        if check_step_condition(step, self.conf.strategy.perturb.start_iteration, self.conf.strategy.perturb.end_iteration, self.conf.strategy.perturb.frequency):
            self.perturb_gaussians()

        return True

    @torch.no_grad()
    def relocate_gaussians(self) -> None:
        # Get the per Gaussian densities and scales (after sigmoid)
        densities = self.model.get_density()
        # Find the dead indices
        dead_idxs = torch.where(densities <= self.conf.strategy.opacity_threshold)[0]
        alive_idxs = torch.where(densities > self.conf.strategy.opacity_threshold)[0]
        n_dead_gaussians = len(dead_idxs)

        if n_dead_gaussians:
            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(n_dead_gaussians, alive_idxs)
            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param[dead_idxs] = param[sampled_idxs]
                return torch.nn.Parameter(param, requires_grad=param.requires_grad)

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v[sampled_idxs] = 0
                return v

            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

        if self.conf.strategy.print_stats:
            logger.info(f"Relocated {n_dead_gaussians} ({n_dead_gaussians / len(densities) * 100:.2f}%) gaussians")


    @torch.no_grad()
    def add_new_gaussians(self) -> None:
        # Get the current number of gaussians
        current_num_gaussians = self.model.num_gaussians
        target_num_gaussians = min(self.conf.strategy.add.max_n_gaussians, int(1.05 * current_num_gaussians))
        num_gaussians_to_add = max(0, target_num_gaussians - current_num_gaussians)

        if num_gaussians_to_add:
            sampled_idxs, new_densities, new_scales = self.sample_new_gaussians(num_gaussians_to_add)

            def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
                if name == "density":
                    param[sampled_idxs] = new_densities
                elif name == "scale":
                    param[sampled_idxs] = new_scales
                param_new = torch.cat([param, param[sampled_idxs]])
                return torch.nn.Parameter(param_new, requires_grad=param.requires_grad)

            def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
                v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
                return torch.cat([v, v_new])

            self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)

        if self.conf.strategy.print_stats:
            logger.info(f"Added {num_gaussians_to_add} ({num_gaussians_to_add / current_num_gaussians * 100:.2f}%) gaussians")

    @torch.no_grad()
    def perturb_gaussians(self) -> None:
        covariance = self.model.get_covariance()
        positions = self.model.get_positions()
        densities = self.model.get_density()

        current_lr = 0.0
        for param_gorup in self.model.optimizer.param_groups:
            if param_gorup["name"] == "positions":
                current_lr = param_gorup["lr"]

        def op_sigmoid(x: torch.Tensor, k: int = 100, x0: float = 0.995) -> torch.Tensor:
            return 1 / (1 + torch.exp(-k * (x - x0)))

        # Current positional learning rate multiplied by the config paramater scale
        noise = torch.randn_like(positions) * (op_sigmoid(1 - densities)) * self.conf.strategy.perturb.noise_lr * current_lr
        noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)

        self.model.positions.add_(noise)

    def sample_new_gaussians(
        self, num_gaussians: int, valid_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        densities = self.model.get_density()
        scales = self.model.get_scale()

        if valid_indices is None:
            valid_indices = torch.arange(0, int(densities.shape[0]), device=densities.device, dtype=torch.int32)

        probabilities = densities[valid_indices].flatten()  # ensure its shape is [N,]

        # Sample the locations to which the dead Gaussians will be moved proportional to the opacity of the alive Gaussians
        sampled_idxs = _multinomial_sample(probabilities, num_gaussians, replacement=True)
        sampled_idxs = valid_indices[sampled_idxs]

        ratios = (torch.bincount(sampled_idxs)[sampled_idxs] + 1).clamp_(min=1, max=self.conf.strategy.binom_n_max).int()

        new_densities, new_scales = _mcmc_plugin.compute_relocation_tensor(
            densities[sampled_idxs].contiguous(),
            scales[sampled_idxs].contiguous(),
            ratios.contiguous(),
            self.binoms,
            self.conf.strategy.binom_n_max,
        )

        new_densities = self.model.density_activation_inv(
            torch.clamp(new_densities, max=1.0 - torch.finfo(torch.float32).eps, min=self.conf.strategy.opacity_threshold)
        )

        new_scales = self.model.scale_activation_inv(new_scales)

        return sampled_idxs, new_densities, new_scales