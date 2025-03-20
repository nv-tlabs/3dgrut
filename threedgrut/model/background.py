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

from abc import ABC, abstractmethod

import omegaconf
import torch
from omegaconf import OmegaConf

from threedgrut.datasets.utils import DEFAULT_DEVICE


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def make(name: str, config):
    match name:
        case "background-color":
            return BackgroundColor(config=config)
        case "skip-background":
            return SkipBackground(config=config)
        case _:
            raise ValueError(f"background {name} not supported, choice must be in [background-color, skip-background]")


class BaseBackground(ABC, torch.nn.Module):
    def __init__(self, config: omegaconf.dictconfig.DictConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.device = DEFAULT_DEVICE

        self.setup(**kwargs)

    @abstractmethod
    def setup(self, **kwargs) -> None:
        raise NotImplementedError("Must override in the child class")

    @abstractmethod
    def forward(self, ray_to_world, rays_d, rgb, opacity):
        raise NotImplementedError("Must override in the child class")

    def linear_to_srgb(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)

    def srgb_to_linear(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class BackgroundColor(BaseBackground):
    def setup(self, **kwargs):
        self.background_color_type = self.config.color

        assert self.background_color_type in [
            "white",
            "black",
            "random",
        ], "Background color must be one of 'white', 'black', 'random'"

        if self.background_color_type == "white":
            self.color = torch.ones((3,), dtype=torch.float32, device=self.device)
        elif self.background_color_type == "black":
            self.color = torch.zeros((3,), dtype=torch.float32, device=self.device)
        elif self.background_color_type == "random":
            # set the stored color to black for random, we use this when not training
            self.color = torch.zeros((3,), dtype=torch.float32, device=self.device)

    @torch.cuda.nvtx.range("background_color.forward")
    def forward(self, ray_to_world, rays_d, rgb, opacity, train: bool):

        color = self.color

        if self.background_color_type == "random":  # only use random color when training
            if train:

                # NOTE: this uses random color PER PIXEL, other codebases use constant random
                color = torch.rand_like(rays_d, dtype=torch.float32, device=self.device)

                # this is set up to statefully remember the last random color, we ise this during trainint
                self.color = color

                rgb = rgb + color * (1.0 - opacity)

        return rgb, opacity


class SkipBackground(BaseBackground):
    def setup(self, **kwargs):
        pass

    @torch.cuda.nvtx.range("skip_background.forward")
    def forward(self, ray_to_world, rays_d, rgb, opacity, train) -> None:
        return rgb, opacity
