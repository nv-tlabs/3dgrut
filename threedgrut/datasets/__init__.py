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

from .dataset_nerf import NeRFDataset
from .dataset_colmap import ColmapDataset
from .dataset_scannetpp import ScannetppDataset


def make(name: str, config, ray_jitter):
    match name:
        case "nerf":
            train_dataset = NeRFDataset(
                config.path,
                split="train",
                return_alphas=False,
                bg_color=config.model.background.color,
                ray_jitter=ray_jitter,
            )
            val_dataset = NeRFDataset(
                config.path,
                split="val",
                return_alphas=False,
                bg_color=config.model.background.color,
            )
        case "colmap":
            train_dataset = ColmapDataset(
                config.path,
                split="train",
                downsample_factor=config.dataset.downsample_factor,
                ray_jitter=ray_jitter,
            )
            val_dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
            )
        case "scannetpp":
            train_dataset = ScannetppDataset(
                config.path,
                split="train",
                ray_jitter=ray_jitter,
            )
            val_dataset = ScannetppDataset(
                config.path,
                split="val",
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "scannetpp"].'
            )

    return train_dataset, val_dataset
