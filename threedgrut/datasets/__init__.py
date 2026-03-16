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

import os
from pathlib import Path
from typing import Optional

from .dataset_colmap import ColmapDataset
from .dataset_nerf import NeRFDataset
from .dataset_nerfstudio import NerfstudioDataset
from .dataset_scannetpp import ScannetppDataset
from .utils import read_colmap_extrinsics_binary, read_colmap_extrinsics_text


def _load_colmap_exif_exposures(
    dataset_path: str,
    downsample_factor: int,
) -> list[Optional[float]]:
    """Load EXIF exposure data for all COLMAP images.

    Reads COLMAP extrinsics to get all image paths, then loads EXIF exposure
    data and returns mean-normalized values. This is called once and shared
    between train and val datasets.

    Args:
        dataset_path: Path to COLMAP dataset root
        downsample_factor: Downsample factor for images folder suffix

    Returns:
        List of mean-normalized log2 exposure values for all images.
    """
    from threedgrut.utils.exif import load_exif_exposures

    # Read COLMAP extrinsics to get image names
    try:
        cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.bin")
        cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.txt")
        cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)

    # Build image paths
    downsample_suffix = "" if downsample_factor == 1 else f"_{downsample_factor}"
    images_folder = f"images{downsample_suffix}"

    image_paths: list[Path] = []
    for extr in cam_extrinsics:
        image_path = Path(dataset_path) / images_folder / extr.name
        image_paths.append(image_path)

    return load_exif_exposures(image_paths)


def make(name: str, config, ray_jitter):
    match name:
        case "nerf":
            train_dataset = NeRFDataset(
                config.path,
                split="train",
                bg_color=config.model.background.color,
                ray_jitter=ray_jitter,
            )
            val_dataset = NeRFDataset(
                config.path,
                split="val",
                bg_color=config.model.background.color,
            )
        case "colmap":
            # Load EXIF exposure data if enabled (shared between train and val)
            if config.dataset.get("load_exif", True):
                exif_exposures = _load_colmap_exif_exposures(
                    config.path,
                    config.dataset.downsample_factor,
                )
            else:
                exif_exposures = None

            train_dataset = ColmapDataset(
                config.path,
                split="train",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                ray_jitter=ray_jitter,
                exif_exposures=exif_exposures,
            )
            val_dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                exif_exposures=exif_exposures,
            )
        case "scannetpp":
            train_dataset = ScannetppDataset(
                config.path,
                split="train",
                ray_jitter=ray_jitter,
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
            val_dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "nerfstudio":
            train_dataset = NerfstudioDataset(
                config.path,
                split="train",
                bg_color=config.model.background.color,
                ray_jitter=ray_jitter,
                test_split_interval=config.dataset.test_split_interval,
            )
            val_dataset = NerfstudioDataset(
                config.path,
                split="val",
                bg_color=config.model.background.color,
                test_split_interval=config.dataset.test_split_interval,
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "nerfstudio", "scannetpp"].'
            )

    return train_dataset, val_dataset


def make_test(name: str, config):
    match name:
        case "nerf":
            dataset = NeRFDataset(
                config.path,
                split="test",
                bg_color=config.model.background.color,
            )
        case "colmap":
            # Load EXIF exposure data if enabled
            if config.dataset.get("load_exif", True):
                exif_exposures = _load_colmap_exif_exposures(
                    config.path,
                    config.dataset.downsample_factor,
                )
            else:
                exif_exposures = None

            dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                exif_exposures=exif_exposures,
            )
        case "scannetpp":
            dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "nerfstudio":
            dataset = NerfstudioDataset(
                config.path,
                split="test",
                bg_color=config.model.background.color,
                test_split_interval=config.dataset.test_split_interval,
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "nerfstudio", "scannetpp"].'
            )
    return dataset
