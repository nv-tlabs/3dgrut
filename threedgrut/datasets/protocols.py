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

from typing import Protocol, Optional, runtime_checkable, final
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class Batch:
    rays_ori: torch.Tensor  # [B, H, W, 3] ray origins in arbitrary space
    rays_dir: torch.Tensor  # [B, H, W, 3] ray directions in arbitrary space
    T_to_world: torch.Tensor  # [B, 4, 4] transformation matrix from the ray space to the world space
    rgb_gt: Optional[torch.Tensor] = None
    intrinsics: Optional[list] = None
    intrinsics_OpenCVPinholeCameraModelParameters: Optional[dict] = None
    intrinsics_OpenCVFisheyeCameraModelParameters: Optional[dict] = None

    def __post_init__(self):
        batch_size = self.T_to_world.shape[0]
        assert self.rays_ori.shape[0] == batch_size, "rays_ori must have the same batch size"
        assert self.rays_dir.shape[0] == batch_size, "rays_dir must have the same batch size"
        if self.rgb_gt is not None:
            assert self.rgb_gt.ndim == 4, "rgb_gt must be a 4D tensor [B, H, W, 3]"
            assert self.rgb_gt.shape[0] == batch_size, "rgb_gt must have the same batch size"
        # assert (
        #     self.intrinsics is not None
        #     or self.intrinsics_OpenCVPinholeCameraModelParameters is not None
        #     or self.intrinsics_OpenCVFisheyeCameraModelParameters is not None
        # ), "At least one of intrinsics must be provided."
        if self.intrinsics:
            assert isinstance(self.intrinsics, list), "intrinsics must be a list"
            assert len(self.intrinsics) == 4, "intrinsics must have 4 elements [fx, fy, cx, cy]"


class BoundedMultiViewDataset(Protocol):
    """Defines the basic functionality required from all datasets that can be used with the 3dgrut Trainer."""

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the bounding box of the scene as a tuple of vec3 (min,max)"""
        ...

    def get_scene_extent(self) -> float: 
        """TODO"""
        ...

    def get_observer_points(self) -> np.ndarray: 
        """TODO"""
        ...

    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch:
        """Add the intrinsics to the batch and move data to GPU."""
        ...

    def __getitem__(self, index: int) -> dict: ...

    def __len__(self) -> int: ...


@runtime_checkable
class DatasetVisualization(Protocol):
    """Defines the basic functionality required from all datasets that can be visualized in the GUI app."""

    def create_dataset_camera_visualization(self): ...
