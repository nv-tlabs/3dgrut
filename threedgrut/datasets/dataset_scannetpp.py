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

from .dataset_colmap import ColmapDataset
from .utils import read_colmap_extrinsics_text, read_colmap_intrinsics_text


class ScannetppDataset(ColmapDataset):

    def __init__(self, path, device="cuda", split="train", ray_jitter=None):
        super(ScannetppDataset, self).__init__(path, device, split, ray_jitter)

    def load_intrinsics_and_extrinsics(self):
        cameras_extrinsic_file = os.path.join(self.path, "colmap", "images.txt")
        cameras_intrinsic_file = os.path.join(self.path, "colmap", "cameras.txt")
        self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
        self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

        # Remove camera distortions because images are already undistorted
        for intr in self.cam_intrinsics:
            intr.params[4:] = 0.0

    def get_images_folder(self):
        return "image_undistorted_fisheye"
