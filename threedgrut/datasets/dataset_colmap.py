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
import copy
import platform

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import (
    create_camera_visualization,
    get_center_and_diag,
    pinhole_camera_rays,
    compute_max_radius,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
    get_worker_id,
)
from .camera_models import (
    ShutterType,
    OpenCVPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    image_points_to_camera_rays,
    pixels_to_image_points,
)


class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        downsample_factor=1,
        test_split_interval=8,
        ray_jitter=None,
    ):
        self.path = path
        self.device = device
        self.split = split
        self.downsample_factor = downsample_factor
        self.ray_jitter = ray_jitter
        self.test_split_interval = test_split_interval

        # Worker-based GPU cache for multiprocessing compatibility
        self._worker_gpu_cache = {}

        # (Re)load intrinsics and extrinsics
        self.reload()

    def reload(self):
        # GPU cache of processed camera intrinsics - now per camera ID
        self.intrinsics = {}

        # Get the scene data
        self.load_intrinsics_and_extrinsics()
        self.n_frames = len(self.cam_extrinsics)
        self.load_camera_data()
        indices = np.arange(self.n_frames)

        # If test_split_interval is set, every test_split_interval frame will be excluded from the training set
        # If test_split_interval is non-positive, all images will be used for training and testing
        if self.test_split_interval > 0:
            if self.split == "train":
                indices = np.mod(indices, self.test_split_interval) != 0
            else:
                indices = np.mod(indices, self.test_split_interval) == 0

        self.cam_extrinsics = [self.cam_extrinsics[i] for i in np.where(indices)[0]]
        self.poses = self.poses[indices].astype(np.float32)
        self.image_paths = self.image_paths[indices]  # numpy str array of image paths
        self.camera_centers = self.camera_centers[indices]
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

        # Clear existing worker caches to force recreation with new intrinsics
        self._worker_gpu_cache.clear()

    def load_intrinsics_and_extrinsics(self):
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

    def get_images_folder(self):
        downsample_suffix = (
            "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        )
        return f"images{downsample_suffix}"

    def load_camera_data(self):
        """
        Load the camera data and generate rays for each camera.
        This function is called on CPU for multiprocessing compatibility
        GPU tensors will be created per-worker as needed
        """
        self._camera_data_params = {}
        self._store_camera_params_cpu()

    def _store_camera_params_cpu(self):
        """Store camera parameters on CPU for multiprocessing compatibility."""

        def create_pinhole_camera(focalx, focaly, w, h):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.int64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([w, h], dtype=np.float32) / 2,
                focal_length=np.array([focalx, focaly], dtype=np.float32),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            rays_o_cam, rays_d_cam = pinhole_camera_rays(
                u, v, focalx, focaly, w, h, self.ray_jitter
            )
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
                type(params).__name__,
            )

        def create_fisheye_camera(params, w, h):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            resolution = np.array([w, h]).astype(np.int64)
            principal_point = params[2:4].astype(np.float32)
            focal_length = params[0:2].astype(np.float32)
            radial_coeffs = params[4:].astype(np.float32)
            # Estimate max angle for fisheye
            max_radius_pixels = compute_max_radius(
                resolution.astype(np.float64), principal_point
            )
            fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
            fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
            max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0

            params = OpenCVFisheyeCameraModelParameters(
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                resolution=resolution,
                max_angle=max_angle,
                shutter_type=ShutterType.GLOBAL,
            )
            pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            image_points = pixels_to_image_points(pixel_coords)
            rays_d_cam = image_points_to_camera_rays(params, image_points)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
            )

        cam_id_to_image_name = {
            extr.camera_id: extr.name for extr in self.cam_extrinsics
        }

        for intr in self.cam_intrinsics.values():
            full_width = intr.width
            full_height = intr.height

            image_name = cam_id_to_image_name[intr.id]
            image_name = os.path.join(os.path.split(image_name)[1], '') if self.get_images_folder() in image_name else image_name
            image_path = os.path.join(self.path, self.get_images_folder(), image_name)

            try:
                # Load the image to get its actual dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
            except FileNotFoundError:
                logger.error(
                    f"Image {image_path} not found. Cannot determine dimensions for intrinsic ID {intr.id}."
                )
                continue

            # Calculate scaling factor to match the image dimensions to the intrinsic dimensions
            scaling_factor = int(round(intr.height / height))
            expected_size = (
                f"{full_width / scaling_factor}x{full_height / scaling_factor}"
            )
            assert (
                abs(full_width / scaling_factor - width) <= 1
            ), f"Scaled image dimension {expected_size} (factor {scaling_factor}x) does not match the actual image dimensions {width}x{height}"
            assert (
                abs(full_height / scaling_factor - height) <= 1
            ), f"Scaled image dimension {expected_size} (factor {scaling_factor}x) does not match the actual image dimensions {width}x{height}"

            if intr.model == "SIMPLE_PINHOLE":
                focal_length = intr.params[0] / scaling_factor
                self.intrinsics[intr.id] = create_pinhole_camera(
                    focal_length, focal_length, width, height
                )

            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0] / scaling_factor
                focal_length_y = intr.params[1] / scaling_factor
                self.intrinsics[intr.id] = create_pinhole_camera(
                    focal_length_x, focal_length_y, width, height
                )

            elif intr.model == "OPENCV_FISHEYE":
                params = copy.deepcopy(intr.params)
                params[:4] = params[:4] / scaling_factor
                self.intrinsics[intr.id] = create_fisheye_camera(params, width, height)

            else:
                assert (
                    False
                ), f"Colmap camera model '{intr.model}' not handled: Only undistorted datasets (PINHOLE, SIMPLE_PINHOLE or OPENCV_FISHEYE cameras) supported!"

        # Load poses and paths
        self.poses = []
        self.image_paths = []
        self.mask_paths = []

        cam_centers = []
        for extr in logger.track(
            self.cam_extrinsics,
            description=f"Load Dataset ({self.split})",
            color="salmon1",
        ):
            R = qvec_to_so3(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.zeros((4, 4), dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)
            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3])

            image_path = os.path.join(self.path, self.get_images_folder(), extr.name)
            self.image_paths.append(image_path)

            # Mask path
            self.mask_paths.append(os.path.splitext(image_path)[0] + "_mask.png")

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses)
        self.image_paths = np.stack(self.image_paths, dtype=str)
        self.mask_paths = np.stack(self.mask_paths, dtype=str)

    def _lazy_worker_intrinsics_cache(self):
        """Create intrinsics cache for a specific worker."""
        worker_id = get_worker_id()

        # Check if this worker already has cached tensors
        if worker_id not in self._worker_gpu_cache:
            # For now, fall back to the original approach for each worker
            # This ensures each worker creates its own GPU tensors
            worker_intrinsics = {}
            for intr_id, (
                params_dict,
                rays_ori,
                rays_dir,
                camera_name,
            ) in self.intrinsics.items():
                # Create new GPU tensors for this worker
                worker_rays_ori = rays_ori.to(self.device, non_blocking=True)
                worker_rays_dir = rays_dir.to(self.device, non_blocking=True)
                worker_intrinsics[intr_id] = (
                    params_dict,
                    worker_rays_ori,
                    worker_rays_dir,
                    camera_name,
                )
            self._worker_gpu_cache[worker_id] = worker_intrinsics

        return self._worker_gpu_cache[worker_id]

    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)  # mean distance between of cameras from center
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale

    def get_center(self):
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 transformation matrices.

        COLMAP Dataset Implementation:
        COLMAP naturally provides poses in a coordinate system compatible with
        3DGRUT's "right down front" convention, so no coordinate conversion is needed.

        The poses are constructed from COLMAP's world-to-camera matrices by:
        1. Building W2C from rotation (qvec_to_so3) and translation (tvec)
        2. Inverting to get camera-to-world: C2W = inv(W2C)

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4) in "right down front" convention
        """
        return self.poses

    def get_intrinsics_idx(self, extr_idx: int):
        return self.cam_extrinsics[extr_idx].camera_id

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx) -> dict:
        # Load image and get its actual dimensions
        image_data = np.asarray(Image.open(self.image_paths[idx]))
        actual_h, actual_w = image_data.shape[:2]

        # Use actual image dimensions for output shape
        out_shape = (1, actual_h, actual_w, 3)

        assert image_data.dtype == np.uint8, "Image data must be of type uint8"

        output_dict = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
        }

        # Only add mask to dictionary if it exists
        if os.path.exists(mask_path := self.mask_paths[idx]):
            mask = torch.from_numpy(
                np.array(Image.open(mask_path).convert("L"))
            ).reshape(1, actual_h, actual_w, 1)
            output_dict["mask"] = mask

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()

        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Get intrinsics for current worker
        worker_intrinsics = self._lazy_worker_intrinsics_cache()

        camera_params_dict, rays_ori, rays_dir, camera_name = worker_intrinsics[intr]

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
        }

        if "mask" in batch:
            mask = batch["mask"][0].to(self.device, non_blocking=True) / 255.0
            mask = (mask > 0.5).to(torch.float32)
            sample["mask"] = mask

        return Batch(**sample)

    def create_dataset_camera_visualization(self):
        """Create a visualization of the dataset cameras."""

        cam_list = []

        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # Camera convention rotation
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = (
                camera_convention_rot @ trans_mat_world_to_camera
            )

            # Get camera ID and corresponding intrinsics
            camera_id = self.get_intrinsics_idx(i_cam)
            intr, _, _, _ = self.intrinsics[camera_id]

            # Load actual image to get dimensions
            image_data = np.asarray(Image.open(self.image_paths[i_cam]))
            h, w = image_data.shape[:2]

            f_w = intr["focal_length"][0]
            f_h = intr["focal_length"][1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            assert image_data.dtype == np.uint8, "Image data must be of type uint8"
            rgb = image_data.reshape(h, w, 3) / np.float32(255.0)
            assert (
                rgb.dtype == np.float32
            ), f"RGB image must be float32, got {rgb.dtype}"

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        create_camera_visualization(cam_list)