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

        # GPU cache of processed camera intrinsics - now per camera ID
        self.intrinsics = {}
        # Store camera ID for each frame
        self.camera_ids = []

        # Get the scene data
        self.load_intrinsics_and_extrinsics()
        self.get_scene_info()
        self.load_camera_data()

        indices = np.arange(self.n_frames)

        # If test_split_interval is set, every test_split_interval frame will be excluded from the training set
        # If test_split_interval is non-positive, all images will be used for training and testing
        if self.test_split_interval > 0:
            if split == "train":
                indices = np.mod(indices, self.test_split_interval) != 0
            else:
                indices = np.mod(indices, self.test_split_interval) == 0
        
        self.poses = self.poses[indices].astype(np.float32)
        self.image_paths = self.image_paths[indices]
        self.camera_ids = np.array(self.camera_ids)[indices]  # Filter camera IDs too
        self.camera_centers = self.camera_centers[indices]
        
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

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

    def get_scene_info(self):
        # For multi-camera, we may need to handle several different image dimensions
        # We'll store dimensions per camera and use the first camera as reference
        self.camera_dimensions = {}
        self.n_frames = len(self.cam_extrinsics)
        
        # Get dimensions for each unique camera
        for intr in self.cam_intrinsics:
            self.camera_dimensions[intr.id] = {
                'width': intr.width,
                'height': intr.height
            }
        
        # Use first camera as reference for dataset-wide dimensions
        first_camera_id = self.cam_intrinsics[0].id
        first_intrinsic = next(intr for intr in self.cam_intrinsics if intr.id == first_camera_id)
        
        # Get actual image dimensions from first image
        image_path = os.path.join(
            self.path,
            self.get_images_folder(),
            self.cam_extrinsics[0].name,
        )
        image = np.asarray(Image.open(image_path))
        self.image_h = image.shape[0]
        self.image_w = image.shape[1]
        
        # Calculate scaling factor based on first camera
        self.scaling_factor = int(round(first_intrinsic.height / self.image_h))

    def create_camera_intrinsics(self, intr, image_h, image_w):
        """Create camera intrinsics for a specific camera with given dimensions."""
        scaling_factor = int(round(intr.height / image_h))
        
        # Generate UV coordinates for this camera's resolution
        u = np.tile(np.arange(image_w), image_h)
        v = np.arange(image_h).repeat(image_w)
        out_shape = (1, image_h, image_w, 3)

        def create_pinhole_camera(focalx, focaly):
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([image_w, image_h], dtype=np.int64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([image_w, image_h], dtype=np.float32) / 2,
                focal_length=np.array([focalx, focaly], dtype=np.float32),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            rays_o_cam, rays_d_cam = pinhole_camera_rays(
                u, v, focalx, focaly, image_w, image_h, self.ray_jitter
            )
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32, device=self.device).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32, device=self.device).reshape(out_shape),
                type(params).__name__,
            )

        def create_fisheye_camera(params):
            resolution = np.array([image_w, image_h]).astype(np.int64)
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
                # NOTE: fixed max angle might not apply to all datasets / better estimate from available data
                max_angle=max_angle,
                shutter_type=ShutterType.GLOBAL,
            )
            pixel_coords = torch.tensor(
                np.stack([u, v], axis=1), dtype=torch.int32, device=self.device
            )
            image_points = pixels_to_image_points(pixel_coords)
            rays_d_cam = image_points_to_camera_rays(
                params, image_points, device=self.device
            )
            rays_o_cam = torch.zeros_like(rays_d_cam)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
            )

        if intr.model == "SIMPLE_PINHOLE":
            focal_length = intr.params[0] / scaling_factor
            return create_pinhole_camera(focal_length, focal_length)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0] / scaling_factor
            focal_length_y = intr.params[1] / scaling_factor
            return create_pinhole_camera(focal_length_x, focal_length_y)
        elif intr.model == "OPENCV_FISHEYE":
            params = copy.deepcopy(intr.params)
            params[:4] = params[:4] / scaling_factor
            return create_fisheye_camera(params)
        else:
            assert False, f"Colmap camera model '{intr.model}' not handled"

    def load_camera_data(self):
        """Load the camera data and generate rays for each camera."""
        
        # Create intrinsics for each unique camera with its actual image dimensions
        for intr in self.cam_intrinsics:
            # For each camera, we need to check what the actual image dimensions are
            # Find a sample image from this camera
            sample_extr = next(ext for ext in self.cam_extrinsics if ext.camera_id == intr.id)
            image_path = os.path.join(
                self.path, self.get_images_folder(), os.path.basename(sample_extr.name)
            )
            
            # Get actual image dimensions for this camera
            if os.path.exists(image_path):
                sample_image = np.asarray(Image.open(image_path))
                cam_image_h, cam_image_w = sample_image.shape[:2]
            else:
                # Fallback to scaled dimensions
                cam_image_h = intr.height // self.scaling_factor
                cam_image_w = intr.width // self.scaling_factor
            
            # Create intrinsics for this specific camera
            self.intrinsics[intr.id] = self.create_camera_intrinsics(intr, cam_image_h, cam_image_w)

        # Load poses and paths
        self.poses = []
        self.image_paths = []
        self.mask_paths = []
        self.camera_ids = []

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
            
            image_path = os.path.join(
                self.path, self.get_images_folder(), extr.name
            )
            self.image_paths.append(image_path)
            self.camera_ids.append(extr.camera_id)  # Store camera ID for each frame

            # Mask path
            self.mask_paths.append(os.path.splitext(image_path)[0] + "_mask.png")

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses)
        self.image_paths = np.stack(self.image_paths, dtype=str)
        self.mask_paths = np.stack(self.mask_paths, dtype=str)

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

    def get_intrinsics_idx(self, frame_idx: int):
        """Get the camera ID for a given frame index."""
        return self.camera_ids[frame_idx]
    
    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx) -> dict:
        # Get the camera ID for this frame
        camera_id = self.camera_ids[idx]
        
        # Load image and get its actual dimensions
        image_data = np.asarray(Image.open(self.image_paths[idx]))
        actual_h, actual_w = image_data.shape[:2]
        
        # Use actual image dimensions for output shape
        out_shape = (1, actual_h, actual_w, 3)
        
        assert image_data.dtype == np.uint8, "Image data must be of type uint8"

        output_dict = {
            "data": torch.tensor(image_data).reshape(out_shape),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": camera_id,  # Use actual camera ID
            "image_dims": (actual_h, actual_w),  # Store actual dimensions
        }

        # Only add mask to dictionary if it exists
        if os.path.exists(mask_path := self.mask_paths[idx]):
            try:
                mask = torch.from_numpy(np.array(Image.open(mask_path).convert('L'))).reshape(
                    1, self.image_h, self.image_w, 1
                )
                output_dict["mask"] = mask
            except:
                print(1)

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""
        
        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        camera_id = batch["intr"][0].item()
        
        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Get intrinsics for this specific camera
        camera_params_dict, rays_ori, rays_dir, camera_name = self.intrinsics[camera_id]

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
            "camera_id": [camera_id],  # Include camera ID in sample
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
            camera_convention_rot = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            # Get camera ID and corresponding intrinsics
            camera_id = self.camera_ids[i_cam]
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
            assert rgb.dtype == np.float32, f"RGB image must be float32, got {rgb.dtype}"

            cam_list.append({
                "ext_mat": trans_mat_world_to_camera,
                "w": w,
                "h": h,
                "fov_w": fov_w,
                "fov_h": fov_h,
                "rgb_img": rgb,
                "split": self.split,
                "camera_id": camera_id,  # Include camera ID for reference
            })

        create_camera_visualization(cam_list)