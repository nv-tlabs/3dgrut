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

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .camera_models import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
    image_points_to_camera_rays,
    opencv_pinhole_image_points_to_camera_rays,
    pixels_to_image_points,
)
from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import compute_max_radius, create_camera_visualization, get_center_and_diag, get_worker_id


class NerfstudioDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    """
    Dataset loader for Nerfstudio format datasets.

    Nerfstudio format uses explicit intrinsics (fl_x, fl_y, cx, cy), OpenCV distortion
    parameters (k1, k2, p1, p2), and supports per-frame intrinsic overrides.

    Expected JSON format:
    {
        "fl_x": 784.57, "fl_y": 587.71,
        "cx": 633.57, "cy": 363.09,
        "w": 1280, "h": 720,
        "k1": -0.054, "k2": 0.061, "p1": -0.0002, "p2": 0.0011,
        "camera_model": "OPENCV",
        "frames": [
            {
                "file_path": "./images/frame.png",
                "transform_matrix": [[...], [...], [...], [0,0,0,1]],
                // Optional per-frame overrides:
                "fl_x": ..., "fl_y": ..., "cx": ..., "cy": ..., "w": ..., "h": ...
            }
        ]
    }
    """

    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        ray_jitter=None,
        bg_color=None,
        test_split_interval=8,
    ):
        self.path = path
        self.device = device
        self.split = split
        self.ray_jitter = ray_jitter
        self.bg_color = bg_color
        self.test_split_interval = test_split_interval
        self.aabb_scale = None

        # Worker-based GPU cache for multiprocessing compatibility
        self._worker_gpu_cache = {}

        # Per-intrinsic-config storage (like COLMAP)
        # Dict[int, (params_dict, rays_o, rays_d, camera_name)]
        self.intrinsics = {}
        # Maps intrinsic key tuple -> integer ID
        self._intrinsic_key_to_id = {}
        # Maps frame idx -> intrinsic integer ID
        self.frame_to_intrinsic_id = []

        # (Re)load intrinsics and extrinsics
        self.reload()

    def reload(self):
        """Reload the dataset metadata."""
        # Clear caches
        self.intrinsics = {}
        self._intrinsic_key_to_id = {}
        self.frame_to_intrinsic_id = []
        self._worker_gpu_cache.clear()

        # Read metadata
        self.read_meta(self.split)
        self.n_frames = len(self.poses)
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

    def _parse_intrinsics(self, data: dict, frame: dict) -> dict:
        """
        Extract intrinsics from global data or per-frame data.
        Per-frame values override global values.
        """
        intr = {
            "fl_x": frame.get("fl_x", data.get("fl_x")),
            "fl_y": frame.get("fl_y", data.get("fl_y")),
            "cx": frame.get("cx", data.get("cx")),
            "cy": frame.get("cy", data.get("cy")),
            "w": frame.get("w", data.get("w")),
            "h": frame.get("h", data.get("h")),
            "k1": frame.get("k1", data.get("k1", 0.0)),
            "k2": frame.get("k2", data.get("k2", 0.0)),
            "k3": frame.get("k3", data.get("k3", 0.0)),
            "k4": frame.get("k4", data.get("k4", 0.0)),
            "k5": frame.get("k5", data.get("k5", 0.0)),
            "k6": frame.get("k6", data.get("k6", 0.0)),
            "p1": frame.get("p1", data.get("p1", 0.0)),
            "p2": frame.get("p2", data.get("p2", 0.0)),
            "s1": frame.get("s1", data.get("s1", 0.0)),
            "s2": frame.get("s2", data.get("s2", 0.0)),
            "s3": frame.get("s3", data.get("s3", 0.0)),
            "s4": frame.get("s4", data.get("s4", 0.0)),
            "camera_model": frame.get("camera_model", data.get("camera_model", "OPENCV")),
        }

        w = intr["w"]
        h = intr["h"]

        # Support camera_angle_x / camera_angle_y (FOV in radians) -> focal lengths
        if intr["fl_x"] is None:
            camera_angle_x = frame.get("camera_angle_x", data.get("camera_angle_x"))
            if camera_angle_x is not None and w is not None:
                intr["fl_x"] = 0.5 * w / np.tan(0.5 * camera_angle_x)
        if intr["fl_y"] is None:
            camera_angle_y = frame.get("camera_angle_y", data.get("camera_angle_y"))
            if camera_angle_y is not None and h is not None:
                intr["fl_y"] = 0.5 * h / np.tan(0.5 * camera_angle_y)

        # Fallback: fl_y = fl_x when not specified
        if intr["fl_y"] is None:
            intr["fl_y"] = intr["fl_x"]

        # Fallback: principal point at image center
        if intr["cx"] is None and w is not None:
            intr["cx"] = w / 2.0
        if intr["cy"] is None and h is not None:
            intr["cy"] = h / 2.0

        return intr

    def _intrinsics_to_key(self, intr: dict) -> tuple:
        """Create a hashable key from intrinsics dict."""
        return (
            intr["fl_x"],
            intr["fl_y"],
            intr["cx"],
            intr["cy"],
            intr["w"],
            intr["h"],
            intr["k1"],
            intr["k2"],
            intr["k3"],
            intr["k4"],
            intr["k5"],
            intr["k6"],
            intr["p1"],
            intr["p2"],
            intr["s1"],
            intr["s2"],
            intr["s3"],
            intr["s4"],
            intr["camera_model"],
        )

    def _create_camera_rays(self, intr: dict) -> tuple:
        """
        Generate rays with distortion handling.

        Returns:
            (params_dict, rays_o, rays_d, camera_name)
        """
        w = int(intr["w"])
        h = int(intr["h"])
        fl_x = float(intr["fl_x"])
        fl_y = float(intr["fl_y"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])

        # Distortion coefficients
        k1 = float(intr["k1"])
        k2 = float(intr["k2"])
        k3 = float(intr["k3"])
        k4 = float(intr["k4"])
        k5 = float(intr["k5"])
        k6 = float(intr["k6"])
        p1 = float(intr["p1"])
        p2 = float(intr["p2"])
        s1 = float(intr["s1"])
        s2 = float(intr["s2"])
        s3 = float(intr["s3"])
        s4 = float(intr["s4"])

        camera_model = intr["camera_model"]

        out_shape = (1, h, w, 3)

        # Generate pixel coordinates
        u = np.tile(np.arange(w), h)
        v = np.arange(h).repeat(w)

        if camera_model == "OPENCV_FISHEYE":
            # Use fisheye camera model with 4 radial coefficients (k1-k4)
            resolution = np.array([w, h], dtype=np.int64)
            principal_point = np.array([cx, cy], dtype=np.float32)
            focal_length = np.array([fl_x, fl_y], dtype=np.float32)
            radial_coeffs = np.array([k1, k2, k3, k4], dtype=np.float32)

            # Estimate max angle for fisheye
            max_radius_pixels = compute_max_radius(resolution.astype(np.float64), principal_point)
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
        else:
            # OPENCV / PERSPECTIVE / default: use pinhole model
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.int64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([fl_x, fl_y], dtype=np.float32),
                radial_coeffs=np.array([k1, k2, k3, k4, k5, k6], dtype=np.float32),
                tangential_coeffs=np.array([p1, p2], dtype=np.float32),
                thin_prism_coeffs=np.array([s1, s2, s3, s4], dtype=np.float32),
            )

            # Check if we have any distortion
            has_distortion = any(abs(x) > 1e-10 for x in [k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4])

            if has_distortion:
                # Use iterative undistortion for distorted cameras
                pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
                image_points = pixels_to_image_points(pixel_coords)

                rays_d_cam = opencv_pinhole_image_points_to_camera_rays(params, image_points, device="cpu")
                rays_o_cam = torch.zeros_like(rays_d_cam)
            else:
                # Use simple pinhole model with actual principal point (not image center)
                if self.ray_jitter is not None:
                    jitter = self.ray_jitter(u.shape).numpy()
                    jitter_xs = jitter[:, 0]
                    jitter_ys = jitter[:, 1]
                else:
                    jitter_xs = jitter_ys = 0.5

                # Compute ray directions using actual principal point (cx, cy)
                xs = ((u + jitter_xs) - cx) / fl_x
                ys = ((v + jitter_ys) - cy) / fl_y

                rays_d_cam = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
                rays_d_cam = rays_d_cam / np.linalg.norm(rays_d_cam, axis=-1, keepdims=True)
                rays_o_cam = np.zeros_like(rays_d_cam)

                rays_o_cam = torch.tensor(rays_o_cam, dtype=torch.float32)
                rays_d_cam = torch.tensor(rays_d_cam, dtype=torch.float32)

        return (
            params.to_dict(),
            rays_o_cam.to(torch.float32).reshape(out_shape),
            rays_d_cam.to(torch.float32).reshape(out_shape),
            type(params).__name__,
        )

    def read_meta(self, split):
        """Load JSON, group frames by intrinsics, apply coord conversion."""
        # Try split-specific file first, then generic
        split_specific_path = os.path.join(self.path, f"transforms_{split}.json")
        generic_path = os.path.join(self.path, "transforms.json")

        data = None
        used_generic = False
        if os.path.exists(split_specific_path):
            with open(split_specific_path, "r") as f:
                data = json.load(f)
        elif os.path.exists(generic_path):
            with open(generic_path, "r") as f:
                data = json.load(f)
            used_generic = True

        if data is None:
            raise FileNotFoundError(f"Could not find transforms JSON file in {self.path}")

        # Parse aabb_scale if present
        if "aabb_scale" in data:
            self.aabb_scale = data["aabb_scale"]

        frames = data.get("frames", [])

        self.poses = []
        self.image_paths = []
        self.mask_paths = []
        self.frame_to_intrinsic_id = []

        cam_centers = []

        for frame in logger.track(frames, description=f"Load Dataset ({split})", color="salmon1"):
            # Parse intrinsics for this frame
            intr = self._parse_intrinsics(data, frame)
            intr_key = self._intrinsics_to_key(intr)

            # Create camera rays if we haven't seen this intrinsic configuration before
            if intr_key not in self._intrinsic_key_to_id:
                intr_id = len(self._intrinsic_key_to_id)
                self._intrinsic_key_to_id[intr_key] = intr_id
                self.intrinsics[intr_id] = self._create_camera_rays(intr)

            self.frame_to_intrinsic_id.append(self._intrinsic_key_to_id[intr_key])

            # Load pose
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            # Ensure 4x4 matrix
            if c2w.shape == (3, 4):
                c2w = np.vstack([c2w, [0, 0, 0, 1]])
            # Convert from nerfstudio convention [right up back] to [right down front]
            c2w[:, 1:3] *= -1
            cam_centers.append(c2w[:3, 3])
            self.poses.append(c2w)

            # Image path
            file_path = frame["file_path"]
            # Handle relative paths
            if file_path.startswith("./"):
                file_path = file_path[2:]
            img_path = os.path.join(self.path, file_path)

            # Check if the image path has an extension, try common extensions
            if not os.path.exists(img_path):
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    if os.path.exists(img_path + ext):
                        img_path = img_path + ext
                        break

            self.image_paths.append(img_path)

            # Mask path: check frame JSON first, then fall back to convention
            mask_path = frame.get("mask_path")
            if mask_path is not None:
                if mask_path.startswith("./"):
                    mask_path = mask_path[2:]
                mask_path = os.path.join(self.path, mask_path)
            else:
                mask_path = os.path.splitext(img_path)[0] + "_mask.png"
            self.mask_paths.append(mask_path)

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses).astype(np.float32)
        self.image_paths = np.array(self.image_paths, dtype=str)
        self.mask_paths = np.array(self.mask_paths, dtype=str)

        # Apply train/val split when using a generic transforms.json
        if used_generic and self.test_split_interval > 0:
            n_frames = len(self.poses)
            indices = np.arange(n_frames)
            if split == "train":
                mask = np.mod(indices, self.test_split_interval) != 0
            else:
                mask = np.mod(indices, self.test_split_interval) == 0
            self.poses = self.poses[mask]
            self.image_paths = self.image_paths[mask]
            self.mask_paths = self.mask_paths[mask]
            self.camera_centers = self.camera_centers[mask]
            self.frame_to_intrinsic_id = [self.frame_to_intrinsic_id[i] for i in np.where(mask)[0]]

    @torch.no_grad()
    def compute_spatial_extents(self):
        """Compute scene center, length scale, and bounding box."""
        camera_origins = torch.FloatTensor(self.poses[:, :3, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values

        if self.aabb_scale is not None:
            bbox_center = (bbox_min + bbox_max) / 2.0
            bbox_half = (bbox_max - bbox_min) / 2.0
            bbox_min = bbox_center - bbox_half * self.aabb_scale
            bbox_max = bbox_center + bbox_half * self.aabb_scale

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

        Nerfstudio Dataset Implementation:
        Converts from Nerfstudio's "right up back" coordinate system to 3DGRUT's
        "right down front" convention by negating Y and Z axes during loading.

        Original Nerfstudio Convention: [right, up, back]
        3DGRUT Convention: [right, down, front]
        Conversion: c2w[:, 1:3] *= -1  # Negate Y and Z columns

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4) in "right down front" convention
        """
        return self.poses

    def get_intrinsics_idx(self, extr_idx: int):
        """Get the intrinsic ID for a given frame index."""
        return self.frame_to_intrinsic_id[extr_idx]

    def _lazy_worker_intrinsics_cache(self):
        """Create intrinsics cache for a specific worker."""
        worker_id = get_worker_id()

        if worker_id not in self._worker_gpu_cache:
            worker_intrinsics = {}
            for intr_key, (params_dict, rays_ori, rays_dir, camera_name) in self.intrinsics.items():
                worker_rays_ori = rays_ori.to(self.device, non_blocking=True)
                worker_rays_dir = rays_dir.to(self.device, non_blocking=True)
                worker_intrinsics[intr_key] = (
                    params_dict,
                    worker_rays_ori,
                    worker_rays_dir,
                    camera_name,
                )
            self._worker_gpu_cache[worker_id] = worker_intrinsics

        return self._worker_gpu_cache[worker_id]

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx) -> dict:
        """Return {data, pose, intr} like COLMAP."""
        # Load image
        image_data = np.asarray(Image.open(self.image_paths[idx]))
        actual_h, actual_w = image_data.shape[:2]

        # Verify dimensions match the pre-computed rays
        intr_id = self.get_intrinsics_idx(idx)
        _, rays_o, _, _ = self.intrinsics[intr_id]
        expected_h, expected_w = rays_o.shape[1], rays_o.shape[2]
        if actual_h != expected_h or actual_w != expected_w:
            raise ValueError(
                f"Image {self.image_paths[idx]} has dimensions {actual_w}x{actual_h} "
                f"but intrinsics specify {expected_w}x{expected_h}. "
                f"Image dimensions must match intrinsics."
            )

        # Handle RGBA images
        if len(image_data.shape) == 3 and image_data.shape[2] == 4:
            # Blend alpha channel
            if self.bg_color is None:
                image_data = image_data[..., :3]
            elif self.bg_color == "black":
                alpha = image_data[..., 3:4].astype(np.float32) / 255.0
                image_data = (image_data[..., :3].astype(np.float32) * alpha).astype(np.uint8)
            elif self.bg_color == "white":
                alpha = image_data[..., 3:4].astype(np.float32) / 255.0
                image_data = (
                    image_data[..., :3].astype(np.float32) * alpha + (1 - alpha) * 255.0
                ).astype(np.uint8)

        assert image_data.dtype == np.uint8, "Image data must be of type uint8"

        # Ensure image has 3 channels
        if len(image_data.shape) == 2:
            # Grayscale - convert to RGB
            image_data = np.stack([image_data] * 3, axis=-1)
        elif len(image_data.shape) == 3 and image_data.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got shape {image_data.shape}")

        output_dict = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
        }

        # Only add mask if it exists
        if os.path.exists(mask_path := self.mask_paths[idx]):
            mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).reshape(
                1, actual_h, actual_w, 1
            )
            output_dict["mask"] = mask

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU. Follow COLMAP pattern."""
        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()  # Integer intrinsic ID

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
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            # Get intrinsics for this frame
            intr_id = self.get_intrinsics_idx(i_cam)
            intr, _, _, _ = self.intrinsics[intr_id]

            # Load actual image to get dimensions
            image_data = np.asarray(Image.open(self.image_paths[i_cam]))
            h, w = image_data.shape[:2]

            f_w = intr["focal_length"][0]
            f_h = intr["focal_length"][1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            # Handle RGBA images
            if image_data.shape[2] == 4:
                image_data = image_data[..., :3]

            assert image_data.dtype == np.uint8, "Image data must be of type uint8"
            rgb = image_data.reshape(h, w, 3) / np.float32(255.0)
            assert rgb.dtype == np.float32, f"RGB image must be float32, got {rgb.dtype}"

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
