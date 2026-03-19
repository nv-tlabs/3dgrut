# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Adapter to integrate NCoreDataset with threedgrut framework.

NCoreDataset uses a different batch format (rays_cam with metadata)
compared to threedgrut's expected format (rays_ori, rays_dir, rgb_gt, etc.).

This adapter wraps NCoreDataset and provides the required interface methods.
"""

import cv2
import numpy as np
import torch

import ncore.data
import ncore.sensors

from threedgrut.datasets.camera_models import FThetaCameraModelParameters, ShutterType
from threedgrut.datasets.datasetNcore import NCoreDataset
from threedgrut.datasets.protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from threedgrut.datasets.utils import create_camera_visualization, create_pixel_coords, get_worker_id
from threedgrut.utils.logger import logger


class NCoreDatasetAdapter(NCoreDataset, BoundedMultiViewDataset, DatasetVisualization):
    """
    Adapter for NCoreDataset to work with threedgrut framework.

    Key adaptations:
    - Converts NCore's rays_cam to threedgrut's rays_ori/rays_dir
    - Implements required protocol methods (get_frames_per_camera, get_gpu_batch_with_intrinsics)
    - Handles coordinate space transformations (NCore uses WORLD_SPACE)
    """

    def __init__(self, *args, device="cuda", **kwargs):
        """Initialize with device parameter for GPU operations"""
        super().__init__(*args, **kwargs)
        self.device = device

        # Store training mode for batch reshaping
        self._sample_full_image = self.sample_full_image
        self._train_patch_size = None

        # Cache for camera intrinsics (populated on first worker init)
        self._camera_intrinsics_cache = {}

        # Per-worker GPU caches: camera-space rays, pixel coordinates
        # Populated lazily on first use in each DataLoader worker process.
        # Keys: worker_id -> dict[(camera_id, w, h) -> (rays_ori, rays_dir, pixel_coords)]
        self._worker_gpu_cache: dict = {}

    def _lazy_worker_gpu_cache(self) -> dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Lazily create and cache camera-space rays + pixel coords on GPU for the current worker.

        Following the same pattern as COLMAP's _lazy_worker_intrinsics_cache (dataset_colmap.py:292).
        Each DataLoader worker creates its own GPU tensors on first use.

        Returns:
            Dict mapping (camera_id, w, h) -> (rays_ori, rays_dir, pixel_coords) on GPU.
        """
        worker_id = get_worker_id()

        if worker_id not in self._worker_gpu_cache:
            cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

            for sequence_id in self.sequence_cameras_all_rays:
                for camera_id, rays_np in self.sequence_cameras_all_rays[sequence_id].items():
                    # rays_np: (H, W, 3) camera-space ray directions (pre-computed at init)
                    h_full, w_full = rays_np.shape[:2]
                    rays_dir = torch.from_numpy(rays_np).to(dtype=torch.float32, device=self.device, non_blocking=True)
                    rays_ori = torch.zeros_like(rays_dir)  # Intrinsic origin at (0,0,0)
                    pixel_coords = create_pixel_coords(w_full, h_full, device=self.device)  # (1, H, W, 2)
                    cache[(camera_id, w_full, h_full)] = (rays_ori, rays_dir, pixel_coords)

                # Also cache subsampled rays for validation (when n_val_image_subsample > 1)
                if self.n_val_image_subsample > 1:
                    for camera_id, rays_sub_np in self.sequence_cameras_rays_subsample[sequence_id].items():
                        camera_model = self.sequence_camera_models[sequence_id][camera_id]
                        w_full = int(camera_model.resolution[0].item())
                        h_full = int(camera_model.resolution[1].item())
                        w_sub = w_full // self.n_val_image_subsample
                        h_sub = h_full // self.n_val_image_subsample
                        rays_dir = torch.from_numpy(rays_sub_np.reshape(h_sub, w_sub, 3)).to(
                            dtype=torch.float32, device=self.device, non_blocking=True
                        )
                        rays_ori = torch.zeros_like(rays_dir)
                        pixel_coords = create_pixel_coords(w_sub, h_sub, device=self.device)
                        cache[(camera_id, w_sub, h_sub)] = (rays_ori, rays_dir, pixel_coords)

            self._worker_gpu_cache[worker_id] = cache

        return self._worker_gpu_cache[worker_id]

    def __getitem__(self, idx) -> dict:
        """Override to return dict instead of NCoreBatch for DataLoader collation.

        The batch dict does NOT contain rays — camera-space rays are cached on GPU per worker
        and looked up by (camera_id, w, h) in get_gpu_batch_with_intrinsics. This eliminates
        the ~61MB CPU->GPU ray transfer per batch.
        """
        ncore_batch = super().__getitem__(idx)

        batch_dict: dict = {}

        # RGB from labels
        if ncore_batch.labels is not None and ncore_batch.labels.rgb is not None:
            batch_dict["rgb"] = ncore_batch.labels.rgb

        # Valid mask (validation only)
        if (
            ncore_batch.labels is not None
            and hasattr(ncore_batch.labels, "valid")
            and ncore_batch.labels.valid is not None
        ):
            batch_dict["valid"] = ncore_batch.labels.valid

        # Camera-to-world poses (start + end for rolling shutter)
        batch_dict["T_camera_to_world"] = ncore_batch.T_camera_to_world
        batch_dict["T_camera_to_world_end"] = ncore_batch.T_camera_to_world_end

        # Camera ID for ray cache + intrinsics lookup
        batch_dict["camera_id"] = ncore_batch.camera_id

        # Image dimensions (required for ray cache lookup and reshaping)
        if ncore_batch.h is not None:
            batch_dict["h"] = ncore_batch.h
        if ncore_batch.w is not None:
            batch_dict["w"] = ncore_batch.w

        # Frame/camera indices for PPISP post-processing
        if ncore_batch.frame_idx is not None:
            batch_dict["frame_idx"] = ncore_batch.frame_idx
        if ncore_batch.camera_idx is not None:
            batch_dict["camera_idx"] = ncore_batch.camera_idx

        return batch_dict

    def get_frames_per_camera(self) -> list[int]:
        """Return split-specific frame counts per camera (training-only for PPISP parameter allocation)."""
        return [int(n) for n in self.get_n_frames_per_camera(unique_sensors=True)]

    def get_camera_names(self) -> list[str]:
        """Return camera names ordered by camera index."""
        return self.get_camera_sensor_ids(unique_sensors=True)

    def create_dataset_camera_visualization(self):
        """Create camera visualization for GUI display."""
        cam_list = []

        # Limit number of visualized frames to avoid GUI clutter
        # For training: subsample heavily to ~50 total views
        # For validation: show more frames (~10 per camera)
        if self.split == "train":
            max_frames_per_camera = max(1, 50 // len(self.camera_ids))
        else:
            max_frames_per_camera = 10  # More frames for validation

        # Iterate in camera-blocked order (all frames for cam0, then cam1, ...)
        # This matches the dataset's camera-blocked frame indexing
        sequence_id = self.sequence_id
        camera_sensors = self.sequence_camera_sensors[sequence_id]
        camera_models = self.sequence_camera_models[sequence_id]

        for camera_id in self.camera_ids:
            if camera_id not in camera_sensors:
                continue

            camera_sensor = camera_sensors[camera_id]
            camera_model = camera_models[camera_id]

            # Get camera resolution (already scaled by downsample in _init_worker)
            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())

            # Apply validation subsampling to match rendered resolution
            w_viz = w // self.n_val_image_subsample
            h_viz = h // self.n_val_image_subsample

            # Calculate field of view (model-dependent)
            if isinstance(camera_model, ncore.sensors.FThetaCameraModel):
                # FTheta has no focal_length; use max_angle as approximate half-FOV
                fov_w = 2.0 * camera_model.max_angle
                fov_h = 2.0 * camera_model.max_angle
            elif hasattr(camera_model, "focal_length"):
                fx = float(camera_model.focal_length[0])
                fy = float(camera_model.focal_length[1])
                fx_viz = fx / self.n_val_image_subsample
                fy_viz = fy / self.n_val_image_subsample
                fov_w = 2.0 * np.arctan(0.5 * w_viz / fx_viz)
                fov_h = 2.0 * np.arctan(0.5 * h_viz / fy_viz)
            else:
                logger.warning(
                    f"Camera {camera_id}: unsupported model type for FOV calculation, skipping visualization"
                )
                continue

            # Get frame range for this camera
            camera_frame_range = self.camera_frame_ranges[camera_id]

            # Calculate frame step based on split
            num_frames = camera_frame_range.stop - camera_frame_range.start
            frame_step = max(1, num_frames // max_frames_per_camera)

            for frame_idx in range(camera_frame_range.start, camera_frame_range.stop, frame_step):
                try:
                    # Get camera pose (sensor-to-world transformation)
                    T_sensor_to_world_flat = camera_sensor.get_frames_T_source_target(
                        source_node=camera_sensor.sensor_id,
                        target_node="world",
                        frame_indices=np.array(frame_idx),
                        frame_timepoint=ncore.data.FrameTimepoint.START,
                    )
                    # Reshape flat array to 4x4 matrix if necessary
                    if T_sensor_to_world_flat.ndim == 1:
                        T_sensor_to_world = T_sensor_to_world_flat.reshape(4, 4)
                    else:
                        T_sensor_to_world = T_sensor_to_world_flat

                    # Convert NCore pose to world-global space
                    T_sensor_to_world_global_batch = self._transform_poses_to_world_global(
                        T_sensor_to_world[None, :, :], self.T_world_to_world_global
                    )
                    # Ensure we have a 4x4 matrix
                    if T_sensor_to_world_global_batch.ndim == 3:
                        T_sensor_to_world_global = T_sensor_to_world_global_batch[0]
                    elif T_sensor_to_world_global_batch.ndim == 2:
                        T_sensor_to_world_global = T_sensor_to_world_global_batch.reshape(4, 4)
                    else:
                        T_sensor_to_world_global = T_sensor_to_world_global_batch

                    # Get world-to-camera (extrinsics for polyscope)
                    T_world_to_sensor = np.linalg.inv(T_sensor_to_world_global)

                    # Polyscope camera convention flip (Y and Z axes)
                    camera_convention_rot = np.array(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    T_world_to_sensor = camera_convention_rot @ T_world_to_sensor

                    # Get image (subsampled for visualization)
                    frame_image = camera_sensor.get_frame_image_array(frame_idx)
                    if self.n_val_image_subsample > 1:
                        frame_image = cv2.resize(frame_image, (w_viz, h_viz), interpolation=cv2.INTER_AREA)
                    rgb_img = frame_image.astype(np.float32) / 255.0

                    cam_list.append(
                        {
                            "ext_mat": T_world_to_sensor,
                            "w": w_viz,
                            "h": h_viz,
                            "fov_w": fov_w,
                            "fov_h": fov_h,
                            "rgb_img": rgb_img,
                            "split": self.split,
                        }
                    )
                except Exception as e:
                    # Skip frames that fail (e.g., out of bounds)
                    continue

        if cam_list:
            create_camera_visualization(cam_list)

    def get_gpu_batch_with_intrinsics(self, batch) -> Batch:
        """Convert NCore batch dict to threedgrut Batch, using GPU-cached camera-space rays.

        Camera-space rays are NOT transferred from CPU each batch. Instead they are
        looked up from the per-worker GPU cache by (camera_id, w, h). Only the per-frame
        pose (128 bytes) and RGB image are transferred to GPU.

        The renderer handles rolling shutter interpolation via T_to_world / T_to_world_end
        and the shutter_type from intrinsics. Rays are always in camera space
        (rays_in_world_space=False).
        """

        # --- Resolve camera_id -------------------------------------------------
        camera_id = batch["camera_id"]
        if isinstance(camera_id, (list, tuple)):
            camera_id = camera_id[0]

        # --- Resolve image dimensions -------------------------------------------
        h = batch["h"][0] if isinstance(batch["h"], (list, torch.Tensor)) else batch["h"]
        w = batch["w"][0] if isinstance(batch["w"], (list, torch.Tensor)) else batch["w"]
        h, w = int(h), int(w)

        # --- Look up cached camera-space rays on GPU (no CPU->GPU transfer) -----
        worker_cache = self._lazy_worker_gpu_cache()
        rays_ori, rays_dir, pixel_coords = worker_cache[(camera_id, w, h)]

        # Normalize ray directions and add batch dimension: (H, W, 3) -> (1, H, W, 3)
        rays_ori = rays_ori.unsqueeze(0)
        rays_dir = rays_dir.unsqueeze(0)

        # --- Transfer poses to GPU (small: 2x 64 bytes) -------------------------
        T_to_world = batch["T_camera_to_world"].to(self.device, non_blocking=True)
        if T_to_world.ndim == 2:
            T_to_world = T_to_world.unsqueeze(0)

        T_to_world_end = batch["T_camera_to_world_end"].to(self.device, non_blocking=True)
        if T_to_world_end.ndim == 2:
            T_to_world_end = T_to_world_end.unsqueeze(0)

        # --- Transfer RGB to GPU -------------------------------------------------
        rgb_gt = None
        if "rgb" in batch and batch["rgb"] is not None:
            rgb = batch["rgb"]
            if not isinstance(rgb, torch.Tensor):
                rgb = torch.from_numpy(rgb)
            rgb_gt = rgb.to(self.device, non_blocking=True)
            if rgb_gt.dim() == 2:
                rgb_gt = rgb_gt.unsqueeze(0)
            rgb_gt = rgb_gt.reshape(1, h, w, 3)

        # --- Build output batch dict --------------------------------------------
        batch_dict = {
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": T_to_world,
            "T_to_world_end": T_to_world_end,
            "rays_in_world_space": False,  # Always camera-space; renderer handles shutter
            "pixel_coords": pixel_coords,
        }

        if rgb_gt is not None:
            batch_dict["rgb_gt"] = rgb_gt

        # --- Intrinsics ----------------------------------------------------------
        intrinsics_result = self._get_camera_model_parameters_for_resolution(camera_id, w, h)
        if intrinsics_result is not None:
            intrinsics_params, model_type_name = intrinsics_result
            batch_dict[f"intrinsics_{model_type_name}"] = intrinsics_params

        # --- Frame / camera indices (PPISP post-processing) ----------------------
        if "frame_idx" in batch:
            frame_idx = batch["frame_idx"]
            batch_dict["frame_idx"] = frame_idx[0].item() if isinstance(frame_idx, torch.Tensor) else int(frame_idx)
        if "camera_idx" in batch:
            camera_idx = batch["camera_idx"]
            batch_dict["camera_idx"] = camera_idx[0].item() if isinstance(camera_idx, torch.Tensor) else int(camera_idx)

        return Batch(**batch_dict)

    def _get_camera_model_parameters_for_resolution(self, camera_id, render_w, render_h):
        """
        Extract camera model parameters scaled to the render resolution.

        Returns a (params_dict, model_type_name) tuple, where model_type_name is the
        Python class name of the NCore parameters object (e.g. "OpenCVPinholeCameraModelParameters").
        The caller uses model_type_name to set the correct Batch field
        (e.g. "intrinsics_OpenCVPinholeCameraModelParameters").

        Supported for OpenCV Pinhole, OpenCV Fisheye, and FTheta cameras; returns None for other models.
        """
        if camera_id is None or not hasattr(self, "sequence_camera_models"):
            return None

        # Check cache first
        cache_key = (camera_id, render_w, render_h, "scaled_params")
        if cache_key in self._camera_intrinsics_cache:
            return self._camera_intrinsics_cache[cache_key]

        for sequence_id, camera_models in self.sequence_camera_models.items():
            if camera_id not in camera_models:
                continue

            camera_model = camera_models[camera_id]

            # Get (already-downsampled) parameters and transform to the render resolution
            model_params = camera_model.get_parameters()
            downsampled_w = int(model_params.resolution[0])
            downsampled_h = int(model_params.resolution[1])
            render_scale = (render_w / downsampled_w, render_h / downsampled_h)
            scaled_params = model_params.transform(
                image_domain_scale=render_scale,
                new_resolution=(render_w, render_h),
            )

            # Use shutter type from camera parameters (renderer handles interpolation natively)
            shutter_type = scaled_params.shutter_type
            logger.info(f"[Shutter] Camera {camera_id}: shutter_type = {shutter_type.name}")

            # Build parameter dict from the properly-scaled NCore parameters.
            # Distortion coefficients are resolution-independent and preserved through transform().
            # Each camera model type produces a dict matching the tracer's expected keys.
            if isinstance(camera_model, ncore.sensors.OpenCVPinholeCameraModel):
                params_dict = {
                    "resolution": scaled_params.resolution,
                    "shutter_type": shutter_type,
                    "principal_point": scaled_params.principal_point,
                    "focal_length": scaled_params.focal_length,
                    "radial_coeffs": scaled_params.radial_coeffs,
                    "tangential_coeffs": scaled_params.tangential_coeffs,
                    "thin_prism_coeffs": scaled_params.thin_prism_coeffs,
                }
            elif isinstance(camera_model, ncore.sensors.OpenCVFisheyeCameraModel):
                params_dict = {
                    "resolution": scaled_params.resolution,
                    "shutter_type": shutter_type,
                    "principal_point": scaled_params.principal_point,
                    "focal_length": scaled_params.focal_length,
                    "radial_coeffs": scaled_params.radial_coeffs,
                    "max_angle": scaled_params.max_angle,
                }
            elif isinstance(camera_model, ncore.sensors.FThetaCameraModel):
                # Map ncore's PolynomialType enum to the repo's FThetaCameraModelParameters.PolynomialType
                # by name, since they are different enum classes with matching member names.
                reference_poly = FThetaCameraModelParameters.PolynomialType[scaled_params.reference_poly.name]
                params_dict = {
                    "resolution": scaled_params.resolution,
                    "shutter_type": shutter_type,
                    "principal_point": scaled_params.principal_point,
                    "reference_poly": reference_poly,
                    "pixeldist_to_angle_poly": scaled_params.pixeldist_to_angle_poly,
                    "angle_to_pixeldist_poly": scaled_params.angle_to_pixeldist_poly,
                    "max_angle": scaled_params.max_angle,
                    "linear_cde": scaled_params.linear_cde,
                }
            else:
                logger.warning(
                    f"Camera {camera_id}: unsupported camera model type {type(camera_model).__name__} for intrinsics extraction"
                )
                return None

            logger.info(f"[Distortion] Camera {camera_id}: Using NCore native lens distortion parameters")

            # Cache for future use
            model_type_name = type(scaled_params).__name__
            result = (params_dict, model_type_name)
            self._camera_intrinsics_cache[cache_key] = result
            return result

        return None
