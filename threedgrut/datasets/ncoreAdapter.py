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

from threedgrut.datasets.camera_models import ShutterType
from threedgrut.datasets.datasetNcore import NCoreDataset
from threedgrut.datasets.protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from threedgrut.datasets.utils import create_camera_visualization, create_pixel_coords
from threedgrut.utils.logger import logger


class NCoreDatasetAdapter(NCoreDataset, BoundedMultiViewDataset, DatasetVisualization):
    """
    Adapter for NCoreDataset to work with threedgrut framework.
    
    Key adaptations:
    - Converts NCore's rays_cam to threedgrut's rays_ori/rays_dir
    - Implements required protocol methods (get_frames_per_camera, get_gpu_batch_with_intrinsics)
    - Handles coordinate space transformations (NCore uses COLMAP_SPACE)
    """
    
    def __init__(self, *args, device="cuda", **kwargs):
        """Initialize with device parameter for GPU operations"""
        super().__init__(*args, **kwargs)
        self.device = device
        
        # Store training mode for batch reshaping
        # When sample_full_image=True, training uses full images (not patches)
        # Random ray sampling uses 1×N format
        self._sample_full_image = self.sample_full_image
        self._train_patch_size = None  # Not used for NCore (full images only)
        
        # Cache for camera intrinsics (populated on first worker init)
        self._camera_intrinsics_cache = {}
    
    def __getitem__(self, idx) -> dict:
        """Override to return dict instead of NCoreBatch for DataLoader collation."""
        ncore_batch = super().__getitem__(idx)
        
        # Convert NCoreBatch to dict with only collatable types (tensors, arrays, scalars)
        # Custom objects like RaysCamMeta and Labels can't be collated by default DataLoader
        batch_dict = {
            "rays_cam": ncore_batch.rays_cam,  # torch.Tensor (N, 6)
            "idx": ncore_batch.idx,  # int or list
        }
        
        # Extract RGB from Labels object (only collatable tensor)
        if ncore_batch.labels is not None and hasattr(ncore_batch.labels, "rgb"):
            if ncore_batch.labels.rgb is not None:
                batch_dict["rgb"] = ncore_batch.labels.rgb  # torch.Tensor (N, 3)
        
        # Add image dimensions and camera info for validation
        if ncore_batch.h is not None:
            batch_dict["h"] = ncore_batch.h
        if ncore_batch.w is not None:
            batch_dict["w"] = ncore_batch.w
        
        # Add camera ID from metadata for intrinsics lookup
        # Extract camera_id from rays_cam_meta
        if hasattr(ncore_batch, 'rays_cam_meta') and ncore_batch.rays_cam_meta is not None:
            if hasattr(ncore_batch.rays_cam_meta, 'unique_sensor_idx'):
                # Get the first ray's camera index (for training, all rays in a patch are from same camera)
                unique_sensor_idx = ncore_batch.rays_cam_meta.unique_sensor_idx
                if isinstance(unique_sensor_idx, torch.Tensor) and len(unique_sensor_idx) > 0:
                    camera_idx = int(unique_sensor_idx[0].item())
                    # Map camera index to camera_id string using the dataset's camera_ids list
                    if hasattr(self, 'camera_ids') and camera_idx < len(self.camera_ids):
                        batch_dict["camera_id"] = self.camera_ids[camera_idx]
        
        # Add camera-to-world transformations (START and END) for rolling shutter support
        if hasattr(ncore_batch, 'T_camera_to_world') and ncore_batch.T_camera_to_world is not None:
            batch_dict["T_camera_to_world"] = ncore_batch.T_camera_to_world
        if hasattr(ncore_batch, 'T_camera_to_world_end') and ncore_batch.T_camera_to_world_end is not None:
            batch_dict["T_camera_to_world_end"] = ncore_batch.T_camera_to_world_end
        
        # Pass frame_idx and camera_idx for PPISP post-processing
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
            elif hasattr(camera_model, 'focal_length'):
                fx = float(camera_model.focal_length[0])
                fy = float(camera_model.focal_length[1])
                fx_viz = fx / self.n_val_image_subsample
                fy_viz = fy / self.n_val_image_subsample
                fov_w = 2.0 * np.arctan(0.5 * w_viz / fx_viz)
                fov_h = 2.0 * np.arctan(0.5 * h_viz / fy_viz)
            else:
                logger.warning(f"Camera {camera_id}: unsupported model type for FOV calculation, skipping visualization")
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

                    # Convert NCore pose to COLMAP space
                    T_sensor_to_world_colmap_batch = self._ncore_world_to_colmap_poses(
                        T_sensor_to_world[None, :, :],
                        self.T_world_to_colmap_world
                    )
                    # Ensure we have a 4x4 matrix
                    if T_sensor_to_world_colmap_batch.ndim == 3:
                        T_sensor_to_world_colmap = T_sensor_to_world_colmap_batch[0]
                    elif T_sensor_to_world_colmap_batch.ndim == 2:
                        T_sensor_to_world_colmap = T_sensor_to_world_colmap_batch.reshape(4, 4)
                    else:
                        T_sensor_to_world_colmap = T_sensor_to_world_colmap_batch

                    # Get world-to-camera (extrinsics for polyscope)
                    T_world_to_sensor = np.linalg.inv(T_sensor_to_world_colmap)

                    # Polyscope camera convention flip (Y and Z axes)
                    camera_convention_rot = np.array([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    T_world_to_sensor = camera_convention_rot @ T_world_to_sensor

                    # Get image (subsampled for visualization)
                    try:
                        frame_image = camera_sensor.get_frame_image_array(frame_idx)
                        if self.n_val_image_subsample > 1:
                            frame_image = cv2.resize(
                                frame_image,
                                (w_viz, h_viz),
                                interpolation=cv2.INTER_LINEAR
                            )
                        rgb_img = frame_image.astype(np.float32) / 255.0
                    except:
                        # If image loading fails, create a placeholder
                        rgb_img = np.zeros((h_viz, w_viz, 3), dtype=np.float32)

                    cam_list.append({
                        "ext_mat": T_world_to_sensor,
                        "w": w_viz,
                        "h": h_viz,
                        "fov_w": fov_w,
                        "fov_h": fov_h,
                        "rgb_img": rgb_img,
                        "split": self.split,
                    })
                except Exception as e:
                    # Skip frames that fail (e.g., out of bounds)
                    continue
        
        if cam_list:
            create_camera_visualization(cam_list)
    
    def get_gpu_batch_with_intrinsics(self, batch) -> Batch:
        """
        Convert NCore batch format (dict from DataLoader) to threedgrut Batch format.
        
        Input batch (dict from DataLoader collation):
        - rays_cam: (B, N, 6) [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z] in COLMAP space
        - rgb: Optional (B, N, 3) RGB values
        - idx: batch index
        
        Output Batch:
        - rays_ori: (B, H, W, 3) ray origins in CAMERA space
        - rays_dir: (B, H, W, 3) ray directions in CAMERA space  
        - T_to_world: (B, 4, 4) camera-to-world transformation matrix (START pose)
        - T_to_world_end: (B, 4, 4) camera-to-world transformation matrix (END pose for rolling shutter)
        - rgb_gt: (B, H, W, 3) ground truth RGB
        """
        # Extract rays from NCore format (already collated by DataLoader)
        # NCore stores rays in COLMAP space (world space)
        if "rays_cam" in batch:
            rays_cam = batch["rays_cam"]  # (B, N, 6) after collation
            
            if not isinstance(rays_cam, torch.Tensor):
                rays_cam = torch.from_numpy(rays_cam)
            rays_cam = rays_cam.to(self.device, non_blocking=True)
            
            # Handle batch dimension from collation
            if rays_cam.dim() == 2:
                # Single sample: (N, 6) -> (1, N, 6)
                rays_cam = rays_cam.unsqueeze(0)
            
            # Split into origins and directions: (B, N, 6) -> (B, N, 3) each
            rays_ori_world = rays_cam[:, :, :3]  # (B, N, 3) in COLMAP/world space
            rays_dir_world = rays_cam[:, :, 3:6]  # (B, N, 3) in COLMAP/world space
            batch_size = rays_ori_world.shape[0]
            
            # Extract camera-to-world transformations (START and END for rolling shutter)
            if "T_camera_to_world" in batch and batch["T_camera_to_world"] is not None:
                T_cam_to_world = batch["T_camera_to_world"].to(self.device)
                if T_cam_to_world.ndim == 2:  # (4, 4) -> (1, 4, 4)
                    T_cam_to_world = T_cam_to_world.unsqueeze(0)
                if T_cam_to_world.shape[0] == 1 and batch_size > 1:
                    T_cam_to_world = T_cam_to_world.expand(batch_size, -1, -1)
                T_to_world = T_cam_to_world  # START pose as primary T_to_world
                
                # Extract END pose for rolling shutter
                if "T_camera_to_world_end" in batch and batch["T_camera_to_world_end"] is not None:
                    T_cam_to_world_end = batch["T_camera_to_world_end"].to(self.device)
                    if T_cam_to_world_end.ndim == 2:  # (4, 4) -> (1, 4, 4)
                        T_cam_to_world_end = T_cam_to_world_end.unsqueeze(0)
                    if T_cam_to_world_end.shape[0] == 1 and batch_size > 1:
                        T_cam_to_world_end = T_cam_to_world_end.expand(batch_size, -1, -1)
                    T_to_world_end = T_cam_to_world_end
                else:
                    # If no END pose, use START pose (global shutter)
                    T_to_world_end = T_cam_to_world
                
                # Check if rays are already in world space (rolling shutter with per-pixel poses)
                rays_in_world_space = batch.get("rays_in_world_space", False)
                
                if rays_in_world_space:
                    # Rays already in world space with correct per-pixel poses baked in
                    # Keep them as-is, renderer will receive identity transform
                    rays_ori = rays_ori_world.reshape(batch_size, -1, 3)  # (B, N, 3)
                    rays_dir = rays_dir_world.reshape(batch_size, -1, 3)  # (B, N, 3)
                    # Still need to normalize directions
                    rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
                else:
                    # Transform rays from world space to camera space
                    # NCore provides rays in world space, but renderer expects camera space
                    # Use START pose for world->camera transformation
                    T_world_to_cam = torch.inverse(T_cam_to_world)  # (B, 4, 4)
                    R = T_world_to_cam[:, :3, :3]  # (B, 3, 3) rotation
                    t = T_world_to_cam[:, :3, 3]   # (B, 3) translation
                    
                    # Transform ray origins: ori_cam = R @ ori_world + t
                    rays_ori_flat = rays_ori_world.reshape(batch_size, -1, 3)  # (B, N, 3)
                    rays_ori = torch.einsum('bnc,bdc->bnd', rays_ori_flat, R) + t.unsqueeze(1)  # (B, N, 3)
                    
                    # Transform ray directions: dir_cam = R @ dir_world (no translation)
                    rays_dir_flat = rays_dir_world.reshape(batch_size, -1, 3)  # (B, N, 3)
                    rays_dir = torch.einsum('bnc,bdc->bnd', rays_dir_flat, R)  # (B, N, 3)
                    
                    # Normalize ray directions (NCore provides non-normalized directions
                    # that include scale/depth information)
                    rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
            else:
                # Fallback: no transformation, assume rays already in camera space
                rays_ori = rays_ori_world.reshape(batch_size, -1, 3)
                rays_dir = rays_dir_world.reshape(batch_size, -1, 3)
                # Still need to normalize directions
                rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
                T_to_world = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
                T_to_world_end = T_to_world  # Same pose for fallback
            
            # Reshape to [B, H, W, 3] format
            # Check if we have image dimensions (validation or full-image training)
            if "h" in batch and "w" in batch:
                # Validation or full-image training: use provided image dimensions
                h = batch["h"][0] if isinstance(batch["h"], (list, torch.Tensor)) else batch["h"]
                w = batch["w"][0] if isinstance(batch["w"], (list, torch.Tensor)) else batch["w"]
                rays_ori = rays_ori.reshape(batch_size, h, w, 3)
                rays_dir = rays_dir.reshape(batch_size, h, w, 3)
            else:
                # Random ray sampling: use 1×N format
                num_rays = rays_ori.shape[1]
                rays_ori = rays_ori.reshape(batch_size, 1, num_rays, 3)
                rays_dir = rays_dir.reshape(batch_size, 1, num_rays, 3)
        else:
            # No camera rays, create empty tensors
            rays_ori = torch.empty((1, 1, 0, 3), dtype=torch.float32, device=self.device)
            rays_dir = torch.empty((1, 1, 0, 3), dtype=torch.float32, device=self.device)
            T_to_world = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
            T_to_world_end = T_to_world  # Same pose for no-rays case
        
        # Extract RGB (already collated by DataLoader, already 0-1 normalized)
        rgb_gt = None
        if "rgb" in batch:
            rgb = batch["rgb"]
            if not isinstance(rgb, torch.Tensor):
                rgb = torch.from_numpy(rgb)
            rgb_gt = rgb.to(self.device, non_blocking=True)
            
            # Add batch dimension if needed
            if rgb_gt.dim() == 2:
                rgb_gt = rgb_gt.unsqueeze(0)  # (1, N, 3)
            
            # Reshape to [B, H, W, 3]
            # Check if we have image dimensions (validation or full-image training)
            if "h" in batch and "w" in batch:
                # Validation or full-image training: use provided image dimensions
                h = batch["h"][0] if isinstance(batch["h"], (list, torch.Tensor)) else batch["h"]
                w = batch["w"][0] if isinstance(batch["w"], (list, torch.Tensor)) else batch["w"]
                batch_size = rgb_gt.shape[0]
                rgb_gt = rgb_gt.reshape(batch_size, h, w, 3)
            else:
                # Random ray sampling: use 1×N format
                batch_size, num_rays, _ = rgb_gt.shape
                rgb_gt = rgb_gt.reshape(batch_size, 1, num_rays, 3)
        
        # Create bernardin Batch object
        batch_dict = {
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": T_to_world,
            "T_to_world_end": T_to_world_end,  # END pose for rolling shutter
            "rays_in_world_space": batch.get("rays_in_world_space", False),  # Flag for renderer
        }
        
        if rgb_gt is not None:
            batch_dict["rgb_gt"] = rgb_gt
        
        # Compute resolution for intrinsics scaling
        if "h" in batch and "w" in batch:
            # Validation: use provided dimensions
            h = batch["h"][0] if isinstance(batch["h"], (list, torch.Tensor)) else batch["h"]
            w = batch["w"][0] if isinstance(batch["w"], (list, torch.Tensor)) else batch["w"]
            render_w, render_h = int(w), int(h)
        elif self._train_patch_size is not None:
            # Training with patches: use window_size
            render_w = render_h = self._train_patch_size
        elif rays_ori.shape[1] == 1:
            # Random rays: use ray count as width, height=1
            render_w, render_h = rays_ori.shape[2], 1
        else:
            # Use actual rendered dimensions
            render_w, render_h = rays_ori.shape[2], rays_ori.shape[1]
        
        # Extract camera intrinsics from NCore camera models
        # The renderer needs real intrinsics to project Gaussians onto the image plane
        # Rays are in world space, but Gaussians are projected using camera intrinsics
        
        # Get camera ID for intrinsics lookup
        camera_id = None
        if "camera_id" in batch:
            camera_id = batch["camera_id"]
            if isinstance(camera_id, (list, tuple)) and len(camera_id) > 0:
                camera_id = camera_id[0]
            elif isinstance(camera_id, torch.Tensor):
                camera_id = camera_id[0].item() if camera_id.numel() > 0 else None
        
        # Try to get full camera model parameters with proper resolution.
        # Intrinsics are required by rasterization-based renderers (3DGUT) but not by
        # ray-traced renderers (3DGRT) which only use pre-computed rays.
        # For unsupported camera models (e.g. FTheta), intrinsics extraction is skipped.
        intrinsics_result = self._get_camera_model_parameters_for_resolution(camera_id, render_w, render_h)
        if intrinsics_result is not None:
            intrinsics_params, model_type_name = intrinsics_result
            batch_dict[f"intrinsics_{model_type_name}"] = intrinsics_params
        
        # Pass frame_idx and camera_idx for PPISP post-processing
        if "frame_idx" in batch:
            frame_idx = batch["frame_idx"]
            batch_dict["frame_idx"] = frame_idx[0].item() if isinstance(frame_idx, torch.Tensor) else int(frame_idx)
        if "camera_idx" in batch:
            camera_idx = batch["camera_idx"]
            batch_dict["camera_idx"] = camera_idx[0].item() if isinstance(camera_idx, torch.Tensor) else int(camera_idx)
        
        # Generate pixel coordinates for post-processing (PPISP)
        batch_dict["pixel_coords"] = create_pixel_coords(render_w, render_h, device=self.device)
        
        return Batch(**batch_dict)
    
    def _get_camera_model_parameters_for_resolution(self, camera_id, render_w, render_h):
        """
        Extract camera model parameters scaled to the render resolution.

        Returns a (params_dict, model_type_name) tuple, where model_type_name is the
        Python class name of the NCore parameters object (e.g. "OpenCVPinholeCameraModelParameters").
        The caller uses model_type_name to set the correct Batch field
        (e.g. "intrinsics_OpenCVPinholeCameraModelParameters").

        Only supported for OpenCV Pinhole/Fisheye cameras; returns None for other models (e.g. FTheta).
        """
        if camera_id is None or not hasattr(self, 'sequence_camera_models'):
            return None

        # Check cache first
        cache_key = (camera_id, render_w, render_h, 'scaled_params')
        if cache_key in self._camera_intrinsics_cache:
            return self._camera_intrinsics_cache[cache_key]

        for sequence_id, camera_models in self.sequence_camera_models.items():
            if camera_id not in camera_models:
                continue

            camera_model = camera_models[camera_id]

            # Only OpenCV Pinhole/Fisheye models are currently supported by the rasterization renderer.
            if isinstance(camera_model, ncore.sensors.FThetaCameraModel):
                return None

            # Get (already-downsampled) parameters and transform to the render resolution
            model_params = camera_model.get_parameters()
            downsampled_w = int(model_params.resolution[0])
            downsampled_h = int(model_params.resolution[1])
            render_scale = (render_w / downsampled_w, render_h / downsampled_h)
            scaled_params = model_params.transform(
                image_domain_scale=render_scale,
                new_resolution=(render_w, render_h),
            )

            # Determine shutter type
            if self.force_global_shutter:
                shutter_type = ShutterType.GLOBAL
                logger.info(f"[Shutter] Camera {camera_id}: Using GLOBAL shutter (force_global_shutter=True)")
            else:
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
            else:
                logger.warning(f"Camera {camera_id}: unsupported camera model type {type(camera_model).__name__} for intrinsics extraction")
                return None

            logger.info(f"[Distortion] Camera {camera_id}: Using NCore native lens distortion parameters")

            # Cache for future use
            model_type_name = type(scaled_params).__name__
            result = (params_dict, model_type_name)
            self._camera_intrinsics_cache[cache_key] = result
            return result

        return None
    

