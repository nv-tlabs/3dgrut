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

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator, NamedTuple, Optional

import cv2
import ncore.data
import ncore.data.v4
import ncore.sensors
import numpy as np
import numpy.typing as npt
import simplejpeg
import torch
import torch.utils.data
from ncore.impl.common.transformations import HalfClosedInterval
from scipy import ndimage

from threedgrut.datasets.protocols import (
    Batch,
    BoundedMultiViewDataset,
    DatasetVisualization,
)
from threedgrut.datasets.utils import (
    PointCloud,
    create_camera_visualization,
    create_pixel_coords,
    get_center_and_diag,
    get_worker_id,
)
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import to_torch


class NCoreDataset(torch.utils.data.Dataset, BoundedMultiViewDataset, DatasetVisualization):

    def __init__(
        self,
        datapath: str,
        seek_offset_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        split: str = "train",
        # Sensors
        camera_ids: list[str] | None = None,
        lidar_ids: list[str] | None = None,
        # Misc
        device: str = "cuda",
        downsample: float = 1.0,  # Training image downsample factor (0.5 = half resolution)
        sample_full_image: bool = True,
        window_size: int = 256,
        n_samples_per_epoch: int = 1000,
        n_train_sample_timepoints: int = 1,
        n_train_sample_camera_rays: int = 4096,
        n_val_image_subsample: int = 4,
        val_frame_interval: int = 8,  # Every Nth frame is validation (like COLMAP test_split_interval)
        n_camera_mask_dilation_iterations: int = 30,
        open_consolidated=True,
        camera_max_fov_deg: float = 190.0,
        # V4 component group names
        poses_component_group: str = "default",
        intrinsics_component_group: str = "default",
        masks_component_group: str = "default",
        # JPEG decoding options
        jpeg_backend_cpu: str = "simplejpeg",  # "simplejpeg" (fast, libjpeg-turbo) or "PIL" (fallback)
        simplejpeg_fastdct: bool = False,  # ~4-5% faster decode, minor quality loss
        simplejpeg_fastupsample: bool = False,  # ~4-5% faster chroma upsampling, minor quality loss
        # Lidar point cloud color data name (if available)
        lidar_color_generic_data_name: str = "rgb",
    ):
        super().__init__()

        self.device = device

        # Cache for camera intrinsics (populated on first worker init)
        self._camera_intrinsics_cache: dict = {}

        # Per-worker GPU caches: camera-space rays, pixel coordinates
        # Populated lazily on first use in each DataLoader worker process.
        # Keys: worker_id -> dict[camera_id -> (rays_ori, rays_dir, pixel_coords)]
        self._worker_gpu_cache: dict = {}

        # JPEG decoding parameters
        self.jpeg_backend_cpu: str = jpeg_backend_cpu
        self.simplejpeg_fastdct: bool = simplejpeg_fastdct
        self.simplejpeg_fastupsample: bool = simplejpeg_fastupsample

        # Lidar point cloud color data name
        self.lidar_color_generic_data_name: str = lidar_color_generic_data_name

        # store relevant parameters from config
        self.split: str = split
        self.downsample: float = downsample
        self.n_samples_per_epoch: int = n_samples_per_epoch
        self.n_train_sample_timepoints: int = n_train_sample_timepoints
        self.n_train_sample_camera_rays: int = n_train_sample_camera_rays

        self.sample_full_image: bool = sample_full_image
        self.window_size: int = window_size

        # sample full image:
        if self.sample_full_image:
            assert self.n_train_sample_timepoints == 1
            self.n_train_sample_camera_rays = self.window_size**2

        self.n_val_image_subsample: int = n_val_image_subsample
        self.val_frame_interval: int = val_frame_interval  # Frame-level train/val split

        self.n_camera_mask_dilation_iterations: int = n_camera_mask_dilation_iterations

        self.camera_max_fov_deg: float = camera_max_fov_deg

        # V4 component group names
        self.poses_component_group: str = poses_component_group
        self.intrinsics_component_group: str = intrinsics_component_group
        self.masks_component_group: str = masks_component_group

        self.init_camera_ids: list[str] | None = camera_ids
        self.init_lidar_ids: list[str] | None = lidar_ids

        self.open_consolidated: bool = open_consolidated

        self.split: str = split

        # load single-sequence NCore V4 meta-file
        assert (path := Path(datapath)).is_file(), f"NCoreDataset: provided path {path} not a file"
        with open(path, "r") as fp:
            try:
                dataset_meta = json.load(fp)
            except ValueError as e:
                raise ValueError(f"NCoreDataset: provided file {path} not a json file")

        assert all(
            (
                key in dataset_meta
                for key in ("sequence_id", "sequence_timestamp_interval_us", "version", "component_stores")
            )
        ), f"NCoreDataset: provided json file {path} not a NCore V4 single-sequence file"

        time_start = dataset_meta["sequence_timestamp_interval_us"]["start"]
        time_stop = dataset_meta["sequence_timestamp_interval_us"]["stop"]

        if seek_offset_sec:
            time_start += int(seek_offset_sec * 1e6)
        # duration_sec = -1 means "use all available frames"
        if duration_sec is not None and duration_sec > 0:
            time_stop = min(time_start + int(duration_sec * 1e6), time_stop)

        chunk_time_range_us = HalfClosedInterval(time_start, time_stop)

        self.sequence_id: str = dataset_meta["sequence_id"]
        self.sequence_meta_file_path: Path = path
        self.time_range_us: HalfClosedInterval = chunk_time_range_us

        # initialize sequence loaders
        self.sequence_loaders: dict[str, ncore.data.SequenceLoaderProtocol] = {}
        self.worker_id: Optional[int] = None
        self._init_worker()

    def _init_worker(self) -> None:
        """Re-initialize worker if IDs changes, making sure data resources are not shared across workers"""

        # Determine current worker / process and whether re-initialization is necessary
        match torch.utils.data.get_worker_info():
            case None:
                # main process case
                if self.sequence_loaders and self.worker_id is None:
                    # skip re-initialization
                    return
                # in case we move back from a worker to the main process we should also re-init
                self.worker_id = None
                self.rng = np.random.default_rng(seed=0)  # deterministic sampling
            case torch.utils.data._utils.worker.WorkerInfo(id=worker_id, seed=worker_seed):  # type: ignore
                # worker process case
                if self.sequence_loaders and self.worker_id is worker_id:
                    # skip re-initialization
                    return
                self.worker_id = worker_id
                self.rng = np.random.default_rng(seed=worker_seed)  # non-deterministic sampling

        # Reload case: check if reloading is sufficient for initialization
        if self.sequence_loaders:
            # only reload data loaders
            for loader in self.sequence_loaders.values():
                loader.reload_resources()

            return

        # Full initial load case
        class UniqueSensorId(NamedTuple):
            """Represents a unique sensor ID along with its unique index for a given sensor type"""

            id: str
            idx: int

        self.n_unique_cameras = 0
        self.camera_unique_ids: dict[str, list[UniqueSensorId]] = defaultdict(list)
        self.sequence_camera_sensors: dict[str, dict[str, ncore.data.CameraSensorProtocol]] = {}
        self.sequence_camera_models: dict[str, dict[str, ncore.sensors.CameraModel]] = {}
        self.sequence_camera_unique_ids: dict[str, dict[str, UniqueSensorId]] = {}
        self.sequence_cameras_all_pixels: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self.sequence_cameras_all_rays: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self.sequence_cameras_valid_pixels_ego_masks: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self.sequence_cameras_frame_valid_pixels_masks: dict[str, dict[str, dict[int, np.ndarray]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.sequence_cameras_pixels_subsample: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        self.sequence_cameras_rays_subsample: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

        self.n_unique_lidars = 0
        self.lidar_unique_ids: dict[str, list[UniqueSensorId]] = defaultdict(list)
        self.sequence_lidar_sensors: dict[str, dict[str, ncore.data.LidarSensorProtocol]] = {}
        self.sequence_lidar_unique_ids: dict[str, dict[str, UniqueSensorId]] = {}

        sequence_id = self.sequence_id

        # Construct V4 sequence loader — SequenceComponentGroupsReader handles
        # expansion of the meta JSON file to component store paths internally.
        sequence_loader = self.sequence_loaders[sequence_id] = ncore.data.v4.SequenceLoaderV4(
            ncore.data.v4.SequenceComponentGroupsReader(
                [self.sequence_meta_file_path], open_consolidated=self.open_consolidated
            ),
            poses_component_group_name=self.poses_component_group,
            intrinsics_component_group_name=self.intrinsics_component_group,
            masks_component_group_name=self.masks_component_group,
        )

        # Auto-detect _single_ sensors if not specified - sensors need to be specified explicitly
        # to avoid ambiguity (e.g., in case of multiple downscaled sensors)
        self.camera_ids: list[str]
        if self.init_camera_ids is None:
            self.camera_ids = sequence_loader.camera_ids
            if len(self.camera_ids) > 1:
                raise ValueError(
                    "NCoreDataset: Multiple camera sensors in dataset, explicit"
                    f" specification of camera sensors required to avoid ambiguity: {self.camera_ids}"
                )
            logger.info(f"Auto-detected camera: {self.camera_ids}")
        else:
            self.camera_ids = self.init_camera_ids
            assert all(
                isinstance(cid, str) for cid in self.camera_ids
            ), f"NCoreDataset: camera_ids should be a list of strings, got {self.camera_ids}"
            logger.info(f"Using cameras: {self.camera_ids}")

        self.lidar_ids: list[str]
        if self.init_lidar_ids is None:
            self.lidar_ids = sequence_loader.lidar_ids
            if len(self.lidar_ids) > 1:
                raise ValueError(
                    "NCoreDataset: Multiple lidar sensors in dataset, explicit"
                    f" specification of lidar sensors required to avoid ambiguity: {self.lidar_ids}"
                )
            logger.info(f"Auto-detected lidar: {self.lidar_ids}")
        else:
            self.lidar_ids = self.init_lidar_ids
            assert all(
                isinstance(lid, str) for lid in self.lidar_ids
            ), f"NCoreDataset: lidar_ids should be a list of strings, got {self.lidar_ids}"
            logger.info(f"Using lidars: {self.lidar_ids}")

        # Load camera sensors
        camera_sensors = self.sequence_camera_sensors[sequence_id] = {
            camera_id: sequence_loader.get_camera_sensor(camera_id) for camera_id in self.camera_ids
        }

        # Load camera models
        camera_models = self.sequence_camera_models[sequence_id] = {}
        for camera_id in self.camera_ids:
            model_params = camera_sensors[camera_id].model_parameters

            # Scale camera model parameters for downsampled images
            if self.downsample < 1.0:
                model_params = model_params.transform(image_domain_scale=self.downsample)

            camera_model = ncore.sensors.CameraModel.from_parameters(model_params, device="cpu", dtype=torch.float32)

            camera_models[camera_id] = camera_model

        # Log per-camera image resolutions
        logger.info(f"NCoreDataset [{self.split}] sequence '{sequence_id}' per-camera image resolutions:")
        for camera_id, camera_model in camera_models.items():
            width = int(camera_model.resolution[0].item())
            height = int(camera_model.resolution[1].item())
            logger.info(f"  {camera_id}: {width}x{height}")

        # Construct unique camera instance ids and indices
        self.sequence_camera_unique_ids[sequence_id] = {
            camera_id: UniqueSensorId("@".join((camera_id, sequence_id)), camera_instance_idx)
            for camera_instance_idx, camera_id in enumerate(self.camera_ids, self.n_unique_cameras)
        }
        self.n_unique_cameras += len(self.camera_ids)
        for camera_id in self.camera_ids:
            self.camera_unique_ids[camera_id].append(self.sequence_camera_unique_ids[sequence_id][camera_id])

        # Restrict effective FOV of omnidirectional cameras
        for camera_model in camera_models.values():
            if not isinstance(camera_model, (ncore.sensors.FThetaCameraModel, ncore.sensors.OpenCVFisheyeCameraModel)):
                continue
            camera_model.max_angle = min(np.deg2rad(self.camera_max_fov_deg) / 2.0, camera_model.max_angle)

        # Determine all pixels and rays per camera
        for camera_id in self.camera_ids:
            camera_sensor = camera_sensors[camera_id]
            camera_model = camera_models[camera_id]

            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())

            camera_pixels_x, camera_pixels_y = np.meshgrid(np.arange(w, dtype=np.int16), np.arange(h, dtype=np.int16))

            self.sequence_cameras_all_pixels[sequence_id] |= {
                camera_id: np.stack([camera_pixels_x.flatten(), camera_pixels_y.flatten()], axis=1)
            }

            if self.n_val_image_subsample > 1:
                assert (
                    w % self.n_val_image_subsample == 0 and h % self.n_val_image_subsample == 0
                ), f"NCoreDataset: Validation subsample factor {self.n_val_image_subsample} invalid for camera {camera_id} with resolution {w}x{h}"
                camera_pixels_x_subsample, camera_pixels_y_subsample = np.meshgrid(
                    np.arange(start=0, stop=w, step=self.n_val_image_subsample, dtype=np.int16),
                    np.arange(start=0, stop=h, step=self.n_val_image_subsample, dtype=np.int16),
                )
            else:
                camera_pixels_x_subsample, camera_pixels_y_subsample = camera_pixels_x, camera_pixels_y

            self.sequence_cameras_pixels_subsample[sequence_id] |= {
                camera_id: np.stack([camera_pixels_x_subsample.flatten(), camera_pixels_y_subsample.flatten()], axis=1)
            }

            # Statically unmasked pixels (ego mask)
            if camera_mask_image := camera_sensor.get_mask_images().get("ego"):
                camera_mask_array = np.asarray(camera_mask_image.convert("L")) != 0
                camera_mask_array = ndimage.binary_dilation(
                    camera_mask_array, iterations=self.n_camera_mask_dilation_iterations
                )
                camera_valid_pixels_ego_mask = np.logical_not(camera_mask_array)
            else:
                camera_valid_pixels_ego_mask = np.ones(
                    (int(camera_model.resolution[1].item()), int(camera_model.resolution[0].item())), dtype=bool
                )
            self.sequence_cameras_valid_pixels_ego_masks[sequence_id] |= {camera_id: camera_valid_pixels_ego_mask}

            # Precompute all rays
            h_rays = h
            w_rays = w
            self.sequence_cameras_all_rays[sequence_id] |= {
                camera_id: camera_model.pixels_to_camera_rays(self.sequence_cameras_all_pixels[sequence_id][camera_id])
                .reshape(h_rays, w_rays, 3)
                .numpy()
            }

            self.sequence_cameras_rays_subsample[sequence_id] |= {
                camera_id: camera_model.pixels_to_camera_rays(
                    self.sequence_cameras_pixels_subsample[sequence_id][camera_id]
                ).numpy()
            }

        # Pre-compute per-camera resolutions for the current split (used for GPU ray cache lookup)
        self._camera_resolutions: dict[str, tuple[int, int]] = {}
        for camera_id in self.camera_ids:
            camera_model = camera_models[camera_id]
            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())
            if not self.split.startswith("train") and self.n_val_image_subsample > 1:
                w = w // self.n_val_image_subsample
                h = h // self.n_val_image_subsample
            self._camera_resolutions[camera_id] = (w, h)

        # Load lidar sensors
        self.sequence_lidar_sensors[sequence_id] = {
            lidar_id: sequence_loader.get_lidar_sensor(lidar_id) for lidar_id in self.lidar_ids
        }

        # Construct unique lidar instance ids and indices
        self.sequence_lidar_unique_ids[sequence_id] = {
            lidar_id: UniqueSensorId("@".join((lidar_id, sequence_id)), lidar_instance_idx)
            for lidar_instance_idx, lidar_id in enumerate(self.lidar_ids, self.n_unique_lidars)
        }
        self.n_unique_lidars += len(self.lidar_ids)
        for lidar_id in self.lidar_ids:
            self.lidar_unique_ids[lidar_id].append(self.sequence_lidar_unique_ids[sequence_id][lidar_id])

        # Determine linear per-sensor-frame index ranges depending on dataset time restrictions,
        # making sure *both* frame start and end-times are fully covered
        def get_sensor_frame_range(frames_timestamps_us: np.ndarray) -> range:
            # make sure end-of-frame times are are covered by the time range
            cover_range = self.time_range_us.cover_range(frames_timestamps_us[:, ncore.data.FrameTimepoint.END])
            # make sure that the first frame's start-of-frame time is also covered - skip frames as required
            # (could be more than a single frame if frame ranges are not exclusively partitioning the time range)
            while len(cover_range) and (
                int(frames_timestamps_us[cover_range.start, ncore.data.FrameTimepoint.START]) not in self.time_range_us
            ):
                cover_range = cover_range[1:]

            return cover_range

        self.camera_frame_ranges: dict[str, range] = {
            camera_id: get_sensor_frame_range(self.sequence_camera_sensors[sequence_id][camera_id].frames_timestamps_us)
            for camera_id in self.camera_ids
        }
        self.lidar_frame_ranges: dict[str, range] = {
            lidar_id: get_sensor_frame_range(self.sequence_lidar_sensors[sequence_id][lidar_id].frames_timestamps_us)
            for lidar_id in self.lidar_ids
        }

        # Pre-compute train/val frame lists for unbiased sampling (frame-level split)
        self.camera_train_frame_indices: dict[str, np.ndarray] = {}
        self.camera_val_frame_indices: dict[str, np.ndarray] = {}
        frames_per_camera_dict: dict[str, int] = {}
        for camera_id, camera_frame_range in self.camera_frame_ranges.items():
            all_frames = np.arange(camera_frame_range.start, camera_frame_range.stop, dtype=np.int32)

            # Validation: every val_frame_interval-th frame (e.g., 0, 8, 16, 24, ...)
            val_mask = (all_frames % self.val_frame_interval) == 0
            self.camera_val_frame_indices[camera_id] = all_frames[val_mask]

            # Training: all other frames (e.g., 1-7, 9-15, 17-23, ...)
            self.camera_train_frame_indices[camera_id] = all_frames[~val_mask]

            if self.split == "train":
                frames_per_camera_dict[camera_id] = len(self.camera_train_frame_indices[camera_id])
            else:
                frames_per_camera_dict[camera_id] = len(self.camera_val_frame_indices[camera_id])

        # Log temporal window and frame counts
        temporal_start_sec = self.time_range_us.start / 1e6
        temporal_end_sec = self.time_range_us.stop / 1e6
        temporal_duration_sec = (self.time_range_us.stop - self.time_range_us.start) / 1e6
        logger.info(
            f"NCoreDataset [{self.split}] - Temporal window: {temporal_start_sec:.2f}s to {temporal_end_sec:.2f}s (duration: {temporal_duration_sec:.2f}s)"
        )

        total_frames = sum(frames_per_camera_dict.values())
        logger.info(f"NCoreDataset [{self.split}] frame counts (after temporal filtering):")
        for camera_id, count in frames_per_camera_dict.items():
            logger.info(f"  {camera_id}: {count} frames")
        logger.info(f"  Total: {total_frames} frames")

        # Store total training frame count for __len__
        self._n_train_frames = total_frames if self.split.startswith("train") else 0

        # Compute camera-blocked linear start frame indices (cumulative offsets per camera)
        self.camera_linear_start_frame_indices: dict[str, int] = {}
        cumulative_offset = 0
        for camera_id in self.camera_ids:
            self.camera_linear_start_frame_indices[camera_id] = cumulative_offset
            cumulative_offset += frames_per_camera_dict[camera_id]

        # Compute world-to-world_global transformation from pose graph
        world_world_global_edge = sequence_loader.pose_graph.get_edge("world", "world_global")
        if world_world_global_edge is not None:
            T_world_base = world_world_global_edge.T_source_target
        else:
            T_world_base = np.eye(4, dtype=np.float64)
        self.T_world_common_world_base: npt.NDArray[np.float64] = T_world_base
        T_world_base_world_common = np.linalg.inv(T_world_base).astype(np.float32)
        self.T_world_to_world_global: np.ndarray = T_world_base_world_common

        # Compute per-frame valid pixels from static ego-car masks
        for camera_id, camera_frame_range in self.camera_frame_ranges.items():
            camera_valid_pixels_ego_mask = self.sequence_cameras_valid_pixels_ego_masks[sequence_id][camera_id]
            for camera_frame_idx in camera_frame_range:
                self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id] |= {
                    camera_frame_idx: camera_valid_pixels_ego_mask.copy()
                }

        # Compute the first frame timestamp for relative time computation
        self.first_frame_timestamp_us = None
        for camera_id in self.camera_ids:
            camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
            camera_frame_range = self.camera_frame_ranges[camera_id]
            if len(camera_frame_range) > 0:
                first_frame_ts = camera_sensor.get_frame_timestamp_us(camera_frame_range.start)
                if self.first_frame_timestamp_us is None or first_frame_ts < self.first_frame_timestamp_us:
                    self.first_frame_timestamp_us = first_frame_ts

        if self.first_frame_timestamp_us is None:
            self.first_frame_timestamp_us = 0

    def get_camera_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor ids"""
        if unique_sensors:
            return [unique_id.id for unique_ids in self.camera_unique_ids.values() for unique_id in unique_ids]
        else:
            return self.camera_ids

    def get_n_frames_per_camera(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of split-specific frame counts per camera sensor instance.

        For training split: returns training-only frame counts (excluding validation frames).
        For validation split: returns validation-only frame counts (excluding training frames).
        """
        self._init_worker()  # make sure worker is initialized at this point

        frame_indices = (
            self.camera_train_frame_indices if self.split.startswith("train") else self.camera_val_frame_indices
        )
        return np.array(
            [len(frame_indices[camera_id]) for camera_id in self.camera_ids],
            dtype=np.int32,
        )

    def _get_camera_centers(self, camera_id: str, frame_indices: np.ndarray) -> np.ndarray:
        """Return camera centers in world-global space for the given frame indices."""
        sequence_id = self.sequence_id
        camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]

        camera_centers = self._transform_poses_to_world_global(
            camera_sensor.get_frames_T_source_target(
                source_node=camera_sensor.sensor_id,
                target_node="world",
                frame_indices=frame_indices,
                frame_timepoint=ncore.data.FrameTimepoint.START,
            ),
            self.T_world_to_world_global,
        )[:, :3, 3]

        return camera_centers

    def get_observer_points(self, camera_id=None):
        """Return camera centers in world-global space for the current split (train or val).

        For the training split, returns only training frame camera centers
        (excluding validation frames), matching the behavior of ColmapDataset.
        """
        self._init_worker()

        assert len(self.camera_ids), "NCoreDataset: no camera sensors loaded"
        camera_id = self.camera_ids[0] if camera_id is None else camera_id

        # Use split-appropriate frame indices (train-only or val-only)
        frame_indices = (
            self.camera_train_frame_indices[camera_id]
            if self.split.startswith("train")
            else self.camera_val_frame_indices[camera_id]
        )
        return self._get_camera_centers(camera_id, frame_indices)

    def get_scene_extent(self):
        """Compute scene extent from ALL camera centers (both train and val).

        This matches ColmapDataset behavior where cameras_extent is computed
        from all frames before the train/val split.
        """
        self._init_worker()

        assert len(self.camera_ids), "NCoreDataset: no camera sensors loaded"
        camera_id = self.camera_ids[0]

        # Use all frame indices (train + val) for scene extent, matching ColmapDataset
        all_frame_indices = np.sort(
            np.concatenate(
                [
                    self.camera_train_frame_indices[camera_id],
                    self.camera_val_frame_indices[camera_id],
                ]
            )
        )
        all_camera_centers = self._get_camera_centers(camera_id, all_frame_indices)

        _, diagonal = get_center_and_diag(all_camera_centers)
        cameras_extent = diagonal * 1.1
        return cameras_extent

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tuple of vec3 (min,max)"""
        # reference implementation
        # pc = PointCloud.from_sequence(list(train_dataset.get_point_clouds(step_frame=10, non_dynamic_points_only=True)), device="cpu")
        # # Scene extend from bbox of point-cloud
        # scene_bbox = (pc.xyz_end.min(0).values, pc.xyz_end.max(0).values)
        # scene_extent = torch.linalg.norm(scene_bbox[1] - scene_bbox[0])
        camera_origins = torch.tensor(self.get_observer_points())
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return (bbox_min, bbox_max)

    def __len__(self) -> int:
        """Returns the total number of samples provided by the dataset (depending on split type and parametrization)"""
        # make sure worker is initialized at this point
        self._init_worker()

        if self.split.startswith("train"):
            return self._n_train_frames

        # Validation: count validation frames across all cameras
        return sum(len(self.camera_val_frame_indices[camera_id]) for camera_id in self.camera_ids)

    def _transform_poses_to_world_global(
        self,
        T_poses_world: np.ndarray,
        T_world_to_world_global: np.ndarray,
    ) -> np.ndarray:
        """
        Transform input poses 'T_poses_world' in NCore world frame (metric units) to
        poses in world-global frame by applying the world-to-world_global coordinate system
        transformation (given by 'T_world_to_world_global' (4,4)).

        Supports both singular (4,4) and batched (N,4,4) input poses.
        """

        T_poses_world_global = T_world_to_world_global @ T_poses_world.reshape((-1, 4, 4))  # (N,4,4)

        return T_poses_world_global.squeeze()

    def _get_start_end_poses_world_global(
        self,
        camera_sensor: ncore.sensors.CameraSensor,
        camera_frame_index: int,
    ) -> np.ndarray:
        """Get start/end poses in world-global frame for a given camera frame.

        Returns (2, 4, 4) array with [start_pose, end_pose].
        """
        return self._transform_poses_to_world_global(
            camera_sensor.get_frames_T_source_target(
                source_node=camera_sensor.sensor_id,
                target_node="world",
                frame_indices=np.array(camera_frame_index),
                frame_timepoint=None,  # both start and end -> (2,4,4)
            ),
            self.T_world_to_world_global,
        )

    def _compute_frame_time_ms(
        self,
        camera_sensor: ncore.sensors.CameraSensor,
        camera_frame_index: int,
    ) -> float:
        """Compute frame time in milliseconds relative to the first frame."""
        frame_timestamp_us = camera_sensor.get_frame_timestamp_us(camera_frame_index)
        return float((frame_timestamp_us - self.first_frame_timestamp_us) / 1000.0)

    def _decode_jpeg_bytes(self, encoded: bytes, target_width: int, target_height: int) -> np.ndarray:
        """Decode JPEG bytes with simplejpeg, applying downscale during decode via min_width/min_height.

        Args:
            encoded: Raw JPEG-encoded image bytes.
            target_width: Desired output width (used as min_width for simplejpeg).
            target_height: Desired output height (used as min_height for simplejpeg).

        Returns:
            Decoded RGB image as uint8 numpy array at approximately the target resolution.
        """
        return simplejpeg.decode_jpeg(
            encoded,
            fastdct=self.simplejpeg_fastdct,
            fastupsample=self.simplejpeg_fastupsample,
            min_width=target_width,
            min_height=target_height,
        )

    def _decode_image(
        self,
        camera_sensor: ncore.data.CameraSensorProtocol,
        frame_index: int,
    ) -> np.ndarray:
        """Decode and downscale a camera frame image.

        Uses pre-loaded cache if available, otherwise decodes from disk.
        When jpeg_backend_cpu="simplejpeg" and the image is JPEG-encoded, uses
        libjpeg-turbo for faster decoding with optional downscale during decode.
        Falls back to PIL-based decoding + cv2.resize otherwise.

        Args:
            camera_sensor: The ncore camera sensor to read from.
            frame_index: The frame index to decode.

        Returns:
            Decoded RGB image as uint8 numpy array at the target (downsampled) resolution.
        """
        camera_model = self.sequence_camera_models[self.sequence_id][camera_sensor.sensor_id]
        target_w = int(camera_model.resolution[0].item())
        target_h = int(camera_model.resolution[1].item())

        # Try simplejpeg fast path for JPEG images
        if self.jpeg_backend_cpu == "simplejpeg":
            try:
                image_data = camera_sensor.get_frame_handle(frame_index).get_data()
                if image_data.get_encoded_image_format().lower() in ("jpeg", "jpg"):
                    encoded = image_data.get_encoded_image_data()
                    return self._decode_jpeg_bytes(encoded, target_w, target_h)
            except Exception:
                pass  # fall through to PIL path

        # PIL fallback: full-resolution decode + optional cv2 resize
        frame_image_array = camera_sensor.get_frame_image_array(frame_index)
        if self.downsample < 1.0:
            frame_image_array = cv2.resize(frame_image_array, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return frame_image_array

    @torch.cuda.nvtx.range("ncore_dataset::_getitem")
    def __getitem__(self, idx) -> dict:
        """Returns a specific sample of the dataset (depending on split type and parametrization).

        Returns a plain dict suitable for PyTorch DataLoader default collation.
        Camera-space rays are NOT included — they are cached on GPU per worker and
        looked up by camera_id in get_gpu_batch_with_intrinsics.
        """
        # make sure worker is initialized
        self._init_worker()

        if self.split.startswith("train"):
            sequence_id = self.sequence_id

            if self.sample_full_image:
                # Select one random camera from available cameras
                valid_camera_ids = [self.rng.choice(self.camera_ids)]
            else:
                valid_camera_ids = self.camera_ids

            sampled_camera_id: Optional[str] = None
            global_frame_idx: int = idx
            rgb: Optional[torch.Tensor] = None

            # Iterate over selected cameras (typically just one for full-image training)
            for camera_id in valid_camera_ids:
                sampled_camera_id = camera_id

                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_all_pixels = self.sequence_cameras_all_pixels[sequence_id][camera_id]

                # Get pre-computed training frame list for unbiased sampling
                train_frames = self.camera_train_frame_indices[camera_id]

                if len(train_frames) == 0:
                    continue

                # Randomly sample a frame index directly from training frames list
                closest_idx = self.rng.integers(0, len(train_frames))
                camera_frame_index = train_frames[closest_idx]

                # Compact, 0-based, contiguous training frame index (camera-blocked)
                global_frame_idx = self.camera_linear_start_frame_indices[camera_id] + closest_idx

                # Decode image (with accelerated JPEG decoding + downscaling if available)
                frame_image_array = self._decode_image(camera_sensor, camera_frame_index)

                # Sample ALL pixels from the image (full image training)
                pixel_samples = camera_all_pixels
                rgb = to_torch(
                    frame_image_array[pixel_samples[:, 1], pixel_samples[:, 0]].astype(np.float32) / 255.0,
                    device="cpu",
                )

                # Get start/end poses in world-global frame (renderer handles shutter interpolation)
                T_sensor_startend = self._get_start_end_poses_world_global(camera_sensor, camera_frame_index)

            frame_time_ms = self._compute_frame_time_ms(camera_sensor, camera_frame_index)
            camera_index = self.camera_ids.index(sampled_camera_id)

            batch_dict: dict[str, Any] = {
                "T_camera_to_world": to_torch(T_sensor_startend[0], device="cpu"),
                "T_camera_to_world_end": to_torch(T_sensor_startend[1], device="cpu"),
                "camera_id": sampled_camera_id,
                "frame_idx": global_frame_idx,
                "camera_idx": camera_index,
            }

            if rgb is not None:
                batch_dict["rgb"] = rgb

            return batch_dict

        else:
            # Decode *linear* global sample index, considering only every Nth frame (validation)
            # This implements frame-level train/val split (like COLMAP)
            sequence_id = self.sequence_id
            run_frames = 0
            for camera_id in self.camera_ids:
                val_frames_in_range = self.camera_val_frame_indices[camera_id]

                if idx >= run_frames + len(val_frames_in_range):
                    # current camera depleted, check next one
                    run_frames += len(val_frames_in_range)
                    continue

                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_model = self.sequence_camera_models[sequence_id][camera_id]
                camera_pixels_subsampled = self.sequence_cameras_pixels_subsample[sequence_id][camera_id]

                # determine frame of current camera (from validation frame list)
                val_frame_list_idx = idx - run_frames
                camera_frame_index = val_frames_in_range[val_frame_list_idx]

                # Decode image (with accelerated JPEG decoding + downscaling if available)
                frame_image_array = self._decode_image(camera_sensor, camera_frame_index)

                # Get full camera model resolution for mask resizing
                w_full = int(camera_model.resolution[0].item())
                h_full = int(camera_model.resolution[1].item())

                # sample image colors at pixel centers
                rgb = (
                    frame_image_array[camera_pixels_subsampled[:, 1], camera_pixels_subsampled[:, 0]].astype(np.float32)
                    / 255.0
                )

                # sample valid pixel mask
                frame_valid_pixel_mask = self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][
                    camera_frame_index
                ]

                if self.downsample < 1.0:
                    frame_valid_pixel_mask = cv2.resize(
                        frame_valid_pixel_mask.astype(np.uint8),
                        (w_full, h_full),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                valid = frame_valid_pixel_mask[camera_pixels_subsampled[:, 1], camera_pixels_subsampled[:, 0]]

                # Get start/end poses in world-global frame (renderer handles shutter interpolation)
                T_sensor_startend = self._get_start_end_poses_world_global(camera_sensor, camera_frame_index)

                frame_time_ms = self._compute_frame_time_ms(camera_sensor, camera_frame_index)
                camera_index = self.camera_ids.index(camera_id)

                return {
                    "rgb": to_torch(rgb, device="cpu"),
                    "valid": to_torch(valid, device="cpu"),
                    "T_camera_to_world": to_torch(T_sensor_startend[0], device="cpu"),
                    "T_camera_to_world_end": to_torch(T_sensor_startend[1], device="cpu"),
                    "camera_id": camera_id,
                    "frame_idx": -1,  # Validation: -1 signals novel-view mode for PPISP
                    "camera_idx": camera_index,
                }

            raise IndexError(f"Out of range validation sample {idx}")

    def _sensor_ids_to_unique_ids(self, input_sensor_ids: list[str], sensor_type: str) -> Generator[str, None, None]:
        """Converts logical or unique sensor ids to the corresponding set of unique ids for a certain sensor type.
        Valid unique ids are provided as is, logical sensor ids are expanded to the corresponding unique ids.
        Raises KeyError if input sensor was not found"""

        match sensor_type:
            case "camera":
                sensor_unique_ids = self.camera_unique_ids
            case "lidar":
                sensor_unique_ids = self.lidar_unique_ids
            case _:
                raise ValueError(f"NCoreDataset: unknown sensor type {sensor_type}")

        for input_sensor_id in input_sensor_ids:
            sensor_found = False
            if input_sensor_id in sensor_unique_ids:
                # input sensor id is a logical sensor name
                for unique_sensor_id in sensor_unique_ids[input_sensor_id]:
                    yield unique_sensor_id.id
                    sensor_found = True
            else:
                # input sensor id might be a unique name - look for it and return if found
                for unique_sensor_ids in sensor_unique_ids.values():
                    for unique_sensor_id in unique_sensor_ids:
                        if unique_sensor_id.id == input_sensor_id:
                            yield input_sensor_id
                            sensor_found = True
            if not sensor_found:
                raise KeyError(f"NCoreDataset: unknown sensor id {input_sensor_id}")

    def get_point_clouds(
        self,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        non_dynamic_points_only: bool = True,
        step_frame: int = 1,
    ) -> Generator[PointCloud, None, None]:
        """Returns a generator for all point-clouds available for point-cloud sensor (lidar / camera), transformed into world-global frame.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Can be parameterized to only return non-dynamic points (default).

        Default point-cloud sensor: *first* logical lidar
        """

        # we only support point clouds from lidar sensors
        if camera_ids is not None and len(camera_ids):
            raise ValueError(
                "NCoreDataset: camera-based point clouds requested, but only lidar-based point clouds supported"
            )

        # make sure we are initialized
        self._init_worker()

        # default to first lidar instance if not provided explicitly
        assert len(
            self.lidar_ids
        ), f"NCoreDataset: At least a single lidar needs to be available for point-cloud generation"
        lidar_ids = [self.lidar_ids[0]] if lidar_ids is None else lidar_ids

        sequence_id = self.sequence_id
        for lidar_id in lidar_ids:
            lidar_sensor = self.sequence_lidar_sensors[sequence_id][lidar_id]

            for lidar_frame_index in self.time_range_us.cover_range(lidar_sensor.get_frames_timestamps_us())[
                ::step_frame
            ]:
                # Load point cloud via compat API
                pc = lidar_sensor.get_frame_point_cloud(
                    frame_index=lidar_frame_index,
                    motion_compensation=True,
                    with_start_points=True,
                    return_index=0,
                )
                assert pc.xyz_m_start is not None, "Expected start points from motion-compensated point cloud"
                xyz_s = pc.xyz_m_start
                xyz_e = pc.xyz_m_end

                # load point color, if available
                if lidar_sensor.has_frame_generic_data(lidar_frame_index, self.lidar_color_generic_data_name):
                    color = lidar_sensor.get_frame_generic_data(lidar_frame_index, self.lidar_color_generic_data_name)
                    assert (
                        color.shape == xyz_s.shape
                    ), "Color data length does not match point cloud length (expecting 3-channel RGB color per point)"
                    assert color.dtype == np.uint8, "Expected color data in uint8 format"
                else:
                    color = None

                # determine point subset to load
                point_filter = ...
                if non_dynamic_points_only:
                    # filter out dynamic points if dynamic_flag is available via generic data
                    if lidar_sensor.has_frame_generic_data(lidar_frame_index, "dynamic_flag"):
                        dynamic_flags = lidar_sensor.get_frame_generic_data(lidar_frame_index, "dynamic_flag")
                        point_filter = dynamic_flags != 1  # 1 ~ DYNAMIC

                xyz_s = xyz_s[point_filter]
                xyz_e = xyz_e[point_filter]
                if color is not None:
                    color = color[point_filter]

                # transform points from sensor to world-global frame
                T_sensor_world_global = self._transform_poses_to_world_global(
                    lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index),
                    self.T_world_to_world_global,
                )

                xyz_s = (T_sensor_world_global[:3, :3] @ xyz_s.transpose() + T_sensor_world_global[:3, 3:4]).transpose()
                xyz_e = (T_sensor_world_global[:3, :3] @ xyz_e.transpose() + T_sensor_world_global[:3, 3:4]).transpose()

                yield PointCloud(
                    xyz_start=to_torch(xyz_s, device="cpu"),
                    xyz_end=to_torch(xyz_e, device="cpu"),
                    color=to_torch(color, device="cpu") if color is not None else None,
                    device="cpu",
                )

    # ---- Framework protocol methods (BoundedMultiViewDataset, DatasetVisualization) ----

    def _lazy_worker_gpu_cache(self) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Lazily create and cache camera-space rays + pixel coords on GPU for the current worker.

        Each DataLoader worker creates its own GPU tensors on first use.

        Returns:
            Dict mapping camera_id -> (rays_ori, rays_dir, pixel_coords) on GPU.
        """
        worker_id = get_worker_id()

        if worker_id not in self._worker_gpu_cache:
            cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

            for sequence_id in self.sequence_cameras_all_rays:
                if self.split.startswith("train"):
                    # Training: use full-resolution rays
                    for camera_id, rays_np in self.sequence_cameras_all_rays[sequence_id].items():
                        h_full, w_full = rays_np.shape[:2]
                        rays_dir = torch.from_numpy(rays_np).to(
                            dtype=torch.float32, device=self.device, non_blocking=True
                        )
                        rays_ori = torch.zeros_like(rays_dir)
                        pixel_coords = create_pixel_coords(w_full, h_full, device=self.device)
                        cache[camera_id] = (rays_ori, rays_dir, pixel_coords)
                else:
                    # Validation: use subsampled rays if n_val_image_subsample > 1, else full
                    if self.n_val_image_subsample > 1:
                        for camera_id, rays_sub_np in self.sequence_cameras_rays_subsample[sequence_id].items():
                            w, h = self._camera_resolutions[camera_id]
                            rays_dir = torch.from_numpy(rays_sub_np.reshape(h, w, 3)).to(
                                dtype=torch.float32, device=self.device, non_blocking=True
                            )
                            rays_ori = torch.zeros_like(rays_dir)
                            pixel_coords = create_pixel_coords(w, h, device=self.device)
                            cache[camera_id] = (rays_ori, rays_dir, pixel_coords)
                    else:
                        for camera_id, rays_np in self.sequence_cameras_all_rays[sequence_id].items():
                            h_full, w_full = rays_np.shape[:2]
                            rays_dir = torch.from_numpy(rays_np).to(
                                dtype=torch.float32, device=self.device, non_blocking=True
                            )
                            rays_ori = torch.zeros_like(rays_dir)
                            pixel_coords = create_pixel_coords(w_full, h_full, device=self.device)
                            cache[camera_id] = (rays_ori, rays_dir, pixel_coords)

            self._worker_gpu_cache[worker_id] = cache

        return self._worker_gpu_cache[worker_id]

    def get_gpu_batch_with_intrinsics(self, batch) -> Batch:
        """Convert batch dict to threedgrut Batch, using GPU-cached camera-space rays.

        Camera-space rays are NOT transferred from CPU each batch. Instead they are
        looked up from the per-worker GPU cache by camera_id. Only the per-frame
        pose (128 bytes) and RGB image are transferred to GPU.

        The renderer handles rolling shutter interpolation via T_to_world / T_to_world_end
        and the shutter_type from intrinsics. Rays are always in camera space
        (rays_in_world_space=False).
        """

        # --- Resolve camera_id -------------------------------------------------
        camera_id = batch["camera_id"]
        if isinstance(camera_id, (list, tuple)):
            camera_id = camera_id[0]

        # --- Look up cached camera-space rays on GPU (no CPU->GPU transfer) -----
        worker_cache = self._lazy_worker_gpu_cache()
        rays_ori, rays_dir, pixel_coords = worker_cache[camera_id]

        # Derive image dimensions from cached ray tensor shape
        h, w = rays_dir.shape[0], rays_dir.shape[1]

        # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
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
        batch_dict: dict[str, Any] = {
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
            logger.info(f"[Shutter] Camera {camera_id}: shutter_type = {scaled_params.shutter_type.name}")

            # Build parameter dict from the properly-scaled NCore parameters.
            # Distortion coefficients are resolution-independent and preserved through transform().
            # Each camera model type produces a dict matching the tracer's expected keys.
            if isinstance(camera_model, ncore.sensors.OpenCVPinholeCameraModel):
                params_dict = {
                    "resolution": scaled_params.resolution,
                    "shutter_type": scaled_params.shutter_type.name,
                    "principal_point": scaled_params.principal_point,
                    "focal_length": scaled_params.focal_length,
                    "radial_coeffs": scaled_params.radial_coeffs,
                    "tangential_coeffs": scaled_params.tangential_coeffs,
                    "thin_prism_coeffs": scaled_params.thin_prism_coeffs,
                }
            elif isinstance(camera_model, ncore.sensors.OpenCVFisheyeCameraModel):
                params_dict = {
                    "resolution": scaled_params.resolution,
                    "shutter_type": scaled_params.shutter_type.name,
                    "principal_point": scaled_params.principal_point,
                    "focal_length": scaled_params.focal_length,
                    "radial_coeffs": scaled_params.radial_coeffs,
                    "max_angle": scaled_params.max_angle,
                }
            elif isinstance(camera_model, ncore.sensors.FThetaCameraModel):
                params_dict = {
                    "resolution": scaled_params.resolution,
                    "shutter_type": scaled_params.shutter_type.name,
                    "principal_point": scaled_params.principal_point,
                    "reference_poly": scaled_params.reference_poly.name,
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
        if self.split == "train":
            max_frames_per_camera = max(1, 50 // len(self.camera_ids))
        else:
            max_frames_per_camera = 10

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

            camera_frame_range = self.camera_frame_ranges[camera_id]
            num_frames = camera_frame_range.stop - camera_frame_range.start
            frame_step = max(1, num_frames // max_frames_per_camera)

            for frame_idx in range(camera_frame_range.start, camera_frame_range.stop, frame_step):
                try:
                    T_sensor_to_world_flat = camera_sensor.get_frames_T_source_target(
                        source_node=camera_sensor.sensor_id,
                        target_node="world",
                        frame_indices=np.array(frame_idx),
                        frame_timepoint=ncore.data.FrameTimepoint.START,
                    )
                    if T_sensor_to_world_flat.ndim == 1:
                        T_sensor_to_world = T_sensor_to_world_flat.reshape(4, 4)
                    else:
                        T_sensor_to_world = T_sensor_to_world_flat

                    T_sensor_to_world_global_batch = self._transform_poses_to_world_global(
                        T_sensor_to_world[None, :, :], self.T_world_to_world_global
                    )
                    if T_sensor_to_world_global_batch.ndim == 3:
                        T_sensor_to_world_global = T_sensor_to_world_global_batch[0]
                    elif T_sensor_to_world_global_batch.ndim == 2:
                        T_sensor_to_world_global = T_sensor_to_world_global_batch.reshape(4, 4)
                    else:
                        T_sensor_to_world_global = T_sensor_to_world_global_batch

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
                except Exception:
                    continue

        if cam_list:
            create_camera_visualization(cam_list)
