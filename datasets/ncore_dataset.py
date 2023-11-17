# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from enum import IntEnum, auto, unique
from pathlib import Path
from typing import Any, Generator, NamedTuple, Optional
from collections import defaultdict

import torch
import torch.utils.data
import torchvision
import numpy as np
import numpy.typing as npt
import PIL.Image as PILImage

from scipy import ndimage
from thirdparty.pytorch_morphological_dilation2d_erosion2d.morphology import Dilation2d

import ncore.data.v3
import ncore.data
import ncore.sensors
import ncore.impl.common.transformations as ncore_transformations
import ncore.impl.common.common as ncore_common
import ncore.impl.data.types as ncore_datatypes

from datasets.ncore_utils import (
    RayFlags,
    HalfClosedInterval,
    SequenceChunk,
    Labels,
    RaysCamMeta,
    RaysLidarMeta,
    Batch,
    FrameConversion,
    AuxShardDataLoader,
    CameraFrustum,
    BoundingBox,
    RigTrajectories,
    object_mask_from_image_points,
)
from datasets.utils import PointCloud

from utils.misc import to_torch, to_torch_optional, unpack_optional


def chunk_sizes(n: int, size: int) -> Generator[int, None, None]:
    """Divides size into n approximately equally sized subdivisions such that they sum to size

    Example: list(chunk_sizes(3, 4096)) == [1365, 1365, 1366]
    """
    start = 0
    for i in range(1, n):
        stop = i * size // n  # evenly sized initial elements
        yield stop - start
        start = stop
    yield size - start  # collect overlap in last element


def test_chunk_size() -> None:
    l = list(chunk_sizes(1, 4096))
    assert sum(l) == 4096

    l = list(chunk_sizes(2, 4096))
    assert sum(l) == 4096

    l = list(chunk_sizes(3, 4096))
    assert sum(l) == 4096


@dataclass(slots=True)
class RangedSequenceChunk(SequenceChunk):
    """Represents a chunk (given by time-range) within a sequence per-sensor frame ranges"""

    camera_frame_ranges: dict[str, range]  # camera-id to range of sensor frame indices
    lidar_frame_ranges: dict[str, range]  # lidar-id to range of sensor frame indices

    camera_linear_start_frame_indices: dict[
        str, int
    ]  # camera-id to global linear start frame index (across all chunks and cameras)
    lidar_linear_start_frame_indices: dict[
        str, int
    ]  # camera-id to global linear start frame index (across all chunks and lidars)

    T_world_local_world_common: np.ndarray  # transformation from local chunk world frame to common world frame across multiple chunks

    # TODO[janickm]: have a form of per-frame probability?


class NCoreDataset(torch.utils.data.Dataset):
    TOTAL_STANDSTILL_DISTANCE_THRESHOLD_M = 0.5  # threshold to classify standstill cases over all chunks

    UNCONDITIONALLY_DYNAMIC_LABELS: set[str] = set(
        [
            "pedestrian",
            "stroller",
            "person",
            "person_group",
            "rider",
            "bicycle_with_rider",
            "bicycle",
            "CYCLIST",
            "motorcycle",
            "motorcycle_with_rider",
            "cycle",
        ]
    )

    @unique
    class ValidPixelsMethod(IntEnum):
        """Different method variants to determine valid pixels"""

        EGO = auto()  # ego camera masks only
        EGO_CUBOIDTRACKS = auto()  # ego camera masks and frame-projected dynamic objects from cuboid tracks
        EGO_SCENEFLOW = auto()  # ego camera masks and scene flow masks
        EGO_CUBOIDTRACKS_SCENEFLOW = (
            auto()
        )  # ego camera masks, frame-projected dynamic objects from cuboid tracks, and scene flow masks

    @dataclass(kw_only=True)
    class ValidPixelsCuboidTracksParams:
        track_mask_dilate_ratio: float
        track_min_speed_ms: float
        track_min_centroid_rig_dist_m: float
        track_extrapolate: bool

        def __post_init__(self):
            assert (
                self.track_mask_dilate_ratio >= 0.0
            ), "[NCoreDataset]: Dilation ratio for track masks needs to be positive"
            assert self.track_min_speed_ms >= 0.0, "[NCoreDataset]: Speed threshold for tracks needs to be positive"
            assert (
                self.track_min_centroid_rig_dist_m >= 0.0
            ), "[NCoreDataset]: Minimum track centroid-to-rig distance needs to be positive"

    @dataclass(kw_only=True)
    class ValidPixelsSceneFlowParams:
        flow_min_speed_ms: float
        flow_dilate_radius: int
        flow_downsample_scale: int  # for speeding up dilation

        def __post_init__(self):
            assert self.flow_min_speed_ms >= 0.0, "[NCoreDataset]: Speed threshold for scene-flow needs to be positive"
            assert self.flow_dilate_radius >= 0, "[NCoreDataset]: Dilation radius for scene-flow needs to be positive"
            assert (
                self.flow_downsample_scale >= 1
            ), "[NCoreDataset]: Downsample scale for scene-flow needs to be greater than or equal to 1"

    def __init__(
        self,
        # TODO: bring into config
        # Data-source specification by a 'datapath' to either
        # - a *single-sequence* NCore meta-file. In this case the sequence time
        #   range can be restricted with additional 'seek_offset_sec' / 'duration_sec'
        #   parameters
        # - a NRE *multi-chunk* meta-file (it's an error to additionally
        #   specify 'seek_offset_sec' / 'duration_sec')
        datapath: str,
        seek_offset_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        split: str = "train",
        # Sensors
        camera_ids=["camera_front_wide_120fov"],
        lidar_ids=["lidar_gt_top_p128_v4p5"],
        # Scale
        aabb_scale: Optional[float] = None,
        max_dist_m: float = 150.0,
        # Misc
        n_samples_per_epoch: int = 1000,
        n_train_sample_timepoints: int = 1,
        n_train_sample_camera_rays: int = 4096,
        n_train_sample_lidar_rays: int = 0,
        n_val_image_subsample: int = 4,
        n_camera_mask_dilation_iterations: int = 30,
        open_consolidated=True,
        aux_data=True,
        camera_max_fov_deg: float = 190.0,
        valid_pixels_method: str = "EGO_CUBOIDTRACKS",
        valid_pixels_cuboid_track_params: dict = {
            "track_mask_dilate_ratio": 1.4,  # Ratio of the mask (1.0 results in no dilation, values smaller than 1.0 shrink the mask)
            "track_min_speed_ms": 1.5,  # Speed threshold to classify objects to be dynamic  [m/s]
            "track_min_centroid_rig_dist_m": 3.0,  # Distance threshold for cubic tracks to be considered self-classifications to skip [m]
            "track_extrapolate": True,  # Extrapolate label pose of dynamic tracks once at the start / end (to improve interpolation coverage)
        },
        valid_pixels_scene_flow_params: dict = {
            "flow_min_speed_ms": 1.4,  # Speed threshold to classify objects to be dynamic  [m/s]
            "flow_dilate_radius": 20,  # dilation radius for the dynamic mask from scene flow (Dilation is done after downsampling the mask to a low-res)
            "flow_downsample_scale": 2,  # downsample scale for dynamic mask from scene flow. This is only for accelerating the dilation operation.) -> None:
        },
    ):
        super().__init__()

        # Init parameters from config

        # store relevant parameters from config
        self.n_samples_per_epoch: int = n_samples_per_epoch
        self.n_train_sample_timepoints: int = n_train_sample_timepoints
        self.n_train_sample_camera_rays: int = n_train_sample_camera_rays
        self.n_train_sample_lidar_rays: int = n_train_sample_lidar_rays

        self.n_val_image_subsample: int = n_val_image_subsample

        self.n_camera_mask_dilation_iterations: int = n_camera_mask_dilation_iterations

        self.camera_max_fov_deg: float = camera_max_fov_deg

        # load valid pixels method and method-dependent parameters
        self.valid_pixels_method = NCoreDataset.ValidPixelsMethod[valid_pixels_method]
        self.valid_pixels_cuboid_tracks_params: NCoreDataset.ValidPixelsCuboidTracksParams | None = None
        self.valid_pixels_pixels_scene_flow_params: NCoreDataset.ValidPixelsSceneFlowParams | None = None
        match self.valid_pixels_method:
            case NCoreDataset.ValidPixelsMethod.EGO:
                pass
            case NCoreDataset.ValidPixelsMethod.EGO_CUBOIDTRACKS:
                self.valid_pixels_cuboid_tracks_params = NCoreDataset.ValidPixelsCuboidTracksParams(
                    **valid_pixels_cuboid_track_params
                )
            case NCoreDataset.ValidPixelsMethod.EGO_SCENEFLOW:
                self.valid_pixels_pixels_scene_flow_params = NCoreDataset.ValidPixelsSceneFlowParams(
                    **valid_pixels_scene_flow_params
                )
            case NCoreDataset.ValidPixelsMethod.EGO_CUBOIDTRACKS_SCENEFLOW:
                self.valid_pixels_cuboid_tracks_params = NCoreDataset.ValidPixelsCuboidTracksParams(
                    **valid_pixels_cuboid_track_params
                )
                self.valid_pixels_pixels_scene_flow_params = NCoreDataset.ValidPixelsSceneFlowParams(
                    **valid_pixels_scene_flow_params
                )
            case _:
                raise ValueError(f"[NCoreDataset]: unsupported valid-pixels method {self.valid_pixels_method}")

        # note: we currently assume these sensors are available for all sequences - can be refined if necessary
        self.camera_ids: list[str] = camera_ids
        self.lidar_ids: list[str] = lidar_ids

        # if aabb_scale is None, the scene will be kept true to scale
        self.aabb_scale: Optional[float] = aabb_scale
        self.max_dist_m: float = max_dist_m

        self.open_consolidated: bool = open_consolidated

        self.aux_data: bool = aux_data

        self.split: str = split

        ## Setup chunks

        self.chunks: list[
            RangedSequenceChunk
        ] = []  # sequence of chunks represented by this dataset (fixed order to provide linear indexing across chunks)

        sequence_chunks: dict  # either loaded from file or constructed for single sequence-chunk variant

        # load and classify input dataset description (input has to be a json file)
        assert (path := Path(datapath)).is_file(), f"NCoreDataset: provided path {path} not a file"
        with open(path, "r") as fp:
            try:
                dataset_meta = json.load(fp)
            except ValueError as e:
                raise ValueError(f"NCoreDataset: provided file {path} not a json file")

        if not "sequences" in dataset_meta:
            # load *single-sequence* NCORE meta to determine single-chunk range

            # sanity check schema
            assert all(
                (key in dataset_meta for key in ("sequence_id", "pose-range", "shards", "shard-ids"))
            ), f"NCoreDataset: provided json file {path} not a NCore single-sequence file"

            # determine time-range of subsection
            chunk_time_range_us = HalfClosedInterval(
                dataset_meta["pose-range"]["start-timestamp_us"], dataset_meta["pose-range"]["end-timestamp_us"]
            )
            if seek_offset_sec := seek_offset_sec:
                chunk_time_range_us.start += int(seek_offset_sec * 1e6)
            if duration_sec := duration_sec:
                chunk_time_range_us.end = min(
                    chunk_time_range_us.start + int(duration_sec * 1e6), chunk_time_range_us.end
                )

            sequence_chunks = {
                "sequences": {
                    dataset_meta["sequence_id"]: {
                        "sequence-meta-file": path,
                        "chunks": [
                            {
                                "start-timestamp_us": chunk_time_range_us.start,
                                "end-timestamp_us": chunk_time_range_us.end,
                            }
                        ],
                    }
                }
            }
        else:
            # sanity check schema (redundant with existing logic, but log proper error-message for user's nonetheless)
            assert all(
                (key in dataset_meta for key in ("sequences",))
            ), f"NCoreDataset: provided json file {path} neither a NCore single-sequence or multi-chunk file"

            # load all chunks from input file
            sequence_chunks = dataset_meta

            # make sure single-sequence overwrites are not falsely specified in this case
            assert seek_offset_sec is None, "NCoreDataset: seek_offset_sec can't be specified for multi-chunk inputs"
            assert duration_sec is None, "NCoreDataset: duration_sec can't be specified for multi-chunk inputs"

        # initialize each chunk for each sequence
        self.sequence_shard_paths: dict[str, list[str]] = {}
        for sequence_id, sequence_data in sequence_chunks["sequences"].items():
            sequence_meta_file = Path(sequence_data["sequence-meta-file"])
            try:
                # try loading absolute / cwd-relative path first
                with open(sequence_meta_file, "r") as fp:
                    sequence_meta = json.load(fp)
            except FileNotFoundError:
                # try loading chunk-file-relative path instead
                sequence_meta_file = path.parent / sequence_meta_file
                with open(sequence_meta_file, "r") as fp:
                    sequence_meta = json.load(fp)

            sequence_path = sequence_meta_file.parent

            chunks: list[dict] = sequence_data["chunks"]

            if not len(chunks):
                logging.warn(f"NCoreDataset: skipping sequence {sequence_id} (empty chunks)")
                continue

            # make sure that the sequence of per-sequence chunks is ordered linearly w.r.t. time and not overlapping
            for chunk, next_chunk in zip(chunks, chunks[1:]):
                assert (
                    chunk["end-timestamp_us"] < next_chunk["start-timestamp_us"]
                ), f"NCoreDataset: neighboring chunks {chunk} / {next_chunk} not ordered linearly or overlapping for sequence {sequence_id}"

            # determine data shards to load for this sequence by all shards that *linearly* span *all* chunks
            sequence_time_range_all_chunks = HalfClosedInterval(
                chunks[0]["start-timestamp_us"], chunks[-1]["end-timestamp_us"]
            )
            self.sequence_shard_paths[sequence_id] = [
                sequence_path / shard["path"]
                for shard in sequence_meta["shards"]
                if sequence_time_range_all_chunks.overlaps(
                    HalfClosedInterval(
                        shard["pose-range"]["start-timestamp_us"], shard["pose-range"]["end-timestamp_us"]
                    )
                )
            ]

            # register chunks for this sequence (per chunk sensor frame ranges will be provided at data init-time)
            self.chunks += [
                RangedSequenceChunk(
                    sequence_id,
                    HalfClosedInterval(chunk["start-timestamp_us"], chunk["end-timestamp_us"]),
                    {},
                    {},
                    {},
                    {},
                    np.empty(0),
                )
                for chunk in chunks
            ]

        assert len(self.chunks), f"NCoreDataset: no chunks loaded"

        # initialize shards once fully
        self.sequence_loaders: dict[str, ncore.data.v3.ShardDataLoader] = {}
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
            case torch.utils.data._utils.worker.WorkerInfo(id=worker_id, seed=worker_seed):  # type:ignore
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
                loader.reload_store_resources()

            for aux_loader in self.sequence_aux_loaders.values():
                aux_loader.reload_store_resources()

            return

        # Full initial load case
        class UniqueSensorId(NamedTuple):
            """Represents a unique sensor ID along with its unique index for a given sensor type"""

            id: str
            idx: int

        self.n_unique_cameras = 0
        self.camera_unique_ids: dict[str, list[UniqueSensorId]] = defaultdict(list)
        self.sequence_camera_sensors: dict[str, dict[str, ncore.data.v3.CameraSensor]] = {}
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
        self.sequence_lidar_sensors: dict[str, dict[str, ncore.data.v3.LidarSensor]] = {}
        self.sequence_lidar_unique_ids: dict[str, dict[str, UniqueSensorId]] = {}

        self.sequence_aux_loaders: dict[str, AuxShardDataLoader] = {}
        self.sequence_camera_sky_class_ids: dict[str, dict[str, int]] = {}

        for sequence_id, shard_paths in self.sequence_shard_paths.items():
            # load all relevant shards
            sequence_loader = self.sequence_loaders[sequence_id] = ncore.data.v3.ShardDataLoader(
                shard_paths=shard_paths, open_consolidated=self.open_consolidated
            )

            # Load sensor data

            # load camera sensors
            camera_sensors = self.sequence_camera_sensors[sequence_id] = {
                camera_id: sequence_loader.get_camera_sensor(camera_id) for camera_id in self.camera_ids
            }

            # load camera models
            camera_models = self.sequence_camera_models[sequence_id] = {
                camera_id: ncore.sensors.CameraModel.from_parameters(
                    camera_sensors[camera_id].get_camera_model_parameters(), device="cpu", dtype=torch.float32
                )
                for camera_id in self.camera_ids
            }

            # construct unique camera instance ids (<camera_id>@<sequence_id>) and instance indices
            # (different sensors across different sequences are considered different instances)
            self.sequence_camera_unique_ids[sequence_id] = {
                camera_id: UniqueSensorId("@".join((camera_id, sequence_id)), camera_instance_idx)
                for camera_instance_idx, camera_id in enumerate(self.camera_ids, self.n_unique_cameras)
            }
            self.n_unique_cameras += len(self.camera_ids)
            for camera_id in self.camera_ids:
                self.camera_unique_ids[camera_id].append(self.sequence_camera_unique_ids[sequence_id][camera_id])

            # restrict effective FOV of omnidirectional cameras for cuboid projections to prevent issues with *invalid* points projected
            # back into the valid image domain FOV - this way they get properly classified as invalid
            for camera_model in camera_models.values():
                if not isinstance(
                    camera_model, (ncore.sensors.FThetaCameraModel, ncore.sensors.OpenCVFisheyeCameraModel)
                ):
                    continue
                camera_model.max_angle = min(np.deg2rad(self.camera_max_fov_deg) / 2.0, camera_model.max_angle)

            # determine all pixels and rays per camera
            for camera_id in self.camera_ids:
                camera_sensor = camera_sensors[camera_id]
                camera_model = camera_models[camera_id]

                # sample pixel ranges
                w = int(camera_model.resolution[0].item())
                h = int(camera_model.resolution[1].item())

                # all pixels
                camera_pixels_x, camera_pixels_y = np.meshgrid(
                    np.arange(w, dtype=np.int16), np.arange(h, dtype=np.int16)
                )  # [0, w-1] x [0, h-1]
                self.sequence_cameras_all_pixels[sequence_id] |= {
                    camera_id: np.stack([camera_pixels_x.flatten(), camera_pixels_y.flatten()], axis=1)
                }

                # subsampled pixels for validation
                assert (
                    w % self.n_val_image_subsample == 0 and h % self.n_val_image_subsample == 0
                ), f"NCoreDataset: Validation subsample factor {self.n_val_image_subsample} invalid for camera {camera_id} with resolution {w}x{h}"
                camera_pixels_x_subsample, camera_pixels_y_subsample = np.meshgrid(
                    np.arange(start=0, stop=w, step=self.n_val_image_subsample, dtype=np.int16),
                    np.arange(start=0, stop=h, step=self.n_val_image_subsample, dtype=np.int16),
                )
                self.sequence_cameras_pixels_subsample[sequence_id] |= {
                    camera_id: np.stack(
                        [camera_pixels_x_subsample.flatten(), camera_pixels_y_subsample.flatten()], axis=1
                    )
                }

                # statically unmasked pixels
                if camera_mask_image := camera_sensor.get_camera_mask_image():
                    # True for parts that we want to mask out
                    camera_mask_array = np.asarray(camera_mask_image) != 0

                    # Dilate mask boundary
                    camera_mask_array = ndimage.binary_dilation(
                        camera_mask_array, iterations=self.n_camera_mask_dilation_iterations
                    )

                    # Subsample valid pixels relative to mask (True for parts that we want to keep)
                    camera_valid_pixels_ego_mask = np.logical_not(camera_mask_array)
                else:
                    # No mask / consider all pixels as valid
                    camera_valid_pixels_ego_mask = np.ones(
                        (int(camera_model.resolution[1].item()), int(camera_model.resolution[0].item())), dtype=bool
                    )
                self.sequence_cameras_valid_pixels_ego_masks[sequence_id] |= {camera_id: camera_valid_pixels_ego_mask}

                # precompute all rays
                self.sequence_cameras_all_rays[sequence_id] |= {
                    camera_id: camera_model.pixels_to_camera_rays(
                        self.sequence_cameras_all_pixels[sequence_id][camera_id]
                    )
                    .reshape(h, w, 3)
                    .numpy()
                }
                self.sequence_cameras_rays_subsample[sequence_id] |= {
                    camera_id: camera_model.pixels_to_camera_rays(
                        self.sequence_cameras_pixels_subsample[sequence_id][camera_id]
                    ).numpy()
                }

            # load lidar sensors
            self.sequence_lidar_sensors[sequence_id] = {
                lidar_id: sequence_loader.get_lidar_sensor(lidar_id) for lidar_id in self.lidar_ids
            }

            # construct unique lidar instance ids (<lidar_id>@<sequence_id>) and instance indices
            # (different sensors across different sequences are considered different instances)
            self.sequence_lidar_unique_ids[sequence_id] = {
                lidar_id: UniqueSensorId("@".join((lidar_id, sequence_id)), lidar_instance_idx)
                for lidar_instance_idx, lidar_id in enumerate(self.lidar_ids, self.n_unique_lidars)
            }
            self.n_unique_lidars += len(self.lidar_ids)
            for lidar_id in self.lidar_ids:
                self.lidar_unique_ids[lidar_id].append(self.sequence_lidar_unique_ids[sequence_id][lidar_id])

            # load aux data shards if enabled
            if self.aux_data:
                aux_loader = self.sequence_aux_loaders[sequence_id] = AuxShardDataLoader(sequence_loader)

                # get index of 'sky' class for each camera
                self.sequence_camera_sky_class_ids[sequence_id] = {
                    camera_id: aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index("sky")
                    for camera_id in self.camera_ids
                }

        # Process chunk-dependent data and compute full scene extent and scales

        # represents the transformation from base world back to common world (to keep coordinates small / centered)
        # - the common world frame is initialized to the world frame of the *first* sequence among all chunks
        T_world_base_world_common: Optional[np.ndarray] = None

        # represents the transformation from common world frame to base world
        self.T_world_common_world_base: npt.NDArray[np.float64]

        chunks_rig_world_positions: list[np.ndarray] = []
        chunk_distances_m: list[float] = []
        cumulative_camera_start_frame_index: int = 0
        cumulative_lidar_start_frame_index: int = 0
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            # determine linear per-sensor-frame index ranges depending on dataset time restrictions
            chunk.camera_frame_ranges = {
                camera_id: chunk.time_range_us.cover_range(
                    self.sequence_camera_sensors[sequence_id][camera_id].get_frames_timestamps_us()
                )
                for camera_id in self.camera_ids
            }
            chunk.lidar_frame_ranges = {
                lidar_id: chunk.time_range_us.cover_range(
                    self.sequence_lidar_sensors[sequence_id][lidar_id].get_frames_timestamps_us()
                )
                for lidar_id in self.lidar_ids
            }

            # determine per-sensor starting index to return unique linear frame indices for each sample
            for camera_id, camera_frame_range in chunk.camera_frame_ranges.items():
                chunk.camera_linear_start_frame_indices[camera_id] = cumulative_camera_start_frame_index
                cumulative_camera_start_frame_index += len(camera_frame_range)

            for lidar_id, lidar_frame_range in chunk.lidar_frame_ranges.items():
                chunk.lidar_linear_start_frame_indices[lidar_id] = cumulative_lidar_start_frame_index
                cumulative_lidar_start_frame_index += len(lidar_frame_range)

            # load all rig poses
            sequence_poses = self.sequence_loaders[sequence_id].get_poses()

            if T_world_base_world_common is None:
                # initialize common world as first sequence's world frame
                self.T_world_common_world_base = sequence_poses.T_rig_world_base
                T_world_base_world_common = ncore_transformations.se3_inverse(self.T_world_common_world_base)

            # construct the transformation from local world to common world
            chunk.T_world_local_world_common = (T_world_base_world_common @ sequence_poses.T_rig_world_base).astype(
                np.float32
            )

            # load chunk-restricted world positions in 'local world'
            chunk_rig_world_local_positions = sequence_poses.T_rig_worlds[
                chunk.time_range_us.cover_range(sequence_poses.T_rig_world_timestamps_us)
            ][:, :3, 3]

            # transform to 'common' world positions
            chunks_rig_world_positions += [
                ncore_transformations.transform_point_cloud(
                    chunk_rig_world_local_positions, chunk.T_world_local_world_common
                )
            ]

            # chunk sampling probability given by total traveled distance
            chunk_distances_m += [np.linalg.norm(np.diff(chunk_rig_world_local_positions, axis=0), axis=1).sum()]

        if (total_distance_m := sum(chunk_distances_m)) >= NCoreDataset.TOTAL_STANDSTILL_DISTANCE_THRESHOLD_M:
            # significant baseline across all chunks - derive per-chunk probabilities relative to chunk-distances
            self.chunk_probabilities = np.hstack(chunk_distances_m) / total_distance_m
        else:
            # insignificant traveled distance - potentially a degenerate case
            logging.warn(
                f"NCOREDataLoader: insufficient traveled distance across all chunks, potentially a degenerate case - assigning uniform probabilities to all chunks"
            )
            self.chunk_probabilities = np.full((len(self.chunks),), 1.0 / len(self.chunks))

        assert np.isclose(self.chunk_probabilities.sum(), 1.0)  # sanity check chunk probabilities

        # merge all common world positions
        rig_world_positions = np.vstack(chunks_rig_world_positions)

        # compute average position and extent (largest axis of the scene's AABB relative to the world frame)
        # to put the scene's center at the origin of the rescaled domain
        mean_rig_world_position_m = rig_world_positions.mean(axis=0).astype(np.float32)

        # make sure that the max distance at the boundary is included when scaling the scene extent to the target AABB scale
        world_diag_extent_m = np.max(rig_world_positions, axis=0) - np.min(rig_world_positions, axis=0)
        world_max_extent_m = np.max(world_diag_extent_m)
        world_to_colmap_scale = (
            1.0 / ((world_max_extent_m / 2 + self.max_dist_m) / (self.aabb_scale / 2)) if self.aabb_scale else 1.0
        )

        self.scene_extent_m = np.linalg.norm(world_diag_extent_m)

        # Setup NCore common world -> colmap transformation
        self.world_to_colmap = FrameConversion.from_origin_scale_axis(
            target_origin=mean_rig_world_position_m,  # put the scene's center at the origin of the rescaled domain
            target_scale=world_to_colmap_scale,
            target_axis=[0, 1, 2],  # xyz[world] -> xyz[colmap]
        )

        # Compute per-frame valid pixels (excluding the dynamic objects, ego-car, ...)
        match self.valid_pixels_method:
            case NCoreDataset.ValidPixelsMethod.EGO:
                self._compute_valid_pixels_ego()
            case NCoreDataset.ValidPixelsMethod.EGO_CUBOIDTRACKS:
                self._compute_valid_pixels_ego()
                self._compute_valid_pixels_cuboidtracks()
            case NCoreDataset.ValidPixelsMethod.EGO_SCENEFLOW:
                self._compute_valid_pixels_ego()
                self._compute_valid_pixels_sceneflow()
            case NCoreDataset.ValidPixelsMethod.EGO_CUBOIDTRACKS_SCENEFLOW:
                self._compute_valid_pixels_ego()
                self._compute_valid_pixels_cuboidtracks()
                self._compute_valid_pixels_sceneflow()
            case _:
                raise ValueError(f"[NCoreDataset]: unsupported valid-pixels method {self.valid_pixels_method}")

    def _compute_valid_pixels_ego(self):
        """Sets static ego camera masks as per-frame valid-pixels masks"""
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            for camera_id, camera_frame_range in chunk.camera_frame_ranges.items():
                camera_valid_pixels_ego_mask = self.sequence_cameras_valid_pixels_ego_masks[sequence_id][camera_id]

                for camera_frame_idx in camera_frame_range:
                    # store copies of ego masks as per-frame valid pixel masks
                    self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id] |= {
                        camera_frame_idx: camera_valid_pixels_ego_mask.copy()
                    }

    def _compute_valid_pixels_cuboidtracks(self) -> None:
        """Computes per-frame projections of cuboid track dynamic objects as valid-pixels masks"""

        assert self.valid_pixels_cuboid_tracks_params is not None

        sequence_all_tracks: dict[str, dict[str, dict]] = defaultdict(dict)
        sequence_dynamic_tracks: dict[str, dict[str, dict]] = defaultdict(dict)

        # First pass over all chunks: collect all dynamic tracks per sequence by iterating over all chunks
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            # Extract the dynamic trajectories for the given data range
            all_sequence_tracks = sequence_all_tracks[sequence_id]
            for lidar_id, lidar_frame_range in chunk.lidar_frame_ranges.items():
                lidar_sensor = self.sequence_lidar_sensors[sequence_id][lidar_id]

                # extend the lidar frame range so that we cover all the images
                extended_lidar_frame_range = range(
                    max(lidar_frame_range.start - 1, 0),
                    min(lidar_frame_range.stop + 1, lidar_sensor.get_frames_count()),
                )

                # iterate over all lidar frames and extract ALL trajectories
                T_sensor_rig = lidar_sensor.get_T_sensor_rig()
                for frame_idx in extended_lidar_frame_range:
                    T_sensor_world = lidar_sensor.get_frame_T_sensor_world(frame_idx)
                    for label in lidar_sensor.get_frame_labels(frame_idx):
                        # skip self-classifications
                        bbox_rig = ncore_transformations.transform_bbox(label.bbox3.to_array(), T_sensor_rig)
                        if (
                            np.linalg.norm(bbox_rig[:3])
                            < self.valid_pixels_cuboid_tracks_params.track_min_centroid_rig_dist_m
                        ):  # skip labels that are too close to the rig center
                            continue

                        if label.track_id not in all_sequence_tracks:
                            # instantiate new track
                            all_sequence_tracks[label.track_id] = {
                                # track-constants:
                                "unconditionally_dynamic": label.label_class
                                in self.UNCONDITIONALLY_DYNAMIC_LABELS,  # some objects are unconditionally dynamic
                                "dimension": label.bbox3.to_array()[3:6],
                                "label_class": label.label_class,
                                # per track instance data:
                                "poses": [],
                                "timestamps_us": [],
                                "max_global_speed": 0.0,
                            }

                        # track to update with this instance's pose / speed data
                        track = all_sequence_tracks[label.track_id]

                        # prevent track instance duplication at perfectly neighboring chunks with overlapping lidar frames due to lidar frame extension
                        # (it's sufficient to check the two most recent timestamps as per-sequence chunks are ordered time-wise)
                        if label.timestamp_us in track["timestamps_us"][-2::]:
                            continue

                        # chunks for a single sequence are ordered time-wise, so these will be time-ordered also
                        track["timestamps_us"].append(label.timestamp_us)
                        track["poses"].append(
                            ncore_transformations.bbox_pose(
                                ncore_transformations.transform_bbox(label.bbox3.to_array(), T_sensor_world)
                            )
                        )

                        track["max_global_speed"] = max(track["max_global_speed"], label.global_speed)

        # Extract ONLY the dynamic trajectories based on the speed threshold / unconditional dynamic flag
        for sequence_id, tracks in sequence_all_tracks.items():
            for track_id, track in tracks.items():
                if (
                    track["max_global_speed"] > self.valid_pixels_cuboid_tracks_params.track_min_speed_ms
                    or track["unconditionally_dynamic"]
                ) and len(track["timestamps_us"]) > 1:
                    # initialize track-associated pose-interpolator
                    poses: list[np.ndarray] = track["poses"]
                    timestamps_us: list[int] = track["timestamps_us"]

                    # perform pose extrapolation by one instance into the past / future if enabled
                    if self.valid_pixels_cuboid_tracks_params.track_extrapolate:
                        # extrapolate first pose to the past
                        poses.insert(
                            0,
                            # extrapolate into pre-time P = (P_1 @ P_0^-1)^-1 @ P_0 = (P_0 @ P_1^-1) @ P_0
                            (poses[0] @ ncore_transformations.se3_inverse(poses[1])) @ poses[0],
                        )
                        timestamps_us.insert(0, timestamps_us[0] - (timestamps_us[1] - timestamps_us[0]))

                        # extrapolate last pose to the future
                        poses.append(
                            # extrapolate into post-time P = (P_N @ P_{N-1}^-1) @ P_N
                            (poses[-1] @ ncore_transformations.se3_inverse(poses[-2]))
                            @ poses[-1],
                        )
                        timestamps_us.append(timestamps_us[-1] + (timestamps_us[-1] - timestamps_us[-2]))

                    sequence_dynamic_tracks[sequence_id][track_id] = track | {
                        "pose_interpolator": ncore_common.PoseInterpolator(np.stack(poses), timestamps_us)
                    }

        # Second pass over all chunks: combine all per-sequence ego + dynamic parts into per-camera valid pixels
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            # combine ego ego car mask with dynamic object bbox projections to the camera image plane
            for camera_id, camera_frame_range in chunk.camera_frame_ranges.items():
                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_model = self.sequence_camera_models[sequence_id][camera_id]

                for camera_frame_idx in camera_frame_range:
                    frame_valid_pixel_mask = self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][
                        camera_frame_idx
                    ]  # this has been initialized before

                    # check which dynamic labels where observed at this frame's time + project
                    frame_start_timestamp_us = camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, ncore.data.FrameTimepoint.START
                    )
                    frame_end_timestamp_us = camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, ncore.data.FrameTimepoint.END
                    )
                    frame_mid_timestamp = (
                        frame_start_timestamp_us + (frame_end_timestamp_us - frame_start_timestamp_us) // 2
                    )

                    T_world_sensor_start = camera_sensor.get_frame_T_world_sensor(
                        camera_frame_idx, ncore.data.FrameTimepoint.START
                    )
                    T_world_sensor_end = camera_sensor.get_frame_T_world_sensor(
                        camera_frame_idx, ncore.data.FrameTimepoint.END
                    )

                    for dynamic_track in sequence_dynamic_tracks[sequence_id].values():
                        # skip track if out of time with current frame
                        if not (
                            dynamic_track["timestamps_us"][0] <= frame_mid_timestamp
                            and frame_mid_timestamp <= dynamic_track["timestamps_us"][-1]
                        ):
                            continue

                        # interpolate track to frame mid-timestamp
                        bbox_pose = dynamic_track["pose_interpolator"].interpolate_to_timestamps(frame_mid_timestamp)[0]

                        bbox = ncore_transformations.pose_bbox(bbox_pose, dynamic_track["dimension"])

                        bbox_corners = ncore_common.get_3d_bbox_coords(bbox)

                        projection = camera_model.world_points_to_image_points_shutter_pose(
                            bbox_corners,
                            T_world_sensor_start,
                            T_world_sensor_end,
                            return_valid_indices=True,
                            return_all_projections=True,
                        )
                        assert projection.valid_indices is not None

                        # extract the mask and set the pixels to invalid, if at least one point projects to the camera image plane
                        if torch.numel(projection.valid_indices) > 0:
                            object_mask = object_mask_from_image_points(
                                projection.image_points,
                                res_x=int(camera_model.resolution[0]),
                                res_y=int(camera_model.resolution[1]),
                                dilate_ratio=self.valid_pixels_cuboid_tracks_params.track_mask_dilate_ratio,
                            )

                            frame_valid_pixel_mask[
                                int(object_mask[0, 0].item()) : int(object_mask[1, 0].item()),
                                int(object_mask[0, 1].item()) : int(object_mask[1, 1].item()),
                            ] = False

    def _compute_valid_pixels_sceneflow(self):
        """Load scene flow based dynamic mask as valid-pixels masks"""

        assert self.valid_pixels_pixels_scene_flow_params is not None
        assert self.sequence_aux_loaders, "NCoreDataset: Auxiliary data was not loaded"

        dilator = Dilation2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.valid_pixels_pixels_scene_flow_params.flow_dilate_radius,
            soft_max=False,
        ).to("cuda")

        downsample_scale = (
            self.valid_pixels_pixels_scene_flow_params.flow_downsample_scale
        )  # downsample the mask before dilating for memory saving

        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            for camera_id, camera_frame_range in chunk.camera_frame_ranges.items():
                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_model = self.sequence_camera_models[sequence_id][camera_id]

                w, h = camera_model.resolution.cpu().numpy()[:]

                resizer1 = torchvision.transforms.Resize(
                    size=(w // downsample_scale, w // downsample_scale),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )  # for speeding up
                resizer2 = torchvision.transforms.Resize(
                    size=(w, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST
                )

                for camera_frame_idx in camera_frame_range:
                    frame_valid_pixel_mask = self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][
                        camera_frame_idx
                    ]  # this has been initialized before

                    # load scene flow magnitude from annotation shard
                    scene_flow_magnitude = np.asarray(
                        self.sequence_aux_loaders[sequence_id].get_scene_flow_magnitude(
                            camera_id, camera_sensor.get_frame_timestamp_us(camera_frame_idx)
                        )
                    )

                    dynamic_pixels_mask = np.greater(
                        scene_flow_magnitude, self.valid_pixels_pixels_scene_flow_params.flow_min_speed_ms
                    )

                    # mask dilation
                    tensor_to_dilate = to_torch(dynamic_pixels_mask, device="cuda").unsqueeze(0).unsqueeze(0).float()
                    tensor_to_dilate_sq = torch.cat([tensor_to_dilate, torch.zeros_like(tensor_to_dilate)], 2)[
                        :, :, :w, :w
                    ]  # pad it to a square image

                    tensor_dilated = resizer2(dilator(resizer1(tensor_to_dilate_sq)))[
                        :, :, :h, :w
                    ]  # dilating and remove the padded region

                    dynamic_pixels_mask = (tensor_dilated.squeeze() > 0.5).cpu().detach().numpy()  # dilated mask

                    # set dynamic pixels as invalid
                    frame_valid_pixel_mask[dynamic_pixels_mask] = False

    def get_camera_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor ids"""
        if unique_sensors:
            return [unique_id.id for unique_ids in self.camera_unique_ids.values() for unique_id in unique_ids]
        else:
            return self.camera_ids

    def get_lidar_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True) or logical (unique_sensors=False) lidar sensor ids"""
        if unique_sensors:
            return [unique_id.id for unique_ids in self.lidar_unique_ids.values() for unique_id in unique_ids]
        else:
            return self.lidar_ids

    def get_n_frames_per_camera(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor instance"""
        self._init_worker()  # make sure worker is initialized at this point

        if unique_sensors:
            n_frames_per_camera = np.zeros((self.n_unique_cameras,), np.int32)

            for chunk in self.chunks:
                for camera_id in self.camera_ids:
                    camera_unique_idx = self.sequence_camera_unique_ids[chunk.sequence_id][camera_id].idx
                    n_frames_per_camera[camera_unique_idx] += len(chunk.camera_frame_ranges[camera_id])

            return n_frames_per_camera
        else:
            return np.array(
                [len(chunk.camera_frame_ranges[camera_id]) for chunk in self.chunks for camera_id in self.camera_ids],
                dtype=np.int32,
            )

    def get_n_frames_per_lidar(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) lidar sensor instance"""
        self._init_worker()  # make sure worker is initialized at this point

        if unique_sensors:
            n_frames_per_lidar = np.zeros((self.n_unique_lidars,), np.int32)

            for chunk in self.chunks:
                for lidar_id in self.lidar_ids:
                    lidar_unique_idx = self.sequence_lidar_unique_ids[chunk.sequence_id][lidar_id].idx
                    n_frames_per_lidar[lidar_unique_idx] += len(chunk.lidar_frame_ranges[lidar_id])

            return n_frames_per_lidar
        else:
            return np.array(
                [len(chunk.lidar_frame_ranges[lidar_id]) for chunk in self.chunks for lidar_id in self.lidar_ids],
                dtype=np.int32,
            )

    def get_observer_points(self, camera_id=None):
        """ Return camera centers in colmap space """
        # make sure we are initialized
        self._init_worker()

        # default to first camera if not provided explicitly
        assert len(self.camera_ids), "NCoreDataset: no camera sensors loaded"
        camera_id = self.camera_ids[0] if camera_id is None else camera_id

        camera_centers = []
        # provided samples are ordered by chunks
        for unique_camera_id in self._sensor_ids_to_unique_ids([camera_id], "camera"):
            for chunk in self.chunks:
                sequence_id = chunk.sequence_id
                for camera_id in self.sequence_camera_sensors[sequence_id].keys():
                    if self.sequence_camera_unique_ids[sequence_id][camera_id].id != unique_camera_id:
                        continue

                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]

                for camera_frame_index in chunk.time_range_us.cover_range(camera_sensor.get_frames_timestamps_us()):
                    T_sensor_colmap = self._ncore_world_to_colmap_poses(
                        camera_sensor.get_frame_T_sensor_world(
                            camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                        ),
                        chunk.T_world_local_world_common,
                    )
                    camera_centers.append(T_sensor_colmap[:3,3])

        return np.array(camera_centers)

    def __len__(self) -> int:
        """Returns the total number of samples provided by the dataset (depending on split type and parametrization)"""
        # we create ray batches internally using randomization -
        # use a fixed number of samples in the public interface
        if self.split.startswith("train"):
            return self.n_samples_per_epoch

        # make sure worker is initialized at this point
        self._init_worker()

        # sum of all per-camera frame-ranges
        return sum(len(chunk.camera_frame_ranges[camera_id]) for chunk in self.chunks for camera_id in self.camera_ids)

    def _ncore_world_to_colmap_poses(
        self,
        T_poses_world_local: np.ndarray,
        T_world_local_world_common: np.ndarray,
    ) -> np.ndarray:
        """
        Transform input poses 'T_poses_world_local' in local NCore world frame (metric units) to
        poses in colmap frame (in *normalized* units), applying local world -> common world transformation
        (given by 'T_world_local_world_common' (4,4)), frame axis conventions, scene rescaling, and origin offsets.

        Supports both singular (4,4) and batched (N,4,4) input poses 'T_poses_world_local'
        """

        # transform from local world to common world [batch dimensions unconditionally]
        T_poses_world_common = T_world_local_world_common @ T_poses_world_local.reshape((-1, 4, 4))  # (N,4,4)

        return self.world_to_colmap.transform_poses(T_poses_world_common)

    def __getitem__(self, idx) -> Batch:
        """Returns a specific sample of the dataset (depending on split type and parametrization)"""
        # make sure worker is initialized
        self._init_worker()

        if self.split.startswith("train"):
            sample: dict[str, Any] = {"idx": idx, "worker_id": self.worker_id}
            labels = Labels()

            # randomly sample chunk according to per-chunk probability ~ induces underlying sequence to sample from
            chunk_idx = self.rng.choice(np.arange(len(self.chunks)), size=1, p=self.chunk_probabilities).item()
            chunk = self.chunks[chunk_idx]
            sequence_id = chunk.sequence_id

            # randomly sample evaluation time from chunk time-range
            evaluation_timestamps_us = self.rng.integers(
                low=chunk.time_range_us.start,
                high=chunk.time_range_us.end,
                size=self.n_train_sample_timepoints,
                endpoint=False,
            )

            # collect camera rays
            rays: list[np.ndarray] = []
            rgbs: list[np.ndarray] = []
            rays_meta: list[RaysCamMeta] = []
            camera_rays: np.ndarray = np.empty((0, 6), dtype=np.float32)
            camera_rays_meta: Optional[RaysCamMeta] = None
            running_n_camera_rays: int = 0
            # create full index range and split according to camera subdiv-count
            for camera_id, evaluation_timestamp_us, n_rays in zip(
                # repeat camera-ids Ntimes times
                [camera_id for _ in range(self.n_train_sample_timepoints) for camera_id in self.camera_ids],
                # repeat evaluation times Ncameras times
                np.repeat(evaluation_timestamps_us, len(self.camera_ids)),
                # chunk each evaluation evenly w.r.t. total target ray count
                chunk_sizes(len(self.camera_ids) * self.n_train_sample_timepoints, self.n_train_sample_camera_rays),
                # make sure all iterators have same length
                strict=True,
            ):
                if n_rays < 1:
                    # skip sampling empty sets
                    continue

                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_model = self.sequence_camera_models[sequence_id][camera_id]
                camera_unique_idx = self.sequence_camera_unique_ids[sequence_id][camera_id].idx
                camera_all_rays = self.sequence_cameras_all_rays[sequence_id][camera_id]
                camera_all_pixels = self.sequence_cameras_all_pixels[sequence_id][camera_id]
                camera_frame_range = chunk.camera_frame_ranges[camera_id]

                # load closest image frame index to evaluation time sample and make sure that the sampled
                # frame index is within the valid frame range given by time bounds
                # (closest-frame sampling could select out-of-time-bound samples)
                camera_frame_index = max(
                    camera_frame_range.start,
                    min(camera_sensor.get_closest_frame_index(evaluation_timestamp_us), camera_frame_range.stop - 1),
                )

                # load the image
                frame_image_array = camera_sensor.get_frame_image_array(camera_frame_index)

                frame_valid_pixels = camera_all_pixels[
                    self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][camera_frame_index].flatten()
                ]

                # the number of valid pixels is a lower bound for the number of independent rays we can produce
                # (this supports zero valid rays in the limit)
                n_rays = min(len(frame_valid_pixels), n_rays)

                # generate random samples in image domain
                # select random samples from valid pixels
                sample_indices = self.rng.choice(len(frame_valid_pixels), size=n_rays, replace=False, shuffle=False)
                pixel_samples = frame_valid_pixels[sample_indices]
                ray_samples = camera_all_rays[pixel_samples[:, 1], pixel_samples[:, 0]]

                # sample image colors at pixel centers
                rgbs.append(frame_image_array[pixel_samples[:, 1], pixel_samples[:, 0]].astype(np.float32) / 255.0)

                # create world rays in colmap domain
                T_sensor_startend_colmap = self._ncore_world_to_colmap_poses(
                    np.stack(
                        [
                            camera_sensor.get_frame_T_sensor_world(
                                camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                            ),
                            camera_sensor.get_frame_T_sensor_world(
                                camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                            ),
                        ]
                    ),
                    chunk.T_world_local_world_common,
                )
                world_rays_return = camera_model.pixels_to_world_rays_shutter_pose(
                    pixel_samples,
                    T_sensor_startend_colmap[0],
                    T_sensor_startend_colmap[1],
                    start_timestamp_us=camera_sensor.get_frame_timestamp_us(
                        camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                    ),
                    end_timestamp_us=camera_sensor.get_frame_timestamp_us(
                        camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                    ),
                    camera_rays=ray_samples,
                    return_timestamps=True,
                )

                rays.append(world_rays_return.world_rays.numpy())

                flags = torch.full(
                    (n_rays,),
                    (RayFlags.CAMERA_SENSOR | RayFlags.COLMAP_SPACE | RayFlags.RGB_LABEL).value,
                    dtype=torch.int32,
                    device="cpu",
                )

                if aux_loader := self.sequence_aux_loaders.get(sequence_id, None):
                    # load semantic segmentation image
                    semantics = np.asarray(
                        aux_loader.get_semantic_segmentation(
                            camera_id, camera_sensor.get_frame_timestamp_us(camera_frame_index)
                        )
                    )

                    # classify sampled rays for sky
                    sky_indices = (
                        semantics[pixel_samples[:, 1], pixel_samples[:, 0]]
                        == self.sequence_camera_sky_class_ids[sequence_id][camera_id]
                    )

                    # update ray types
                    flags[sky_indices] |= RayFlags.SKY_SEMANTIC.value

                rays_meta.append(
                    RaysCamMeta(
                        flags=flags,
                        unique_sensor_idx=torch.full((n_rays,), camera_unique_idx, dtype=torch.int16, device="cpu"),
                        unique_frame_idx=torch.full(
                            (n_rays,),
                            chunk.camera_linear_start_frame_indices[camera_id]
                            + (camera_frame_index - camera_frame_range.start),
                            dtype=torch.int32,
                            device="cpu",
                        ),
                        timestamp_us=unpack_optional(world_rays_return.timestamps_us),
                        trajectory_idx=torch.full((n_rays,), chunk_idx, dtype=torch.int16, device="cpu"),
                    )
                )

                # Update the running num of lidar rays
                running_n_camera_rays += n_rays

            if len(rays):
                camera_rays = np.vstack(rays)
                camera_rays_meta = RaysCamMeta.collate_fn(rays_meta)
                labels.rgb = to_torch(np.vstack(rgbs), device="cpu")

            # collect lidar rays
            rays_dist: list[np.ndarray] = []
            rays_lidar: list[np.ndarray] = []
            rays_lidar_meta: list[RaysLidarMeta] = []
            lidar_rays: Optional[np.ndarray] = None
            lidar_rays_meta: Optional[RaysLidarMeta] = None
            running_n_lidar_rays: int = 0

            # create full index range and split according to lidar subdiv-count
            if len(self.lidar_ids) and self.n_train_sample_lidar_rays:
                for lidar_id, evaluation_timestamp_us, n_rays in zip(
                    # repeat lidar-ids Ntimes times
                    [lidar_id for _ in range(self.n_train_sample_timepoints) for lidar_id in self.lidar_ids],
                    # repeat evaluation times Nlidars times
                    np.repeat(evaluation_timestamps_us, len(self.lidar_ids)),
                    # chunk each evaluation evenly w.r.t. total target ray count
                    chunk_sizes(len(self.lidar_ids) * self.n_train_sample_timepoints, self.n_train_sample_lidar_rays),
                    # make sure all iterators have same length
                    strict=True,
                ):
                    if n_rays < 1:
                        # skip sampling empty sets
                        continue

                    lidar_sensor = self.sequence_lidar_sensors[sequence_id][lidar_id]
                    lidar_unique_idx = self.sequence_lidar_unique_ids[sequence_id][lidar_id].idx
                    lidar_frame_range = chunk.lidar_frame_ranges[lidar_id]

                    # load closest image frame index to evaluation time sample and make sure that the sampled
                    # frame index is within the valid frame range given by time bounds
                    # (closest-frame sampling could select out-of-time-bound samples)
                    lidar_frame_index = max(
                        lidar_frame_range.start,
                        min(lidar_sensor.get_closest_frame_index(evaluation_timestamp_us), lidar_frame_range.stop - 1),
                    )

                    # filter out dynamic points
                    non_dynamic_point_indices = (
                        lidar_sensor.get_frame_data(lidar_frame_index, "dynamic_flag")
                        != ncore.data.DynamicFlagState.DYNAMIC.value
                    )

                    # load the point clouds (represented in sensor-frame)
                    xyz_s = lidar_sensor.get_frame_data(lidar_frame_index, "xyz_s")[non_dynamic_point_indices]
                    xyz_e = lidar_sensor.get_frame_data(lidar_frame_index, "xyz_e")[non_dynamic_point_indices]
                    timestamp_us = lidar_sensor.get_frame_data(lidar_frame_index, "timestamp_us")[
                        non_dynamic_point_indices
                    ]

                    # subsample rays
                    sample_indices = self.rng.choice(
                        len(xyz_s), size=min(n_rays, len(xyz_s)), replace=False, shuffle=False
                    )
                    xyz_s = xyz_s[sample_indices]
                    xyz_e = xyz_e[sample_indices]
                    timestamp_us = timestamp_us[sample_indices]

                    # transform points from sensor to colmap frame, rescaling from meter to colmap scale
                    T_sensor_colmap = self._ncore_world_to_colmap_poses(
                        lidar_sensor.get_frame_T_sensor_world(lidar_frame_index),
                        chunk.T_world_local_world_common,
                    )

                    xyz_s = (
                        (self.world_to_colmap.target_scale * T_sensor_colmap[:3, :3]) @ xyz_s.transpose()
                        + T_sensor_colmap[:3, 3:4]
                    ).transpose()
                    xyz_e = (
                        (self.world_to_colmap.target_scale * T_sensor_colmap[:3, :3]) @ xyz_e.transpose()
                        + T_sensor_colmap[:3, 3:4]
                    ).transpose()

                    xyz_v = xyz_e - xyz_s  # vector between start -> end points
                    dist = np.linalg.norm(xyz_v, axis=1)

                    # mask out rays that are outside of the maximum distance range
                    dist_mask = dist < (self.world_to_colmap.target_scale * self.max_dist_m)

                    xyz_s = xyz_s[dist_mask]
                    xyz_e = xyz_e[dist_mask]
                    xyz_v = xyz_v[dist_mask]
                    dist = dist[dist_mask]
                    timestamp_us = timestamp_us[dist_mask]

                    n_rays = len(xyz_s)

                    # setup samples
                    rays_dist.append(dist)  # lengths of vectors

                    # 6d: [start point], [normalized directions]
                    rays_lidar.append(np.hstack([xyz_s, xyz_v / dist[..., np.newaxis]]))

                    rays_lidar_meta.append(
                        RaysLidarMeta(
                            flags=torch.full(
                                (n_rays,),
                                (RayFlags.LIDAR_SENSOR | RayFlags.COLMAP_SPACE).value,
                                dtype=torch.int32,
                                device="cpu",
                            ),
                            unique_sensor_idx=torch.full((n_rays,), lidar_unique_idx, dtype=torch.int16, device="cpu"),
                            unique_frame_idx=torch.full(
                                (n_rays,),
                                chunk.lidar_linear_start_frame_indices[lidar_id]
                                + (lidar_frame_index - lidar_frame_range.start),
                                dtype=torch.int32,
                                device="cpu",
                            ),
                            trajectory_idx=torch.full((n_rays,), chunk_idx, dtype=torch.int16, device="cpu"),
                            timestamp_us=to_torch(timestamp_us.astype(np.int64), device="cpu"),
                        )
                    )

                    # Update the running num of lidar rays
                    running_n_lidar_rays += n_rays

                if len(rays_dist):
                    lidar_rays = np.vstack(rays_lidar)
                    lidar_rays_meta = RaysLidarMeta.collate_fn(rays_lidar_meta)
                    labels.lidar = to_torch(np.concatenate(rays_dist), device="cpu")

            sample |= {
                "rays_cam": to_torch(camera_rays, device="cpu"),
                "rays_lidar": to_torch_optional(lidar_rays, device="cpu"),
                "rays_cam_meta": camera_rays_meta,
                "rays_lidar_meta": lidar_rays_meta,
                "labels": labels,
            }

            return Batch(**sample)

        else:
            # decode *linear* global sample index over all camera sensors + chunks into local camera + camera_frame_index (ordered by "camera-major" / "chunk-minor")
            run_frames = 0
            for camera_id in self.camera_ids:
                for chunk_idx, chunk in enumerate(self.chunks):
                    if idx >= run_frames + len(camera_frame_range := chunk.camera_frame_ranges[camera_id]):
                        # current chunk / camera depleted, check next one
                        run_frames += len(camera_frame_range)
                        continue

                    sequence_id = chunk.sequence_id

                    camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                    camera_model = self.sequence_camera_models[sequence_id][camera_id]
                    camera_pixels_subsampled = self.sequence_cameras_pixels_subsample[sequence_id][camera_id]
                    camera_rays_subsampled = self.sequence_cameras_rays_subsample[sequence_id][camera_id]

                    # determine frame of current camera in current chunk
                    camera_frame_index = (idx - run_frames) + camera_frame_range.start

                    # load the image
                    frame_image_array = camera_sensor.get_frame_image_array(camera_frame_index)

                    # sample image colors at pixels centers
                    rgb = (
                        frame_image_array[camera_pixels_subsampled[:, 1], camera_pixels_subsampled[:, 0]].astype(
                            np.float32
                        )
                        / 255.0
                    )

                    # sample valid pixel mask
                    frame_valid_pixel_mask = self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][
                        camera_frame_index
                    ]
                    valid = frame_valid_pixel_mask[camera_pixels_subsampled[:, 1], camera_pixels_subsampled[:, 0]]

                    # create world rays in colmap domain
                    T_sensor_startend_colmap = self._ncore_world_to_colmap_poses(
                        np.stack(
                            [
                                camera_sensor.get_frame_T_sensor_world(
                                    camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frame_T_sensor_world(
                                    camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                                ),
                            ]
                        ),
                        chunk.T_world_local_world_common,
                    )

                    world_rays_return = camera_model.pixels_to_world_rays_shutter_pose(
                        camera_pixels_subsampled,
                        T_sensor_startend_colmap[0],
                        T_sensor_startend_colmap[1],
                        start_timestamp_us=camera_sensor.get_frame_timestamp_us(
                            camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                        ),
                        end_timestamp_us=camera_sensor.get_frame_timestamp_us(
                            camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                        ),
                        camera_rays=camera_rays_subsampled,
                        return_timestamps=True,
                    )

                    camera_rays = world_rays_return.world_rays.numpy()

                    camera_rays_meta = RaysCamMeta(
                        flags=torch.full(
                            (len(camera_rays),),
                            (RayFlags.CAMERA_SENSOR | RayFlags.COLMAP_SPACE | RayFlags.RGB_LABEL).value,
                            dtype=torch.int32,
                            device="cpu",
                        ),
                        unique_sensor_idx=torch.full(
                            (len(camera_rays),), self.camera_ids.index(camera_id), dtype=torch.int16, device="cpu"
                        ),
                        unique_frame_idx=torch.full(
                            (len(camera_rays),), camera_frame_index, dtype=torch.int32, device="cpu"
                        ),
                        trajectory_idx=torch.full((len(camera_rays),), chunk_idx, dtype=torch.int16, device="cpu"),
                        timestamp_us=unpack_optional(world_rays_return.timestamps_us),
                    )

                    return Batch(
                        rays_cam=to_torch(camera_rays, device="cpu"),
                        labels=Labels(rgb=to_torch(rgb, device="cpu"), valid=to_torch(valid, device="cpu")),
                        rays_cam_meta=camera_rays_meta,
                        w=int(camera_model.resolution[0].item() / self.n_val_image_subsample),
                        h=int(camera_model.resolution[1].item() / self.n_val_image_subsample),
                        idx=idx,
                        worker_id=self.worker_id,
                    )

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
        """Returns a generator for all point-clouds available for point-cloud sensor (lidar / camera), transformed into colmap frame.

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

        # provided samples are ordered by sensors, then chunks
        for unique_lidar_id in self._sensor_ids_to_unique_ids(lidar_ids, "lidar"):
            for chunk in self.chunks:
                sequence_id = chunk.sequence_id
                for lidar_id in self.sequence_lidar_sensors[sequence_id].keys():
                    if self.sequence_lidar_unique_ids[sequence_id][lidar_id].id != unique_lidar_id:
                        continue

                    lidar_sensor = self.sequence_lidar_sensors[sequence_id][lidar_id]

                    for lidar_frame_index in chunk.time_range_us.cover_range(lidar_sensor.get_frames_timestamps_us())[
                        ::step_frame
                    ]:
                        # determine point subset to load
                        if non_dynamic_points_only:
                            # filter out dynamic points
                            point_filter = (
                                lidar_sensor.get_frame_data(lidar_frame_index, "dynamic_flag")
                                != ncore.data.DynamicFlagState.DYNAMIC.value
                            )
                        else:
                            # take all points
                            point_filter = ...

                        # load the point clouds (represented in sensor-frame)
                        xyz_s = lidar_sensor.get_frame_data(lidar_frame_index, "xyz_s")[point_filter]
                        xyz_e = lidar_sensor.get_frame_data(lidar_frame_index, "xyz_e")[point_filter]

                        # transform points from sensor to colmap frame, rescaling from meter to colmap scale
                        T_sensor_colmap = self._ncore_world_to_colmap_poses(
                            lidar_sensor.get_frame_T_sensor_world(lidar_frame_index),
                            chunk.T_world_local_world_common,
                        )

                        xyz_s = (
                            (self.world_to_colmap.target_scale * T_sensor_colmap[:3, :3]) @ xyz_s.transpose()
                            + T_sensor_colmap[:3, 3:4]
                        ).transpose()
                        xyz_e = (
                            (self.world_to_colmap.target_scale * T_sensor_colmap[:3, :3]) @ xyz_e.transpose()
                            + T_sensor_colmap[:3, 3:4]
                        ).transpose()

                        yield PointCloud(
                            xyz_start=to_torch(xyz_s, device="cpu"), xyz_end=to_torch(xyz_e, device="cpu"), device="cpu"
                        )

    def get_camera_frusta_and_bbox(
        self,
        camera_id: Optional[str] = None,
        near_plane_depth: float = 0.1,
        far_plane_depth: float = 150.0,
        step_frame: int = 1,
        extent: Optional[torch.Tensor] = None,
    ) -> Generator[tuple[CameraFrustum, BoundingBox], None, None]:
        """Returns a generator for all camera frusta and visibility bboxes for a given camera sensor, transformed into colmap frame.

        Camera sensor are specified by by either logical or unique sensor IDs.

        A single camera sensor needs to be specified - defaults to first camera sensor if not specified."""

        assert (
            near_plane_depth < far_plane_depth
        ), "NCoreDataset: Near plane depth of camera frustum is larger than far plane depth"

        # make sure we are initialized
        self._init_worker()

        # default to first camera if not provided explicitly
        assert len(self.camera_ids), "NCoreDataset: no camera sensors loaded"
        camera_id = self.camera_ids[0] if camera_id is None else camera_id

        # use default extent if not specified
        if extent is None:
            extent = torch.tensor([[-150.0, 150.0], [-150.0, 150.0], [-5.0, 50.0]])

        # provided samples are ordered by chunks
        for unique_camera_id in self._sensor_ids_to_unique_ids([camera_id], "camera"):
            for chunk in self.chunks:
                sequence_id = chunk.sequence_id
                for camera_id in self.sequence_camera_sensors[sequence_id].keys():
                    if self.sequence_camera_unique_ids[sequence_id][camera_id].id != unique_camera_id:
                        continue

                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_model = self.sequence_camera_models[sequence_id][camera_id]

                w = int(camera_model.resolution[0].item())
                h = int(camera_model.resolution[1].item())

                # Initialize the corner uv values
                corner_pixels = torch.tensor([[0, h], [0, 0], [w, 0], [w, h]], dtype=torch.int32)

                for camera_frame_index in chunk.time_range_us.cover_range(camera_sensor.get_frames_timestamps_us())[
                    ::step_frame
                ]:
                    # Extract the rays in the camera coordinate system and compute the distance along the ray
                    camera_rays = camera_model.pixels_to_camera_rays(corner_pixels)

                    # Camera rays are already normalized
                    near_plane_dists = near_plane_depth / camera_rays[:, 2:3]
                    far_plane_dists = far_plane_depth / camera_rays[:, 2:3]

                    T_sensor_startend_colmap = self._ncore_world_to_colmap_poses(
                        np.stack(
                            [
                                camera_sensor.get_frame_T_sensor_world(
                                    camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frame_T_sensor_world(
                                    camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                                ),
                            ]
                        ),
                        chunk.T_world_local_world_common,
                    )
                    rays = camera_model.pixels_to_world_rays_mean_pose(
                        corner_pixels,
                        T_sensor_startend_colmap[0],
                        T_sensor_startend_colmap[1],
                        camera_rays=camera_rays,
                    ).world_rays

                    # For origin we take the mean value of the ray origins
                    corners = torch.cat(
                        [
                            rays[:, :3] + rays[:, 3:6] * near_plane_dists * self.world_to_colmap.target_scale,
                            rays[:, :3] + rays[:, 3:6] * far_plane_dists * self.world_to_colmap.target_scale,
                        ],
                        dim=0,
                    )

                    T_rig_colmap = self._ncore_world_to_colmap_poses(
                        camera_sensor.get_frame_T_rig_world(
                            camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                        ),
                        chunk.T_world_local_world_common,
                    )

                    yield CameraFrustum(corners=corners), BoundingBox(
                        extent=extent * self.world_to_colmap.target_scale,
                        T_rig_world=to_torch(T_rig_colmap, device="cpu"),
                    )

    def get_sky_rays(
        self,
        camera_id: Optional[str] = None,
        step_frame: int = 1,
        step_pixel: int = 1,
    ) -> Generator[torch.Tensor, None, None]:
        """Returns a generator for all camera rays (Nx6) belonging to a given semantic class, transformed into colmap frame.

        Camera sensor are specified by by either logical or unique sensor IDs.

        Each generated sample represents a different frame."""

        # make sure we are initialized
        self._init_worker()

        # aux data is required for sky rays
        assert self.sequence_aux_loaders, "NCoreDataset: Auxiliary data was not loaded"

        # default to first camera if not provided explicitly
        assert len(self.camera_ids), "NCoreDataset: no camera sensors loaded"
        camera_id = self.camera_ids[0] if camera_id is None else camera_id

        # provided samples are ordered by chunks
        for unique_camera_id in self._sensor_ids_to_unique_ids([camera_id], "camera"):
            for chunk in self.chunks:
                sequence_id = chunk.sequence_id
                for camera_id in self.sequence_camera_sensors[sequence_id].keys():
                    if self.sequence_camera_unique_ids[sequence_id][camera_id].id != unique_camera_id:
                        continue

                camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]
                camera_model = self.sequence_camera_models[sequence_id][camera_id]

                # get rays of the given class
                camera_all_pixels = self.sequence_cameras_all_pixels[sequence_id][camera_id]
                camera_all_rays = self.sequence_cameras_all_rays[sequence_id][camera_id]

                for camera_frame_index in chunk.time_range_us.cover_range(camera_sensor.get_frames_timestamps_us())[
                    ::step_frame
                ]:
                    # load semantic segmentation image
                    semantics = np.asarray(
                        self.sequence_aux_loaders[sequence_id].get_semantic_segmentation(
                            camera_id, camera_sensor.get_frame_timestamp_us(camera_frame_index)
                        )
                    )

                    valid_pixels = camera_all_pixels[
                        self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][
                            camera_frame_index
                        ].flatten()
                    ]
                    valid_rays = camera_all_rays[valid_pixels[:, 1], valid_pixels[:, 0]]

                    # extract valid ray indices and subsample them with step_pixel if needed
                    ray_indices = (
                        semantics[valid_pixels[:, 1], valid_pixels[:, 0]]
                        == self.sequence_camera_sky_class_ids[sequence_id][camera_id]
                    )

                    T_sensor_startend_colmap = self._ncore_world_to_colmap_poses(
                        np.stack(
                            [
                                camera_sensor.get_frame_T_sensor_world(
                                    camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frame_T_sensor_world(
                                    camera_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                                ),
                            ]
                        ),
                        chunk.T_world_local_world_common,
                    )
                    rays = camera_model.pixels_to_world_rays_shutter_pose(
                        valid_pixels[ray_indices, :][::step_pixel],
                        T_sensor_startend_colmap[0],
                        T_sensor_startend_colmap[1],
                        camera_rays=valid_rays[ray_indices, :][::step_pixel],
                    ).world_rays

                    yield rays

    def get_visibility_bbox(
        self,
        extent: Optional[torch.Tensor] = None,
        step_frame: int = 1,
    ) -> Generator[BoundingBox, None, None]:
        """Returns a generator for all visibility bounding boxes (area that is observed from the sensors) at each sampled rig pose of each chunk."""

        # make sure we are initialized
        self._init_worker()

        # use default extent if not specified
        if extent is None:
            extent = torch.tensor([[-150.0, 150.0], [-150.0, 150.0], [-5.0, 50.0]])

        # provided samples are ordered by chunks
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            poses_data = self.sequence_loaders[sequence_id].get_poses()

            for pose in poses_data.T_rig_worlds[chunk.time_range_us.cover_range(poses_data.T_rig_world_timestamps_us)][
                ::step_frame
            ]:
                T_rig_colmap = self._ncore_world_to_colmap_poses(
                    pose,
                    chunk.T_world_local_world_common,
                )

                yield BoundingBox(
                    extent=extent * self.world_to_colmap.target_scale, T_rig_world=to_torch(T_rig_colmap, device="cpu")
                )

    # Data export functionality
    @dataclass(slots=True)
    class CameraFrameExport:
        """Represents one exported camera-associated frame"""

        sequence_id: str
        timestamp_start_us: int
        timestamp_end_us: int
        T_sensor_to_world_start: npt.NDArray[np.float64]
        T_sensor_to_world_end: npt.NDArray[np.float64]
        camera_model_parameters: ncore_datatypes.ConcreteCameraModelParametersUnion
        image_data: ncore_datatypes.EncodedImageData  # TODO(janickm): should be exported publicly in NCore API
        valid_pixels_mask: npt.NDArray[np.bool_]

        # Aux data - exported if aux is enabled
        sem_seg_meta: Optional[dict]
        sem_seg_image: Optional[PILImage.Image]

    def export_camera_frames(self, camera_id: str) -> Generator[NCoreDataset.CameraFrameExport, None, None]:
        """Returns a generator for all camera frame data of a given sensor"""

        # make sure we are initialized
        self._init_worker()

        # provided frames are ordered by chunks
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            camera_sensor = self.sequence_camera_sensors[sequence_id][camera_id]

            sem_seg_meta: dict | None = None
            if (aux_loader := self.sequence_aux_loaders.get(sequence_id)) is not None:
                sem_seg_meta = aux_loader.get_semantic_segmentation_meta(camera_id)

            for camera_frame_idx in chunk.camera_frame_ranges[camera_id]:
                sem_seg_image: PILImage.Image | None = None
                if aux_loader is not None:
                    sem_seg_image = aux_loader.get_semantic_segmentation(
                        camera_id, camera_sensor.get_frame_timestamp_us(camera_frame_idx)
                    )

                yield NCoreDataset.CameraFrameExport(
                    sequence_id=sequence_id,
                    timestamp_start_us=camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    ),
                    timestamp_end_us=camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                    ),
                    T_sensor_to_world_start=camera_sensor.get_frame_T_sensor_world(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    ),
                    T_sensor_to_world_end=camera_sensor.get_frame_T_sensor_world(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                    ),
                    camera_model_parameters=camera_sensor.get_camera_model_parameters(),
                    image_data=camera_sensor.get_frame_data(camera_frame_idx),
                    valid_pixels_mask=self.sequence_cameras_frame_valid_pixels_masks[sequence_id][camera_id][
                        camera_frame_idx
                    ],
                    sem_seg_meta=sem_seg_meta,
                    sem_seg_image=sem_seg_image,
                )

    @dataclass(slots=True)
    class LidarFrameExport:
        """Represents one exported lidar-associated frame"""

        sequence_id: str
        timestamp_start_us: int
        timestamp_end_us: int
        T_sensor_to_world_start: npt.NDArray[np.float64]
        T_sensor_to_world_end: npt.NDArray[np.float64]
        xyz_s: npt.NDArray[np.float32]  # in sensor frame at end time
        xyz_e: npt.NDArray[np.float32]  # in sensor frame at end time
        intensity: npt.NDArray[np.float32]
        dynamic_flag: npt.NDArray[np.int8]

    def export_lidar_frames(self, lidar_id: str) -> Generator[NCoreDataset.LidarFrameExport, None, None]:
        """Returns a generator for all lidar frame data of a given sensor"""

        # make sure we are initialized
        self._init_worker()

        # provided frames are ordered by chunks
        for chunk in self.chunks:
            sequence_id = chunk.sequence_id

            lidar_sensor = self.sequence_lidar_sensors[sequence_id][lidar_id]

            for lidar_frame_idx in chunk.lidar_frame_ranges[lidar_id]:
                xyz_s = lidar_sensor.get_frame_data(lidar_frame_idx, "xyz_s")
                xyz_e = lidar_sensor.get_frame_data(lidar_frame_idx, "xyz_e")
                intensity = lidar_sensor.get_frame_data(lidar_frame_idx, "intensity")
                dynamic_flag = lidar_sensor.get_frame_data(lidar_frame_idx, "dynamic_flag")

                yield NCoreDataset.LidarFrameExport(
                    sequence_id=sequence_id,
                    timestamp_start_us=lidar_sensor.get_frame_timestamp_us(
                        lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    ),
                    timestamp_end_us=lidar_sensor.get_frame_timestamp_us(
                        lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                    ),
                    T_sensor_to_world_start=lidar_sensor.get_frame_T_sensor_world(
                        lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    ),
                    T_sensor_to_world_end=lidar_sensor.get_frame_T_sensor_world(
                        lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                    ),
                    xyz_s=xyz_s,
                    xyz_e=xyz_e,
                    intensity=intensity,
                    dynamic_flag=dynamic_flag,
                )

    def get_rig_trajectories(self, end_frame_timestamps_only: bool = False) -> RigTrajectories:
        """Returns all rig trajectories associated with the dataset

        - end_frame_timestamps_only (default = False):
            only return frame-end timestamps if True (frame-start timestamps times will be -1),
            faster, as not all individual frames need to be loaded
        """

        # make sure we are initialized
        self._init_worker()

        def rig_trajectories_generator() -> Generator[RigTrajectories.RigTrajectory, None, None]:
            """Produces individual rig trajectories"""

            # generate one rig trajectory for each chunk
            for chunk in self.chunks:
                # collect chunk-associated rig poses
                loader = self.sequence_loaders[sequence_id := chunk.sequence_id]

                sequence_poses = loader.get_poses()

                # subselect and transform local world to common world
                T_rig_world_common: np.ndarray = (
                    chunk.T_world_local_world_common
                    @ sequence_poses.T_rig_worlds[
                        poses_range := chunk.time_range_us.cover_range(sequence_poses.T_rig_world_timestamps_us)
                    ]
                )
                T_rig_world_timestamps_us = sequence_poses.T_rig_world_timestamps_us[poses_range]

                def get_sensor_frame_timestamps_us(sensor: ncore.data.v3.Sensor, frame_range: range) -> torch.Tensor:
                    if not end_frame_timestamps_only:
                        # return both start and end timestamps (requires loading each individual frame)
                        return torch.tensor(
                            [
                                [
                                    sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.START),
                                    sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.END),
                                ]
                                for frame_idx in frame_range
                            ],
                            # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                            dtype=torch.int64,
                            device="cpu",
                        )
                    else:
                        # return end-timestamps only (fill start times with -1)
                        frame_end_timestamps_us = to_torch(
                            sensor.get_frames_timestamps_us()[frame_range]
                            # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                            .astype(np.int64),
                            device="cpu",
                        ).reshape(-1, 1)
                        return torch.cat(
                            (torch.full_like(frame_end_timestamps_us, -1), frame_end_timestamps_us),
                            dim=1,
                        )

                yield RigTrajectories.RigTrajectory(
                    T_rig_worlds=to_torch(T_rig_world_common, device="cpu"),
                    T_rig_world_timestamps_us=to_torch(
                        T_rig_world_timestamps_us
                        # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                        .astype(np.int64),
                        device="cpu",
                    ),
                    cameras_frame_timestamps_us={
                        self.sequence_camera_unique_ids[sequence_id][camera_id].id: get_sensor_frame_timestamps_us(
                            self.sequence_camera_sensors[sequence_id][camera_id], chunk.camera_frame_ranges[camera_id]
                        )
                        for camera_id in self.camera_ids
                    },
                    lidars_frame_timestamps_us={
                        self.sequence_lidar_unique_ids[sequence_id][lidar_id].id: get_sensor_frame_timestamps_us(
                            self.sequence_lidar_sensors[sequence_id][lidar_id], chunk.lidar_frame_ranges[lidar_id]
                        )
                        for lidar_id in self.lidar_ids
                    },
                )

        def camera_calibrations_generator() -> Generator[tuple[str, RigTrajectories.CameraCalibration], None, None]:
            """Produces individual camera calibrations"""

            for sequence_id, camera_sensors in self.sequence_camera_sensors.items():
                for camera_id, camera_sensor in camera_sensors.items():
                    unique_id = self.sequence_camera_unique_ids[sequence_id][camera_id]
                    yield (
                        unique_id.id,  # unique sensor id
                        RigTrajectories.CameraCalibration(
                            logical_sensor_name=camera_id,  # logical sensor name
                            unique_sensor_idx=unique_id.idx,  # unique sensor index
                            T_sensor_rig=to_torch(camera_sensor.get_T_sensor_rig(), device="cpu"),
                            camera_model_parameters=camera_sensor.get_camera_model_parameters(),
                        ),
                    )

        def lidar_calibrations_generator() -> Generator[tuple[str, RigTrajectories.LidarCalibration], None, None]:
            """Produces individual lidar calibrations"""

            for sequence_id, lidar_sensors in self.sequence_lidar_sensors.items():
                for lidar_id, lidar_sensor in lidar_sensors.items():
                    unique_id = self.sequence_lidar_unique_ids[sequence_id][lidar_id]
                    yield (
                        unique_id.id,  # unique sensor id
                        RigTrajectories.LidarCalibration(
                            logical_sensor_name=lidar_id,  # logical sensor name
                            unique_sensor_idx=unique_id.idx,  # unique sensor index
                            T_sensor_rig=to_torch(lidar_sensor.get_T_sensor_rig(), device="cpu"),
                        ),
                    )

        return RigTrajectories(
            T_world_base=to_torch(self.T_world_common_world_base, device="cpu"),
            world_to_colmap=self.world_to_colmap,
            rig_trajectories=list(rig_trajectories_generator()),
            camera_calibrations=dict(list(camera_calibrations_generator())),
            lidar_calibrations=dict(list(lidar_calibrations_generator())),
        )
