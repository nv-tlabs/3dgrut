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

import dataclasses

from collections import defaultdict

from typing import Any, Optional, Sequence, Union
from enum import IntFlag, auto

import numpy as np
import numpy.typing as npt
import torch
import dataclasses_json


class RayFlags(IntFlag):
    """Bitmask flags of per-ray properties (note: limited to 32 variants)"""

    # the ray's coordinates frame [mutually exclusive]
    COLMAP_SPACE = auto()  # set if the ray's 6d coords are relative to the metric COLMAP space

    # the ray's sensor [mutually exclusive]
    CAMERA_SENSOR = auto()  # set if the ray originates from a camera sensor

    # general ray-associated attributes [non-mutually exclusive]
    RGB_LABEL = auto()  # set if the ray has associated RGB values


@dataclasses.dataclass(slots=True)
class HalfClosedInterval:
    """Represents a closed interval [start, end)"""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start <= self.end

    def cover_range(self, sorted_samples: np.ndarray) -> range:
        """Given a set of *sorted* samples (not validated), return the corresponding range for samples
        that are within the interval"""
        cover_range_start = np.argmax(self.start <= sorted_samples).item()
        cover_range_stop = (
            np.argmin(sorted_samples < self.end).item() if self.end < sorted_samples[-1] else len(sorted_samples)
        )  # full range of frames

        return range(cover_range_start, cover_range_stop)




@dataclasses.dataclass(slots=False, kw_only=True)
class Labels:
    """
    Contains
        - rgb: supervision data coming for camera images
        - valid: valid flag for each ray. the rays can be invalid due to e.g. motion state, mask, ...
    """

    rgb: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None

    @staticmethod
    def collate_fn(labels: Union[Labels, list[Labels]]) -> Labels:
        if isinstance(labels, Labels):
            return labels

        out_default = defaultdict(list)
        for sample in labels:
            for k, v in sample.__dict__.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    v = v.unsqueeze(0)
                if v is not None:
                    out_default[k].append(v)
        out = {k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else None for k, v in out_default.items()}

        return Labels(**out)


@dataclasses.dataclass(slots=True, kw_only=True)
class RaysMeta:
    """
    Contains meta data common to all rays irrespective of the sensor type they originate from:
        - flags: bitmask integer value of each ray's flags (see RayFlags)                                        (n_rays,) [int32]
        - unique_sensor_idx: the unique sensor index (of this specific sensor type!) the ray originates from     (n_rays,) [int16]
                             This means that both unique camera and lidar sensor indices are enumerated with
                             [0,1..], respectively.
        - unique_frame_idx: the unique frame idx (across *all* sensors of this type!) the ray originates from    (n_rays,) [int32]
                            This means that both unique camera and lidar frame indices are enumerated with
                            [0,1..], respectively.
        - timestamp_us:     the time-of-measurement of the ray                                                   (n_rays,) [int64]
                            (signed integer only because torch doesn't support large unsigned values yet - note
                            that int64 can't represent full unix epoch timestamps)
    """

    # mandatory fields
    flags: torch.Tensor
    unique_sensor_idx: torch.Tensor
    unique_frame_idx: torch.Tensor

    # optional fields
    timestamp_us: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # make sure all members have same shape and expected types

        assert self.flags.dtype == torch.int32
        assert self.flags.dim() == 1
        assert self.unique_sensor_idx.dtype == torch.int16
        assert self.unique_sensor_idx.shape == self.flags.shape
        assert self.unique_frame_idx.dtype == torch.int32
        assert self.unique_frame_idx.shape == self.flags.shape
        if (timestamp_us := self.timestamp_us) is not None:
            assert timestamp_us.dtype == torch.int64
            assert timestamp_us.shape == self.flags.shape

    def _getitem_basedict(self, indices: torch.Tensor | slice) -> dict[str, torch.Tensor]:
        """Base case to mimic indexing for subsampling and chunking"""

        # mandatory fields are always returned
        out = {
            "flags": self.flags[indices],
            "unique_sensor_idx": self.unique_sensor_idx[indices],
            "unique_frame_idx": self.unique_frame_idx[indices],
        }

        # return optional fields only if present
        if (timestamp_us := self.timestamp_us) is not None:
            out |= {"timestamp_us": timestamp_us[indices]}

        return out

    @staticmethod
    def _collate_fn_basedict(rays_meta: Union[RaysMeta, Sequence[RaysMeta]]) -> dict[str, torch.Tensor]:
        """Collate function for base ray meta properties (extended by derived types)"""
        if isinstance(rays_meta, RaysMeta):
            return rays_meta.__dict__

        # mandatory fields are always returned
        assert len(rays_meta), "Sequence of rays meta is empty"
        out = {
            "flags": torch.cat([s.flags for s in rays_meta], dim=0),
            "unique_sensor_idx": torch.cat([s.unique_sensor_idx for s in rays_meta], dim=0),
            "unique_frame_idx": torch.cat([s.unique_frame_idx for s in rays_meta], dim=0),
        }

        # return optional fields only if present
        if timestamp_us_list := [s.timestamp_us for s in rays_meta if s.timestamp_us is not None]:
            out |= {
                "timestamp_us": torch.cat(timestamp_us_list, dim=0),
            }

        return out


class RaysCamMeta(RaysMeta):
    """
    Contains additional meta-data specific to camera rays only [currently empty]
    """

    def __post_init__(self) -> None:
        super().__post_init__()

    def __getitem__(self, indices: torch.Tensor | slice) -> RaysCamMeta:
        """Mimic indexing for subsampling and chunking"""

        # no additional properties to extend base-type for now
        return RaysCamMeta(**self._getitem_basedict(indices))

    @staticmethod
    def collate_fn(rays_meta: Union[RaysCamMeta, Sequence[RaysCamMeta]]) -> RaysCamMeta:
        """Collate function for camera ray meta properties"""
        if isinstance(rays_meta, RaysCamMeta):
            return rays_meta

        # no additional properties to extend base-type for now
        return RaysCamMeta(**RaysMeta._collate_fn_basedict(rays_meta))


@dataclasses.dataclass(slots=False, kw_only=True)
class Batch:
    """
    Contains
        - rays_cam: origins and directions of the camera rays used for training [float] (n_rays, 6)
        - rays_cam_meta: cam ray's metadata (see RaysCamMeta)
        - idx: batch idx sampled by the dataloader (only valid during val/test time) [int]
        - labels: supervision signal for the rays (color or distance) [float] (n_rays, 3)
        - worker_id: ID of the worker that generated this batch (None if batch is not generated in a multi-worker env) [int]
        - h: height of the image - used in validation and test mode [int]
        - w: width of the image - used in validation and test mode [int]
        - T_camera_to_world: camera-to-world transformation matrix (4, 4) in COLMAP space [float]
    """

    rays_cam: torch.Tensor
    rays_cam_meta: RaysCamMeta
    idx: list[int]
    labels: Labels
    worker_id: Union[list[int], int, None] = None
    T_camera_to_world: Optional[torch.Tensor] = None  # (4, 4) camera-to-world in COLMAP space (START pose)
    T_camera_to_world_end: Optional[torch.Tensor] = None  # (4, 4) camera-to-world END pose for rolling shutter
    rays_in_world_space: bool = False  # True if rays are already in world space (no transform needed)
    h: Union[list[int], int, None] = None
    w: Union[list[int], int, None] = None
    batch_size: Optional[int] = None
    camera_id: Union[str, list[str], None] = None  # Camera ID string for intrinsics lookup
    frame_time: Union[float, list[float], None] = None  # Frame time in milliseconds, relative to first frame (starts at 0.0)
    frame_idx: Union[int, list[int], None] = None  # 0-based contiguous training frame index (camera-blocked); -1 for validation
    camera_idx: Union[int, list[int], None] = None  # Camera index

    @staticmethod
    def collate_fn(batch: Union[Batch, list[Batch]]) -> Batch:
        # Even if the batch is returned directly, we should still process it to have consistent types
        if isinstance(batch, Batch):
            batch = [batch]

        out_default = defaultdict(list)
        out: dict[str, Any] = {}

        for sample in batch:
            for k, v in sample.__dict__.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    v = v.unsqueeze(0)
                if v is not None:
                    out_default[k].append(v)

        for k, v in out_default.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.cat(v, dim=0)
            elif v[0] is None:
                out[k] = None
            else:
                out[k] = v

        out["rays_cam_meta"] = RaysCamMeta.collate_fn(out["rays_cam_meta"])

        out["labels"] = Labels.collate_fn(out["labels"])

        out["batch_size"] = len(batch)

        return Batch(**out)


def field_numpy_array(dtype_like: npt.DTypeLike, shape: tuple[int, ...], *args, **kwargs):
    """Provides encoder / decoder functionality for numpy arrays or dicts or arrays into field types compatible with dataclass-JSON

    Args:
    - dtype_like: required for encoding / decoding (encoded inputs are verified to have this datatype, and decoded arrays will be returned with this type)
    - shape: tuple of array dimensions, used for consistency checks (one dimension is allowed to be -1, similar to reshape)
    - remaining args/kwargs are forward to dataclasses.field()
    """

    dtype = np.dtype(dtype_like)

    def validate_shape(array: np.ndarray) -> np.ndarray:
        assert len(array.shape) == len(shape) and all(
            [dim == expected_dim if expected_dim != -1 else True for (dim, expected_dim) in zip(array.shape, shape)]
        ), f"Array {array} not having expected shape {shape}"
        return array

    def decoder(input: list | dict[Any, list]) -> np.ndarray | dict[Any, np.ndarray]:
        match input:
            case dict():
                # decode dict[<key-type>, list]
                return {key: validate_shape(np.array(value, dtype=dtype)) for (key, value) in input.items()}
            case list():
                # encode list
                return validate_shape(np.array(input, dtype=dtype))
            case _:
                raise ValueError(f"field_numpy_array: unsupported decoder input type {type(input)}")

    def encoder(input: np.ndarray | dict[Any, np.ndarray]) -> list | dict[Any, list]:
        match input:
            case dict():
                # encode as dict[<key-type>, list]
                assert all(
                    [array.dtype == dtype for array in input.values()]
                ), f"Not all arrays in {input} of expected dtype {dtype}"
                return {key: np.ndarray.tolist(validate_shape(array)) for (key, array) in input.items()}
            case np.ndarray():
                # encode as list
                assert input.dtype == dtype, f"Provided array {input} is not of expected dtype {dtype}"
                return np.ndarray.tolist(validate_shape(input))
            case _:
                raise ValueError(f"field_numpy_array: unsupported encoder input type {type(input)}")

    return dataclasses.field(metadata=dataclasses_json.config(encoder=encoder, decoder=decoder), *args, **kwargs)


@dataclasses.dataclass(slots=True, kw_only=True)
class FrameConversion(dataclasses_json.DataClassJsonMixin):
    """Represents parameters and functions to convert frame-associated data between different (potentially uniformly scaled) canonical 3d frames"""

    # Homogeneous source -> target transformation of the form
    #
    # ⎡ R  -o ⎤
    # ⎣ 0 1/s ⎦
    #
    # with
    # - R: source -> target frame orientation with det(R)=1 (3,3)
    # - o: origin of the target frame in the source frame (in source-frame units) (3,1)
    # - s: the source -> target scale
    matrix: npt.NDArray[np.float32] = field_numpy_array(np.float32, (4, 4))

    @classmethod
    def from_origin_scale_axis(
        cls, target_origin: npt.NDArray[np.float32], target_scale: float, target_axis: list[int]
    ):
        """Construct FrameConversion from
        - target_origin: origin of the target frame relative to the source frame (in source-frame units)
        - target_scale: uniform scale of the target frame relative to the source frame
        - target_axis: The target's frame axis order relative to the source frame using axis indices.
                       For instance, an axis conversion of xyz[source] -> yzx[target] would be represented by [1, 2, 0]
        """
        # Construct homogeneous transformation matrix from translation / scale / orientation components
        matrix = np.eye(4, dtype=np.float32)

        # translation
        matrix[:3, 3] = -target_origin

        # scale
        matrix[3, 3] = 1 / target_scale

        # axis swap
        assert len(np.unique(target_axis)) == 3
        matrix = matrix[target_axis + [3]]

        return cls(matrix=matrix)

    def __post_init__(self):
        assert self.matrix.shape == (4, 4)
        assert self.matrix.dtype == np.float32
        assert self.matrix[3, 3] > 0.0
        assert np.isclose(np.linalg.det(self.matrix[:3, :3]), 1.0)

    @property
    def target_scale(self) -> float:
        """The uniform scale of the target frame relative to the source frame"""
        return 1 / self.matrix[3, 3]

    def get_transformation_matrices(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Returns scale-aware (4,4) matrices T / S, which can be used to transform

        - source *points* / *vectors* x_source to the target frame via

          x_target = T @ x_source

        - source *poses* P_source to the target frame via

          P_target = T @ P_source @ S

          Resulting poses have target frame scale when incorporating S, or source frame scale if omitting S
        """

        # T has the form
        # ⎡ s*R -s*o ⎤
        # ⎣ 0    1   ⎦
        T = self.matrix.copy()
        T *= self.target_scale

        # S has the form
        # ⎡ 1/s*I 0 ⎤
        # ⎣ 0     1 ⎦
        inv_s = self.matrix[3, 3]
        S = np.zeros((4, 4), dtype=np.float32)
        np.fill_diagonal(S, [inv_s, inv_s, inv_s, 1.0])

        return (T, S)

    def transform_poses(
        self,
        T_poses_source: np.ndarray,
    ) -> np.ndarray:
        """Transforms poses in the source frame to corresponding poses in the target frame.

        Returned poses have target frame units.

        Supports both singular (4,4) and batched (N,4,4) input poses 'T_poses_source'
        """

        # batch dimensions unconditionally
        T_poses = T_poses_source.reshape((-1, 4, 4))  # (N,4,4)

        # apply transformation
        T, S = self.get_transformation_matrices()
        T_poses = T @ T_poses @ S

        # unbatch dimensions conditionally
        return T_poses.squeeze()  # (N,4,4) or (4,4)

