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

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import dataclasses_json


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

    def __contains__(self, item: int | HalfClosedInterval) -> bool:
        if isinstance(item, int):
            return self.start <= item < self.end
        elif isinstance(item, HalfClosedInterval):
            return (self.start <= item.start) and (item.end <= self.end)
        else:
            raise TypeError(f"Expected int or HalfClosedInterval, got {type(item).__name__}")


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


@dataclasses.dataclass(slots=False, kw_only=True)
class Batch:
    """
    Contains
        - idx: batch idx sampled by the dataloader (only valid during val/test time) [int]
        - labels: supervision signal for the rays (color or distance) [float] (n_rays, 3)
        - T_camera_to_world: camera-to-world START pose (4, 4) in world-global space [float]
        - T_camera_to_world_end: camera-to-world END pose (4, 4) for rolling shutter interpolation [float]
        - camera_id: camera identifier string for intrinsics / ray cache lookup
        - h: height of the image [int]
        - w: width of the image [int]

    Camera-space rays are NOT included in the batch. Instead, they are cached on GPU per
    worker in NCoreDatasetAdapter and looked up by (camera_id, w, h) in get_gpu_batch_with_intrinsics.
    The renderer handles rolling shutter interpolation via T_camera_to_world / T_camera_to_world_end
    and the shutter_type from intrinsics.
    """

    idx: list[int]
    labels: Labels
    T_camera_to_world: torch.Tensor  # (4, 4) camera-to-world in world-global space (START pose)
    T_camera_to_world_end: torch.Tensor  # (4, 4) camera-to-world END pose (for rolling shutter interpolation)
    camera_id: Union[str, list[str], None] = None  # Camera ID string for intrinsics / ray cache lookup
    h: Union[list[int], int, None] = None
    w: Union[list[int], int, None] = None
    worker_id: Union[list[int], int, None] = None
    batch_size: Optional[int] = None
    frame_time: Union[float, list[float], None] = (
        None  # Frame time in milliseconds, relative to first frame (starts at 0.0)
    )
    frame_idx: Union[int, list[int], None] = (
        None  # 0-based contiguous training frame index (camera-blocked); -1 for validation
    )
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
