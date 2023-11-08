# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import concurrent
import io
import dataclasses

from collections import defaultdict
from pathlib import Path
from types import NoneType

from typing import Any, DefaultDict, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable
from enum import IntFlag, auto

import numpy as np
import numpy.typing as npt
import torch
import torchvision
import zarr
import PIL.Image as PILImage
import dataclasses_json

from thirdparty.pytorch_morphological_dilation2d_erosion2d.morphology import Erosion2d

import ncore.data.v3
import ncore.impl.data.stores as ncore_stores
import ncore.impl.data.types as ncore_types


class RayFlags(IntFlag):
    """Bitmask flags of per-ray properties (note: limited to 32 variants)"""

    # the ray's coordinates frame [mutually exclusive]
    WORLD_SPACE = auto()  # set if the ray's 6d coords are relative to the metric WORLD space
    COLMAP_SPACE = auto()  # set if the ray's 6d coords are relative to the metric COLMAP space

    # the ray's sensor [mutually exclusive]
    CAMERA_SENSOR = auto()  # set if the ray originates from a camera sensor
    LIDAR_SENSOR = auto()  # set if the ray originates from a lidar sensor

    # general ray-associated attributes [non-mutually exclusive]
    RGB_LABEL = auto()  # set if the ray has associated RGB values
    SKY_SEMANTIC = auto()  # set if the ray is classified to be a sky ray


@dataclasses.dataclass(slots=True)
class HalfClosedInterval:
    """Represents a closed interval [start, end)"""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start <= self.end

    def __contains__(self, item) -> bool:
        return self.start <= item < self.end

    def intersection(self, other: HalfClosedInterval) -> Optional[HalfClosedInterval]:
        """Computes the intersection of two half-closed interval"""
        if other.start >= self.end or other.end <= self.start:
            return None

        return HalfClosedInterval(max(self.start, other.start), min(self.end, other.end))

    def overlaps(self, other: HalfClosedInterval) -> bool:
        """Checks if the interval has a non-zero overlap with an other closed interval"""
        return self.intersection(other) is not None

    def cover_range(self, sorted_samples: np.ndarray) -> range:
        """Given a set of *sorted* samples (not validated), return the corresponding range for samples
        that are within the interval"""
        cover_range_start = np.argmax(self.start <= sorted_samples).item()
        cover_range_stop = (
            np.argmin(sorted_samples < self.end).item() if self.end < sorted_samples[-1] else len(sorted_samples)
        )  # full range of frames

        return range(cover_range_start, cover_range_stop)


@dataclasses.dataclass
class SequenceChunk:
    """Represents a chunk (given by time-range) within a sequence"""

    sequence_id: str
    time_range_us: HalfClosedInterval

    def time_length_us(self) -> int:
        return self.time_range_us.end - self.time_range_us.start

    def time_length_sec(self) -> float:
        return self.time_length_us() / 1e6


@dataclasses.dataclass(slots=False, kw_only=True)
class Labels:
    """
    Contains
        - rgb: supervision data coming for camera images
        - lidar: distance supervision labels based on LiDAR measurements (3d offsets along the ray's direction) [NGP scale]
        - semantic: semantic labels inferred for camera rays/pixels using a pretrained semantic seg network
        - distance: supervision from, e.g., a depth camera [NGP scale]
          (note that this is *not* the depth offset along the camera's principal axis, but a 3d offsets along the ray's direction)
        - valid: valid flag for each ray. the rays can be invalid due to e.g. motion state, mask, ...
    """

    rgb: Optional[torch.Tensor] = None
    lidar: Optional[torch.Tensor] = None
    semantic: Optional[torch.Tensor] = None
    distance: Optional[torch.Tensor] = None
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


@runtime_checkable
class Chunkable(Protocol):
    """Marks classes as chunkable"""

    def __getitem__(self: Chunkable, key: Any) -> Any:
        pass


@dataclasses.dataclass(slots=True, kw_only=True)
class RaysMeta(Chunkable):
    """
    Contains meta data common to all rays irrespective of the sensor type they originate from:
        - flags: bitmask integer value of each ray's flags (see RayFlags)                                        (n_rays,) [int32]
        - unique_sensor_idx: the unique sensor index (of this specific sensor type!) the ray originates from     (n_rays,) [int16]
                             This means that both unique camera and lidar sensor indices are enumerated with
                             [0,1..], respectively.
        - unique_frame_idx: the unique frame idx (across *all* sensors of this type!) the ray originates from    (n_rays,) [int32]
                            This means that both unique camera and lidar frame indices are enumerated with
                            [0,1..], respectively.
        - trajectory_idx:   the index of the trajectory the ray originates from                                  (n_rays,) [int16]
        - timestamp_us:     the time-of-measurement of the ray                                                   (n_rays,) [int64]
                            (signed integer only because torch doesn't support large unsigned values yet - note
                            that int64 can't represent full unix epoch timestamps)
    """

    # mandatory fields
    flags: torch.Tensor
    unique_sensor_idx: torch.Tensor
    unique_frame_idx: torch.Tensor

    # optional fields
    trajectory_idx: torch.Tensor | None = None
    timestamp_us: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # make sure all members have same shape and expected types

        assert self.flags.dtype == torch.int32
        assert self.flags.dim() == 1
        assert self.unique_sensor_idx.dtype == torch.int16
        assert self.unique_sensor_idx.shape == self.flags.shape
        assert self.unique_frame_idx.dtype == torch.int32
        assert self.unique_frame_idx.shape == self.flags.shape
        if (trajectory_idx := self.trajectory_idx) is not None:
            assert trajectory_idx.dtype == torch.int16
            assert trajectory_idx.shape == self.flags.shape
        if (timestamp_us := self.timestamp_us) is not None:
            assert timestamp_us.dtype == torch.int64
            assert timestamp_us.shape == self.flags.shape

    def get_mask_flags_all(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *all* flag bits of 'flags' set"""
        return torch.bitwise_and(self.flags, flags.value).eq(flags.value)

    def get_mask_flags_any(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *any* flag bits of 'flags' set"""
        return torch.bitwise_and(self.flags, flags.value).ne(0)

    def get_mask_flags_none(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *none* of the flag bits of 'flags' set"""
        return torch.bitwise_and(self.flags, flags.value).eq(0)

    def _getitem_basedict(self, indices: torch.Tensor | slice) -> dict[str, torch.Tensor]:
        """Base case to mimic indexing for subsampling and chunking"""

        # mandatory fields are always returned
        out = {
            "flags": self.flags[indices],
            "unique_sensor_idx": self.unique_sensor_idx[indices],
            "unique_frame_idx": self.unique_frame_idx[indices],
        }

        # return optional fields only if present
        if (trajectory_idx := self.trajectory_idx) is not None:
            out |= {"trajectory_idx": trajectory_idx[indices]}
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
        if trajectory_idx_list := [s.trajectory_idx for s in rays_meta if s.trajectory_idx is not None]:
            out |= {
                "trajectory_idx": torch.cat(trajectory_idx_list, dim=0),
            }
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


class RaysLidarMeta(RaysMeta):
    """
    Contains additional meta-data specific to lidar rays only [currently empty]
    """

    def __post_init__(self) -> None:
        super().__post_init__()

    def __getitem__(self, indices: torch.Tensor | slice) -> RaysLidarMeta:
        """Mimic indexing for subsampling and chunking"""

        # no additional properties to extend base-type for now
        return RaysLidarMeta(**self._getitem_basedict(indices))

    @staticmethod
    def collate_fn(rays_meta: Union[RaysLidarMeta, Sequence[RaysLidarMeta]]) -> RaysLidarMeta:
        """Collate function for lidar ray meta properties"""
        if isinstance(rays_meta, RaysLidarMeta):
            return rays_meta

        # no additional properties to extend base-type for now
        return RaysLidarMeta(**RaysMeta._collate_fn_basedict(rays_meta))


@dataclasses.dataclass(slots=False, kw_only=True)
class Batch:
    """
    Contains
        - rays_cam: origins and directions of the camera rays used for training [float] (n_rays, 6)
        - rays_lidar: origins and directions of the lidar rays used for training [float] (n_rays, 6)
        - rays_cam_meta: cam ray's metadata (see RaysCamMeta)
        - rays_lidar_meta: lidar ray's metadata (see RaysLidarMeta)
        - idx: batch idx sampled by the dataloader (only valid during val/test time) [int]
        - labels: supervision signal for the rays (color or distance) [float] (n_rays, 3)
        - worker_id: ID of the worker that generated this batch (None if batch is not generated in a multi-worker env) [int]
        - h: height of the image - used in validation and test mode [int]
        - w: width of the image - used in validation and test mode [int]
        - n_patches: number of patches - used when sampling the patches and not random rays [int]
    """

    rays_cam: torch.Tensor
    rays_lidar: Optional[torch.Tensor] = None
    rays_cam_meta: RaysCamMeta
    rays_lidar_meta: Optional[RaysLidarMeta] = None
    idx: list[int]
    labels: Labels
    worker_id: Union[list[int], int, None] = None
    h: Union[list[int], int, None] = None
    w: Union[list[int], int, None] = None
    n_patches: Union[list[int], int, None] = None
    batch_size: Optional[int] = None

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

        if "rays_lidar" in out:
            if isinstance(out["rays_lidar"], NoneType):
                out["rays_lidar_meta"] = None
            else:
                out["rays_lidar_meta"] = RaysLidarMeta.collate_fn(out["rays_lidar_meta"])

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


def field_torch_tensor(dtype: torch.dtype, shape: tuple[int, ...], device: Optional[str] = None, *args, **kwargs):
    """Provides encoder / decoder functionality for torch tensors arrays or dicts or tensors into field types compatible with dataclass-JSON

    Args:
    - dtype: required for encoding / decoding (encoded inputs are verified to have this datatype, and decoded arrays will be returned with this type)
    - shape: tuple of tensor dimensions, used for consistency checks (one dimension is allowed to be -1, similar to reshape)
    - device: optional device for the deserialized data
    - remaining args/kwargs are forward to dataclasses.field()
    """

    def validate(tensor: torch.Tensor) -> torch.Tensor:
        assert len(tensor.shape) == len(shape) and all(
            [dim == expected_dim if expected_dim != -1 else True for (dim, expected_dim) in zip(tensor.shape, shape)]
        ), f"Tensor {tensor} not having expected shape {shape}"
        if device is not None:
            assert tensor.device == torch.device(device), f"Tensor {tensor} not having expected device '{device}'"
        return tensor

    def decoder(input: list | dict[Any, list]) -> torch.Tensor | dict[Any, torch.Tensor]:
        match input:
            case dict():
                # decode dict[<key-type>, list]
                return {
                    key: validate(torch.tensor(value, dtype=dtype, device=device)) for (key, value) in input.items()
                }
            case list():
                # encode list
                return validate(torch.tensor(input, dtype=dtype, device=device))
            case _:
                raise ValueError(f"field_torch_tensor: unsupported decoder input type {type(input)}")

    def encoder(input: torch.Tensor | dict[Any, torch.Tensor]) -> list | dict[Any, list]:
        match input:
            case dict():
                # encode as dict[<key-type>, list]
                assert all(
                    [tensor.dtype == dtype for tensor in input.values()]
                ), f"Not all tensors in {input} of expected dtype {dtype}"
                return {key: validate(tensor).tolist() for (key, tensor) in input.items()}
            case torch.Tensor():
                # encode as list
                assert input.dtype == dtype, f"Provided tensor {input} is not of expected dtype {dtype}"
                return validate(input).tolist()
            case _:
                raise ValueError(f"field_torch_tensor: unsupported encoder input type {type(input)}")

    return dataclasses.field(metadata=dataclasses_json.config(encoder=encoder, decoder=decoder), *args, **kwargs)


def field_camera_model_parameters(*args, **kwargs):
    """Provides encoder / decoder functionality for NCore CameraModelParameters types compatible with dataclass-JSON

    Encoded camera-model-parameters will be encoded as

    "camera_model": {
        "parameters": {
            <TYPE_SPECIFIC_PARAMETERS>
        },
        "type": "<TYPE_ID>"
    },
    """

    def decoder(input: dict) -> ncore_types.ConcreteCameraModelParametersUnion:
        # deserialize based on encoded camera model type
        match type := input["type"]:
            case "ftheta":
                return ncore_types.FThetaCameraModelParameters.from_dict(input["parameters"])
            case "opencv-pinhole":
                return ncore_types.OpenCVPinholeCameraModelParameters.from_dict(input["parameters"])
            case "opencv-fisheye":
                return ncore_types.OpenCVFisheyeCameraModelParameters.from_dict(input["parameters"])
            case _:
                raise ValueError(f"field_camera_model_parameters: unknown camera model_type '{type}'")

    def encoder(input: ncore_types.ConcreteCameraModelParametersUnion) -> dict:
        # serialize camera model type and parameters
        return {"type": input.type(), "parameters": input.to_dict()}

    return dataclasses.field(
        metadata=dataclasses_json.config(encoder=encoder, decoder=decoder, field_name="camera_model"), *args, **kwargs
    )


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
    def target_origin(self) -> npt.NDArray[np.float32]:
        """The origin of the target frame relative to the source frame (in source-frame units)"""
        return -self.matrix[:3, 3]

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

class CameraFrustum:
    """
    We use the following convention for the frustum corners

              5--------6
             /|       /|
            1-+------2 |
            | |      | |
            | 4------+-7
            |/       |/
            0--------3

    """

    def __init__(self, corners: torch.Tensor, device: str = "cuda") -> None:
        """Represents a camera frustum through the near plane corners, the 4 vectors and depth along the normal"""
        self.corners: torch.Tensor = corners.to(device)  # [8,3]
        self.device = device
        self.edges: torch.Tensor = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        ).to(self.device)

        self.planes: torch.Tensor = torch.tensor(
            [[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0], [4, 0, 3, 7], [3, 2, 6, 7], [1, 5, 6, 2]]
        ).to(self.device)
        assert self.corners.shape == (8, 3), "Frustum is defined by 8 corners"
        self.check_input_conformity()

    def check_input_conformity(self) -> None:
        """We make two assumptions on the input: 1) first and second 4 points define the near, far plane respectively. Hence,
        their corners need to be coplanar, 2) the two planes are parallel"""

        # Near plane
        v = self.corners[1:4] - self.corners[0:1]
        v /= v.norm(dim=-1, keepdim=True)
        assert torch.isclose(
            torch.det(v), torch.zeros(1, device=self.device), atol=1e-4
        ), "The corners of near plane are not coplanar"

        # Far plane
        v = self.corners[5:] - self.corners[4:5]
        v /= v.norm(dim=-1, keepdim=True)
        assert torch.isclose(
            torch.det(v), torch.zeros(1, device=self.device), atol=1e-4
        ), "The corners of far plane are not coplanar"

        # Compute the normal vectors of all planes and check that the ones of near/far plane are parallel
        self.compute_normal_vectors()
        assert torch.isclose(
            torch.dot(self.normals[0], self.normals[0]), torch.ones(1, device=self.device)
        ), "The near and far plane are not parallel"

    def compute_normal_vectors(self):
        """Get the indices of the points used to compute the normal vectors. Planes are defined such that the first three indices
        always form a normal vector that points into the frustum"""
        normal_idx = self.planes[:, :3].flatten()
        points = self.corners[normal_idx].reshape(6, 3, 3)
        vectors = points[:, [2, 0]] - points[:, 1:2]
        normals = torch.linalg.cross(vectors[:, 0], vectors[:, 1])
        self.normals = normals / normals.norm(dim=1, keepdim=True)

    def points_in_frustum(self, points: torch.Tensor) -> torch.Tensor:
        points_on_plane = self.corners[self.planes[:, 0]]
        w = points.unsqueeze(1).to(self.device) - points_on_plane
        dist_to_plane = torch.sum(w * self.normals.unsqueeze(0), dim=-1)

        return (dist_to_plane >= 0.0).all(dim=1)

    def get_aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.corners.min(0).values, self.corners.max(0).values


class BoundingBox:
    def __init__(self, extent: torch.Tensor, T_rig_world: torch.Tensor = torch.eye(4), device: str = "cuda") -> None:
        """Represents a bounding box through its 8 corners"""

        # TODO[ZG]: split into length, width, heigh and check if 1 or 2 values are passed
        self.device: str = device
        self.extent: torch.Tensor = extent.to(device)  # [3,2]
        self.T_rig_world: torch.Tensor = T_rig_world.to(device)
        self.edges: torch.Tensor = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        ).to(device)
        self.get_corners()

    def get_corners(self) -> None:
        local_corners = torch.tensor(
            [
                [self.extent[0, 0], self.extent[1, 0], self.extent[2, 0]],  # Back bottom left
                [self.extent[0, 0], self.extent[1, 0], self.extent[2, 1]],  # Back top left
                [self.extent[0, 0], self.extent[1, 1], self.extent[2, 1]],  # Back top right
                [self.extent[0, 0], self.extent[1, 1], self.extent[2, 0]],  # Back bottom right
                [self.extent[0, 1], self.extent[1, 0], self.extent[2, 0]],  # Front bottom left
                [self.extent[0, 1], self.extent[1, 0], self.extent[2, 1]],  # Front top left
                [self.extent[0, 1], self.extent[1, 1], self.extent[2, 1]],  # Front top right
                [self.extent[0, 1], self.extent[1, 1], self.extent[2, 0]],
            ]
        ).to(
            self.device
        )  # Front bottom right

        self.corners = (self.T_rig_world[:3, :3] @ local_corners.transpose(0, 1) + self.T_rig_world[:3, 3:4]).transpose(
            0, 1
        )

@dataclasses.dataclass(slots=True, kw_only=True)
class RigTrajectories(dataclasses_json.DataClassJsonMixin):
    """Represents a list of rig trajectories (using NCore frame conventions)"""

    # NCore world frame -> base frame rigid transformation (potentially geo-located)
    T_world_base: torch.Tensor = field_torch_tensor(torch.float64, (4, 4), device="cpu", kw_only=True)

    # NCore world -> colmap frame conversion
    world_to_colmap: FrameConversion

    @dataclasses.dataclass(slots=True, kw_only=True)
    class RigTrajectory(dataclasses_json.DataClassJsonMixin):
        """Represents a single rig trajectory with associated sensor and frame timestamps"""

        cameras_frame_timestamps_us: dict[str, torch.Tensor] = field_torch_tensor(
            torch.int64, (-1, 2), device="cpu", kw_only=True
        )  # map of *unique* camera sensor ids to start-/end-of-frame timestamps Nx2
        lidars_frame_timestamps_us: dict[str, torch.Tensor] = field_torch_tensor(
            torch.int64, (-1, 2), device="cpu", kw_only=True
        )  # map of *unique* camera sensor indices to start-/end-of-frame timestamps Nx2

        # Timestamped trajectory of the rig frame in NCore world coordinates
        T_rig_worlds: torch.Tensor = field_torch_tensor(torch.float64, (-1, 4, 4), device="cpu", kw_only=True)  # Nx4x4
        T_rig_world_timestamps_us: torch.Tensor = field_torch_tensor(
            torch.int64, (-1,), device="cpu", kw_only=True
        )  # N

        def __post_init__(self):
            assert len(self.T_rig_worlds) == len(self.T_rig_world_timestamps_us)

    rig_trajectories: list[RigTrajectory]  # indexed by trajectory index

    @dataclasses.dataclass(slots=True, kw_only=True)
    class SensorCalibration(dataclasses_json.DataClassJsonMixin):
        """Represents a generic sensor-associated calibration"""

        logical_sensor_name: str  # logical sensor name
        unique_sensor_idx: int  # unique sensor index

        T_sensor_rig: torch.Tensor = field_torch_tensor(
            torch.float32, (4, 4), device="cpu", kw_only=True
        )  # extrinsics 4x4

    @dataclasses.dataclass(slots=True, kw_only=True)
    class CameraCalibration(SensorCalibration):
        """Represents a camera-associated calibration"""

        camera_model_parameters: ncore_types.ConcreteCameraModelParametersUnion = field_camera_model_parameters()  # intrinsics

    camera_calibrations: dict[str, CameraCalibration]  # indexed by *unique* camera sensor ids

    @dataclasses.dataclass(slots=True, kw_only=True)
    class LidarCalibration(SensorCalibration):
        """Represents a lidar-associated calibration"""

        pass

    lidar_calibrations: dict[str, LidarCalibration]  # indexed by *unique* lidar sensor ids

    def __post_init__(self):
        # make sure sensors referenced by trajectories are available
        for rig_trajectory in self.rig_trajectories:
            for camera_id in rig_trajectory.cameras_frame_timestamps_us.keys():
                assert camera_id in self.camera_calibrations, f"Missing camera {camera_id} in camera calibrations"
            for lidar_id in rig_trajectory.lidars_frame_timestamps_us.keys():
                assert lidar_id in self.lidar_calibrations, f"Missing lidar {lidar_id} in lidar calibrations"


def object_mask_from_image_points(
    image_points: torch.Tensor, res_x: int, res_y: int, dilate_ratio: float
) -> torch.Tensor:
    """Given a set of image points, computes the smallest axis aligned bounding box that contains all the points.
    The resolution of the bbox is in pixels (values will be floored, ceiled accordingly.). The bbox is represented
    with top-left, bottom-right coordinates.
    """

    image_points[:, 0] = torch.clamp(image_points[:, 0], min=0, max=res_x)
    image_points[:, 1] = torch.clamp(image_points[:, 1], min=0, max=res_y)

    min_x, min_y = image_points.min(0)[0]
    max_x, max_y = image_points.max(0)[0]

    mask_width_padding = torch.ceil((max_x - min_x) * (dilate_ratio - 1.0) / 2).to(torch.int32)
    mask_height_padding = torch.ceil((max_y - min_y) * (dilate_ratio - 1.0) / 2).to(torch.int32)

    max_x_int, max_y_int = (
        torch.ceil(max_x).to(torch.int32) + mask_width_padding,
        torch.ceil(max_y).to(torch.int32) + mask_height_padding,
    )
    min_x_int, min_y_int = (
        torch.floor(min_x).to(torch.int32) - mask_width_padding,
        torch.floor(min_y).to(torch.int32) - mask_height_padding,
    )

    min_x = torch.clamp(min_x_int, min=0, max=res_x)
    min_y = torch.clamp(min_y_int, min=0, max=res_y)
    max_x = torch.clamp(max_x_int, min=0, max=res_x)
    max_y = torch.clamp(max_y_int, min=0, max=res_y)

    return torch.tensor([[min_y, min_x], [max_y, max_x]])


class AuxShardDataLoader:
    # Common aux base-group names used by both the aux data writer and loader
    SEMANTIC_SEG_BASE_GROUP = "semantic_segmentation"
    INSTANCE_SEG_BASE_GROUP = "instance_segmentation"
    OPTICAL_FLOW_BASE_GROUP = "optical_flow"
    SCENE_FLOW_BASE_GROUP = "scene_flow"

    """Very simple (~dumb / linear) annotation data loader for NCore multi-shard-associated aux data"""

    def __init__(self, sequence_loader: ncore.data.v3.ShardDataLoader, open_consolidated=True) -> None:
        ## collect store candidate paths for a given sequence
        store_paths: list[Path] = []

        # inferred store paths from data shard paths
        for shard_path_str in sequence_loader.get_shard_paths():
            # Make sure paths are absolute at this point - in the future we might have fully-resolved URLs instead here, too
            shard_path = Path(shard_path_str).absolute()

            # Map data shard file name to corresponding annotation file-name (this is fragile)
            shard_base_name = shard_path.stem.split(".")[0]

            # find matching stores paths
            for path in shard_path.parent.iterdir():
                if path.is_file():
                    if not path.name.endswith(".zarr.itar"):
                        # not a supported file-based store format
                        continue

                    # check for matching base names
                    if (
                        # new-style aux data <session-shard>.aux.<signal>.zarr.itar
                        path.name.startswith(shard_base_name + ".aux.")
                        or
                        # backwards-compatibility <session-shard>-annotations.zarr.itar
                        path.name.startswith(shard_base_name + "-annotations")
                    ):
                        store_paths.append(path)
                if path.is_dir():
                    if not path.name.endswith(".zarr"):
                        # not a supported directory store format
                        continue

                    if path.name.startswith(
                        shard_base_name + ".aux."
                    ):  # new-style aux data <session-shard>.aux.<signal>.zarr
                        store_paths.append(path)

        ## load stores concurrently
        self.aux_shard_stores: list[zarr.storage.Store] = []
        self.base_groups: DefaultDict[str, list[zarr.Group]] = defaultdict(
            list
        )  # maps from base-group-name to *unordered* list of groups per shard

        with concurrent.futures.ThreadPoolExecutor() as executor:

            def thread_load_aux_store(store_path):
                """Thread-executed shard opening"""

                if store_path.is_file():
                    # load itar store
                    aux_shard_store = ncore_stores.IndexedTarStore(store_path, mode="r")
                else:
                    # load directory store
                    aux_shard_store = zarr.storage.DirectoryStore(store_path)

                aux_shard_root = (
                    ncore_stores.open_compressed_consolidated(store=aux_shard_store, mode="r")
                    if open_consolidated
                    else zarr.open(store=aux_shard_store, mode="r")
                )

                return aux_shard_root, aux_shard_store

            loaded_base_groups: set[Tuple[str, int]] = set()  # sanity check loaded data for consistency
            for future in concurrent.futures.as_completed(
                [executor.submit(thread_load_aux_store, store_path) for store_path in store_paths]
            ):
                # Note: thread completion order is not relevant here
                aux_shard_root, aux_shard_store = future.result()

                aux_sequence_id = aux_shard_root.attrs.get("sequence_id")
                aux_shard_id = aux_shard_root.attrs.get("shard_id")
                aux_shard_count = aux_shard_root.attrs.get("shard_count")
                aux_root_group_name = aux_shard_root.attrs.get(
                    "aux_root_group_name",
                    # backwards-compatibility <session-shard>-annotations.zarr.itar fallback
                    "annotations",
                )

                # setup consistency checks
                if not len(self.base_groups):
                    self._sequence_id: str = aux_sequence_id
                    self._shard_count: int = aux_shard_count

                if not self._sequence_id == aux_sequence_id:
                    raise ValueError("Can't load aux data for different sequences")
                if not (source_sequence_id := sequence_loader.get_sequence_id()) == aux_sequence_id:
                    raise ValueError(
                        f"Loaded aux data for sequence {aux_sequence_id} not compatible with source sequence {source_sequence_id}"
                    )
                if not self._shard_count == aux_shard_count:
                    raise ValueError("Can't load aux data from different sequence subdivisions")

                # register loaded groups within this store per shard
                for base_group_name, base_group in aux_shard_root[aux_root_group_name].items():
                    # only register groups, not datasets
                    if not isinstance(base_group, zarr.Group):
                        continue

                    if (base_group_key := (base_group_name, aux_shard_id)) in loaded_base_groups:
                        raise ValueError(f"Group {base_group_name} loaded multiple times for shard ID {aux_shard_id}")
                    loaded_base_groups.add(base_group_key)

                    self.base_groups[base_group_name].append(base_group)

                self.aux_shard_stores.append(aux_shard_store)

    def reload_store_resources(self) -> None:
        """Trigger a reload of the resources of each shard store - useful to, e.g., re-open file objects in multi-process settings"""
        for aux_shard_store in self.aux_shard_stores:
            # only need to reload itar-based stores
            if isinstance(aux_shard_store, ncore_stores.IndexedTarStore):
                aux_shard_store.reload_resources()

    def get_semantic_segmentation_meta(self, camera_id: str) -> dict:
        if self.SEMANTIC_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no semantic segmentation data loaded")

        # Take meta from first shard
        return dict(self.base_groups[self.SEMANTIC_SEG_BASE_GROUP][0][camera_id].attrs)

    def get_semantic_segmentation(self, camera_id: str, frame_timestamps_us: int) -> PILImage.Image:
        if self.SEMANTIC_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no semantic segmentation data loaded")

        # find sample by linearly going through available shards samples
        # TODO(@janickm): this can be done much more efficiently and will be slow for a lot of shards
        for base_group in self.base_groups[self.SEMANTIC_SEG_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
            except KeyError:
                # it's ok if the key isn't in the current shard - continue look in next shard
                continue

            return PILImage.open(io.BytesIO(ds[()]), formats=[ds.attrs["format"]])

        raise KeyError(f"semantic segmentation not found for {camera_id} and timestamp {frame_timestamps_us}")

    def get_instance_segmentation_meta(self, camera_id: str) -> dict:
        if self.INSTANCE_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no instance segmentation data loaded")

        # Take meta from first shard
        return dict(self.base_groups[self.INSTANCE_SEG_BASE_GROUP][0][camera_id].attrs)

    def get_instance_segmentation(self, camera_id: str, frame_timestamps_us: int) -> dict:
        if self.INSTANCE_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no instance segmentation data loaded")

        for base_group in self.base_groups[self.INSTANCE_SEG_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
                w, h = base_group[camera_id].attrs["resolution"]
            except KeyError:
                continue

            instance_masks = np.unpackbits(ds["instance_masks"]).reshape(-1, h, w)

            return {
                "instance_masks": instance_masks,
                "scores": np.array(ds["scores"]),
                "classes": np.array(ds["classes"][...]),
            }

        raise KeyError(f"instance segmentation not found for {camera_id} and timestamp {frame_timestamps_us}")

    def get_optical_flow_meta(self, camera_id: str) -> dict:
        if self.OPTICAL_FLOW_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no optical flow data loaded")

        # Take meta from first shard
        return dict(self.base_groups[self.OPTICAL_FLOW_BASE_GROUP][0][camera_id].attrs)

    def get_optical_flow(self, camera_id: str, frame_timestamps_us: int) -> np.ndarray:
        if self.OPTICAL_FLOW_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no optical flow data loaded")

        for base_group in self.base_groups[self.OPTICAL_FLOW_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
                w, h = base_group[camera_id].attrs["resolution"]

                if base_group[camera_id].attrs["store_as_png"]:
                    shift_to_positive = base_group[camera_id].attrs["shift_to_positive"]

                    image_data = np.asarray(
                        PILImage.open(io.BytesIO(ds[()]), formats=[ds.attrs["format"]]).convert("RGBA")
                    ).astype(np.float32)

                    flow_x = image_data[:, :, 0:1] * 256.0 + image_data[:, :, 1:2] - shift_to_positive
                    flow_y = image_data[:, :, 2:3] * 256.0 + image_data[:, :, 3:4] - shift_to_positive

                    flow = np.concatenate([flow_x, flow_y], axis=2)
                else:
                    flow = np.asarray(ds)

                backward_flow = torch.nn.functional.interpolate(
                    torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda(),
                    size=(h, w),
                    mode="bilinear",
                )
                backward_flow = backward_flow[0].permute(1, 2, 0).cpu().numpy()

            except KeyError:
                continue

            return backward_flow  # the optical flow is from frame t to frame t-offset

        raise KeyError(f"optical flow not found for {camera_id} and timestamp {frame_timestamps_us}")

    def get_scene_flow_meta(self, camera_id: str) -> dict:
        if self.SCENE_FLOW_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no scene flow data loaded")

        # Take meta from first shard
        return dict(self.base_groups[self.SCENE_FLOW_BASE_GROUP][0][camera_id].attrs)

    def get_scene_flow(self, camera_id: str, frame_timestamps_us: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.SCENE_FLOW_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no scene flow data loaded")

        for base_group in self.base_groups[self.SCENE_FLOW_BASE_GROUP]:
            try:
                meta_data = self.get_scene_flow_meta(camera_id)
                scale_to_int = meta_data[
                    "scale_to_int"
                ]  # "scale" value timed to the float flow before converting to int16
                w, h = meta_data["resolution"]

                data = base_group[camera_id][str(frame_timestamps_us)]

                if meta_data["store_as_png"]:
                    shift_to_positive = meta_data["shift_to_positive"]

                    image_data = np.asarray(
                        PILImage.open(io.BytesIO(data[()]), formats=[data.attrs["format"]]).convert("RGBA")
                    ).astype(np.float32)
                    image_h = image_data.shape[0]

                    image_data_1 = image_data[: image_h // 2, :, :]
                    image_data_2 = image_data[image_h // 2 :, :, :]

                    flow_x = image_data_1[:, :, 0:1] * 256.0 + image_data_1[:, :, 1:2] - shift_to_positive
                    flow_y = image_data_1[:, :, 2:3] * 256.0 + image_data_1[:, :, 3:4] - shift_to_positive
                    flow_z = image_data_2[:, :, 0:1] * 256.0 + image_data_2[:, :, 1:2] - shift_to_positive
                    lidar_d = image_data_2[:, :, 2:3] * 256.0 + image_data_2[:, :, 3:4] - shift_to_positive

                    scene_flow_data = (
                        np.concatenate([flow_x, flow_y, flow_z, lidar_d], axis=2).astype(np.float32) / scale_to_int
                    )

                    # upsample to original size
                    scene_flow_data = (
                        torch.nn.functional.interpolate(
                            torch.from_numpy(scene_flow_data).permute(2, 0, 1).unsqueeze(0).cuda(),
                            size=(h, w),
                            mode="bilinear",
                        )[0]
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )

                else:
                    scene_flow_data = np.zeros([h, w, 4], dtype=np.float32)
                    scene_flow_data[data[:, 0], data[:, 1], :] = np.array(data[:, 2:]).astype(np.float32) / scale_to_int

            except KeyError:
                continue

            scene_flow = scene_flow_data[:, :, :3]  # the scene flow is forward, from frame t-offset to frame t
            lidar_dist = scene_flow_data[:, :, 3:]

            return scene_flow, lidar_dist

        raise KeyError(f"scene flow not found for {camera_id} and timestamp {frame_timestamps_us}")

    def get_scene_flow_magnitude(
        self,
        camera_id: str,
        frame_timestamps_us: int,
        mask_erode_radius: int = 0,  # radius of mask erosion (before median vote)
        instance_dist_threshold: float = 100,  # instances with distance to ego car larger than the threshold will be labeled as dynamic
    ) -> np.ndarray:
        instances_meta = self.get_instance_segmentation_meta(camera_id)
        instances = self.get_instance_segmentation(camera_id, frame_timestamps_us)
        scene_flow_data = self.get_scene_flow(camera_id, frame_timestamps_us)
        scene_flow = torch.from_numpy(scene_flow_data[0]).to("cuda")
        lidar_dist = torch.from_numpy(scene_flow_data[1]).to("cuda")

        w: int = instances_meta["resolution"][0]
        h: int = instances_meta["resolution"][1]

        dynamic_mag = np.zeros([h, w], np.float32)

        # erode the mask of all instances together
        mask_all = torch.sum(torch.from_numpy(instances["instance_masks"]).to("cuda"), dim=0) > 0.5

        if mask_erode_radius > 0:
            downsample_scale = 2  # downsample the mask before erode for memory saving
            eroder = Erosion2d(
                in_channels=1, out_channels=1, kernel_size=mask_erode_radius // downsample_scale + 1, soft_max=False
            ).to("cuda")

            tensor_to_erode = mask_all.unsqueeze(0).unsqueeze(0).float()
            tensor_to_erode_sq = torch.cat([tensor_to_erode, torch.zeros_like(tensor_to_erode)], dim=2)[
                :, :, :w, :w
            ]  # pad it to a square image
            resizer1 = torchvision.transforms.Resize(
                size=(w // downsample_scale, w // downsample_scale),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )  # for speeding up
            resizer2 = torchvision.transforms.Resize(
                size=(w, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
            tensor_eroded = resizer2(eroder(resizer1(tensor_to_erode_sq)))[
                :, :, :h, :w
            ]  # eroding and remove the padded zeros

            mask_all_eroded = tensor_eroded.squeeze() > 0.5
        else:
            mask_all_eroded = mask_all

        # get instance masks
        for id in range(instances["instance_masks"].shape[0]):
            MAX_VELOCITY = 10.0

            instance_mask = torch.from_numpy(instances["instance_masks"][id]).to("cuda") * mask_all_eroded
            instance_pixels = instance_mask.nonzero().cpu().numpy()

            # undo the erosion if number of instance_pixels is smaller than 100
            if instance_pixels.shape[0] < 100:
                instance_mask = torch.from_numpy(instances["instance_masks"][id]).to("cuda")
                instance_pixels = instance_mask.nonzero().cpu().numpy()

            class_id = instances["classes"][id]
            is_vehicle = instances_meta["thing_classes"][class_id] in ["car", "truck", "bus", "train"]

            if instance_pixels.shape[0] > 1 and is_vehicle:
                distance_on_mask = lidar_dist[
                    instance_pixels[:, 0], instance_pixels[:, 1], :
                ]  # distance of each pixel to ego car
                if (
                    torch.median(distance_on_mask).item() < instance_dist_threshold
                ):  # if the median distance is smaller than threshold, we compute the dynamic using scene flow
                    scene_flow_on_mask = scene_flow[instance_pixels[:, 0], instance_pixels[:, 1], :]
                    magn = torch.median(torch.norm(scene_flow_on_mask, dim=1)).item()
                else:
                    magn = MAX_VELOCITY  # if the median distance is larger than threshold, the instance is treated as dynamic
            else:
                magn = MAX_VELOCITY  # instances that are not vehicles are always dynamic

            dynamic_mag[instances["instance_masks"][id].astype(bool)] = magn

        return dynamic_mag
