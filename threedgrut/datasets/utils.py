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

from __future__ import annotations

import math
import struct
import collections
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

DEFAULT_DEVICE = torch.device("cuda")


def fov2focal(fov_radians: float, pixels: int):
    return pixels / (2 * math.tan(fov_radians / 2))


def focal2fov(focal: float, pixels: int):
    return 2 * math.atan(pixels / (2 * focal))


def pinhole_camera_rays(x, y, f_x, f_y, w, h, ray_jitter=None):
    """
    return:
        ray_origin (sz_y, sz_x, 3)
        normalized ray_direction (sz_y, sz_x, 3)
    """

    if ray_jitter is not None:
        jitter = ray_jitter(x.shape).numpy()
        jitter_xs = jitter[:, 0]
        jitter_ys = jitter[:, 1]
    else:
        jitter_xs = jitter_ys = 0.5

    xs = ((x + jitter_xs) - 0.5 * w) / f_x
    ys = ((y + jitter_ys) - 0.5 * h) / f_y

    ray_lookat = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    ray_origin = np.zeros_like(ray_lookat)

    return ray_origin, ray_lookat / np.linalg.norm(ray_lookat, axis=-1, keepdims=True)


def camera_to_world_rays(ray_o, ray_d, poses):
    """
    input:
        ray_o_cam [n, 3] - ray origins in the camera coordinate system
        ray_d_cam [n, 3] - ray origins in the camera coordinate system
        poses [n, 4,4] - camera to world transformation matrices

    return:
        ray_o [n, 3] - ray origins in the world coordinate system
        ray_d [n, 3] - ray directions in the world coordinate system
    """
    if isinstance(poses, torch.Tensor):
        ray_o = torch.einsum("ijk,ik->ij", poses[:, :3, :3], ray_o) + poses[:, :3, 3]
        ray_d = torch.einsum("ijk,ik->ij", poses[:, :3, :3], ray_d)
    else:
        ray_o = np.einsum("ijk,ik->ij", poses[:, :3, :3], ray_o) + poses[:, :3, 3]
        ray_d = np.einsum("ijk,ik->ij", poses[:, :3, :3], ray_d)

    return ray_o, ray_d

@dataclass(slots=True, kw_only=True)
class PointCloud:
    """Represents a 3d point cloud consisting of corresponding start and end points"""

    xyz_start: torch.Tensor  # [N,3]
    xyz_end: torch.Tensor  # [N,3]
    device: str
    dtype = torch.float32
    color: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert len(self.xyz_start) == len(self.xyz_end)
        assert self.xyz_start.shape[1] == self.xyz_end.shape[1] == 3

        self.xyz_start.to(self.device, dtype=self.dtype)
        self.xyz_end.to(self.device, dtype=self.dtype)

        if self.color is not None:
            assert self.color.shape[1] == 3
            assert len(self.color) == len(self.xyz_end)

            self.color.to(self.device, dtype=self.dtype)

    @staticmethod
    def from_sequence(point_clouds: Sequence[PointCloud], device: str) -> PointCloud:
        point_clouds_list = list(point_clouds)

        return PointCloud(
            xyz_start=torch.cat([pc.xyz_start for pc in point_clouds_list]),
            xyz_end=torch.cat([pc.xyz_end for pc in point_clouds_list]),
            color=torch.cat([pc.color for pc in point_clouds_list]) if point_clouds_list[0].color is not None else None,
            device=device,
        )

    def selected_idxs(self, idxs):
        return PointCloud(
            xyz_start=self.xyz_start[idxs],
            xyz_end=self.xyz_end[idxs],
            color=self.color[idxs] if self.color is not None else None,
            device=self.device,
        )


def get_center_and_diag(cam_centers):
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary."""
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()


def create_camera_visualization(cam_list):
    """
    Given a list-of-dicts of camera & image info, register them in polyscope
    to create a visualization
    """

    import polyscope as ps

    for i_cam, cam in enumerate(cam_list):

        ps_cam_param = ps.CameraParameters(
            ps.CameraIntrinsics(
                fov_vertical_deg=np.degrees(cam["fov_h"]),
                fov_horizontal_deg=np.degrees(cam["fov_w"]),
            ),
            ps.CameraExtrinsics(mat=cam["ext_mat"]),
        )

        cam_color = (1.0, 1.0, 1.0)
        if cam["split"] == "train":
            cam_color = (1.0, 0.7, 0.7)
        elif cam["split"] == "val":
            cam_color = (0.7, 0.1, 0.7)

        ps_cam = ps.register_camera_view(f"{cam['split']}_view_{i_cam:03d}", ps_cam_param, widget_color=cam_color)

        ps_cam.add_color_image_quantity("target image", cam["rgb_img"][:, :, :3], enabled=True)

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_colmap_points3D_text(path):
    """
    Read points3D.txt file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []
    
    # Single file read
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # Convert directly to numpy arrays while appending
                xyzs.append([float(x) for x in elems[1:4]])
                rgbs.append([int(x) for x in elems[4:7]])
                errors.append(float(elems[7]))
    
    # Convert lists to numpy arrays all at once
    return (np.array(xyzs, dtype=np.float64),
            np.array(rgbs, dtype=np.int32),
            np.array(errors, dtype=np.float64).reshape(-1, 1))

def read_colmap_points3D_binary(path_to_model_file):
    """
    Read points3D.bin file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []
    
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_points):
            # Read the point data
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # Append coordinates, colors, and error
            xyzs.append(binary_point_line_properties[1:4])
            rgbs.append(binary_point_line_properties[4:7])
            errors.append(binary_point_line_properties[7])
            
            # Skip track length and elements as they're not used
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.seek(8 * track_length, 1)

    # Convert lists to numpy arrays all at once
    return (np.array(xyzs, dtype=np.float64),
            np.array(rgbs, dtype=np.int32),
            np.array(errors, dtype=np.float64).reshape(-1, 1))




def read_colmap_intrinsics_text(path):
    """
    Read camera intrinsics from a COLMAP text file.
    
    Args:
        path: Path to the cameras.txt file
        
    Returns:
        List of Camera objects sorted by camera ID
    """
    cameras = []
    with open(path, "r") as fid:
        # Skip comment lines at the start
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        
        for line in lines:
            # Unpack elements directly using split with maxsplit
            camera_id, model, width, height, *params = line.split()
            cameras.append(Camera(
                id=int(camera_id),
                model=model,
                width=int(width),
                height=int(height),
                params=np.array([float(p) for p in params])
            ))
    
    return sorted(cameras, key=lambda x: x.id)


def read_colmap_intrinsics_binary(path_to_model_file):
    """
    Read camera intrinsics from a COLMAP binary file.
    
    Args:
        path_to_model_file: Path to the cameras.bin file
        
    Returns:
        List of Camera objects sorted by camera ID
        
    Raises:
        ValueError: If the number of cameras read doesn't match the expected count
        KeyError: If an invalid camera model ID is encountered
    """
    cameras = []
    with open(path_to_model_file, "rb") as fid:
        # Read number of cameras
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_cameras):
            # Read fixed-size camera properties
            camera_id, model_id, width, height = read_next_bytes(
                fid, 
                num_bytes=24, 
                format_char_sequence="iiQQ"
            )
            
            # Get camera model information
            try:
                camera_model = CAMERA_MODEL_IDS[model_id]
            except KeyError:
                raise KeyError(f"Invalid camera model ID: {model_id}")
                
            # Read camera parameters
            params = read_next_bytes(
                fid, 
                num_bytes=8 * camera_model.num_params,
                format_char_sequence="d" * camera_model.num_params
            )
            
            # Create camera object
            cameras.append(Camera(
                id=camera_id,
                model=camera_model.model_name,
                width=width,
                height=height,
                params=np.array(params)
            ))
    
    # Verify camera count
    if len(cameras) != num_cameras:
        raise ValueError(
            f"Expected {num_cameras} cameras, but read {len(cameras)}"
        )
    
    return sorted(cameras, key=lambda x: x.id)

def qvec_to_so3(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

class Image(BaseImage):
    def qvec_to_so3(self):
        return qvec_to_so3(self.qvec)
    
def read_colmap_extrinsics_binary(path_to_model_file):
    """
    Read camera extrinsics from a COLMAP binary file.
    
    Args:
        path_to_model_file: Path to the images.bin file
        
    Returns:
        List of Image objects sorted by image name
        
    Raises:
        ValueError: If string parsing or data reading fails
    """
    images = []
    with open(path_to_model_file, "rb") as fid:
        # Read number of registered images
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_reg_images):
            # Read image properties (id, rotation, translation, camera_id)
            props = read_next_bytes(
                fid, 
                num_bytes=64, 
                format_char_sequence="idddddddi"
            )
            
            image_id, *qvec_tvec, camera_id = props
            qvec = np.array(qvec_tvec[:4])
            tvec = np.array(qvec_tvec[4:7])
            
            # Read image name (null-terminated string)
            image_name = ""
            while True:
                current_char = read_next_bytes(fid, 1, "c")[0]
                if current_char == b"\x00":
                    break
                try:
                    image_name += current_char.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"Invalid character in image name at position {len(image_name)}")
            
            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            point_data = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D
            )
            
            # Parse point data into coordinates and IDs
            xys = np.array([
                (point_data[i], point_data[i + 1])
                for i in range(0, len(point_data), 3)
            ])
            point3D_ids = np.array([
                int(point_data[i + 2])
                for i in range(0, len(point_data), 3)
            ])
            
            # Create image object
            images.append(Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            ))
    
    return sorted(images, key=lambda x: x.name)

def read_colmap_extrinsics_text(path):
    """
    Read camera extrinsics from a COLMAP text file.
    
    Args:
        path: Path to the images.txt file
        
    Returns:
        List of Image objects sorted by image name
        
    Raises:
        ValueError: If file format is invalid or data parsing fails
    """
    images = []
    with open(path, "r") as fid:
        # Skip comment lines and get valid lines
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        
        # Process lines in pairs (image info + points info)
        try:
            while True:
                # Read image info line
                image_line = next(lines, None)
                if image_line is None:
                    break
                    
                # Parse image properties
                elems = image_line.split()
                if len(elems) < 10:  # Minimum required elements
                    raise ValueError(f"Invalid image line format: {image_line}")
                
                image_id = int(elems[0])
                qvec = np.array([float(x) for x in elems[1:5]])
                tvec = np.array([float(x) for x in elems[5:8]])
                camera_id = int(elems[8])
                image_name = elems[9]
                
                # Read points line
                points_line = next(lines, None)
                if points_line is None:
                    raise ValueError(f"Missing points data for image {image_name}")
                
                # Parse 2D points and 3D point IDs
                point_elems = points_line.split()
                if len(point_elems) % 3 != 0:
                    raise ValueError(f"Invalid points format for image {image_name}")
                
                xys = np.array([
                    (float(point_elems[i]), float(point_elems[i + 1]))
                    for i in range(0, len(point_elems), 3)
                ])
                point3D_ids = np.array([
                    int(point_elems[i + 2])
                    for i in range(0, len(point_elems), 3)
                ])
                
                # Create image object
                images.append(Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                ))
                
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing extrinsics file: {e}")
    
    return sorted(images, key=lambda x: x.name)
