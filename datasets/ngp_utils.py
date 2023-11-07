# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

from typing import Union, Tuple

import io
import lzma
import struct
import numpy as np


def nerf_matrix_to_ngp(
    pose: np.ndarray, scale: float = 0.33, offset: Union[np.ndarray, list] = [0, 0, 0]
) -> np.ndarray:
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    return new_pose


def nerf_ray_to_ngp(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    scale: float = 0.33,
    offset: Union[np.ndarray, list] = [0, 0, 0],
    scale_direction: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    ray_o = ray_o * scale + offset
    if scale_direction:
        ray_d *= scale

    tmp = ray_o[:, 0].copy()
    ray_o[:, 0] = ray_o[:, 1]
    ray_o[:, 1] = ray_o[:, 2]
    ray_o[:, 2] = tmp

    tmp = ray_d[:, 0].copy()
    ray_d[:, 0] = ray_d[:, 1]
    ray_d[:, 1] = ray_d[:, 2]
    ray_d[:, 2] = tmp

    return ray_o, ray_d

def nerf_matrix_to_colmap(
    pose: np.ndarray, scale: float = 0.33, offset: Union[np.ndarray, list] = [0, 0, 0]
) -> np.ndarray:
    new_pose = np.array(
        [
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[0]],
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[1]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    return new_pose


def nerf_ray_to_colmap(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    scale: float = 0.33,
    offset: Union[np.ndarray, list] = [0, 0, 0],
    scale_direction: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    ray_o = ray_o * scale + offset
    if scale_direction:
        ray_d *= scale
    return ray_o, ray_d




def load_pc_dat(file_path: str, allow_lookup_fallback: bool = True) -> np.ndarray:
    """
    Loads binary .dat / .dat.xz files representing a 2D single-precision array.
    Serialized 2D arrays usually represent a point-clouds with columns defined as

    [x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic_flag]

    - xys_s / xyz_e: the start / end point of world rays
    - dist: the norm of the ray
    - intensity: lidar intensity response value for this point
    - dynamic_flag:
      - -1: if the information is not available,
      -  0: static
      -  1: = dynamic

    Args:
        file_path: path to .dat / .dat.xz file to load.
        allow_lookup_fallback: If enabled, will fall back to .dat.xz/.dat, resp., in case loading .dat/.dat.xz fails (for backwards-compatibility).
    Return:
        lidar_data: loaded 2D single-precision array
    """

    def load(file: Union[io.BufferedReader, lzma.LZMAFile]) -> np.ndarray:
        # The first number denotes the number of points
        n_rows, n_columns = struct.unpack("<ii", file.read(8))
        # The remaining data are floats saved in little endian
        # Columns usually contain: x_s, y_s, z_s, x_e, y_e, z_e, d, intensity, dynamic_flag
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
        return np.array(struct.unpack("<%sf" % (n_rows * n_columns), file.read()), dtype=np.float32).reshape(
            n_rows, n_columns
        )

    if file_path.endswith(".dat"):
        try:
            with open(file_path, "rb") as file:
                lidar_data = load(file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with lzma.open(file_path + ".xz", "rb") as lzma_file:
                    lidar_data = load(lzma_file)
            else:
                raise e
    elif file_path.endswith(".dat.xz"):
        try:
            with lzma.open(file_path, "rb") as lzma_file:
                lidar_data = load(lzma_file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with open(file_path.replace(".dat.xz", ".dat"), "rb") as file:
                    lidar_data = load(file)
            else:
                raise e
    else:
        raise ValueError("invalid file format provided, supporting .dat / .dat.xz files only")

    return lidar_data
