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
Transform utilities for export.

Provides coordinate system transforms and normalizing transforms
for exporting Gaussian models to various formats.
"""

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom


@dataclass(frozen=True)
class USDTransformSamples:
    """A composed USD transform sampled at default time and optional time codes."""

    default: Gf.Matrix4d
    time_samples: tuple[tuple[float, Gf.Matrix4d], ...] = ()


def estimate_normalizing_transform(poses: np.ndarray) -> np.ndarray:
    """Estimate transform to normalize camera poses.

    Moves the average camera position to the origin and aligns the average
    down direction with world Y-axis.

    Args:
        poses: Camera poses with shape (N, 4, 4)
    Returns:
        4x4 transformation matrix
    """
    if len(poses) == 0:
        return np.eye(4)

    # Extract camera positions (translation vectors)
    positions = poses[:, :3, 3]  # Shape: (N, 3)
    avg_position = np.mean(positions, axis=0)

    # Extract down vectors (Y-axis) directly from all camera poses
    down_vectors = poses[:, :3, 1]  # Shape: (N, 3)

    # Compute average down direction
    avg_down = np.mean(down_vectors, axis=0)
    avg_down = avg_down / np.linalg.norm(avg_down)  # Normalize

    # Target down direction (world Y-axis)
    target_down = np.array([0, 1, 0])

    # Compute rotation to align avg_down with target_down
    # Using cross product and Rodrigues' rotation formula
    v = np.cross(avg_down, target_down)
    s = np.linalg.norm(v)
    c = np.dot(avg_down, target_down)

    if s < 1e-6:  # Vectors are already aligned
        rotation_matrix = np.eye(3)
    else:
        # Skew-symmetric matrix
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # Rodrigues' rotation formula
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    # Apply rotation then translation
    transform[:3, 3] = -rotation_matrix @ avg_position

    return transform


def get_3dgrut_to_usd_transform() -> np.ndarray:
    """Get the coordinate system transform from 3DGRUT to USD.

    3DGRUT uses "right-down-front" camera convention.
    USD uses Y-up world space.

    Returns:
        4x4 transformation matrix
    """
    # Flip Y and Z to convert from 3DGRUT's coordinate system
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_3dgrut_to_usdz_coordinate_transform() -> np.ndarray:
    """Get the 3DGRUT-to-USDZ (Omniverse) coordinate transform.

    Same matrix used by NuRec when apply_coordinate_transform is True.
    Use for both Lightfield and NuRec when aligning with Omniverse convention.

    Returns:
        4x4 transformation matrix (column-vector convention)
    """
    return np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def column_vector_4x4_to_usd_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    """Build USD Gf.Matrix4d from a column-vector 4x4 (p' = M @ p).

    Uses Gf.Matrix4d.SetTransform so the result matches USD row-vector convention
    (translation in bottom row). Use for any column-vector 4x4 (e.g. normalizing
    transform, camera c2w).

    Args:
        matrix: 4x4 numpy array (column-vector convention)

    Returns:
        Gf.Matrix4d for USD (row-vector convention)
    """
    R = matrix[:3, :3].astype(np.float64)
    t = matrix[:3, 3]
    m = Gf.Matrix4d()
    m.SetTransform(Gf.Matrix3d(*R.T.flatten()), Gf.Vec3d(t[0], t[1], t[2]))
    return m


def usd_matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    """Return a row-major numpy representation of a USD matrix."""
    return np.array([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def _is_identity_usd_matrix(matrix: Gf.Matrix4d, atol: float = 1e-12) -> bool:
    return np.allclose(usd_matrix_to_numpy(matrix), np.eye(4), atol=atol)


def collect_local_to_world_transform_samples(prim: Usd.Prim) -> USDTransformSamples:
    """Collect a prim's composed local-to-world transform at all authored xform sample times."""
    sample_times: set[float] = set()
    current = prim
    while current and current.IsValid() and str(current.GetPath()) != "/":
        xformable = UsdGeom.Xformable(current)
        if xformable:
            for op in xformable.GetOrderedXformOps():
                attr = op.GetAttr()
                if attr.IsValid():
                    sample_times.update(float(t) for t in attr.GetTimeSamples())
        current = current.GetParent()

    xformable = UsdGeom.Xformable(prim)
    default = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    samples = tuple(
        (time_code, xformable.ComputeLocalToWorldTransform(Usd.TimeCode(time_code)))
        for time_code in sorted(sample_times)
    )
    return USDTransformSamples(default=default, time_samples=samples)


def apply_usd_transform_samples(
    xformable: UsdGeom.Xformable,
    transform_samples: USDTransformSamples | None,
    *,
    op_suffix: str = "sourcePose",
) -> None:
    """Author transform samples on an xformable prim, skipping static identity transforms."""
    if transform_samples is None:
        return
    if not transform_samples.time_samples and _is_identity_usd_matrix(transform_samples.default):
        return

    transform_op = xformable.AddTransformOp(opSuffix=op_suffix)
    transform_op.Set(transform_samples.default)
    for time_code, matrix in transform_samples.time_samples:
        transform_op.Set(matrix, Usd.TimeCode(time_code))
