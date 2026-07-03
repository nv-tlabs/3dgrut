# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""GSplat-compatible COLMAP world normalization."""

import numpy as np


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine transform to an Nx3 point array."""
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a (4, 4) transform, got {matrix.shape}.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}.")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """Apply a similarity transform to camera-to-world matrices."""
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a (4, 4) transform, got {matrix.shape}.")
    if camtoworlds.ndim != 3 or camtoworlds.shape[1:] != (4, 4):
        raise ValueError(f"Expected cameras with shape (N, 4, 4), got {camtoworlds.shape}.")

    transformed = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    if np.any(scaling <= 0) or not np.all(np.isfinite(scaling)):
        raise ValueError("Invalid camera transform scaling while normalizing COLMAP scene.")
    transformed[:, :3, :3] /= scaling[:, None, None]
    return transformed


def similarity_from_cameras(camtoworlds: np.ndarray) -> np.ndarray:
    """Compute GSplat's focus-centered camera similarity transform."""
    if camtoworlds.ndim != 3 or camtoworlds.shape[1:] != (4, 4) or camtoworlds.shape[0] == 0:
        raise ValueError(f"Expected cameras with shape (N, 4, 4), got {camtoworlds.shape}.")
    if not np.all(np.isfinite(camtoworlds)):
        raise ValueError("Cannot normalize COLMAP scene with non-finite cameras.")

    t = camtoworlds[:, :3, 3].astype(np.float64)
    rotations = camtoworlds[:, :3, :3].astype(np.float64)

    ups = np.sum(rotations * np.array([0.0, -1.0, 0.0], dtype=np.float64), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up_norm = np.linalg.norm(world_up)
    if world_up_norm <= 0 or not np.isfinite(world_up_norm):
        raise ValueError("Cannot normalize COLMAP scene with degenerate camera up vectors.")
    world_up /= world_up_norm

    up_camspace = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cosine = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    if cosine > -1:
        rotation_align = np.eye(3, dtype=np.float64) + skew + (skew @ skew) / (1 + cosine)
    else:
        rotation_align = np.diag([-1.0, 1.0, 1.0])

    rotations = rotation_align @ rotations
    forwards = np.sum(rotations * np.array([0.0, 0.0, 1.0], dtype=np.float64), axis=-1)
    t = (rotation_align @ t[..., None])[..., 0]

    nearest = t + (forwards * -t).sum(-1)[:, None] * forwards
    translate = -np.median(nearest, axis=0)
    median_dist = np.median(np.linalg.norm(t + translate, axis=-1))
    if median_dist <= 0 or not np.isfinite(median_dist):
        raise ValueError("Cannot normalize COLMAP scene with degenerate camera distances.")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translate
    transform[:3, :3] = rotation_align
    transform[:3, :] *= 1.0 / median_dist
    return transform


def align_principal_axes(points: np.ndarray) -> np.ndarray:
    """Compute GSplat's median-centered principal-axis alignment."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"Expected at least three points with shape (N, 3), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Cannot PCA-align COLMAP scene with non-finite points.")

    centroid = np.median(points, axis=0)
    covariance = np.cov(points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    rotation = eigenvectors.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def normalize_world_space(
    camtoworlds: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize COLMAP cameras and points using the reference GSplat protocol."""
    camera_transform = similarity_from_cameras(camtoworlds)
    normalized_cameras = transform_cameras(camera_transform, camtoworlds)
    normalized_points = transform_points(camera_transform, points)

    axes_transform = align_principal_axes(normalized_points)
    normalized_cameras = transform_cameras(axes_transform, normalized_cameras)
    normalized_points = transform_points(axes_transform, normalized_points)
    transform = axes_transform @ camera_transform

    if np.median(normalized_points[:, 2]) > np.mean(normalized_points[:, 2]):
        flip_transform = np.diag([1.0, -1.0, -1.0, 1.0])
        normalized_cameras = transform_cameras(flip_transform, normalized_cameras)
        normalized_points = transform_points(flip_transform, normalized_points)
        transform = flip_transform @ transform

    return normalized_cameras, normalized_points, transform


def scene_scale(camtoworlds: np.ndarray) -> float:
    """Return GSplat's maximum camera distance from the mean camera center."""
    if camtoworlds.ndim != 3 or camtoworlds.shape[1:] != (4, 4) or camtoworlds.shape[0] == 0:
        raise ValueError(f"Expected cameras with shape (N, 4, 4), got {camtoworlds.shape}.")
    if not np.all(np.isfinite(camtoworlds)):
        raise ValueError("Cannot compute scene scale from non-finite cameras.")
    camera_locations = camtoworlds[:, :3, 3]
    center = np.mean(camera_locations, axis=0)
    return float(np.max(np.linalg.norm(camera_locations - center, axis=1)))
