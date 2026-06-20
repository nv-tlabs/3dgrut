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


# ---------------------------------------------------------------------------
# Canonical object-frame estimation (geometry-based / PCA)
# ---------------------------------------------------------------------------

# Index of the up axis in the canonical frame.
_UP_INDEX = {"y": 1, "z": 2}


def _subsample(points: np.ndarray, weights: np.ndarray, max_samples: int, seed: int = 0):
    """Deterministically subsample points/weights for cheap frame estimation at huge N."""
    n = points.shape[0]
    if n <= max_samples:
        return points, weights
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return points[idx], weights[idx]


def _robust_weighted_moments(points: np.ndarray, weights: np.ndarray, trim_percentile: float):
    """Weighted mean + covariance after trimming far outliers (robust to floaters)."""
    w = np.clip(weights.astype(np.float64), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    if trim_percentile < 100.0:
        keep = dist <= np.percentile(dist, trim_percentile)
        if keep.sum() >= 3:
            points, w = points[keep], w[keep]
    wsum = w.sum()
    mean = (w[:, None] * points).sum(axis=0) / wsum
    centered = points - mean
    cov = (w[:, None, None] * np.einsum("ni,nj->nij", centered, centered)).sum(axis=0) / wsum
    return mean, cov, points, w


def _build_canonical_rotation(eigvecs: np.ndarray, up_axis: str) -> np.ndarray:
    """Rows = canonical axes in world coords. Two largest PCs in-plane, smallest along up.

    ``eigvecs`` columns are eigenvectors sorted by ascending eigenvalue (numpy ``eigh``).
    """
    smallest, mid, largest = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
    if up_axis == "y":
        # canonical x, y(up), z
        rows = [largest, smallest, mid]
    elif up_axis == "z":
        # canonical x, y, z(up)
        rows = [largest, mid, smallest]
    else:
        raise ValueError(f"up_axis must be 'y' or 'z', got '{up_axis}'")
    R = np.stack(rows, axis=0)
    # Deterministic sign: make each axis point with the dominant world component positive.
    for i in range(3):
        if R[i, np.argmax(np.abs(R[i]))] < 0:
            R[i] = -R[i]
    # Right-handed.
    if np.linalg.det(R) < 0:
        R[_UP_INDEX[up_axis]] = -R[_UP_INDEX[up_axis]]
    return R


def estimate_pca_frame(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    up_axis: str = "y",
    origin: str = "centroid",
    up_min_percentile: float = 2.0,
    trim_percentile: float = 99.5,
    max_samples: int = 2_000_000,
) -> np.ndarray:
    """Estimate a canonical object frame from points via robust, opacity-weighted PCA.

    Returns a 4x4 (column-vector) world->canonical transform: ``p_canon = R (p - origin)``.
    The two largest principal axes span the in-plane (x,z for y-up) and the smallest-variance
    axis is the up direction. ``origin='centroid'`` puts 0 at the weighted centroid;
    ``origin='plane'`` keeps the in-plane centroid but sets up=0 at a robust low percentile
    (so an object rests on its base instead of floating at mean height).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.eye(4)
    if weights is None:
        weights = np.ones(points.shape[0], dtype=np.float64)
    points_s, weights_s = _subsample(points, np.asarray(weights, dtype=np.float64), max_samples)
    mean, cov, kept, kept_w = _robust_weighted_moments(points_s, weights_s, trim_percentile)

    evals, evecs = np.linalg.eigh(cov)  # ascending
    R = _build_canonical_rotation(evecs, up_axis)

    t = -R @ mean
    if origin == "plane":
        up_idx = _UP_INDEX[up_axis]
        up_coords = (kept - mean) @ R[up_idx]
        t[up_idx] -= float(np.percentile(up_coords, up_min_percentile))
    elif origin != "centroid":
        raise ValueError(f"origin must be 'centroid' or 'plane', got '{origin}'")

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def estimate_frame_from_fields(
    fields,
    *,
    up_axis: str = "y",
    origin: str = "centroid",
    up_min_percentile: float = 2.0,
    trim_percentile: float = 99.5,
    per_field_samples: int = 1_000_000,
) -> np.ndarray:
    """Estimate one global frame from several sources' world points (no merge of attributes).

    ``fields`` is a list of ``(world_points[N,3], weights[N])``. Each is subsampled, then the
    subsamples are pooled for a single PCA — so multi-prim assets get one shared frame.
    """
    pts, wts = [], []
    for points, weights in fields:
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        weights = np.ones(points.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
        ps, ws = _subsample(points, weights, per_field_samples)
        pts.append(ps)
        wts.append(ws)
    if not pts:
        return np.eye(4)
    return estimate_pca_frame(
        np.concatenate(pts, axis=0),
        np.concatenate(wts, axis=0),
        up_axis=up_axis,
        origin=origin,
        up_min_percentile=up_min_percentile,
        trim_percentile=trim_percentile,
        max_samples=per_field_samples * max(len(pts), 1),
    )


def resolve_frame_transform(
    mode: str,
    *,
    fields=None,
    poses: np.ndarray | None = None,
    up_axis: str = "y",
    origin: str = "centroid",
    up_min_percentile: float = 2.0,
) -> np.ndarray:
    """Resolve the scene-normalizing frame transform from the selected estimator.

    mode: 'none' (identity), 'cameras' (from dataset poses), 'pca' (geometry-based).
    """
    if mode == "none":
        return np.eye(4)
    if mode == "cameras":
        if poses is None:
            raise ValueError("frame mode 'cameras' requires camera poses (a dataset).")
        return estimate_normalizing_transform(poses)
    if mode == "pca":
        if not fields:
            raise ValueError("frame mode 'pca' requires Gaussian fields.")
        return estimate_frame_from_fields(
            fields, up_axis=up_axis, origin=origin, up_min_percentile=up_min_percentile
        )
    raise ValueError(f"Unknown frame mode '{mode}' (expected none/cameras/pca).")
