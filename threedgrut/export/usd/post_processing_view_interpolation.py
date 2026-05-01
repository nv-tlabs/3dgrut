# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""View samplers for SH-bake fitting.

The default fit loop iterates the training dataloader, so the optimizer
only sees the discrete set of training poses. This module adds two
interpolation-based samplers:

* ``random-pair-slerp`` -- pick two distinct training views uniformly at
  random, slerp between them at a random ``s in [0, 1]``. Cheap, no
  global structure.

* ``trajectory`` -- order the training views along a smooth path using
  nearest-neighbour + 2-opt with a position+direction distance, then
  arc-length-parameterise the path on ``[0, 1]``. Each sample picks a
  random ``t in [0, 1]``, locates the bracketing pair, and slerps inside
  the segment. Closer to the kind of camera continuum a viewer would
  fly through; better for fitting a residual that is supposed to
  generalise to nearby novel views.

Both samplers reuse the dataset's per-intrinsic camera-space rays and
pixel-coordinate grid -- only ``T_to_world`` changes per sample. PPISP's
``FixedPPISP`` ignores the per-frame indices on the synthetic batch, so
camera/frame indices on the template are kept as-is.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from threedgrut.datasets.protocols import Batch

logger = logging.getLogger(__name__)


VIEW_SAMPLING_TRAINING = "training"
VIEW_SAMPLING_RANDOM_PAIR_SLERP = "random-pair-slerp"
VIEW_SAMPLING_TRAJECTORY = "trajectory"
VIEW_SAMPLING_MODES = {
    VIEW_SAMPLING_TRAINING,
    VIEW_SAMPLING_RANDOM_PAIR_SLERP,
    VIEW_SAMPLING_TRAJECTORY,
}


def normalize_view_sampling_mode(mode: Optional[str]) -> str:
    normalized = VIEW_SAMPLING_TRAINING if mode is None else str(mode).strip().lower()
    if normalized not in VIEW_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported view sampling mode '{mode}'. "
            f"Expected one of: {sorted(VIEW_SAMPLING_MODES)}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Pose interpolation primitives (numpy, double precision for stability)
# ---------------------------------------------------------------------------


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion [w, x, y, z] (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = (q / np.linalg.norm(q)).tolist()
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """Standard quaternion slerp; falls back to lerp+normalise when nearly parallel."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:  # take the short arc
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + s * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = math.acos(max(min(d, 1.0), -1.0))
    sin_theta = math.sin(theta)
    a = math.sin((1.0 - s) * theta) / sin_theta
    b = math.sin(s * theta) / sin_theta
    return a * q0 + b * q1


def slerp_pose(pose_a: np.ndarray, pose_b: np.ndarray, s: float) -> np.ndarray:
    """Interpolate a 4x4 c2w pose between ``pose_a`` and ``pose_b`` at ``s in [0, 1]``.

    Rotation: quaternion slerp. Translation: linear lerp. Lower row is left as
    ``[0, 0, 0, 1]``.
    """
    s = float(np.clip(s, 0.0, 1.0))
    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    q_a = _R_to_quat(pose_a[:3, :3])
    q_b = _R_to_quat(pose_b[:3, :3])
    q = _slerp_quat(q_a, q_b, s)
    R = _quat_to_R(q)
    t = (1.0 - s) * pose_a[:3, 3] + s * pose_b[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


# ---------------------------------------------------------------------------
# Trajectory ordering: nearest-neighbour + 2-opt on a position+direction metric
# ---------------------------------------------------------------------------


def _pose_distance_matrix(
    poses: np.ndarray,
    weight_position: float,
    weight_rotation: float,
) -> np.ndarray:
    """``D[i, j]`` = weighted (position L2 + 1 - cos(forward angle))."""
    n = poses.shape[0]
    pos = poses[:, :3, 3]                     # (N, 3)
    fwd = poses[:, :3, 2]                     # (N, 3)  RDF: +Z = forward
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-12)

    # vectorised pairwise position distance
    diff = pos[:, None, :] - pos[None, :, :]
    d_pos = np.linalg.norm(diff, axis=2)
    # normalise by mean pairwise so the rotation term lives on a comparable scale
    mean_pos = max(float(d_pos[d_pos > 0].mean()) if (d_pos > 0).any() else 1.0, 1e-9)

    cos_ang = np.clip(fwd @ fwd.T, -1.0, 1.0)
    d_rot = 1.0 - cos_ang  # in [0, 2]

    return weight_position * (d_pos / mean_pos) + weight_rotation * d_rot


def _nearest_neighbour_order(D: np.ndarray, start: int = 0) -> List[int]:
    n = D.shape[0]
    visited = [False] * n
    order = [start]
    visited[start] = True
    while len(order) < n:
        last = order[-1]
        # mask visited with +inf
        candidates = np.where(visited, np.inf, D[last])
        nxt = int(np.argmin(candidates))
        order.append(nxt)
        visited[nxt] = True
    return order


def _two_opt(order: List[int], D: np.ndarray, max_passes: int = 50) -> List[int]:
    """In-place 2-opt swap loop. Stops when a full pass yields no improvement
    or when ``max_passes`` is reached."""
    n = len(order)
    if n < 4:
        return order
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[j + 1]
                # original edges (a,b) + (c,d)
                # candidate after reverse: (a,c) + (b,d)
                if D[a, c] + D[b, d] + 1e-12 < D[a, b] + D[c, d]:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return order


def order_views_along_trajectory(
    poses: np.ndarray,
    *,
    weight_position: float = 1.0,
    weight_rotation: float = 0.5,
    start_index: int = 0,
    two_opt_passes: int = 50,
) -> Tuple[List[int], np.ndarray]:
    """Order ``poses`` along an approximate Hamiltonian path.

    Returns ``(ordered_indices, cum_t)`` where ``cum_t[k] in [0, 1]`` is the
    arc-length parameter at the k-th ordered pose. ``cum_t[0] = 0`` and
    ``cum_t[-1] = 1``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must be (N, 4, 4), got {poses.shape}")
    n = poses.shape[0]
    if n < 2:
        return list(range(n)), np.zeros(max(n, 1), dtype=np.float64)

    D = _pose_distance_matrix(poses, weight_position, weight_rotation)
    order = _nearest_neighbour_order(D, start=start_index)
    order = _two_opt(order, D, max_passes=two_opt_passes)

    cum = np.zeros(n, dtype=np.float64)
    for k in range(1, n):
        cum[k] = cum[k - 1] + D[order[k - 1], order[k]]
    if cum[-1] > 0:
        cum = cum / cum[-1]
    return order, cum


# ---------------------------------------------------------------------------
# Sampler driving the fit loop
# ---------------------------------------------------------------------------


class InterpolatedViewSampler:
    """Yields ``Batch`` objects with synthetic interpolated poses.

    The sampler grabs one template batch from the training dataset to
    cache the per-intrinsic camera-space rays, pixel coords and any
    intrinsic dictionaries; only ``T_to_world`` (and ``T_to_world_end``,
    which we set to the same pose -- no rolling shutter on synthetic
    poses) changes per sample.

    Args:
        train_dataset: must implement
            :meth:`~threedgrut.datasets.protocols.BoundedMultiViewDataset.get_poses`
            and :meth:`get_gpu_batch_with_intrinsics`.
        mode: ``"random-pair-slerp"`` or ``"trajectory"``.
        steps_per_epoch: how many synthetic batches to emit per pass.
        seed: optional RNG seed for reproducibility.
        weight_position / weight_rotation: trajectory mode only.
        start_index: trajectory mode only.
    """

    def __init__(
        self,
        train_dataset,
        template_gpu_batch: Batch,
        mode: str,
        steps_per_epoch: int,
        *,
        seed: Optional[int] = None,
        weight_position: float = 1.0,
        weight_rotation: float = 0.5,
        start_index: int = 0,
    ) -> None:
        mode = normalize_view_sampling_mode(mode)
        if mode == VIEW_SAMPLING_TRAINING:
            raise ValueError("InterpolatedViewSampler is only for non-training modes.")
        if not hasattr(train_dataset, "get_poses"):
            raise TypeError(
                "InterpolatedViewSampler requires a dataset exposing get_poses(); "
                f"got {type(train_dataset).__name__}."
            )
        if not isinstance(template_gpu_batch, Batch):
            raise TypeError("template_gpu_batch must be a threedgrut Batch instance.")
        self.dataset = train_dataset
        self.mode = mode
        self.steps_per_epoch = int(steps_per_epoch)
        self._rng = np.random.default_rng(seed)
        self._template = template_gpu_batch

        poses = np.asarray(train_dataset.get_poses(), dtype=np.float64)
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(f"dataset.get_poses() must be (N, 4, 4), got {poses.shape}")
        if poses.shape[0] < 2:
            raise ValueError("Need at least 2 training views to interpolate.")
        self._poses = poses

        if mode == VIEW_SAMPLING_TRAJECTORY:
            self._ordered_indices, self._cum_t = order_views_along_trajectory(
                poses,
                weight_position=weight_position,
                weight_rotation=weight_rotation,
                start_index=start_index,
            )
            logger.info(
                "Built %d-view trajectory (NN + 2-opt) for SH-bake interpolation.",
                len(self._ordered_indices),
            )
        else:
            self._ordered_indices = None
            self._cum_t = None

    # ------------------------------------------------------------------
    # Pose sampling
    # ------------------------------------------------------------------

    def _sample_pose_random_pair(self) -> np.ndarray:
        n = self._poses.shape[0]
        i = int(self._rng.integers(0, n))
        j = int(self._rng.integers(0, n - 1))
        if j >= i:
            j += 1  # ensures j != i without bias
        s = float(self._rng.random())
        return slerp_pose(self._poses[i], self._poses[j], s)

    def _sample_pose_trajectory(self) -> np.ndarray:
        t = float(self._rng.random())
        cum = self._cum_t
        # Find segment k s.t. cum[k-1] <= t <= cum[k] (with cum[0]=0).
        k = int(np.searchsorted(cum, t, side="left"))
        k = max(1, min(k, len(cum) - 1))
        denom = max(cum[k] - cum[k - 1], 1e-12)
        local_s = float((t - cum[k - 1]) / denom)
        a = self._ordered_indices[k - 1]
        b = self._ordered_indices[k]
        return slerp_pose(self._poses[a], self._poses[b], local_s)

    def _sample_pose(self) -> np.ndarray:
        if self.mode == VIEW_SAMPLING_RANDOM_PAIR_SLERP:
            return self._sample_pose_random_pair()
        return self._sample_pose_trajectory()

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _make_batch(self, pose_np: np.ndarray) -> Batch:
        device = self._template.T_to_world.device
        dtype = self._template.T_to_world.dtype
        T = torch.from_numpy(pose_np).to(device=device, dtype=dtype).unsqueeze(0)
        # Same pose for start and end -- no rolling shutter on synthetic views.
        return replace(self._template, T_to_world=T, T_to_world_end=T)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Batch]:
        for _ in range(self.steps_per_epoch):
            yield self._make_batch(self._sample_pose())

    def __len__(self) -> int:
        return self.steps_per_epoch
