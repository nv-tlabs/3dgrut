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
Spatial partitioning of a Gaussian scene into volume partitions.

Slices a trained Gaussian scene into spatial partitions, each holding at most a
configurable number of Gaussians, so that large scenes can be exported as several
``ParticleField3DGaussianSplat`` UsdVol prims (and several PLY point clouds) instead
of one monolithic volume.

Design goals (see plan):
- **Capped:** no partition exceeds ``max_per_volume`` Gaussians.
- **Balanced:** partitions have similar Gaussian counts (KD-tree median split).
- **Non-overlapping:** axis-aligned, disjoint partitions; optionally split oversized
  Gaussians that straddle boundaries into smaller, footprint-preserving children.
- **GPU-first:** all heavy math runs on the tensors' device (CUDA in production) so the
  pipeline scales to scenes with up to billions of Gaussians. Per-partition NumPy
  ``GaussianAttributes`` are materialized only at the writer boundary, one at a time.

When ``max_per_volume >= num_gaussians`` no partitioning is performed: the result holds a
single partition (``is_partitioned == False``) and exporters reproduce the regular,
unpartitioned output exactly (no partition id leaks into prim paths or file names).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from threedgrut.export.accessor import GaussianAttributes, GaussianExportAccessor, ModelCapabilities
from threedgrut.export.base import ExportableModel
from threedgrut.export.sh_rotation import rotate_specular
from threedgrut.utils.misc import inverse_sigmoid, quaternion_to_so3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers (torch, batched, device-agnostic)
# ---------------------------------------------------------------------------


def gaussian_covariances(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian 3x3 covariance ``Sigma = R S Sᵀ Rᵀ``.

    Mirrors ``MixtureOfGaussians.get_covariance``. ``scales`` are post-activation
    (linear) and ``rotations`` are wxyz quaternions (normalized internally).
    """
    R = quaternion_to_so3(rotations)  # [N, 3, 3]
    # M = R @ diag(scales): scale column j by scales[:, j].
    M = R * scales.unsqueeze(1)
    return M @ M.transpose(1, 2)


def gaussian_extents(scales: torch.Tensor, rotations: torch.Tensor, n_sigma: float = 3.0) -> torch.Tensor:
    """Per-Gaussian axis-aligned half-extent (``n_sigma`` standard deviations).

    Returns ``[N, 3]`` half-widths of the world-axis-aligned bounding box that
    encloses the ``n_sigma`` iso-surface of each Gaussian.
    """
    cov = gaussian_covariances(scales, rotations)
    diag = torch.diagonal(cov, dim1=1, dim2=2)  # [N, 3] world-axis variances
    return n_sigma * torch.sqrt(diag.clamp_min(0.0))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def so3_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Batched rotation-matrix -> wxyz quaternion (numerically stable).

    ``matrix`` is ``[N, 3, 3]`` proper rotations; returns ``[N, 4]`` (w, x, y, z).
    Method matches the standard four-candidate formulation used by pytorch3d.
    """
    batch = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(*batch, 9), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    best = torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    return quat_candidates[best, :].reshape(*batch, 4)


# ---------------------------------------------------------------------------
# KD-tree partition
# ---------------------------------------------------------------------------


def kdtree_partition(positions: torch.Tensor, max_per_volume: int) -> Tuple[torch.Tensor, int]:
    """Partition Gaussian centers via recursive median splits.

    Repeatedly splits the longest-spread axis at its median (balanced by sorted
    midpoint) until every leaf holds at most ``max_per_volume`` points. Produces
    disjoint, axis-aligned, near-equal-count partitions.

    Returns a ``(labels [int64 N], num_partitions)`` tuple. Labels run ``0..num-1``.
    """
    n = positions.shape[0]
    device = positions.device
    labels = torch.empty(n, dtype=torch.long, device=device)

    if n <= max_per_volume:
        labels.zero_()
        return labels, 1

    stack = [torch.arange(n, device=device)]
    next_label = 0
    while stack:
        idx = stack.pop()
        if idx.numel() <= max_per_volume:
            labels[idx] = next_label
            next_label += 1
            continue
        pts = positions.index_select(0, idx)
        spans = pts.amax(dim=0) - pts.amin(dim=0)
        axis = int(torch.argmax(spans).item())
        order = torch.argsort(pts[:, axis])
        half = idx.numel() // 2
        stack.append(idx.index_select(0, order[:half]))
        stack.append(idx.index_select(0, order[half:]))

    return labels, next_label


# ---------------------------------------------------------------------------
# Oversized-Gaussian splitting (deterministic, moment-preserving)
# ---------------------------------------------------------------------------


def split_large_gaussians(
    post: Dict[str, torch.Tensor],
    target_size: float,
    n_sigma: float = 3.0,
    max_splits: int = 4,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Split Gaussians whose footprint exceeds ``target_size`` into smaller children.

    Each oversized Gaussian ``N(mu, Sigma)`` is split along its principal axis into two
    equally-weighted children whose combined mean and covariance equal the parent's
    (moment-preserving, the 3DGS-style split). The principal-axis variance is reduced to
    a quarter and the children are offset by ``sqrt(0.75 * lambda)`` along that axis, so
    the dominant extent halves each pass. Opacity and SH are copied to both children.

    Operates on post-activation tensors (linear scales, wxyz quaternions). Iterates up to
    ``max_splits`` times, re-evaluating footprints, until none exceed ``target_size``.

    Returns ``(post, num_added)`` where ``post`` is the expanded tensor dict and
    ``num_added`` is the net number of Gaussians introduced.
    """
    initial_count = post["positions"].shape[0]

    for _ in range(max_splits):
        scales = post["scales"]
        rotations = post["rotations"]
        cov = gaussian_covariances(scales, rotations)
        # eigh returns ascending eigenvalues; last is the principal (largest) axis.
        evals, evecs = torch.linalg.eigh(cov)
        principal_extent = n_sigma * torch.sqrt(evals[:, -1].clamp_min(0.0))
        big = principal_extent > target_size
        if not bool(big.any()):
            break

        b = big.nonzero(as_tuple=False).squeeze(1)
        keep = (~big).nonzero(as_tuple=False).squeeze(1)

        evb_vals = evals.index_select(0, b)  # [B, 3]
        evb_vecs = evecs.index_select(0, b)  # [B, 3, 3], columns are eigenvectors
        lam = evb_vals[:, -1]
        axis = evb_vecs[:, :, -1]  # principal direction (unit) [B, 3]
        offset = torch.sqrt((0.75 * lam).clamp_min(0.0))

        # Child covariance: principal eigenvalue reduced to a quarter, basis unchanged.
        new_vals = evb_vals.clone()
        new_vals[:, -1] = 0.25 * lam
        child_scales = torch.sqrt(new_vals.clamp_min(1e-24))

        # Build a proper rotation from the eigenbasis (flip a column if reflection).
        rot_mat = evb_vecs.clone()
        det = torch.linalg.det(rot_mat)
        rot_mat[det < 0, :, 0] = -rot_mat[det < 0, :, 0]
        child_quat = so3_to_quaternion_wxyz(rot_mat)

        pos_b = post["positions"].index_select(0, b)
        child0_pos = pos_b + offset.unsqueeze(1) * axis
        child1_pos = pos_b - offset.unsqueeze(1) * axis

        def _cat_children(name: str, child_value: torch.Tensor) -> torch.Tensor:
            return torch.cat([post[name].index_select(0, keep), child_value, child_value], dim=0)

        dens_b = post["densities"].index_select(0, b)
        alb_b = post["albedo"].index_select(0, b)
        spec_b = post["specular"].index_select(0, b)

        post = {
            "positions": torch.cat(
                [post["positions"].index_select(0, keep), child0_pos, child1_pos], dim=0
            ),
            "scales": _cat_children("scales", child_scales),
            "rotations": _cat_children("rotations", child_quat),
            "densities": torch.cat([post["densities"].index_select(0, keep), dens_b, dens_b], dim=0),
            "albedo": torch.cat([post["albedo"].index_select(0, keep), alb_b, alb_b], dim=0),
            "specular": torch.cat([post["specular"].index_select(0, keep), spec_b, spec_b], dim=0),
        }

    return post, post["positions"].shape[0] - initial_count


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class PartitionResult:
    """Outcome of :func:`partition_scene`.

    ``labels`` (GPU tensor) assign each exported Gaussian to a partition id. When
    ``is_partitioned`` is False there is a single partition spanning the whole scene and
    exporters must reproduce the regular unpartitioned output.
    """

    labels: torch.Tensor
    num_partitions: int
    is_partitioned: bool
    capabilities: ModelCapabilities
    metrics: Dict = field(default_factory=dict)

    # Source field's local-to-world (M_f), authored on each of this result's USD prims so a
    # multi-prim asset's per-prim poses are preserved. None == identity.
    source_transform: Optional["np.ndarray"] = None

    # Exactly one source of attributes is set:
    #  - _accessor: not split, attributes come 1:1 from the original model.
    #  - _split_post: split, attributes come from the expanded post-activation tensors.
    _accessor: Optional[GaussianExportAccessor] = None
    _split_post: Optional[Dict[str, torch.Tensor]] = None

    @property
    def labels_np(self) -> np.ndarray:
        return self.labels.detach().cpu().numpy()

    def full_attributes(self, preactivation: bool) -> GaussianAttributes:
        """All exported Gaussians as a single ``GaussianAttributes`` (order matches labels).

        Built once; callers slice it per-partition with ``filter_by_mask`` to avoid
        re-extracting the whole scene for every partition.
        """
        if self._split_post is not None:
            return _attrs_from_post(self._split_post, preactivation=preactivation)
        assert self._accessor is not None
        return self._accessor.get_attributes(preactivation=preactivation)

    def iter_partitions(self, preactivation: bool) -> Iterator[Tuple[int, GaussianAttributes]]:
        """Yield ``(partition_id, GaussianAttributes)`` for each partition.

        Groups the scene in a single ``argsort`` pass (O(N log N)) and slices contiguous index
        ranges, instead of re-scanning all labels once per partition (O(N*K)).
        """
        attrs = self.full_attributes(preactivation=preactivation)
        if self.num_partitions <= 1:
            yield 0, attrs
            return
        labels = self.labels_np
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=self.num_partitions)
        bounds = np.zeros(self.num_partitions + 1, dtype=np.int64)
        np.cumsum(counts, out=bounds[1:])
        for pid in range(self.num_partitions):
            idx = order[bounds[pid] : bounds[pid + 1]]
            yield pid, attrs.filter_by_mask(idx)


def _gather_post_tensors(model: ExportableModel) -> Dict[str, torch.Tensor]:
    """Post-activation tensors straight off the model (kept on their original device)."""
    densities = model.get_density(preactivation=False).detach()
    if densities.ndim == 1:
        densities = densities.unsqueeze(1)
    return {
        "positions": model.get_positions().detach(),
        "scales": model.get_scale(preactivation=False).detach(),
        "rotations": model.get_rotation(preactivation=False).detach(),
        "densities": densities,
        "albedo": model.get_features_albedo().detach(),
        "specular": model.get_features_specular().detach(),
    }


def _attrs_from_post(post: Dict[str, torch.Tensor], preactivation: bool) -> GaussianAttributes:
    """Convert post-activation tensors to a NumPy ``GaussianAttributes``.

    For ``preactivation`` (PLY) the default ``exp``/``sigmoid`` activations are inverted
    (``log``/``inverse_sigmoid``); rotations are stored normalized (PLY re-normalizes on
    load). Used only for split children, whose values are synthetic anyway.
    """
    scales = post["scales"]
    densities = post["densities"]
    if preactivation:
        scales = torch.log(scales.clamp_min(1e-12))
        densities = inverse_sigmoid(densities.clamp(1e-6, 1.0 - 1e-6))

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    return GaussianAttributes(
        positions=to_np(post["positions"]),
        rotations=to_np(post["rotations"]),
        scales=to_np(scales),
        densities=to_np(densities).reshape(-1, 1),
        albedo=to_np(post["albedo"]),
        specular=to_np(post["specular"]),
    )


def _default_target_size(positions: torch.Tensor, num_partitions: int, fraction: float) -> float:
    """Estimate a split target footprint as a fraction of the expected cell edge."""
    span = (positions.amax(dim=0) - positions.amin(dim=0)).clamp_min(1e-8)
    cell_volume = float(span.prod().item()) / max(num_partitions, 1)
    cell_edge = cell_volume ** (1.0 / 3.0)
    return fraction * cell_edge


def _compute_metrics(
    positions: torch.Tensor,
    extents: torch.Tensor,
    labels: torch.Tensor,
    num_partitions: int,
    num_split_added: int,
    max_pairwise_partitions: int = 2048,
) -> Dict:
    """Balance and overlap metrics for the partitioning (all on-device)."""
    counts = torch.bincount(labels, minlength=num_partitions).float()

    expand = labels.unsqueeze(1).expand(-1, 3)
    box_min = torch.full((num_partitions, 3), float("inf"), device=positions.device, dtype=positions.dtype)
    box_max = torch.full((num_partitions, 3), float("-inf"), device=positions.device, dtype=positions.dtype)
    box_min = box_min.scatter_reduce(0, expand, positions - extents, reduce="amin", include_self=True)
    box_max = box_max.scatter_reduce(0, expand, positions + extents, reduce="amax", include_self=True)

    metrics = {
        "num_partitions": num_partitions,
        "total_exported": int(positions.shape[0]),
        "num_split_added": int(num_split_added),
        "count_min": int(counts.min().item()),
        "count_max": int(counts.max().item()),
        "count_mean": float(counts.mean().item()),
        "count_std": float(counts.std(unbiased=False).item()),
        "overlap_ratio": None,
    }

    # Pairwise AABB-overlap ratio (volume of overlaps / total partition volume). O(P^2);
    # only computed when the partition count is small enough to fit comfortably.
    if 1 < num_partitions <= max_pairwise_partitions:
        vol = (box_max - box_min).clamp_min(0.0).prod(dim=1)
        lo = torch.maximum(box_min[:, None, :], box_min[None, :, :])
        hi = torch.minimum(box_max[:, None, :], box_max[None, :, :])
        inter = (hi - lo).clamp_min(0.0).prod(dim=2)
        inter = torch.triu(inter, diagonal=1).sum()
        total = vol.sum().clamp_min(1e-12)
        metrics["overlap_ratio"] = float((inter / total).item())

    return metrics


def partition_scene(
    model: ExportableModel,
    max_per_volume: Optional[int],
    *,
    conf=None,
    split: bool = False,
    split_target_size: Optional[float] = None,
    split_target_fraction: float = 0.5,
    max_splits: int = 4,
    n_sigma: float = 3.0,
    frame_transform: Optional["np.ndarray"] = None,
) -> PartitionResult:
    """Partition one Gaussian source into volume partitions of at most ``max_per_volume``.

    When ``max_per_volume`` is None or >= the source's Gaussian count, no subdivision is
    done and a single-partition result is returned (exporters then reproduce the regular
    unpartitioned output). Each call handles a single source (one model, one ParticleField
    prim, or one input file); callers process multiple sources independently and never merge
    them into one array.

    Args:
        model: Source model (``ExportableModel``).
        max_per_volume: Maximum Gaussians per partition, or None for no subdivision.
        conf: Optional config (forwarded to the accessor for capabilities/activations).
        split: If True, run the oversized-Gaussian split pass before partitioning.
        split_target_size: Footprint threshold (world units) above which a Gaussian is
            split. Defaults to ``split_target_fraction`` of the estimated cell edge.
        split_target_fraction: Fraction of the cell edge used for the default threshold.
        max_splits: Maximum split iterations.
        n_sigma: Sigma multiplier for footprint/extent computation.
    """
    if max_per_volume is not None and max_per_volume < 1:
        raise ValueError(f"max_per_volume must be >= 1, got {max_per_volume}")

    accessor = GaussianExportAccessor(model, conf)
    capabilities = accessor.get_capabilities()
    num_gaussians = accessor.get_num_gaussians()

    # No partitioning needed: single partition, regular export reproduced exactly.
    if max_per_volume is None or num_gaussians <= max_per_volume:
        labels = torch.zeros(num_gaussians, dtype=torch.long, device=model.get_positions().device)
        logger.info("Source has %d Gaussians within budget; exporting a single unpartitioned volume", num_gaussians)
        return PartitionResult(
            labels=labels,
            num_partitions=1,
            is_partitioned=False,
            capabilities=capabilities,
            metrics={"num_partitions": 1, "total_exported": num_gaussians, "num_split_added": 0},
            _accessor=accessor,
        )

    num_split_added = 0
    split_post: Optional[Dict[str, torch.Tensor]] = None

    if split:
        post = _gather_post_tensors(model)
        target = split_target_size
        if target is None:
            est_partitions = -(-num_gaussians // max_per_volume)  # ceil
            target = _default_target_size(post["positions"], est_partitions, split_target_fraction)
        logger.info("Splitting oversized Gaussians (target footprint=%.4g, max_splits=%d)", target, max_splits)
        post, num_split_added = split_large_gaussians(post, target_size=target, n_sigma=n_sigma, max_splits=max_splits)
        if num_split_added:
            logger.info("Split pass added %d Gaussians (%d -> %d)", num_split_added, num_gaussians, post["positions"].shape[0])
        split_post = post
        positions = post["positions"]
        extents = gaussian_extents(post["scales"], post["rotations"], n_sigma=n_sigma)
    else:
        positions = model.get_positions().detach()
        extents = gaussian_extents(
            model.get_scale(preactivation=False).detach(),
            model.get_rotation(preactivation=False).detach(),
            n_sigma=n_sigma,
        )

    # Partition in the canonical (framed) space so KD-tree splits align to the chosen axes.
    # Labels still index the original Gaussians; the exporter authors the frame on the root.
    if frame_transform is not None:
        Tt = torch.as_tensor(np.asarray(frame_transform), dtype=positions.dtype, device=positions.device)
        positions = positions @ Tt[:3, :3].transpose(0, 1) + Tt[:3, 3]

    labels, num_partitions = kdtree_partition(positions, max_per_volume)
    metrics = _compute_metrics(positions, extents, labels, num_partitions, num_split_added)
    logger.info(
        "Partitioned %d Gaussians into %d volumes (counts %d-%d, mean %.0f, overlap_ratio=%s)",
        positions.shape[0],
        num_partitions,
        metrics["count_min"],
        metrics["count_max"],
        metrics["count_mean"],
        f"{metrics['overlap_ratio']:.4f}" if metrics["overlap_ratio"] is not None else "n/a",
    )

    return PartitionResult(
        labels=labels,
        num_partitions=num_partitions,
        is_partitioned=True,
        capabilities=capabilities,
        metrics=metrics,
        _accessor=None if split_post is not None else accessor,
        _split_post=split_post,
    )


def apply_frame_to_attributes(attrs: GaussianAttributes, transform, max_sh_degree: int) -> GaussianAttributes:
    """Bake a 4x4 frame transform into Gaussian attributes (for formats with no root xform, e.g. PLY).

    Rotates+translates positions, composes the rotation into per-Gaussian orientations, and rotates
    the view-dependent SH coefficients (so specular stays correct). Scales/opacity/DC are unchanged.
    The rotation used for orientations/SH is the orthonormalized linear part (assumes a rigid frame).
    """
    transform = np.asarray(transform)
    if np.allclose(transform, np.eye(4)):
        return attrs  # identity: leave attributes (and quaternion signs) untouched
    T = torch.as_tensor(transform, dtype=torch.float64)
    R_full = T[:3, :3]
    t = T[:3, 3]
    # Orthonormalized rotation for orientation/SH (robust to tiny non-orthogonality / uniform scale).
    U, _, Vh = torch.linalg.svd(R_full)
    R = U @ Vh
    if torch.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]
        R = U @ Vh

    positions = torch.from_numpy(np.asarray(attrs.positions, dtype=np.float64))
    new_positions = positions @ R_full.transpose(0, 1) + t

    quats = torch.from_numpy(np.asarray(attrs.rotations, dtype=np.float64))
    new_rot_mats = R.unsqueeze(0) @ quaternion_to_so3(quats)
    new_quats = so3_to_quaternion_wxyz(new_rot_mats)

    specular = torch.from_numpy(np.asarray(attrs.specular, dtype=np.float64))
    new_specular = rotate_specular(specular, R, max_sh_degree)

    return GaussianAttributes(
        positions=new_positions.to(torch.float32).numpy(),
        rotations=new_quats.to(torch.float32).numpy(),
        scales=attrs.scales,
        densities=attrs.densities,
        albedo=attrs.albedo,
        specular=new_specular.to(torch.float32).numpy(),
    )
