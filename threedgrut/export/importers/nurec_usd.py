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

"""NuRec USD format importer for Gaussian splatting data."""

import gzip
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import msgpack
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.importers.base import FormatImporter

logger = logging.getLogger(__name__)

# State dict keys written by fill_3dgut_template / _fill_state_dict_tensors
_STATE_POSITIONS = ".gaussians_nodes.gaussians.positions"
_STATE_ROTATIONS = ".gaussians_nodes.gaussians.rotations"
_STATE_SCALES = ".gaussians_nodes.gaussians.scales"
_STATE_DENSITIES = ".gaussians_nodes.gaussians.densities"
_STATE_FEATURES_ALBEDO = ".gaussians_nodes.gaussians.features_albedo"
_STATE_FEATURES_SPECULAR = ".gaussians_nodes.gaussians.features_specular"
_STATE_N_ACTIVE = ".gaussians_nodes.gaussians.n_active_features"
_STATE_EXTRA_SIGNAL = ".gaussians_nodes.gaussians.extra_signal"


def _find_nurec_volume_prim(stage: Usd.Stage) -> Optional[Usd.Prim]:
    """Find the NuRec Volume prim (UsdVol::Volume with omni:nurec:isNuRecVolume)."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Volume":
            continue
        attr = prim.GetAttribute("omni:nurec:isNuRecVolume")
        if attr.IsValid() and attr.Get():
            return prim
    return None


def _get_nurec_file_path(volume_prim: Usd.Prim) -> Optional[Path]:
    """Get the .nurec asset path from a child OmniNuRecFieldAsset (filePath attribute)."""
    for child in volume_prim.GetChildren():
        if child.GetTypeName() != "OmniNuRecFieldAsset":
            continue
        attr = child.GetAttribute("filePath")
        if not attr.IsValid():
            continue
        val = attr.Get()
        if val is None:
            continue
        # Sdf.AssetPath or string
        path_str = str(val.path if hasattr(val, "path") else val).strip()
        if path_str.startswith("./"):
            path_str = path_str[2:]
        if path_str.lower().endswith(".nurec"):
            return Path(path_str)
    return None


def _load_nurec_bytes(resolution_root: Path, file_path: Path) -> bytes:
    """Resolve file_path relative to resolution_root and load gzip-compressed bytes."""
    candidates = [
        resolution_root / file_path,
        resolution_root / file_path.name,
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return gzip.decompress(f.read())
    raise FileNotFoundError(f"NuRec data file not found. Tried: {[str(c) for c in candidates]}")


def _decode_state_dict(raw: bytes) -> dict:
    """Unpack msgpack payload (expects nre_data.state_dict structure)."""
    data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    if "nre_data" in data and "state_dict" in data["nre_data"]:
        return data["nre_data"]["state_dict"]
    raise ValueError("NuRec msgpack missing nre_data.state_dict")


def _tensor_from_state(state: dict, key: str, dtype=np.float16, shape_key: Optional[str] = None) -> np.ndarray:
    """Decode a tensor from state_dict (bytes + shape)."""
    if shape_key is None:
        shape_key = key + ".shape"
    raw = state.get(key)
    shape = state.get(shape_key)
    if raw is None:
        raise KeyError(f"Missing state key: {key}")
    arr = np.frombuffer(raw, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr.astype(np.float32)


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion (one quat)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / ((1.0 + trace) ** 0.5 + 1e-8)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def _apply_volume_transform(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    matrix: Gf.Matrix4d,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Volume local-to-world transform to positions, rotations, and scales."""
    m = matrix
    # Build 4x4 numpy (row-major as in Gf)
    t = np.array(
        [
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ],
        dtype=np.float64,
    )
    # Positions: p' = p * M^T (row vectors), then take xyz
    ones = np.ones((positions.shape[0], 1), dtype=np.float64)
    p4 = np.hstack([positions.astype(np.float64), ones])
    p4_new = (p4 @ t.T)[:, :3]
    positions_out = p4_new.astype(np.float32)

    # 3x3 linear part: column norms = scale factors
    lin = np.array(
        [
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        ],
        dtype=np.float64,
    )
    scale_factors = np.array(
        [
            np.linalg.norm(lin[:, 0]),
            np.linalg.norm(lin[:, 1]),
            np.linalg.norm(lin[:, 2]),
        ],
        dtype=np.float32,
    )
    scale_factors = np.maximum(scale_factors, 1e-8)
    scales_out = scales * scale_factors

    # Rotation: divide 3x3 by scale to get R, then R_vol * q_local
    lin_scaled = lin / scale_factors.astype(np.float64)
    q_vol = _rotation_matrix_to_quat_wxyz(lin_scaled)
    qw, qx, qy, qz = q_vol[0], q_vol[1], q_vol[2], q_vol[3]
    rw, rx, ry, rz = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    nw = qw * rw - qx * rx - qy * ry - qz * rz
    nx = qw * rx + qx * rw + qy * rz - qz * ry
    ny = qw * ry - qx * rz + qy * rw + qz * rx
    nz = qw * rz + qx * ry - qy * rx + qz * rw
    rotations_out = np.stack([nw, nx, ny, nz], axis=1).astype(np.float32)
    return positions_out, rotations_out, scales_out


class NuRecUSDImporter(FormatImporter):
    """Importer for NuRec USD/USDZ (UsdVol::Volume + .nurec gzip-msgpack payload).

    Reads Omniverse NuRec format and returns pre-activation Gaussian attributes.
    """

    @property
    def stores_preactivation(self) -> bool:
        return True

    def load(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load NuRec USD or USDZ into GaussianAttributes."""
        path = Path(path)
        if path.suffix.lower() == ".usdz":
            return self._load_usdz(path)
        return self._load_usd(path)

    def _load_usdz(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir_path)
            usd_files = list(tmpdir_path.glob("*.usd*"))
            root_file = None
            for f in usd_files:
                if f.stem == "default":
                    root_file = f
                    break
            if root_file is None:
                root_file = usd_files[0] if usd_files else None
            if root_file is None:
                raise ValueError(f"No USD file found in USDZ: {path}")
            return self._load_stage(root_file, resolution_root=tmpdir_path)

    def _load_usd(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        return self._load_stage(path, resolution_root=path.parent)

    def _load_stage(self, stage_path: Path, resolution_root: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        stage = Usd.Stage.Open(str(stage_path))
        if not stage:
            raise ValueError(f"Failed to open USD stage: {stage_path}")

        volume_prim = _find_nurec_volume_prim(stage)
        if volume_prim is None:
            raise ValueError(f"No NuRec Volume prim (UsdVol::Volume with omni:nurec:isNuRecVolume) in {stage_path}")

        nurec_path = _get_nurec_file_path(volume_prim)
        if nurec_path is None:
            raise ValueError(f"NuRec Volume has no OmniNuRecFieldAsset with filePath: {volume_prim.GetPath()}")

        raw = _load_nurec_bytes(resolution_root, nurec_path)
        state = _decode_state_dict(raw)

        positions = _tensor_from_state(state, _STATE_POSITIONS)
        rotations = _tensor_from_state(state, _STATE_ROTATIONS)
        scales = _tensor_from_state(state, _STATE_SCALES)
        densities = _tensor_from_state(state, _STATE_DENSITIES)
        features_albedo = _tensor_from_state(state, _STATE_FEATURES_ALBEDO)
        features_specular = _tensor_from_state(state, _STATE_FEATURES_SPECULAR)

        n_active = state.get(_STATE_N_ACTIVE)
        if n_active is not None:
            sh_degree = int(np.frombuffer(n_active, dtype=np.int64)[0])
        else:
            # Infer from features_specular shape: (N, (degree+1)^2 - 1) * 3
            n_spec = features_specular.shape[1]
            # (deg+1)^2 - 1 = n_spec/3  =>  deg = sqrt(n_spec/3 + 1) - 1
            sh_degree = int(round((n_spec // 3 + 1) ** 0.5 - 1))
            sh_degree = max(0, min(3, sh_degree))

        num_gaussians = positions.shape[0]
        if densities.ndim == 1:
            densities = densities.reshape(-1, 1)

        # Apply Volume local-to-world transform
        xformable = UsdGeom.Xformable(volume_prim)
        world_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        positions, rotations, scales = _apply_volume_transform(positions, rotations, scales, world_matrix)

        attrs = GaussianAttributes(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            albedo=features_albedo,
            specular=features_specular,
        )

        caps = ModelCapabilities(
            has_spherical_harmonics=True,
            sh_degree=sh_degree,
            num_gaussians=num_gaussians,
            is_surfel=False,
            density_activation="sigmoid",
            scale_activation="exp",
        )

        logger.info(f"Loaded {num_gaussians} Gaussians from NuRec, SH degree {sh_degree}")
        return attrs, caps
