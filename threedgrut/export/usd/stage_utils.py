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
USD Stage utilities for Gaussian export.

Provides functions for creating and configuring USD stages,
coordinate transforms, and USDZ packaging.
"""

import logging
import os
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from pxr import Gf, Usd, UsdGeom

from threedgrut.export.transforms import (
    USDTransformSamples,
    apply_usd_transform_samples,
    column_vector_4x4_to_usd_matrix,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_FRAME_RATE = 24.0
USD_WORLD_PATH = "/World"
_USDZ_ALIGNMENT = 64
_USDZ_PADDING_EXTRA_ID = 0x1986


def _write_usdz_entry(zip_file: zipfile.ZipFile, filename: str, data: Union[str, bytes]) -> None:
    if isinstance(data, str):
        data = data.encode("utf-8")

    header_offset = zip_file.fp.tell()
    filename_size = len(filename.encode("utf-8"))
    unpadded_data_offset = header_offset + 30 + filename_size
    padding_size = (-unpadded_data_offset) % _USDZ_ALIGNMENT

    # ZIP extra fields need a 4-byte header. If the needed padding is smaller,
    # add one full alignment period and keep the same modulo.
    if 0 < padding_size < 4:
        padding_size += _USDZ_ALIGNMENT

    zip_info = zipfile.ZipInfo(filename)
    zip_info.compress_type = zipfile.ZIP_STORED
    if padding_size:
        zip_info.extra = struct.pack("<HH", _USDZ_PADDING_EXTRA_ID, padding_size - 4)
        zip_info.extra += b"\0" * (padding_size - 4)

    zip_file.writestr(zip_info, data)


@dataclass(kw_only=True)
class NamedUSDStage:
    """Container for a USD stage with an associated filename."""

    filename: str
    stage: Usd.Stage

    def save(self, out_dir: Path):
        """Save the stage to a directory."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.stage.Export(str(out_dir / self.filename))

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        """Save the stage to a zip file."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        _write_usdz_entry(zip_file, self.filename, usd_data)
        os.unlink(temp_file_path)


@dataclass(kw_only=True)
class NamedSerialized:
    """Container for serialized data with a filename."""

    filename: str
    serialized: Union[str, bytes]

    def save(self, out_dir: Path):
        """Save the serialized data to a directory."""
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(self.serialized, bytes) else "w"
        with open(out_dir / self.filename, mode) as f:
            f.write(self.serialized)

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        """Save the serialized data to a zip file."""
        _write_usdz_entry(zip_file, self.filename, self.serialized)


def initialize_usd_stage(up_axis: str = "Y") -> Usd.Stage:
    """
    Initialize a new USD stage with standard settings.

    Args:
        up_axis: Up axis for the stage ("Y" or "Z")

    Returns:
        Usd.Stage: A new USD stage with standard settings
    """
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", up_axis)
    stage.SetTimeCodesPerSecond(DEFAULT_FRAME_RATE)

    # Define xform containing everything
    UsdGeom.Xform.Define(stage, USD_WORLD_PATH)
    stage.SetMetadata("defaultPrim", USD_WORLD_PATH[1:])

    return stage


def create_gaussian_model_root(
    stage: Usd.Stage,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    flip_z_axis: bool = False,
    root_path: str = "/World/Gaussians",
    normalizing_transform: np.ndarray = None,
    coordinate_transform: np.ndarray = None,
    source_transform_samples: Optional[USDTransformSamples] = None,
    canonical_frame_transform: np.ndarray = None,
) -> str:
    """
    Create the root Xform for Gaussian content with optional coordinate transforms.

    Args:
        stage: USD stage to create the root on
        flip_x_axis: Negate X coordinates
        flip_y_axis: Negate Y coordinates
        flip_z_axis: Negate Z coordinates
        root_path: USD path for the root prim
        normalizing_transform: Optional 4x4 normalizing transform matrix
        coordinate_transform: Optional 4x4 (e.g. 3DGRUT-to-USDZ). Applied after normalizing and scale.
        source_transform_samples: Optional source Gaussian local-to-world transform samples,
            authored before normalization / coordinate transform to match the NuRec exporter.
        canonical_frame_transform: Optional 4x4 canonical object frame. Authored as its own named
            ``xformOp:transform:canonicalFrame`` (outermost), so it stays independently recoverable
            and re-authorable, and lives on this content root (not /World) to be composition-safe.

    Returns:
        The root path string
    """
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    apply_usd_transform_samples(root_xform, source_transform_samples)

    # Build scale matrix for axis flipping
    scale_x = -1.0 if flip_x_axis else 1.0
    scale_y = -1.0 if flip_y_axis else 1.0
    scale_z = -1.0 if flip_z_axis else 1.0

    # Compute combined transform
    has_scale = scale_x != 1.0 or scale_y != 1.0 or scale_z != 1.0
    has_normalizing = normalizing_transform is not None
    has_coordinate = coordinate_transform is not None

    if has_scale or has_normalizing or has_coordinate:
        transform_op = root_xform.AddTransformOp()

        # Start with identity
        combined = Gf.Matrix4d(1.0)

        # Apply normalizing transform first (if provided)
        if has_normalizing:
            combined = column_vector_4x4_to_usd_matrix(normalizing_transform)

        # Then apply scale
        if has_scale:
            scale_mat = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_x, scale_y, scale_z))
            combined = combined * scale_mat

        # Then apply optional 3DGRUT-to-USDZ coordinate transform
        if has_coordinate:
            combined = column_vector_4x4_to_usd_matrix(coordinate_transform) * combined

        transform_op.Set(combined)

    # Canonical object frame as its own named op, applied outermost (after the combined op).
    if is_effective_frame(canonical_frame_transform):
        _set_or_add_canonical_frame(root_xform, column_vector_4x4_to_usd_matrix(np.asarray(canonical_frame_transform)))

    return root_path


def is_effective_frame(transform) -> bool:
    """True if ``transform`` is a non-None, non-identity 4x4 (worth authoring)."""
    return transform is not None and not np.allclose(np.asarray(transform), np.eye(4))


_CANONICAL_FRAME_OP = "xformOp:transform:canonicalFrame"


def _set_or_add_canonical_frame(xformable: "UsdGeom.Xformable", frame_mat: Gf.Matrix4d) -> None:
    """Set the canonicalFrame op if the prim already has one, else append it (idempotent).

    Re-authoring rather than appending keeps re-framing an already-framed asset from stacking
    multiple canonicalFrame ops (which would apply the frame more than once).
    """
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == _CANONICAL_FRAME_OP:
            op.Set(frame_mat)
            return
    xformable.AddTransformOp(opSuffix="canonicalFrame").Set(frame_mat)


def apply_canonical_frame_to_scene(stage: Usd.Stage, frame_transform, skip_paths) -> None:
    """Author the ``canonicalFrame`` named op on top of each /World scene subtree.

    Used to move copied/foreign source prims (cameras, rig, …) by the canonical frame without
    re-parenting or remapping — the op is set on the topmost Xformable of each subtree, so any
    source hierarchy and its references are preserved. ``skip_paths`` are the content-root prim
    paths that already carry the frame themselves (e.g. the authored ``/World/Gaussians`` or the
    NuRec ``/World/gauss`` reference) — identified by path, not by name, so the skip can't drift
    from where the frame was actually authored. The op is idempotent (replaces, never stacks).
    """
    if not is_effective_frame(frame_transform):
        return
    frame_mat = column_vector_4x4_to_usd_matrix(np.asarray(frame_transform))
    skip = {str(p) for p in skip_paths}

    def _apply(prim):
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            _set_or_add_canonical_frame(xformable, frame_mat)
        else:
            for child in prim.GetChildren():
                _apply(child)

    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        for child in world.GetChildren():
            if str(child.GetPath()) not in skip:
                _apply(child)


def compose_default_stage(
    stages: List[NamedUSDStage],
    render_settings: Optional[dict] = None,
) -> NamedUSDStage:
    """
    Create a composition stage that references all provided stages.

    Args:
        stages: List of USD stages to compose
        render_settings: Optional render settings to add

    Returns:
        NamedUSDStage containing the composition
    """
    stage = initialize_usd_stage()

    if render_settings:
        stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    for named_stage in stages:
        filename_stem = Path(named_stage.filename).stem
        prim_path = f"{USD_WORLD_PATH}/{filename_stem}"
        prim = stage.OverridePrim(prim_path)
        prim.GetReferences().AddReference(named_stage.filename)

    return NamedUSDStage(filename="default.usda", stage=stage)


def write_to_usdz(
    output_path: Path,
    stages: List[NamedUSDStage],
    files: Optional[List[NamedSerialized]] = None,
) -> None:
    """
    Package USD stages and files into a USDZ archive.

    Args:
        output_path: Path for the output USDZ file
        stages: List of USD stages to include
        files: Optional list of additional files (HDR textures, etc.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for stage in stages:
            stage.save_to_zip(zip_file)
        if files:
            for file in files:
                file.save_to_zip(zip_file)

    logger.info(f"USDZ file created: {output_path}")
