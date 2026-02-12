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
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from pxr import Gf, Usd, UsdGeom

from threedgrut.export.transforms import column_vector_4x4_to_usd_matrix

logger = logging.getLogger(__name__)

# Constants
DEFAULT_FRAME_RATE = 24.0
USD_WORLD_PATH = "/World"


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
        zip_file.writestr(self.filename, usd_data)
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
        zip_file.writestr(self.filename, self.serialized)


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

    Returns:
        The root path string
    """
    root_xform = UsdGeom.Xform.Define(stage, root_path)

    # Build scale matrix for axis flipping
    scale_x = -1.0 if flip_x_axis else 1.0
    scale_y = -1.0 if flip_y_axis else 1.0
    scale_z = -1.0 if flip_z_axis else 1.0

    # Compute combined transform
    has_scale = scale_x != 1.0 or scale_y != 1.0 or scale_z != 1.0
    has_normalizing = normalizing_transform is not None

    if has_scale or has_normalizing:
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

        transform_op.Set(combined)

    return root_path


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
