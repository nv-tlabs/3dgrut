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

import os
import tempfile
import zipfile
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf, UsdVol, UsdUtils, Vt

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.stage.Export(str(out_dir / self.filename))

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)


def initialize_usd_stage():
    """
    Initialize a new USD stage with standard settings.

    Returns:
        Usd.Stage: A new USD stage with standard settings
    """
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")

    # Define xform containing everything.
    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage


def serialize_usd_stage_to_bytes(stage: Usd.Stage) -> bytes:
    """
    Export a USD stage to a temporary file and read it back as bytes.

    Args:
        stage: The USD stage to export

    Returns:
        bytes: The exported USD stage content
    """
    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as temp_file:
        temp_file_path = temp_file.name

    stage.GetRootLayer().Export(temp_file_path)

    with open(temp_file_path, "rb") as f:
        content = f.read()

    os.unlink(temp_file_path)
    return content


def serialize_nurec_usd(model_file, positions: np.ndarray, normalizing_transform: np.ndarray = np.eye(4)) -> NamedUSDStage:
    """
    Create a USD file for the 3DGS model.

    Args:
        model_file: NamedSerialized object containing the compressed msgpack data
        positions: Positions extracted from PLY file for AABB calculation
        normalizing_transform: 4x4 transformation matrix to normalize the scene (defaults to identity)

    Returns:
        NamedUSDStage object containing the USD stage
    """
    logger.info("Creating USD file containing NuRec model")

    # Calculate AABB from positions
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)
    logger.info(f"Model bounding box: min={min_coord}, max={max_coord}")

    # Convert numpy values to Python floats
    min_x, min_y, min_z = float(min_coord[0]), float(
        min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(
        max_coord[1]), float(max_coord[2])

    min_list = [min_x, min_y, min_z]
    max_list = [max_x, max_y, max_z]

    # Initialize the USD stage with standard settings
    stage = initialize_usd_stage()

    # Set up render settings
    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True
    }
    stage.SetMetadataByDictKey(
        "customLayerData", "renderSettings", render_settings)

    # Define UsdVol::Volume
    gauss_path = "/World/gauss"
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()

    # Apply normalizing transform (identity by default)
    # Default conversion matrix from 3DGRUT to USDZ
    default_conv_tf = np.array([
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ])

    normalizing_inverse = np.linalg.inv(normalizing_transform)
    corrected_matrix = normalizing_inverse @ default_conv_tf

    # Apply transform directly to the gauss volume
    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(*corrected_matrix.flatten()))

    # Define nurec volume properties
    gauss_prim.CreateAttribute(
        "omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)

    # Enable transform of UsdVol::Volume to take effect
    gauss_prim.CreateAttribute(
        "omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define field assets and link to volumetric Gaussians prim
    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(
        emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship(
        "emissiveColor", emissive_color_field_path)

    # Set file paths for field assets
    nurec_relative_path = "./" + model_file.filename
    density_field.CreateAttribute(
        "filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute(
        "fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute(
        "fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute(
        "fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute(
        "filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute(
        "fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute(
        "fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute(
        "fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

    # Set identity color correction matrix
    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([1.0, 0.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 1.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 0.0, 1.0, 0.0])
    )

    # Set extent and crop boundaries
    gauss_prim.GetAttribute("extent").Set([min_list, max_list])

    # Set zero offset
    gauss_offset = [0.0, 0.0, 0.0]
    gauss_prim.CreateAttribute(
        "omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(gauss_offset))

    # Set crop bounds
    min_vec = Gf.Vec3d(min_x, min_y, min_z)
    max_vec = Gf.Vec3d(max_x, max_y, max_z)
    gauss_prim.CreateAttribute(
        "omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(min_vec)
    gauss_prim.CreateAttribute(
        "omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(max_vec)

    # Create empty proxy mesh relationship for forward compatibility
    gauss_prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)


def update_render_settings(stage: Usd.Stage, referenced_layer: Sdf.Layer) -> None:
    """
    Update render settings from a referenced layer.

    Args:
        stage: The stage to update
        referenced_layer: The layer containing render settings to copy
    """
    if "renderSettings" not in referenced_layer.customLayerData:
        return  # Do nothing if render settings are not present in the referenced layer

    new_render_settings = referenced_layer.customLayerData["renderSettings"]
    current_render_settings = stage.GetRootLayer(
    ).customLayerData.get("renderSettings", {})
    if current_render_settings is None:
        current_render_settings = {}

    current_render_settings.update(new_render_settings)
    stage.SetMetadataByDictKey(
        "customLayerData", "renderSettings", current_render_settings)


def serialize_usd_default_layer(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    """
    Create a default USD layer that references the gauss stage.

    Args:
        gauss_stage: The NamedUSDStage object containing the gauss USD stage

    Returns:
        NamedUSDStage: The default USD stage with the gauss reference
    """
    stage = initialize_usd_stage()

    # The delegate captures all errors about dangling references, effectively silencing them.
    delegate = UsdUtils.CoalescingDiagnosticDelegate()

    # Create a reference to the gauss stage
    prim = stage.OverridePrim(f"/World/{Path(gauss_stage.filename).stem}")
    # Assume that all reference paths are in the same directory, so that they are also valid relative file paths.
    prim.GetReferences().AddReference(gauss_stage.filename)

    # Copy render settings from the gauss stage's layer
    gauss_layer = gauss_stage.stage.GetRootLayer()
    if "renderSettings" in gauss_layer.customLayerData:
        update_render_settings(stage, gauss_layer)

    # Return as NamedUSDStage
    return NamedUSDStage(filename="default.usda", stage=stage)


def write_to_usdz(file_path: Path, model_file, gauss_usd: NamedUSDStage, default_usd: NamedUSDStage) -> None:
    """
    Write the USDZ file containing the model data and USD stages.

    Args:
        file_path: Path to write the USDZ file to
        model_file: The compressed model data
        gauss_usd: The gauss USD stage
        default_usd: The default USD stage
    """
    # Make sure path to usdz-file exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        # Save default.usda first (required by USDZ spec)
        default_usd.save_to_zip(zip_file)

        # Save the model file and gauss USD stage
        model_file.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)

    logger.info(f"USDZ file created successfully at {file_path}")
