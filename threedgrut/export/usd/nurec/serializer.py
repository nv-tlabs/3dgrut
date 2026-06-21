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

"""NuRec USD serialization utilities."""

import logging
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol

from threedgrut.export.transforms import (
    USDTransformSamples,
    apply_usd_transform_samples,
    column_vector_4x4_to_usd_matrix,
    get_3dgrut_to_usdz_coordinate_transform,
)
from threedgrut.export.usd.nurec.templates import NamedSerialized
from threedgrut.export.usd.stage_utils import NamedUSDStage
from threedgrut.export.usd.stage_utils import (
    initialize_usd_stage as _initialize_usd_stage,
)

logger = logging.getLogger(__name__)

# NuRec uses Z-up axis (Omniverse convention)
NUREC_UP_AXIS = "Z"


def initialize_usd_stage(up_axis: str = NUREC_UP_AXIS) -> Usd.Stage:
    """
    Initialize a new USD stage with NuRec/Omniverse settings (Z-up by default).

    Returns:
        Usd.Stage: A new USD stage
    """
    return _initialize_usd_stage(up_axis=up_axis)


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


def _author_render_settings(
    stage: Usd.Stage,
    *,
    invert_registered_compositing: bool,
    skip_gaussian_tonemapping: bool,
) -> None:
    """Author the NuRec default renderer settings on a stage."""
    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": invert_registered_compositing,
        "rtx:post:registeredCompositing:invertColorCorrection": invert_registered_compositing,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    if skip_gaussian_tonemapping:
        render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"] = False
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)


def _author_nurec_volume(
    stage: Usd.Stage,
    model_file: NamedSerialized,
    positions: np.ndarray,
    *,
    volume_path: str,
    normalizing_transform: np.ndarray,
    apply_coordinate_transform: bool,
    source_gaussian_transform: USDTransformSamples | None,
    canonical_frame_transform=None,
) -> None:
    """Author a single NuRec ``UsdVol.Volume`` (referencing ``model_file``) into ``stage``."""
    # Calculate AABB from positions
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)

    # Convert numpy values to Python floats
    min_x, min_y, min_z = float(min_coord[0]), float(min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(max_coord[1]), float(max_coord[2])

    min_list = [min_x, min_y, min_z]
    max_list = [max_x, max_y, max_z]

    gauss_path = volume_path
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()
    logger.info(
        f"  {gauss_path}: {len(positions)} gaussians, extent "
        f"min=[{min_x:.4g}, {min_y:.4g}, {min_z:.4g}] max=[{max_x:.4g}, {max_y:.4g}, {max_z:.4g}]"
    )

    # Apply normalizing transform (identity by default). Optionally apply 3DGRUT-to-USDZ coordinate transform.
    normalizing_inverse = np.linalg.inv(normalizing_transform)
    if apply_coordinate_transform:
        coord_tf = get_3dgrut_to_usdz_coordinate_transform()
        corrected_matrix = normalizing_inverse @ coord_tf
    else:
        corrected_matrix = normalizing_inverse

    # Apply transform directly to the gauss volume
    apply_usd_transform_samples(gauss_volume, source_gaussian_transform)
    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(*corrected_matrix.flatten()))

    # Canonical object frame as its own named op (outermost), on the volume — composition-safe
    # (not authored on /World) and independently recoverable.
    if canonical_frame_transform is not None and not np.allclose(np.asarray(canonical_frame_transform), np.eye(4)):
        gauss_volume.AddTransformOp(opSuffix="canonicalFrame").Set(
            column_vector_4x4_to_usd_matrix(np.asarray(canonical_frame_transform))
        )

    # Define nurec volume properties
    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)

    # Enable transform of UsdVol::Volume to take effect
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define field assets and link to volumetric Gaussians prim
    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    # Set file paths for field assets
    nurec_relative_path = "./" + model_file.filename
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

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
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(gauss_offset))

    # Set crop bounds
    min_vec = Gf.Vec3d(min_x, min_y, min_z)
    max_vec = Gf.Vec3d(max_x, max_y, max_z)
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(min_vec)
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(max_vec)

    # Create empty proxy mesh relationship for forward compatibility
    gauss_prim.CreateRelationship("proxy")


def serialize_nurec_usd(
    model_file: NamedSerialized,
    positions: np.ndarray,
    normalizing_transform: np.ndarray = np.eye(4),
    apply_coordinate_transform: bool = False,
    source_gaussian_transform: USDTransformSamples | None = None,
    author_render_settings: bool = True,
    invert_registered_compositing: bool = True,
    skip_gaussian_tonemapping: bool = False,
    world_frame_transform=None,
    up_axis: str = NUREC_UP_AXIS,
) -> NamedUSDStage:
    """Create a USD stage with a single NuRec ``UsdVol.Volume`` at ``/World/gauss``.

    See :func:`serialize_nurec_usd_partitions` for the multi-volume variant.
    """
    logger.info("Creating USD file containing NuRec model")
    stage = initialize_usd_stage(up_axis=up_axis)
    if author_render_settings:
        _author_render_settings(
            stage,
            invert_registered_compositing=invert_registered_compositing,
            skip_gaussian_tonemapping=skip_gaussian_tonemapping,
        )
    _author_nurec_volume(
        stage,
        model_file,
        positions,
        volume_path="/World/gauss",
        normalizing_transform=normalizing_transform,
        apply_coordinate_transform=apply_coordinate_transform,
        source_gaussian_transform=source_gaussian_transform,
        canonical_frame_transform=world_frame_transform,
    )
    return NamedUSDStage(filename="gauss.usda", stage=stage)


def serialize_nurec_usd_partitions(
    volume_specs: list,
    normalizing_transform: np.ndarray = np.eye(4),
    apply_coordinate_transform: bool = False,
    source_gaussian_transform: USDTransformSamples | None = None,
    author_render_settings: bool = True,
    invert_registered_compositing: bool = True,
    skip_gaussian_tonemapping: bool = False,
    world_frame_transform=None,
    up_axis: str = NUREC_UP_AXIS,
) -> NamedUSDStage:
    """Create a USD stage with one NuRec ``UsdVol.Volume`` per partition.

    ``volume_specs`` is a list of ``(model_file, positions)`` — each becomes a Volume at
    ``/World/Partition_NNN/gauss`` referencing its own ``.nurec`` payload. Volumes are never
    fused; each carries its own field assets and crop bounds.
    """
    logger.info("Creating USD file containing %d NuRec volumes", len(volume_specs))
    stage = initialize_usd_stage(up_axis=up_axis)
    if author_render_settings:
        _author_render_settings(
            stage,
            invert_registered_compositing=invert_registered_compositing,
            skip_gaussian_tonemapping=skip_gaussian_tonemapping,
        )
    width = max(3, len(str(max(len(volume_specs) - 1, 0))))
    for i, (model_file, positions) in enumerate(volume_specs):
        _author_nurec_volume(
            stage,
            model_file,
            positions,
            volume_path=f"/World/Partition_{i:0{width}d}/gauss",
            normalizing_transform=normalizing_transform,
            apply_coordinate_transform=apply_coordinate_transform,
            source_gaussian_transform=source_gaussian_transform,
            canonical_frame_transform=world_frame_transform,
        )
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
    current_render_settings = stage.GetRootLayer().customLayerData.get("renderSettings", {})
    if current_render_settings is None:
        current_render_settings = {}

    current_render_settings.update(new_render_settings)
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", current_render_settings)


def serialize_usd_default_layer(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    """
    Create a default USD layer that references the gauss stage.

    Args:
        gauss_stage: The NamedUSDStage object containing the gauss USD stage

    Returns:
        NamedUSDStage: The default USD stage with the gauss reference
    """
    stage = initialize_usd_stage()
    if getattr(gauss_stage.stage, "HasAuthoredTimeCodeRange", None) and gauss_stage.stage.HasAuthoredTimeCodeRange():
        stage.SetStartTimeCode(gauss_stage.stage.GetStartTimeCode())
        stage.SetEndTimeCode(gauss_stage.stage.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(gauss_stage.stage.GetTimeCodesPerSecond())

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


def write_to_usdz(
    file_path: Path,
    model_file,
    gauss_usd: NamedUSDStage,
    default_usd: NamedUSDStage,
    extra_files: list[NamedSerialized] | None = None,
) -> None:
    """
    Write the USDZ file containing the model data and USD stages.

    Args:
        file_path: Path to write the USDZ file to
        model_file: The compressed model data — a single NamedSerialized or a list (one
            ``.nurec`` payload per volume for multi-volume exports)
        gauss_usd: The gauss USD stage
        default_usd: The default USD stage
    """
    model_files = model_file if isinstance(model_file, (list, tuple)) else [model_file]

    # Make sure path to usdz-file exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        # Save default.usda first (required by USDZ spec)
        default_usd.save_to_zip(zip_file)

        # Save the model file(s), gauss USD stage, and optional shader sidecars.
        for mf in model_files:
            mf.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)
        for extra_file in extra_files or []:
            extra_file.save_to_zip(zip_file)

    logger.info(f"USDZ file created successfully at {file_path}")
