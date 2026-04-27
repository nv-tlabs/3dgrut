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
PPISP USD Writer.

Export PPISP (Physically Plausible Image Signal Processing) as a UsdShade
Shader prim on each camera's RenderProduct. Adapted from
nre-fermat/nre/utils/io/export/ppisp_usd_writer.py, replacing the
rig/timestamp frame-mapping with 3DGRUT integer frame indices.

PPISP pipeline stages:
1. Exposure compensation (per-frame, time-sampled)
2. Vignetting correction (per-camera, static)
3. Color correction via ZCA-based homography (per-frame, time-sampled)
4. Camera Response Function (per-camera, static)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from pxr import Gf, Sdf, Usd, UsdShade

if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]

log = logging.getLogger(__name__)

NUM_CHANNELS = 3
COLOR_PARAMS_PER_FRAME = 8
CHANNEL_SUFFIXES = ["R", "G", "B"]

PPISP_SPG_USDA_FILE = "ppisp_usd_spg.slang.usda"
PPISP_SPG_SLANG_FILE = "ppisp_usd_spg.slang"
PPISP_INPUT_RENDER_VAR = "HdrColor"
PPISP_OUTPUT_RENDER_VAR = "PPISPColor"
LDR_COLOR_RENDER_VAR = "LdrColor"


# ---------------------------------------------------------------------------
# Dataset frame-mapping helpers
# ---------------------------------------------------------------------------


def build_camera_frame_mapping(dataset) -> Tuple[List[str], Dict[str, List[int]]]:
    """Build per-camera frame lists from a 3DGRUT dataset.

    Returns:
        (camera_names, {camera_name: [frame_idx, ...]}) where frame_idx values
        are the global training indices used as USD time codes.
    """
    num_frames = len(dataset)

    camera_names: List[str]
    if hasattr(dataset, "get_camera_names"):
        camera_names = dataset.get_camera_names()
    else:
        camera_names = ["camera_0"]

    camera_frames: Dict[str, List[int]] = {name: [] for name in camera_names}

    for frame_idx in range(num_frames):
        if hasattr(dataset, "get_camera_idx"):
            cam_idx = dataset.get_camera_idx(frame_idx)
        else:
            cam_idx = 0
        if 0 <= cam_idx < len(camera_names):
            camera_frames[camera_names[cam_idx]].append(frame_idx)

    return camera_names, camera_frames


# ---------------------------------------------------------------------------
# Shader prim creation
# ---------------------------------------------------------------------------


def _add_ldr_color_render_var(
    stage: Usd.Stage,
    render_product_path: str,
    ppisp_output_path: Sdf.Path,
) -> str:
    """Create a LdrColor RenderVar wired to the PPISP output."""
    render_var_path = f"{render_product_path}/{LDR_COLOR_RENDER_VAR}"
    render_var = stage.DefinePrim(render_var_path, "RenderVar")
    render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(LDR_COLOR_RENDER_VAR)
    aov_attr = render_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    aov_attr.SetConnections([ppisp_output_path])
    return render_var_path


def _create_shader_prim(stage: Usd.Stage, render_product_path: str) -> UsdShade.Shader:
    """Create the PPISP Shader prim on a RenderProduct.

    Wires HdrColor → PPISP → LdrColor and appends LdrColor to orderedVars.
    Returns the UsdShade.Shader for parameter setting.
    """
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    # Mark HdrColor RenderVar input as an opaque AOV (no connection needed here)
    input_var_path = f"{render_product_path}/{PPISP_INPUT_RENDER_VAR}"
    input_var_prim = stage.GetPrimAtPath(input_var_path)
    if input_var_prim.IsValid():
        input_var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)

    # PPISP Shader prim referencing the SPG asset definition
    ppisp_shader_path = f"{render_product_path}/PPISP"
    shader = UsdShade.Shader.Define(stage, ppisp_shader_path)
    shader.GetPrim().GetReferences().AddReference(PPISP_SPG_USDA_FILE)
    # Duplicate the source metadata on the instance. Some Kit SPG/Fabric paths
    # do not resolve referenced shader metadata when opening packaged USDZ files.
    shader.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set(
        "sourceAsset"
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
        Sdf.AssetPath(PPISP_SPG_SLANG_FILE)
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        "ppispProcess"
    )

    # HdrColor opaque input wired to the input RenderVar's AOV
    hdr_input = shader.CreateInput(PPISP_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{PPISP_INPUT_RENDER_VAR}.omni:rtx:aov")])

    # PPISPColor opaque output
    shader.CreateOutput(PPISP_OUTPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)

    # LdrColor RenderVar connected to the PPISP output. This intentionally
    # replaces the display AOV with PPISP's LDR output.
    ppisp_output_path = shader.GetPath().AppendProperty(f"outputs:{PPISP_OUTPUT_RENDER_VAR}")
    ldr_var_path = _add_ldr_color_render_var(stage, render_product_path, ppisp_output_path)

    # Append LdrColor to orderedVars
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        targets = list(ordered_vars_rel.GetTargets())
        targets.append(Sdf.Path(LDR_COLOR_RENDER_VAR))
        ordered_vars_rel.SetTargets(targets)

    return shader


# ---------------------------------------------------------------------------
# Static parameter setters (per-camera)
# ---------------------------------------------------------------------------


def _set_vignetting_params(shader: UsdShade.Shader, ppisp: PPISP, camera_index: int) -> None:
    """Set per-camera vignetting parameters (static).

    ppisp.vignetting_params[camera_index] has shape [3, 5]:
    [cx, cy, alpha1, alpha2, alpha3] per channel.
    """
    vig = ppisp.vignetting_params[camera_index].cpu().numpy()  # [3, 5]
    for ch in range(NUM_CHANNELS):
        s = CHANNEL_SUFFIXES[ch]
        shader.CreateInput(f"vignettingCenter{s}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(vig[ch, 0]), float(vig[ch, 1]))
        )
        shader.CreateInput(f"vignettingAlpha1{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 2]))
        shader.CreateInput(f"vignettingAlpha2{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 3]))
        shader.CreateInput(f"vignettingAlpha3{s}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 4]))


def _set_crf_params(shader: UsdShade.Shader, ppisp: PPISP, camera_index: int) -> None:
    """Set per-camera CRF raw parameters (static).

    ppisp.crf_params[camera_index] has shape [3, 4]:
    [toe, shoulder, gamma, center] per channel (raw, activations applied in shader).
    """
    crf = ppisp.crf_params[camera_index].cpu().numpy()  # [3, 4]
    for ch in range(NUM_CHANNELS):
        s = CHANNEL_SUFFIXES[ch]
        shader.CreateInput(f"crfToe{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 0]))
        shader.CreateInput(f"crfShoulder{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 1]))
        shader.CreateInput(f"crfGamma{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 2]))
        shader.CreateInput(f"crfCenter{s}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 3]))


# ---------------------------------------------------------------------------
# Animated parameter setters (per-frame, time-sampled)
# ---------------------------------------------------------------------------


def _set_animated_exposure_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_indices: List[int],
) -> None:
    """Write time-sampled exposure offset; default = mean across this camera's frames.

    ppisp.exposure_params has shape [num_frames].
    Time code = float(frame_idx).
    """
    exposure = ppisp.exposure_params.cpu().numpy()  # [num_frames]

    valid = [i for i in frame_indices if i < len(exposure)]
    mean_val = float(np.mean(exposure[valid])) if valid else 0.0

    exposure_input = shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float)
    attr = exposure_input.GetAttr()
    attr.Set(mean_val)

    for frame_idx in valid:
        attr.Set(float(exposure[frame_idx]), float(frame_idx))


def _set_animated_color_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_indices: List[int],
) -> None:
    """Write time-sampled color latent offsets; default = mean across this camera's frames.

    ppisp.color_params has shape [num_frames, 8]:
    [db_r, db_g, dr_r, dr_g, dg_r, dg_g, dgray_r, dgray_g].
    Written as 4 float2 attributes.
    Time code = float(frame_idx).
    """
    color = ppisp.color_params.cpu().numpy()  # [num_frames, 8]

    valid = [i for i in frame_indices if i < len(color)]
    mean_color = np.mean(color[valid], axis=0) if valid else np.zeros(8)

    control_point_names = ["colorLatentBlue", "colorLatentRed", "colorLatentGreen", "colorLatentNeutral"]
    attrs = []
    for i, name in enumerate(control_point_names):
        inp = shader.CreateInput(name, Sdf.ValueTypeNames.Float2)
        attr = inp.GetAttr()
        attr.Set(Gf.Vec2f(float(mean_color[i * 2]), float(mean_color[i * 2 + 1])))
        attrs.append(attr)

    for frame_idx in valid:
        frame_color = color[frame_idx]
        for i, attr in enumerate(attrs):
            attr.Set(
                Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1])),
                float(frame_idx),
            )


# ---------------------------------------------------------------------------
# Per-camera entry point
# ---------------------------------------------------------------------------


def add_ppisp_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: PPISP,
    frame_indices: List[int],
) -> Usd.Prim:
    """Add a PPISP Shader to a RenderProduct for one physical camera.

    Per-camera parameters (vignetting, CRF) are written as static USD
    attributes. Per-frame parameters (exposure, color latents) are written
    with a mean-based default value and one time sample per training frame
    at time_code = float(frame_idx).

    Args:
        stage: USD stage containing the RenderProduct.
        render_product_path: Path to the RenderProduct prim.
        camera_index: Index of this camera in the PPISP model.
        ppisp: Trained PPISP module.
        frame_indices: Global frame indices belonging to this camera.

    Returns:
        The created PPISP Shader prim.
    """
    assert camera_index < ppisp.num_cameras, (
        f"camera_index {camera_index} >= ppisp.num_cameras {ppisp.num_cameras}"
    )
    if not frame_indices:
        log.warning(f"No frames for camera {camera_index} at {render_product_path}, skipping")
        return stage.GetPseudoRoot()

    shader = _create_shader_prim(stage, render_product_path)
    _set_vignetting_params(shader, ppisp, camera_index)
    _set_crf_params(shader, ppisp, camera_index)
    _set_animated_exposure_params(shader, ppisp, frame_indices)
    _set_animated_color_params(shader, ppisp, frame_indices)

    log.info(
        f"Added PPISP shader to {render_product_path} "
        f"(camera {camera_index}, {len(frame_indices)} frame(s))"
    )
    return shader.GetPrim()


# ---------------------------------------------------------------------------
# Batch export over all RenderProducts
# ---------------------------------------------------------------------------


def add_ppisp_to_all_render_products(
    stage: Usd.Stage,
    ppisp: PPISP,
    camera_names: List[str],
    camera_frame_mapping: Dict[str, List[int]],
    render_scope_path: str = "/Render",
) -> List[Usd.Prim]:
    """Add PPISP shaders to every RenderProduct in the Render scope.

    Args:
        stage: USD stage with a populated /Render scope.
        ppisp: Trained PPISP module.
        camera_names: Ordered list of camera names (index = camera_idx in ppisp).
        camera_frame_mapping: ``{camera_name: [frame_idx, ...]}`` from
            :func:`build_camera_frame_mapping`.
        render_scope_path: Path to the /Render Scope (default ``/Render``).

    Returns:
        List of created PPISP Shader prims.
    """
    from threedgrut.export.usd.writers.camera import _make_usd_prim_name

    render_scope = stage.GetPrimAtPath(render_scope_path)
    if not render_scope.IsValid():
        log.warning(f"Render scope not found at {render_scope_path}, skipping PPISP export")
        return []

    camera_name_to_index = {name: idx for idx, name in enumerate(camera_names)}
    created: List[Usd.Prim] = []

    for child in render_scope.GetChildren():
        if child.GetTypeName() != "RenderProduct":
            continue

        # RenderProduct prim name matches _make_usd_prim_name(camera_name)
        prim_name = child.GetName()
        # Reverse-lookup original camera_name by prim name
        camera_name = next(
            (n for n in camera_names if _make_usd_prim_name(n) == prim_name),
            None,
        )
        if camera_name is None:
            log.warning(f"RenderProduct '{prim_name}' has no matching camera name, skipping")
            continue

        camera_index = camera_name_to_index.get(camera_name)
        if camera_index is None:
            log.warning(f"Camera '{camera_name}' not in camera_names list, skipping")
            continue

        frame_indices = camera_frame_mapping.get(camera_name, [])

        shader_prim = add_ppisp_shader_to_render_product(
            stage=stage,
            render_product_path=str(child.GetPath()),
            camera_index=camera_index,
            ppisp=ppisp,
            frame_indices=frame_indices,
        )
        created.append(shader_prim)

    log.info(f"Added PPISP shaders to {len(created)} RenderProduct(s)")
    return created
