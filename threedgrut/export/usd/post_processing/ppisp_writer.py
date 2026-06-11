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
Shader prim on each camera's RenderProduct, using 3DGRUT integer frame
indices for the frame-mapping.

PPISP pipeline stages:
1. Exposure compensation (per-frame, time-sampled)
2. Vignetting correction (per-camera, static)
3. Color correction via ZCA-based homography (per-frame, time-sampled)
4. Camera Response Function (per-camera, static)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]

log = logging.getLogger(__name__)

NUM_CHANNELS = 3
COLOR_PARAMS_PER_FRAME = 8
CHANNEL_SUFFIXES = ["R", "G", "B"]
COLOR_LATENT_INPUT_NAMES = ("colorLatentBlue", "colorLatentRed", "colorLatentGreen", "colorLatentNeutral")

DEFAULT_PPISP_RESPONSIVITY = 1.0
PPISP_RESPONSIVITY_INPUT_NAME = "responsivity"
PPISP_CAMERA_SUFFIX = "_ppisp"
PPISP_ATTR_NAMESPACE = "ppisp:"
PPISP_TILE_COUNT_INPUT_NAMES = ("tileCountX", "tileCountY")
DEFAULT_PPISP_TILE_COUNT = 1

PPISP_SPG_USDA_FILE = "ppisp_usd_spg.usda"
PPISP_SPG_CU_FILE = "ppisp_usd_spg.cu"
PPISP_INPUT_RENDER_VAR = "HdrColor"
PPISP_CONTROLLER_INPUT = "ControllerParams"
PPISP_OUTPUT_RENDER_VAR = "PPISPColor"
LDR_COLOR_RENDER_VAR = "LdrColor"
PPISP_CAMERA_EXPOSURE = 0.0
PPISP_CAMERA_EXPOSURE_FSTOP = 1.0
PPISP_CAMERA_EXPOSURE_ISO = 0.0
PPISP_CAMERA_EXPOSURE_RESPONSIVITY = 1.0
PPISP_CAMERA_EXPOSURE_TIME = 1.0
PPISP_CAMERA_AUTO_EXPOSURE_ENABLED = False
PPISP_CAMERA_API_SCHEMAS = [
    "OmniRtxCameraAutoExposureAPI_1",
    "OmniRtxCameraExposureAPI_1",
]


# ---------------------------------------------------------------------------
# Dataset frame-mapping helpers
# ---------------------------------------------------------------------------


def build_camera_frame_mapping(dataset) -> Tuple[List[str], Dict[str, List[int]]]:
    """Build per-camera frame lists from a 3DGRUT dataset.

    Returns:
        (camera_names, {camera_name: [frame_idx, ...]}) where frame_idx values
        are dataset indices used to look up trained PPISP parameters.
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


def build_camera_time_mapping(dataset) -> Tuple[List[str], Dict[str, List[float]]]:
    """Build per-camera USD time-code lists keyed by global dataset frame index.

    Each camera's time-code list is the subsequence of global frame indices
    that belong to that camera.
    """
    num_frames = len(dataset)

    camera_names: List[str]
    if hasattr(dataset, "get_camera_names"):
        camera_names = dataset.get_camera_names()
    else:
        camera_names = ["camera_0"]

    camera_times: Dict[str, List[float]] = {name: [] for name in camera_names}

    for frame_idx in range(num_frames):
        if hasattr(dataset, "get_camera_idx"):
            cam_idx = dataset.get_camera_idx(frame_idx)
        else:
            cam_idx = 0
        if 0 <= cam_idx < len(camera_names):
            camera_times[camera_names[cam_idx]].append(float(frame_idx))

    return camera_names, camera_times


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


def _ensure_render_var(stage: Usd.Stage, render_product_path: str, render_var_name: str) -> str:
    """Ensure a child RenderVar exists and is declared on orderedVars."""
    render_product = stage.GetPrimAtPath(render_product_path)
    render_var_path = f"{render_product_path}/{render_var_name}"
    render_var = stage.DefinePrim(render_var_path, "RenderVar")
    render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(render_var_name)
    render_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)

    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        _append_ordered_var_target_once(ordered_vars_rel, render_product.GetPath(), Sdf.Path(render_var_path))
    return render_var_path


def _append_ordered_var_target_once(
    ordered_vars_rel: Usd.Relationship,
    render_product_path: Sdf.Path,
    target_path: Sdf.Path,
) -> None:
    """Append a RenderVar target if not already present, accepting relative authored paths."""
    target_path = target_path.MakeAbsolutePath(render_product_path)
    targets = list(ordered_vars_rel.GetTargets())
    absolute_targets = [target.MakeAbsolutePath(render_product_path) for target in targets]
    if target_path not in absolute_targets:
        targets.append(target_path)
        ordered_vars_rel.SetTargets(targets)


def _create_shader_prim(stage: Usd.Stage, render_product_path: str) -> UsdShade.Shader:
    """Create the static-parameter PPISP CUDA shader prim on a RenderProduct.

    Authors the static CUDA shader whose exposure/color parameters are USD inputs.
    """
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    # HdrColor shader input RenderVar
    _ensure_render_var(stage, render_product_path, PPISP_INPUT_RENDER_VAR)

    # PPISP Shader prim referencing the SPG asset definition
    ppisp_shader_path = f"{render_product_path}/PPISP"
    shader = UsdShade.Shader.Define(stage, ppisp_shader_path)
    shader.GetPrim().GetReferences().AddReference(PPISP_SPG_USDA_FILE)
    # Duplicate the source metadata on the instance.
    shader.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set(
        "sourceAsset"
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
        Sdf.AssetPath(PPISP_SPG_CU_FILE)
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        "ppispProcess"
    )

    # HdrColor opaque input wired to the input RenderVar's AOV
    hdr_input = shader.CreateInput(PPISP_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{PPISP_INPUT_RENDER_VAR}.omni:rtx:aov")])

    # PPISPColor opaque output
    shader.CreateOutput(PPISP_OUTPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)

    # LdrColor RenderVar connected to the PPISP output, replacing the display AOV.
    ppisp_output_path = shader.GetPath().AppendProperty(f"outputs:{PPISP_OUTPUT_RENDER_VAR}")
    ldr_var_path = _add_ldr_color_render_var(stage, render_product_path, ppisp_output_path)

    # Append LdrColor to orderedVars
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        _append_ordered_var_target_once(
            ordered_vars_rel,
            render_product.GetPath(),
            Sdf.Path(ldr_var_path),
        )

    return shader


# ---------------------------------------------------------------------------
# Static parameter setters (per-camera)
# ---------------------------------------------------------------------------


def _set_responsivity_params(shader: UsdShade.Shader, responsivity: float) -> None:
    """Author the achromatic ``inputs:responsivity`` input.

    The shader premultiplies it with the input HDR before the rest of the
    PPISP pipeline runs.
    """
    _validate_ppisp_responsivity(responsivity)
    shader.CreateInput(PPISP_RESPONSIVITY_INPUT_NAME, Sdf.ValueTypeNames.Float).Set(float(responsivity))


def _set_tile_count_params(shader: UsdShade.Shader) -> None:
    """Author untiled-default ``inputs:tileCount{X,Y}`` = 1 on the shader."""
    for name in PPISP_TILE_COUNT_INPUT_NAMES:
        shader.CreateInput(name, Sdf.ValueTypeNames.Int).Set(DEFAULT_PPISP_TILE_COUNT)


def _validate_ppisp_responsivity(responsivity: object) -> None:
    if isinstance(responsivity, bool) or not isinstance(responsivity, (int, float)):
        raise TypeError(f"responsivity must be a real number (int or float), got {type(responsivity).__name__}")
    if not math.isfinite(float(responsivity)):
        raise ValueError(f"responsivity must be finite, got {responsivity!r}")
    if float(responsivity) <= 0.0:
        raise ValueError(f"responsivity must be strictly positive, got {responsivity!r}")


def _set_vignetting_params(shader: UsdShade.Shader, ppisp: PPISP, camera_index: int) -> None:
    """Set per-camera vignetting parameters (static).

    ppisp.vignetting_params[camera_index] has shape [3, 5]:
    [cx, cy, alpha1, alpha2, alpha3] per channel.
    """
    vig = ppisp.vignetting_params[camera_index].detach().cpu().numpy()  # [3, 5]
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
    crf = ppisp.crf_params[camera_index].detach().cpu().numpy()  # [3, 4]
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
    time_codes: List[float] | None = None,
) -> None:
    """Write time-sampled exposure offset; default = mean across this camera's frames.

    ppisp.exposure_params has shape [num_frames].
    Time codes default to float(frame_idx).
    """
    exposure = ppisp.exposure_params.detach().cpu().numpy()  # [num_frames]

    if time_codes is None:
        time_codes = [float(i) for i in frame_indices]
    if len(time_codes) != len(frame_indices):
        raise ValueError("time_codes length must match frame_indices length")

    valid = [(i, t) for i, t in zip(frame_indices, time_codes) if i < len(exposure)]
    valid_indices = [i for i, _ in valid]
    mean_val = float(np.mean(exposure[valid_indices])) if valid_indices else 0.0

    exposure_input = shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float)
    attr = exposure_input.GetAttr()
    attr.Set(mean_val)

    for frame_idx, time_code in valid:
        attr.Set(float(exposure[frame_idx]), float(time_code))


def _set_static_exposure_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_index: int,
) -> None:
    """Write one fixed exposure offset without USD time samples."""
    exposure = ppisp.exposure_params.detach().cpu().numpy()
    if frame_index < 0 or frame_index >= len(exposure):
        raise ValueError(f"frame_index must be in [0, {len(exposure) - 1}], got {frame_index}.")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(float(exposure[frame_index]))


def _set_animated_color_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_indices: List[int],
    time_codes: List[float] | None = None,
) -> None:
    """Write time-sampled color latent offsets; default = mean across this camera's frames.

    ppisp.color_params has shape [num_frames, 8]:
    [db_r, db_g, dr_r, dr_g, dg_r, dg_g, dgray_r, dgray_g].
    Written as 4 float2 attributes. Time codes default to float(frame_idx).
    """
    color = ppisp.color_params.detach().cpu().numpy()  # [num_frames, 8]

    if time_codes is None:
        time_codes = [float(i) for i in frame_indices]
    if len(time_codes) != len(frame_indices):
        raise ValueError("time_codes length must match frame_indices length")

    valid = [(i, t) for i, t in zip(frame_indices, time_codes) if i < len(color)]
    valid_indices = [i for i, _ in valid]
    mean_color = np.mean(color[valid_indices], axis=0) if valid_indices else np.zeros(8)

    attrs = []
    for i, name in enumerate(COLOR_LATENT_INPUT_NAMES):
        inp = shader.CreateInput(name, Sdf.ValueTypeNames.Float2)
        attr = inp.GetAttr()
        attr.Set(Gf.Vec2f(float(mean_color[i * 2]), float(mean_color[i * 2 + 1])))
        attrs.append(attr)

    for frame_idx, time_code in valid:
        frame_color = color[frame_idx]
        for i, attr in enumerate(attrs):
            attr.Set(
                Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1])),
                float(time_code),
            )


def _set_static_color_params(
    shader: UsdShade.Shader,
    ppisp: PPISP,
    frame_index: int,
) -> None:
    """Write one fixed color latent state without USD time samples."""
    color = ppisp.color_params.detach().cpu().numpy()
    if frame_index < 0 or frame_index >= len(color):
        raise ValueError(f"frame_index must be in [0, {len(color) - 1}], got {frame_index}.")

    frame_color = color[frame_index]
    for i, name in enumerate(COLOR_LATENT_INPUT_NAMES):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1]))
        )


def _set_neutral_frame_params(shader: UsdShade.Shader) -> None:
    """Write default exposure/color state for validation or novel-view products."""
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    for name in COLOR_LATENT_INPUT_NAMES:
        shader.CreateInput(name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))


# ---------------------------------------------------------------------------
# Camera source-of-truth authoring
# ---------------------------------------------------------------------------


def _ppisp_camera_attr(camera_prim: Usd.Prim, input_name: str, value_type: Sdf.ValueTypeName) -> Usd.Attribute:
    """Create or fetch a namespaced ``ppisp:<input_name>`` custom attribute."""
    return camera_prim.CreateAttribute(f"{PPISP_ATTR_NAMESPACE}{input_name}", value_type, custom=True)


def _author_camera_vignetting(camera_prim: Usd.Prim, ppisp: PPISP, camera_index: int) -> None:
    """Author per-camera vignetting parameters as ``ppisp:*`` attributes."""
    vig = ppisp.vignetting_params[camera_index].detach().cpu().numpy()
    for ch in range(NUM_CHANNELS):
        suffix = CHANNEL_SUFFIXES[ch]
        _ppisp_camera_attr(camera_prim, f"vignettingCenter{suffix}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(vig[ch, 0]), float(vig[ch, 1]))
        )
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha1{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 2]))
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha2{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 3]))
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha3{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 4]))


def _author_camera_crf(camera_prim: Usd.Prim, ppisp: PPISP, camera_index: int) -> None:
    """Author per-camera CRF parameters as ``ppisp:*`` attributes."""
    crf = ppisp.crf_params[camera_index].detach().cpu().numpy()
    for ch in range(NUM_CHANNELS):
        suffix = CHANNEL_SUFFIXES[ch]
        _ppisp_camera_attr(camera_prim, f"crfToe{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 0]))
        _ppisp_camera_attr(camera_prim, f"crfShoulder{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 1]))
        _ppisp_camera_attr(camera_prim, f"crfGamma{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 2]))
        _ppisp_camera_attr(camera_prim, f"crfCenter{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 3]))


def _normalized_time_codes(frame_indices: List[int], time_codes: List[float] | None) -> List[float]:
    if time_codes is None:
        return [float(i) for i in frame_indices]
    if len(time_codes) != len(frame_indices):
        raise ValueError("time_codes length must match frame_indices length")
    return [float(t) for t in time_codes]


def _author_camera_animated_exposure(
    camera_prim: Usd.Prim,
    ppisp: PPISP,
    frame_indices: List[int],
    time_codes: List[float] | None,
) -> None:
    """Author time-sampled ``ppisp:exposureOffset`` with per-camera mean as default."""
    exposure = ppisp.exposure_params.detach().cpu().numpy()
    resolved_times = _normalized_time_codes(frame_indices, time_codes)
    valid = [(i, t) for i, t in zip(frame_indices, resolved_times) if i < len(exposure)]
    valid_indices = [i for i, _ in valid]
    mean_exposure = float(np.mean(exposure[valid_indices])) if valid_indices else 0.0
    attr = _ppisp_camera_attr(camera_prim, "exposureOffset", Sdf.ValueTypeNames.Float)
    attr.Set(mean_exposure)
    for frame_idx, time_code in valid:
        attr.Set(float(exposure[frame_idx]), float(time_code))


def _author_camera_animated_color(
    camera_prim: Usd.Prim,
    ppisp: PPISP,
    frame_indices: List[int],
    time_codes: List[float] | None,
) -> None:
    """Author time-sampled ``ppisp:colorLatent*`` attributes with mean defaults."""
    color = ppisp.color_params.detach().cpu().numpy()
    resolved_times = _normalized_time_codes(frame_indices, time_codes)
    valid = [(i, t) for i, t in zip(frame_indices, resolved_times) if i < len(color)]
    valid_indices = [i for i, _ in valid]
    mean_color = np.mean(color[valid_indices], axis=0) if valid_indices else np.zeros(COLOR_PARAMS_PER_FRAME)

    attrs: List[Usd.Attribute] = []
    for i, name in enumerate(COLOR_LATENT_INPUT_NAMES):
        attr = _ppisp_camera_attr(camera_prim, name, Sdf.ValueTypeNames.Float2)
        attr.Set(Gf.Vec2f(float(mean_color[i * 2]), float(mean_color[i * 2 + 1])))
        attrs.append(attr)

    for frame_idx, time_code in valid:
        frame_color = color[frame_idx]
        for i, attr in enumerate(attrs):
            attr.Set(Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1])), float(time_code))


def _author_camera_static_exposure(camera_prim: Usd.Prim, ppisp: PPISP, frame_index: int) -> None:
    """Author one fixed ``ppisp:exposureOffset`` sample."""
    exposure = ppisp.exposure_params.detach().cpu().numpy()
    if frame_index < 0 or frame_index >= len(exposure):
        raise ValueError(f"frame_index must be in [0, {len(exposure) - 1}], got {frame_index}.")
    _ppisp_camera_attr(camera_prim, "exposureOffset", Sdf.ValueTypeNames.Float).Set(float(exposure[frame_index]))


def _author_camera_static_color(camera_prim: Usd.Prim, ppisp: PPISP, frame_index: int) -> None:
    """Author the four fixed ``ppisp:colorLatent*`` attributes."""
    color = ppisp.color_params.detach().cpu().numpy()
    if frame_index < 0 or frame_index >= len(color):
        raise ValueError(f"frame_index must be in [0, {len(color) - 1}], got {frame_index}.")
    frame_color = color[frame_index]
    for i, name in enumerate(COLOR_LATENT_INPUT_NAMES):
        _ppisp_camera_attr(camera_prim, name, Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(frame_color[i * 2]), float(frame_color[i * 2 + 1]))
        )


def _author_camera_neutral_frame_params(camera_prim: Usd.Prim) -> None:
    """Author neutral exposure/color PPISP frame attributes."""
    _ppisp_camera_attr(camera_prim, "exposureOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    for name in COLOR_LATENT_INPUT_NAMES:
        _ppisp_camera_attr(camera_prim, name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))


def _author_ppisp_camera_attributes(
    stage: Usd.Stage,
    camera_path: Sdf.Path,
    ppisp: PPISP,
    camera_index: int,
    *,
    responsivity: float,
    frame_indices: List[int],
    time_codes: List[float] | None,
    fixed_frame_index: int | None,
    neutral_frame_params: bool,
) -> None:
    """Author source-of-truth ``ppisp:*`` attributes on ``<cam>_ppisp``."""
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise ValueError(f"PPISP camera prim not found at path: {camera_path}")

    _ppisp_camera_attr(camera_prim, PPISP_RESPONSIVITY_INPUT_NAME, Sdf.ValueTypeNames.Float).Set(float(responsivity))
    _author_camera_vignetting(camera_prim, ppisp, camera_index)
    _author_camera_crf(camera_prim, ppisp, camera_index)

    if neutral_frame_params:
        _author_camera_neutral_frame_params(camera_prim)
    elif fixed_frame_index is None:
        _author_camera_animated_exposure(camera_prim, ppisp, frame_indices, time_codes)
        _author_camera_animated_color(camera_prim, ppisp, frame_indices, time_codes)
    else:
        _author_camera_static_exposure(camera_prim, ppisp, fixed_frame_index)
        _author_camera_static_color(camera_prim, ppisp, fixed_frame_index)


# ---------------------------------------------------------------------------
# Per-camera entry point
# ---------------------------------------------------------------------------


def add_ppisp_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: PPISP,
    frame_indices: List[int],
    time_codes: List[float] | None = None,
    fixed_frame_index: int | None = None,
    responsivity: float = 1.0,
    neutral_frame_params: bool = False,
    ppisp_camera_path: Optional[Sdf.Path] = None,
) -> Usd.Prim:
    """Add a PPISP Shader to a RenderProduct for one physical camera.

    Per-camera parameters (vignetting, CRF) are written as static USD
    attributes. Per-frame parameters (exposure, color latents) are either:
    - written as neutral exposure/color state when ``neutral_frame_params``
      is set, for validation products that should use only per-camera PPISP
      state, or
    - written with mean-based defaults plus per-frame time samples, or
    - written from one fixed frame when ``fixed_frame_index`` is set.

    Args:
        stage: USD stage containing the RenderProduct.
        render_product_path: Path to the RenderProduct prim.
        camera_index: Index of this camera in the PPISP model.
        ppisp: Trained PPISP module.
        frame_indices: Global frame indices belonging to this camera.
        time_codes: USD time codes matching frame_indices. Defaults to frame_indices.
        fixed_frame_index: If set, write this one PPISP frame state as static
            shader inputs instead of authoring animated time samples.
        responsivity: Achromatic input HDR multiplier authored on the PPISP
            shader as a user-overridable default.
        neutral_frame_params: If True, author exposure 0 and zero color
            latents instead of per-frame/fixed PPISP exposure/color state.
        ppisp_camera_path: Optional path to the hidden ``<cam>_ppisp`` camera
            that should receive matching source-of-truth ``ppisp:*`` attrs.

    Returns:
        The created PPISP Shader prim.
    """
    assert camera_index < ppisp.num_cameras, f"camera_index {camera_index} >= ppisp.num_cameras {ppisp.num_cameras}"
    if not frame_indices and fixed_frame_index is None and not neutral_frame_params:
        log.warning(f"No frames for camera {camera_index} at {render_product_path}, skipping")
        return stage.GetPseudoRoot()

    shader = _create_shader_prim(stage, render_product_path)
    _set_tile_count_params(shader)

    if ppisp_camera_path is not None:
        _author_ppisp_camera_attributes(
            stage=stage,
            camera_path=ppisp_camera_path,
            ppisp=ppisp,
            camera_index=camera_index,
            responsivity=responsivity,
            frame_indices=frame_indices,
            time_codes=time_codes,
            fixed_frame_index=fixed_frame_index,
            neutral_frame_params=neutral_frame_params,
        )

    _set_responsivity_params(shader, responsivity)
    _set_vignetting_params(shader, ppisp, camera_index)
    _set_crf_params(shader, ppisp, camera_index)
    if neutral_frame_params:
        _set_neutral_frame_params(shader)
    elif fixed_frame_index is None:
        _set_animated_exposure_params(shader, ppisp, frame_indices, time_codes)
        _set_animated_color_params(shader, ppisp, frame_indices, time_codes)
    else:
        _set_static_exposure_params(shader, ppisp, fixed_frame_index)
        _set_static_color_params(shader, ppisp, fixed_frame_index)

    log.info(f"Added PPISP shader to {render_product_path} " f"(camera {camera_index}, {len(frame_indices)} frame(s))")
    return shader.GetPrim()


def _create_ppisp_camera(stage: Usd.Stage, render_product: Usd.Prim) -> Optional[Sdf.Path]:
    """Create a hidden neutral-exposure ``<cam>_ppisp`` camera shim as a sibling
    of the source camera, inheriting from it, and retarget the RenderProduct."""
    camera_rel = render_product.GetRelationship("camera")
    camera_targets = camera_rel.GetTargets() if camera_rel else []
    if not camera_targets:
        log.warning(
            "RenderProduct %s has no camera target; skipping PPISP camera override",
            render_product.GetPath(),
        )
        return None

    source_camera_path = camera_targets[0]
    if source_camera_path.name.endswith(PPISP_CAMERA_SUFFIX):
        return source_camera_path

    shim_parent_path = source_camera_path.GetParentPath()
    ppisp_camera_path = shim_parent_path.AppendChild(f"{source_camera_path.name}{PPISP_CAMERA_SUFFIX}")

    stage.OverridePrim(shim_parent_path)
    ppisp_camera_prim = stage.DefinePrim(ppisp_camera_path, "Camera")
    ppisp_camera_prim.SetHidden(True)
    ppisp_camera_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=PPISP_CAMERA_API_SCHEMAS))
    UsdGeom.Imageable(ppisp_camera_prim).CreateVisibilityAttr().Set("invisible")
    ppisp_camera_prim.GetInherits().AddInherit(source_camera_path)
    ppisp_camera_prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE)
    ppisp_camera_prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE_FSTOP)
    ppisp_camera_prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE_ISO)
    ppisp_camera_prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(
        PPISP_CAMERA_EXPOSURE_RESPONSIVITY
    )
    ppisp_camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(PPISP_CAMERA_EXPOSURE_TIME)
    ppisp_camera_prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(
        PPISP_CAMERA_AUTO_EXPOSURE_ENABLED
    )
    camera_rel.SetTargets([ppisp_camera_path])
    return ppisp_camera_path


# ---------------------------------------------------------------------------
# Batch export over all RenderProducts
# ---------------------------------------------------------------------------


def add_ppisp_to_all_render_products(
    stage: Usd.Stage,
    ppisp: PPISP,
    camera_names: List[str],
    camera_frame_mapping: Dict[str, List[int]],
    camera_time_mapping: Dict[str, List[float]] | None = None,
    render_scope_path: str = "/Render",
    fixed_camera_index: int | None = None,
    fixed_frame_index: int | None = None,
    use_controller: bool = False,
    responsivity: float = 1.0,
    neutral_frame_params: bool = False,
) -> List[Usd.Prim]:
    """Add PPISP shaders to every RenderProduct in the Render scope.

    Args:
        stage: USD stage with a populated /Render scope.
        ppisp: Trained PPISP module.
        camera_names: Ordered list of camera names (index = camera_idx in ppisp).
        camera_frame_mapping: ``{camera_name: [frame_idx, ...]}`` from
            :func:`build_camera_frame_mapping`.
        camera_time_mapping: ``{camera_name: [time_code, ...]}`` parallel to
            camera_frame_mapping. Defaults to frame indices as time codes.
        render_scope_path: Path to the /Render Scope (default ``/Render``).
        fixed_camera_index: If set, use this PPISP camera state for every
            RenderProduct instead of matching the RenderProduct camera.
        fixed_frame_index: If set, use this PPISP frame state as static shader
            inputs instead of authoring animated exposure/color samples.
        use_controller: If True, author a per-camera PPISP controller shader
            and wire its output into the PPISP shader, replacing the static /
            time-sampled exposure & colour inputs.
        responsivity: Achromatic input HDR multiplier authored on every PPISP
            shader as a user-overridable default.
        neutral_frame_params: If True, author exposure 0 and zero color
            latents for non-controller products instead of optimized
            per-frame/fixed exposure/color state.

    Returns:
        List of created PPISP Shader prims.
    """
    from threedgrut.export.usd.writers.camera import _make_usd_prim_name

    if use_controller:
        from threedgrut.export.usd.post_processing.ppisp_controller_writer import (
            add_ppisp_auto_shader_to_render_product,
        )
    elif fixed_frame_index is not None and fixed_camera_index is None:
        raise ValueError(
            "ppisp_reference_frame_id was set without ppisp_reference_camera_id "
            "in spg-runtime export mode. Frame-only fixing is ambiguous because "
            "vignetting and CRF live on the camera axis."
        )

    render_scope = stage.GetPrimAtPath(render_scope_path)
    if not render_scope.IsValid():
        log.warning(f"Render scope not found at {render_scope_path}, skipping PPISP export")
        return []

    camera_name_to_index = {name: idx for idx, name in enumerate(camera_names)}
    created: List[Usd.Prim] = []

    for child in render_scope.GetChildren():
        if child.GetTypeName() != "RenderProduct":
            continue

        # Reverse-lookup camera_name whose _make_usd_prim_name matches the prim name
        prim_name = child.GetName()
        camera_name = next(
            (n for n in camera_names if _make_usd_prim_name(n) == prim_name),
            None,
        )
        if camera_name is None:
            log.warning(f"RenderProduct '{prim_name}' has no matching camera name, skipping")
            continue

        camera_index = fixed_camera_index if fixed_camera_index is not None else camera_name_to_index.get(camera_name)
        if camera_index is None:
            log.warning(f"Camera '{camera_name}' not in camera_names list, skipping")
            continue
        if camera_index < 0 or camera_index >= ppisp.num_cameras:
            raise ValueError(f"fixed_camera_index must be in [0, {ppisp.num_cameras - 1}], got {camera_index}.")

        frame_indices = camera_frame_mapping.get(camera_name, [])
        time_codes = None if camera_time_mapping is None else camera_time_mapping.get(camera_name)
        ppisp_camera_path = _create_ppisp_camera(stage, child)

        if use_controller:
            controllers = getattr(ppisp, "controllers", None)
            if controllers is None or int(camera_index) >= len(controllers):
                log.warning(
                    "PPISP controllers missing for camera %s (idx=%d); falling back to "
                    "static parameters for this RenderProduct.",
                    camera_name,
                    int(camera_index),
                )
            else:
                shader_prim = add_ppisp_auto_shader_to_render_product(
                    stage=stage,
                    render_product_path=str(child.GetPath()),
                    camera_index=int(camera_index),
                    ppisp=ppisp,
                    controller=controllers[int(camera_index)],
                    responsivity=responsivity,
                    ppisp_camera_path=ppisp_camera_path,
                )
                created.append(shader_prim)
                continue

        shader_prim = add_ppisp_shader_to_render_product(
            stage=stage,
            render_product_path=str(child.GetPath()),
            camera_index=camera_index,
            ppisp=ppisp,
            frame_indices=frame_indices,
            time_codes=time_codes,
            fixed_frame_index=fixed_frame_index,
            responsivity=responsivity,
            neutral_frame_params=neutral_frame_params,
            ppisp_camera_path=ppisp_camera_path,
        )
        created.append(shader_prim)

    log.info(f"Added PPISP shaders to {len(created)} RenderProduct(s)")
    return created
