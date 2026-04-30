# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Headless slangpy runtime for the PPISP SPG shader chain.

This is *not* a full Kit SPG simulator. It executes the compute stages of
the PPISP SPG sidecars directly against a supplied HDR image so the
exported asset can be validated end-to-end without booting Omniverse.

The harness uses slangpy's low-level pipeline API: load a Slang module,
create a compute pipeline from a chosen entry point, and dispatch with
resources bound through a ``ShaderCursor`` over the root
``ShaderObject``. This matches how SPG itself binds the same shaders.

Three entry points are available:

- :func:`run_controller` — ``ppisp_controller_<cam>.slang`` →
  9-element ``[exposureOffset, blue.xy, red.xy, green.xy, neutral.xy]``.
- :func:`run_ppisp_dyn` — ``ppisp_usd_spg_dyn.slang``, takes the
  controller output texture; returns an HxWx4 uint8 LDR image.
- :func:`run_ppisp_static` — ``ppisp_usd_spg.slang`` (no controller).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import slangpy as spy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VignetteParams:
    """Per-camera vignetting parameters in shader storage order."""
    center_r: Tuple[float, float] = (0.0, 0.0)
    alpha1_r: float = 0.0
    alpha2_r: float = 0.0
    alpha3_r: float = 0.0
    center_g: Tuple[float, float] = (0.0, 0.0)
    alpha1_g: float = 0.0
    alpha2_g: float = 0.0
    alpha3_g: float = 0.0
    center_b: Tuple[float, float] = (0.0, 0.0)
    alpha1_b: float = 0.0
    alpha2_b: float = 0.0
    alpha3_b: float = 0.0


@dataclasses.dataclass
class CrfParams:
    """Per-camera per-channel toe/shoulder/gamma/center raw parameters."""
    toe_r: float = 0.013659
    shoulder_r: float = 0.013659
    gamma_r: float = 0.378165
    center_r: float = 0.0
    toe_g: float = 0.013659
    shoulder_g: float = 0.013659
    gamma_g: float = 0.378165
    center_g: float = 0.0
    toe_b: float = 0.013659
    shoulder_b: float = 0.013659
    gamma_b: float = 0.378165
    center_b: float = 0.0


# ---------------------------------------------------------------------------
# Device + pipeline helpers
# ---------------------------------------------------------------------------


def _make_device(slang_dir: Path) -> spy.Device:
    return spy.create_device(include_paths=[str(slang_dir)])


_VK_BINDING_RE = __import__("re").compile(r"\[\[vk::binding\([^\]]+\)\]\]\s*")


def _build_pipeline(device: spy.Device, slang_path: Path, entry_point_name: str):
    """Compile a Slang file and return its compute pipeline.

    The PPISP SPG shaders carry ``[[vk::binding(slot, set)]]`` annotations
    that match Kit's SPG descriptor layout. Slangpy uses its own automatic
    binding scheme, and the explicit annotations make resource binding
    silently miss (the dispatch runs but reads zeroed buffers). We strip
    the annotations *for slangpy dispatch only*; the on-disk slang file
    used by SPG keeps them.
    """
    session = device.slang_session
    src = _VK_BINDING_RE.sub("", slang_path.read_text())
    module = session.load_module_from_source(
        slang_path.stem,
        src,
        path=str(slang_path),
    )
    entry_point = module.entry_point(entry_point_name)
    program = session.link_program([module], [entry_point])
    pipeline = device.create_compute_pipeline(program)
    return pipeline, program


def _create_hdr_input_texture(device: spy.Device, hdr_image: np.ndarray) -> spy.Texture:
    if hdr_image.ndim != 3 or hdr_image.shape[2] not in (3, 4):
        raise ValueError(f"hdr_image must be HxWx3 or HxWx4, got shape {hdr_image.shape}")
    if hdr_image.dtype != np.float32:
        hdr_image = hdr_image.astype(np.float32, copy=False)
    h, w, c = hdr_image.shape
    if c == 3:
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[..., :3] = hdr_image
        rgba[..., 3] = 1.0
        hdr_image = rgba
    return device.create_texture(
        width=w,
        height=h,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=np.ascontiguousarray(hdr_image),
    )


def _create_controller_texture(device: spy.Device, values: np.ndarray) -> spy.Texture:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size != 9:
        raise ValueError(f"controller values must be 9 floats, got {flat.size}")
    # 9x1 single-channel float texture, indexed at (0..8, 0).
    return device.create_texture(
        width=9,
        height=1,
        format=spy.Format.r32_float,
        usage=spy.TextureUsage.shader_resource,
        data=np.ascontiguousarray(flat.reshape(1, 9)),
    )


def _read_r32f_row(tex: spy.Texture) -> np.ndarray:
    """Read back a 1-row r32_float texture as a flat float32 numpy array."""
    arr = tex.to_numpy()
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _read_rgba8(tex: spy.Texture, h: int, w: int) -> np.ndarray:
    arr = tex.to_numpy()
    return np.asarray(arr, dtype=np.uint8).reshape(h, w, 4)


# ---------------------------------------------------------------------------
# Cursor binding helpers
# ---------------------------------------------------------------------------


def _set_param_block(cursor: spy.ShaderCursor, block_name: str, fields: dict) -> None:
    """Populate a slang ParameterBlock<T> by field name. The cursor we get
    from the root object is itself name-addressable, so ``cursor[name]``
    walks into the parameter block automatically."""
    block = cursor[block_name]
    for k, v in fields.items():
        block[k] = v


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Controller dispatch
# ---------------------------------------------------------------------------


def run_controller(
    slang_path: str | Path,
    hdr_image: np.ndarray,
    weights: np.ndarray,
    prior_exposure: float = 0.0,
    *,
    device: spy.Device | None = None,
) -> np.ndarray:
    """Dispatch the PPISP controller shader and return its 9 outputs.

    ``weights`` must be a flat float32 buffer matching the layout encoded
    in ``ppisp_controller.slang`` (see
    :data:`threedgrut.export.usd.writers.ppisp_controller_writer.EXPECTED_WEIGHTS_LEN`).
    """
    slang_path = Path(slang_path)
    if device is None:
        device = _make_device(slang_path.parent)

    pipeline, _ = _build_pipeline(device, slang_path, "controllerProcess")
    in_tex = _create_hdr_input_texture(device, hdr_image)
    out_tex = device.create_texture(
        width=9,
        height=1,
        format=spy.Format.r32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    flat_weights = np.ascontiguousarray(weights.astype(np.float32, copy=False).reshape(-1))
    weights_buf = device.create_buffer(
        element_count=int(flat_weights.size),
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=flat_weights,
    )

    encoder = device.create_command_encoder()
    with encoder.begin_compute_pass() as cp:
        shader_obj = cp.bind_pipeline(pipeline)
        cur = spy.ShaderCursor(shader_obj)
        _set_param_block(cur, "g_Params", {"priorExposure": float(prior_exposure)})
        cur["g_InTex"] = in_tex
        cur["weights"] = weights_buf
        cur["g_OutTex"] = out_tex
        cp.dispatch(spy.math.uint3(32, 1, 1))
    device.submit_command_buffer(encoder.finish())
    device.wait()

    return _read_r32f_row(out_tex)[:9]


# ---------------------------------------------------------------------------
# PPISP dispatches
# ---------------------------------------------------------------------------


def _vignette_dict(v: VignetteParams) -> dict:
    return {
        "vignettingCenterR": list(v.center_r),
        "vignettingAlpha1R": v.alpha1_r,
        "vignettingAlpha2R": v.alpha2_r,
        "vignettingAlpha3R": v.alpha3_r,
        "vignettingCenterG": list(v.center_g),
        "vignettingAlpha1G": v.alpha1_g,
        "vignettingAlpha2G": v.alpha2_g,
        "vignettingAlpha3G": v.alpha3_g,
        "vignettingCenterB": list(v.center_b),
        "vignettingAlpha1B": v.alpha1_b,
        "vignettingAlpha2B": v.alpha2_b,
        "vignettingAlpha3B": v.alpha3_b,
    }


def _crf_dict(c: CrfParams) -> dict:
    return {
        "crfToeR": c.toe_r,
        "crfShoulderR": c.shoulder_r,
        "crfGammaR": c.gamma_r,
        "crfCenterR": c.center_r,
        "crfToeG": c.toe_g,
        "crfShoulderG": c.shoulder_g,
        "crfGammaG": c.gamma_g,
        "crfCenterG": c.center_g,
        "crfToeB": c.toe_b,
        "crfShoulderB": c.shoulder_b,
        "crfGammaB": c.gamma_b,
        "crfCenterB": c.center_b,
    }


def run_ppisp_dyn(
    slang_path: str | Path,
    hdr_image: np.ndarray,
    controller_output: np.ndarray,
    vignette: VignetteParams,
    crf: CrfParams,
    *,
    device: spy.Device | None = None,
) -> np.ndarray:
    """Run ``ppisp_usd_spg_dyn.slang`` and return an HxWx4 uint8 LDR image."""
    slang_path = Path(slang_path)
    if device is None:
        device = _make_device(slang_path.parent)

    pipeline, _ = _build_pipeline(device, slang_path, "ppispProcessDyn")
    h, w = hdr_image.shape[:2]

    in_tex = _create_hdr_input_texture(device, hdr_image)
    ctrl_tex = _create_controller_texture(device, controller_output)
    out_tex = device.create_texture(
        width=w,
        height=h,
        format=spy.Format.rgba8_unorm,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )

    encoder = device.create_command_encoder()
    with encoder.begin_compute_pass() as cp:
        shader_obj = cp.bind_pipeline(pipeline)
        cur = spy.ShaderCursor(shader_obj)
        _set_param_block(cur, "g_Params",
                         {**_vignette_dict(vignette), **_crf_dict(crf)})
        cur["g_InTex"] = in_tex
        cur["g_ControllerOut"] = ctrl_tex
        cur["g_OutTex"] = out_tex
        cp.dispatch(spy.math.uint3(_ceildiv(w, 16) * 16,
                                   _ceildiv(h, 16) * 16, 1))
    device.submit_command_buffer(encoder.finish())
    device.wait()

    return _read_rgba8(out_tex, h, w)


def run_ppisp_static(
    slang_path: str | Path,
    hdr_image: np.ndarray,
    exposure_offset: float,
    color_latents: Sequence[float],
    vignette: VignetteParams,
    crf: CrfParams,
    *,
    device: spy.Device | None = None,
) -> np.ndarray:
    """Run ``ppisp_usd_spg.slang`` (no controller) and return an LDR uint8 image."""
    slang_path = Path(slang_path)
    if len(color_latents) != 8:
        raise ValueError(f"color_latents must have 8 entries, got {len(color_latents)}")
    if device is None:
        device = _make_device(slang_path.parent)

    pipeline, _ = _build_pipeline(device, slang_path, "ppispProcess")

    h, w = hdr_image.shape[:2]
    in_tex = _create_hdr_input_texture(device, hdr_image)
    out_tex = device.create_texture(
        width=w,
        height=h,
        format=spy.Format.rgba8_unorm,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )

    fields = {
        "exposureOffset": float(exposure_offset),
        "colorLatentBlue":    [float(color_latents[0]), float(color_latents[1])],
        "colorLatentRed":     [float(color_latents[2]), float(color_latents[3])],
        "colorLatentGreen":   [float(color_latents[4]), float(color_latents[5])],
        "colorLatentNeutral": [float(color_latents[6]), float(color_latents[7])],
        **_vignette_dict(vignette),
        **_crf_dict(crf),
    }
    encoder = device.create_command_encoder()
    with encoder.begin_compute_pass() as cp:
        shader_obj = cp.bind_pipeline(pipeline)
        cur = spy.ShaderCursor(shader_obj)
        _set_param_block(cur, "g_Params", fields)
        cur["g_InTex"] = in_tex
        cur["g_OutTex"] = out_tex
        cp.dispatch(spy.math.uint3(_ceildiv(w, 16) * 16,
                                   _ceildiv(h, 16) * 16, 1))
    device.submit_command_buffer(encoder.finish())
    device.wait()

    return _read_rgba8(out_tex, h, w)
