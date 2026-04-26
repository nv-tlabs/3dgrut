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

"""Omniverse USD post-processing fallback writer for PPISP exports.

This writer is a degraded fallback for Kit versions where SPG is unavailable or
unreliable. It authors Omniverse USD render settings only; exact PPISP export
remains the SPG path.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd

from threedgrut.export.usd.writers.camera import _make_usd_prim_name

log = logging.getLogger(__name__)

MODE_PPISP_OMNI_FALLBACK_NONE = "none"
MODE_PPISP_OMNI_FALLBACK_EXPOSURE = "ppisp-exposure-fallback"
MODE_PPISP_OMNI_FALLBACK_FITTED_POST_PROCESSING = "ppisp-fitted-post-processing-fallback"
MODE_PPISP_OMNI_FALLBACK_SPG_PLUS_FITTED_POST_PROCESSING = "ppisp-spg-plus-fitted-post-processing-fallback"

PPISP_OMNI_POST_PROCESSING_FALLBACK_MODES = {
    MODE_PPISP_OMNI_FALLBACK_NONE,
    MODE_PPISP_OMNI_FALLBACK_EXPOSURE,
    MODE_PPISP_OMNI_FALLBACK_FITTED_POST_PROCESSING,
    MODE_PPISP_OMNI_FALLBACK_SPG_PLUS_FITTED_POST_PROCESSING,
}

_BASE_EXPOSURE_TIME_SECONDS = 0.02
_DEFAULT_EXPOSURE_FSTOP = 5.0
_DEFAULT_EXPOSURE_ISO = 100.0
_DEFAULT_EXPOSURE_RESPONSIVITY = 1.10267091
_DEFAULT_RENDER_RESOLUTION = (1280, 720)

_CAMERA_EXPOSURE_APIS = ["OmniRtxCameraExposureAPI_1"]
_RENDER_PRODUCT_APIS = [
    "OmniRtxPostTonemapIrayReinhardAPI_1",
    "OmniRtxPostColorGradingAPI_1",
    "OmniRtxPostTvNoiseAPI_1",
]

_ZCA_BLUE = np.array([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]], dtype=np.float64)
_ZCA_RED = np.array([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]], dtype=np.float64)
_ZCA_GREEN = np.array([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]], dtype=np.float64)
_ZCA_NEUTRAL = np.array([[0.0128369, -0.0034654], [-0.0034654, 0.0128158]], dtype=np.float64)


def normalize_ov_post_processing_mode(mode: str | None) -> str:
    """Normalize and validate the ``export_usd.ov-post-processing`` value."""
    normalized = MODE_PPISP_OMNI_FALLBACK_NONE if mode is None else str(mode).strip().lower()
    if normalized not in PPISP_OMNI_POST_PROCESSING_FALLBACK_MODES:
        raise ValueError(
            f"Unsupported ov-post-processing mode '{mode}'. "
            f"Expected one of: {sorted(PPISP_OMNI_POST_PROCESSING_FALLBACK_MODES)}"
        )
    return normalized


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _prepend_api_schemas(prim: Usd.Prim, schemas: Iterable[str]) -> None:
    """Apply schemas by authoring the same listOp shape used by Kit examples."""
    schemas = [schema for schema in schemas if schema]
    if not schemas:
        return
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=schemas))


def _create_float_attr(prim: Usd.Prim, name: str, value: float):
    attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
    attr.Set(float(value))
    return attr


def _create_bool_attr(prim: Usd.Prim, name: str, value: bool):
    attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool)
    attr.Set(bool(value))
    return attr


def _create_color_attr(prim: Usd.Prim, name: str, value) -> Usd.Attribute:
    vec = Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
    attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Color3f)
    attr.Set(vec)
    return attr


def _compute_homography(color_latent: np.ndarray) -> np.ndarray:
    """Compute PPISP's RGI homography from one 8-float color latent vector."""
    b_lat = color_latent[0:2]
    r_lat = color_latent[2:4]
    g_lat = color_latent[4:6]
    n_lat = color_latent[6:8]

    bd = _ZCA_BLUE @ b_lat
    rd = _ZCA_RED @ r_lat
    gd = _ZCA_GREEN @ g_lat
    nd = _ZCA_NEUTRAL @ n_lat

    t_blue = np.array([bd[0], bd[1], 1.0], dtype=np.float64)
    t_red = np.array([1.0 + rd[0], rd[1], 1.0], dtype=np.float64)
    t_green = np.array([gd[0], 1.0 + gd[1], 1.0], dtype=np.float64)
    t_gray = np.array([1.0 / 3.0 + nd[0], 1.0 / 3.0 + nd[1], 1.0], dtype=np.float64)

    target = np.stack([t_blue, t_red, t_green], axis=1)
    skew = np.array(
        [
            [0.0, -t_gray[2], t_gray[1]],
            [t_gray[2], 0.0, -t_gray[0]],
            [-t_gray[1], t_gray[0], 0.0],
        ],
        dtype=np.float64,
    )
    matrix = skew @ target
    lam = np.cross(matrix[0], matrix[1])
    if np.dot(lam, lam) < 1.0e-20:
        lam = np.cross(matrix[0], matrix[2])
        if np.dot(lam, lam) < 1.0e-20:
            lam = np.cross(matrix[1], matrix[2])

    source_inv = np.array([[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    homography = target @ np.diag(lam) @ source_inv
    if abs(homography[2, 2]) > 1.0e-20:
        homography = homography / homography[2, 2]
    return homography


def _apply_color_homography(rgb: np.ndarray, homography: np.ndarray) -> np.ndarray:
    intensity = np.sum(rgb, axis=1)
    rgi = np.stack([rgb[:, 0], rgb[:, 1], intensity], axis=1)
    corrected = rgi @ homography.T
    corrected = corrected * (intensity / (corrected[:, 2] + 1.0e-5))[:, None]
    return np.stack([corrected[:, 0], corrected[:, 1], corrected[:, 2] - corrected[:, 0] - corrected[:, 1]], axis=1)


def _fit_grade_gain(color_latent: np.ndarray) -> np.ndarray:
    """Fit USD color grade gain to PPISP's cross-channel homography."""
    homography = _compute_homography(color_latent)
    values = np.linspace(0.05, 1.0, 5, dtype=np.float64)
    rgb = np.array(np.meshgrid(values, values, values), dtype=np.float64).T.reshape(-1, 3)
    target = np.clip(_apply_color_homography(rgb, homography), 0.0, 4.0)
    denom = np.maximum(np.sum(rgb * rgb, axis=0), 1.0e-8)
    gain = np.sum(target * rgb, axis=0) / denom
    return np.clip(gain, 0.25, 4.0)


def _bounded_softplus(raw: np.ndarray, min_value: float) -> np.ndarray:
    return min_value + np.log1p(np.exp(raw))


def _sigmoid(raw: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-raw))


def _apply_crf(x: np.ndarray, raw_params: np.ndarray) -> np.ndarray:
    toe = _bounded_softplus(raw_params[0], 0.3)
    shoulder = _bounded_softplus(raw_params[1], 0.3)
    gamma = _bounded_softplus(raw_params[2], 0.1)
    center = _sigmoid(raw_params[3])
    lerp_val = (shoulder - toe) * center + toe
    a = (shoulder * center) / lerp_val
    b = 1.0 - a
    y = np.where(
        x <= center,
        a * np.power(x / center, toe),
        1.0 - b * np.power((1.0 - x) / (1.0 - center), shoulder),
    )
    return np.power(np.maximum(0.0, y), gamma)


def _fit_grade_gamma(crf_params: np.ndarray) -> np.ndarray:
    """Fit USD grade gamma to PPISP's per-channel CRF."""
    x = np.linspace(0.02, 0.98, 96, dtype=np.float64)
    candidates = np.linspace(0.25, 4.0, 128, dtype=np.float64)
    result = []
    for channel in range(3):
        target = _apply_crf(x, crf_params[channel])
        errors = [np.mean((np.power(x, 1.0 / gamma) - target) ** 2) for gamma in candidates]
        result.append(float(candidates[int(np.argmin(errors))]))
    return np.asarray(result, dtype=np.float64)


def _ppisp_vignette_luminance(vig_params: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    sample_w = 48
    sample_h = max(8, int(round(sample_w * max(height, 1) / max(width, 1))))
    xs = (np.arange(sample_w, dtype=np.float64) + 0.5 - sample_w * 0.5) / sample_w
    ys = (np.arange(sample_h, dtype=np.float64) + 0.5 - sample_h * 0.5) / sample_w
    grid_x, grid_y = np.meshgrid(xs, ys)
    uv = np.stack([grid_x, grid_y], axis=-1)

    rgb_falloff = []
    for channel in range(3):
        center = vig_params[channel, 0:2]
        delta = uv - center
        r2 = np.sum(delta * delta, axis=-1)
        falloff = 1.0 + vig_params[channel, 2] * r2 + vig_params[channel, 3] * r2**2 + vig_params[channel, 4] * r2**3
        rgb_falloff.append(np.clip(falloff, 0.0, 1.0))
    rgb_falloff = np.stack(rgb_falloff, axis=-1)
    luminance = np.dot(rgb_falloff, np.array([0.2126, 0.7152, 0.0722], dtype=np.float64))

    org_u = (np.arange(sample_w, dtype=np.float64) + 0.5) / sample_w
    org_v = (np.arange(sample_h, dtype=np.float64) + 0.5) / sample_h
    org_x, org_y = np.meshgrid(org_u, org_v)
    org_uv = np.stack([org_x, org_y], axis=-1)
    return luminance, org_uv


def _fit_tv_vignette(vig_params: np.ndarray, width: int, height: int) -> Tuple[bool, float, float]:
    target, org_uv = _ppisp_vignette_luminance(vig_params, width, height)
    if float(np.max(np.abs(target - 1.0))) < 1.0e-3:
        return False, 107.0, 0.7

    uv2 = org_uv * (1.0 - org_uv)
    base = uv2[..., 0] * uv2[..., 1]
    best_error = float("inf")
    best_size = 107.0
    best_strength = 0.7

    for size in np.linspace(1.0, 180.0, 72):
        raw = np.maximum(base * (size + 14.0), 1.0e-8)
        for strength in np.linspace(0.2, 2.0, 73):
            candidate = np.power(raw, strength)
            error = float(np.mean((candidate - target) ** 2))
            if error < best_error:
                best_error = error
                best_size = float(size)
                best_strength = float(strength)

    return True, best_size, best_strength


def _author_camera_exposure(
    stage: Usd.Stage,
    camera_path: str,
    frame_indices: List[int],
    exposure_params: np.ndarray,
) -> None:
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        log.warning("Cannot author Omniverse fallback exposure: missing camera prim %s", camera_path)
        return

    _prepend_api_schemas(camera_prim, _CAMERA_EXPOSURE_APIS)
    _create_float_attr(camera_prim, "exposure:fStop", _DEFAULT_EXPOSURE_FSTOP)
    _create_float_attr(camera_prim, "exposure:iso", _DEFAULT_EXPOSURE_ISO)
    _create_float_attr(camera_prim, "exposure:responsivity", _DEFAULT_EXPOSURE_RESPONSIVITY)

    valid = [frame_idx for frame_idx in frame_indices if frame_idx < len(exposure_params)]
    exposure_values = np.exp2(exposure_params[valid]) * _BASE_EXPOSURE_TIME_SECONDS if valid else np.asarray([])
    default_value = float(np.mean(exposure_values)) if len(exposure_values) else _BASE_EXPOSURE_TIME_SECONDS

    exposure_time = camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float)
    exposure_time.Set(default_value)
    for frame_idx, value in zip(valid, exposure_values):
        exposure_time.Set(float(value), float(frame_idx))


def _author_tv_vignette(render_product: Usd.Prim, vig_params: np.ndarray, width: int, height: int) -> None:
    enabled, size, strength = _fit_tv_vignette(vig_params, width, height)
    _create_bool_attr(render_product, "omni:rtx:post:tvNoise:enabled", enabled)
    _create_bool_attr(render_product, "omni:rtx:post:tvNoise:vignetting:enabled", enabled)
    _create_float_attr(render_product, "omni:rtx:post:tvNoise:vignetting:size", size)
    _create_float_attr(render_product, "omni:rtx:post:tvNoise:vignetting:strength", strength)

    for attr_name in (
        "omni:rtx:post:tvNoise:filmGrain:enabled",
        "omni:rtx:post:tvNoise:ghostFlickering:enabled",
        "omni:rtx:post:tvNoise:randomSplotches:enabled",
        "omni:rtx:post:tvNoise:scanlines:enabled",
        "omni:rtx:post:tvNoise:scrollBug:enabled",
        "omni:rtx:post:tvNoise:verticalLines:enabled",
        "omni:rtx:post:tvNoise:vignetting:flickering:enabled",
        "omni:rtx:post:tvNoise:waveDistortion:enabled",
    ):
        _create_bool_attr(render_product, attr_name, False)


def _author_color_grade(
    render_product: Usd.Prim,
    frame_indices: List[int],
    color_params: np.ndarray,
    crf_params: np.ndarray,
) -> None:
    valid = [frame_idx for frame_idx in frame_indices if frame_idx < len(color_params)]
    gains = np.stack([_fit_grade_gain(color_params[frame_idx]) for frame_idx in valid], axis=0) if valid else np.ones((0, 3))
    default_gain = np.mean(gains, axis=0) if len(gains) else np.ones(3, dtype=np.float64)
    gamma = _fit_grade_gamma(crf_params)

    grade_enabled = bool(np.max(np.abs(default_gain - 1.0)) > 1.0e-3 or np.max(np.abs(gamma - 1.0)) > 1.0e-3)
    _create_bool_attr(render_product, "omni:rtx:post:grade:enabled", grade_enabled)
    gain_attr = _create_color_attr(render_product, "omni:rtx:post:grade:gain", default_gain)
    _create_color_attr(render_product, "omni:rtx:post:grade:gamma", gamma)
    _create_color_attr(render_product, "omni:rtx:post:grade:offset", (0.0, 0.0, 0.0))
    _create_color_attr(render_product, "omni:rtx:post:grade:contrast", (1.0, 1.0, 1.0))
    _create_color_attr(render_product, "omni:rtx:post:grade:saturation", (1.0, 1.0, 1.0))

    for frame_idx, gain in zip(valid, gains):
        gain_attr.Set(Gf.Vec3f(float(gain[0]), float(gain[1]), float(gain[2])), float(frame_idx))


def add_ov_post_processing(
    stage: Usd.Stage,
    camera_names: List[str],
    camera_prim_paths: Dict[str, str],
    camera_frame_mapping: Dict[str, List[int]],
    render_product_entries: Dict[str, Tuple[str, int, int]],
    post_processing,
    mode: str,
    render_scope_path: str = "/Render",
) -> None:
    """Author Omniverse USD post-processing settings for PPISP fallback export."""
    normalized_mode = normalize_ov_post_processing_mode(mode)
    if normalized_mode == MODE_PPISP_OMNI_FALLBACK_NONE:
        return

    exposure_params = _as_numpy(post_processing.exposure_params)
    color_params = _as_numpy(post_processing.color_params)
    vignetting_params = _as_numpy(post_processing.vignetting_params)
    crf_params = _as_numpy(post_processing.crf_params)

    camera_name_to_index = {name: idx for idx, name in enumerate(camera_names)}
    writes_fitted_post_processing = normalized_mode in {
        MODE_PPISP_OMNI_FALLBACK_FITTED_POST_PROCESSING,
        MODE_PPISP_OMNI_FALLBACK_SPG_PLUS_FITTED_POST_PROCESSING,
    }

    for camera_name in camera_names:
        frame_indices = camera_frame_mapping.get(camera_name, [])
        camera_path = camera_prim_paths.get(camera_name)
        if camera_path is None:
            log.warning("Skipping Omniverse post-processing fallback for %s: missing camera prim", camera_name)
            continue

        _author_camera_exposure(stage, camera_path, frame_indices, exposure_params)

        if not writes_fitted_post_processing:
            continue

        camera_index = camera_name_to_index[camera_name]
        render_product_name = _make_usd_prim_name(camera_name)
        render_product_path = f"{render_scope_path}/{render_product_name}"
        render_product = stage.GetPrimAtPath(render_product_path)
        if not render_product.IsValid():
            log.warning("Skipping Omniverse post-processing fallback for %s: missing RenderProduct", camera_name)
            continue

        _prepend_api_schemas(render_product, _RENDER_PRODUCT_APIS)
        _, width, height = render_product_entries.get(camera_name, ("", *_DEFAULT_RENDER_RESOLUTION))
        width = width or _DEFAULT_RENDER_RESOLUTION[0]
        height = height or _DEFAULT_RENDER_RESOLUTION[1]

        _author_tv_vignette(render_product, vignetting_params[camera_index], width, height)
        _author_color_grade(render_product, frame_indices, color_params, crf_params[camera_index])

    log.warning(
        "Authored Omniverse USD post-processing PPISP fallback mode '%s'. "
        "This is approximate and not SPG-fidelity.",
        normalized_mode,
    )
