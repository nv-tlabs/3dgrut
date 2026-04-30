#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Render a PPISP-bearing USD RenderProduct chain via slangpy.

Given a USD/USDZ that was exported by ``threedgrut.export.usd.exporter``
with PPISP Omniverse-native mode (and optionally the controller), and a
folder of HDR input images, this tool walks each ``/Render/<cam>``
RenderProduct, finds its ``PPISP[+ Controller]`` Shader prims, resolves
their parameter values for every authored time sample, and dispatches
the matching ``.slang`` files via :mod:`tools.render_ppisp_spg.spg_runtime`.

For a controllerless export the per-frame exposure / colour latents are
read off the time-sampled USD attributes. With a controller, the
``priorExposure`` value is read once and the controller shader is
dispatched per frame against the supplied HDR input.

Required layout for the input HDR images:

    <hdr_dir>/<camera_name>/<frame_index>.exr|.png|.npy

Outputs are written to ``<out_dir>/<camera_name>/<frame_index>.png``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from pxr import Sdf, Usd, UsdShade

# Allow running as a script without installing the tool package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.render_ppisp_spg.spg_runtime import (  # noqa: E402
    CrfParams,
    VignetteParams,
    run_controller,
    run_ppisp_dyn,
    run_ppisp_static,
)

logger = logging.getLogger("render_ppisp_spg")


CHANNELS = ("R", "G", "B")


# ---------------------------------------------------------------------------
# USD parsing
# ---------------------------------------------------------------------------


def _resolve_attr_at_time(prim: Usd.Prim, attr_name: str, t: Usd.TimeCode):
    attr = prim.GetAttribute(attr_name)
    if attr is None or not attr.IsValid():
        return None
    return attr.Get(t)


def _read_vignetting(ppisp_prim: Usd.Prim, t: Usd.TimeCode) -> VignetteParams:
    p = VignetteParams()

    def _f(name: str, default: float) -> float:
        v = _resolve_attr_at_time(ppisp_prim, f"inputs:{name}", t)
        return float(v) if v is not None else default

    def _f2(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        v = _resolve_attr_at_time(ppisp_prim, f"inputs:{name}", t)
        if v is None:
            return default
        return (float(v[0]), float(v[1]))

    for ch in CHANNELS:
        setattr(p, f"center_{ch.lower()}", _f2(f"vignettingCenter{ch}", (0.0, 0.0)))
        setattr(p, f"alpha1_{ch.lower()}", _f(f"vignettingAlpha1{ch}", 0.0))
        setattr(p, f"alpha2_{ch.lower()}", _f(f"vignettingAlpha2{ch}", 0.0))
        setattr(p, f"alpha3_{ch.lower()}", _f(f"vignettingAlpha3{ch}", 0.0))
    return p


def _read_crf(ppisp_prim: Usd.Prim, t: Usd.TimeCode) -> CrfParams:
    c = CrfParams()

    def _f(name: str, default: float) -> float:
        v = _resolve_attr_at_time(ppisp_prim, f"inputs:{name}", t)
        return float(v) if v is not None else default

    for ch in CHANNELS:
        chl = ch.lower()
        setattr(c, f"toe_{chl}", _f(f"crfToe{ch}", getattr(c, f"toe_{chl}")))
        setattr(c, f"shoulder_{chl}", _f(f"crfShoulder{ch}", getattr(c, f"shoulder_{chl}")))
        setattr(c, f"gamma_{chl}", _f(f"crfGamma{ch}", getattr(c, f"gamma_{chl}")))
        setattr(c, f"center_{chl}", _f(f"crfCenter{ch}", getattr(c, f"center_{chl}")))
    return c


def _read_color_latents(ppisp_prim: Usd.Prim, t: Usd.TimeCode) -> List[float]:
    out: List[float] = []
    for name in ("colorLatentBlue", "colorLatentRed", "colorLatentGreen", "colorLatentNeutral"):
        v = _resolve_attr_at_time(ppisp_prim, f"inputs:{name}", t)
        if v is None:
            out.extend([0.0, 0.0])
        else:
            out.extend([float(v[0]), float(v[1])])
    return out


def _read_exposure(ppisp_prim: Usd.Prim, t: Usd.TimeCode) -> float:
    v = _resolve_attr_at_time(ppisp_prim, "inputs:exposureOffset", t)
    return float(v) if v is not None else 0.0


def _slang_asset_path(prim: Usd.Prim) -> Optional[str]:
    attr = prim.GetAttribute("info:spg:sourceAsset")
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    if val is None:
        return None
    return val.path if hasattr(val, "path") else str(val)


def _find_render_products(stage: Usd.Stage) -> List[Usd.Prim]:
    render_scope = stage.GetPrimAtPath("/Render")
    if not render_scope.IsValid():
        return []
    return [c for c in render_scope.GetChildren() if c.GetTypeName() == "RenderProduct"]


def _find_ppisp_and_controller(rp: Usd.Prim) -> Tuple[Optional[Usd.Prim], Optional[Usd.Prim]]:
    ppisp = None
    controller = None
    for child in rp.GetChildren():
        if child.GetName() == "PPISP":
            ppisp = child
        elif child.GetName().startswith("PPISPController"):
            controller = child
    return ppisp, controller


def _frame_indices_for_prim(prim: Usd.Prim) -> List[float]:
    """Union of authored time samples over the animated PPISP attributes."""
    samples: set = set()
    for attr_name in (
        "inputs:exposureOffset",
        "inputs:colorLatentBlue",
        "inputs:colorLatentRed",
        "inputs:colorLatentGreen",
        "inputs:colorLatentNeutral",
    ):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            samples.update(attr.GetTimeSamples() or [])
    return sorted(samples)


# ---------------------------------------------------------------------------
# HDR image I/O
# ---------------------------------------------------------------------------


def _load_hdr(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        img = Image.open(path).convert("RGB")
        return (np.asarray(img).astype(np.float32) / 255.0)
    if path.suffix.lower() == ".exr":
        try:
            import OpenEXR  # type: ignore[import-not-found]
            import Imath  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(f"OpenEXR/Imath required to read {path}: {e}")
        f = OpenEXR.InputFile(str(path))
        dw = f.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r, g, b = (np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
                   for c in ("R", "G", "B"))
        return np.stack([r, g, b], axis=-1)
    raise RuntimeError(f"unsupported HDR format: {path.suffix}")


def _save_png(out_path: Path, image_rgba: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgba, mode="RGBA").save(out_path)


# ---------------------------------------------------------------------------
# Per-camera execution
# ---------------------------------------------------------------------------


def _process_render_product(
    rp: Usd.Prim,
    usd_dir: Path,
    hdr_dir: Path,
    out_dir: Path,
    *,
    frames: Optional[Iterable[int]] = None,
) -> None:
    cam_name = rp.GetName()
    ppisp_prim, controller_prim = _find_ppisp_and_controller(rp)
    if ppisp_prim is None:
        logger.warning("RenderProduct %s has no PPISP shader prim, skipping", cam_name)
        return

    ppisp_slang = _slang_asset_path(ppisp_prim)
    ctrl_slang = _slang_asset_path(controller_prim) if controller_prim is not None else None
    if ppisp_slang is None:
        logger.warning("RenderProduct %s PPISP shader has no info:spg:sourceAsset", cam_name)
        return

    ppisp_slang_path = (usd_dir / ppisp_slang).resolve()
    if not ppisp_slang_path.exists():
        logger.error("PPISP slang sidecar not found at %s", ppisp_slang_path)
        return
    ctrl_slang_path = None
    if ctrl_slang is not None:
        ctrl_slang_path = (usd_dir / ctrl_slang).resolve()
        if not ctrl_slang_path.exists():
            logger.error("Controller slang sidecar not found at %s", ctrl_slang_path)
            return

    hdr_cam_dir = hdr_dir / cam_name
    if not hdr_cam_dir.exists():
        logger.warning("No HDR inputs for camera %s under %s, skipping", cam_name, hdr_dir)
        return

    sample_times = _frame_indices_for_prim(ppisp_prim)
    if not sample_times and controller_prim is not None:
        # Controller-only path: time samples are encoded in the HDR folder names.
        sample_times = sorted(
            int(p.stem) for p in hdr_cam_dir.iterdir() if p.stem.isdigit()
        )
    if frames is not None:
        sample_times = [t for t in sample_times if int(t) in set(int(f) for f in frames)]
    if not sample_times:
        logger.warning("Camera %s has no frames to render", cam_name)
        return

    logger.info("Rendering %s (%d frames%s)",
                cam_name, len(sample_times),
                " + controller" if ctrl_slang_path else "")

    for t in sample_times:
        frame_index = int(t)
        candidates = [
            hdr_cam_dir / f"{frame_index}.npy",
            hdr_cam_dir / f"{frame_index}.exr",
            hdr_cam_dir / f"{frame_index}.png",
        ]
        hdr_path = next((c for c in candidates if c.exists()), None)
        if hdr_path is None:
            logger.warning("No HDR input for %s frame %d", cam_name, frame_index)
            continue

        hdr_image = _load_hdr(hdr_path)
        timecode = Usd.TimeCode(float(t))
        vignette = _read_vignetting(ppisp_prim, timecode)
        crf = _read_crf(ppisp_prim, timecode)

        if ctrl_slang_path is not None:
            prior = _resolve_attr_at_time(controller_prim, "inputs:priorExposure", timecode) or 0.0
            weights_attr = controller_prim.GetAttribute("inputs:weights")
            weights_val = weights_attr.Get(timecode) if weights_attr and weights_attr.IsValid() else None
            if weights_val is None:
                logger.error("Controller for %s has no inputs:weights value, skipping frame", cam_name)
                continue
            # USD's VtArray-backed ndarray comes back read-only / OWNDATA=False;
            # slangpy.create_buffer rejects those, so force a writable copy.
            weights = np.array(weights_val, dtype=np.float32, copy=True)
            controller_out = run_controller(ctrl_slang_path, hdr_image, weights,
                                            prior_exposure=float(prior))
            ldr = run_ppisp_dyn(ppisp_slang_path, hdr_image, controller_out,
                                vignette=vignette, crf=crf)
        else:
            exposure = _read_exposure(ppisp_prim, timecode)
            color_latents = _read_color_latents(ppisp_prim, timecode)
            ldr = run_ppisp_static(ppisp_slang_path, hdr_image,
                                   exposure_offset=exposure,
                                   color_latents=color_latents,
                                   vignette=vignette, crf=crf)

        _save_png(out_dir / cam_name / f"{frame_index}.png", ldr)
        logger.debug("  wrote %s/%s/%d.png", out_dir, cam_name, frame_index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_usd_dir(usd_path: Path) -> Path:
    """Slang/usda asset paths are relative to the USD file. For ``.usdz`` we
    extract a temporary copy because the SPG sidecars are stored inside the
    archive and slangpy needs them on disk."""
    if usd_path.suffix.lower() != ".usdz":
        return usd_path.parent

    import tempfile
    import zipfile

    target = Path(tempfile.mkdtemp(prefix="ppisp_usdz_"))
    with zipfile.ZipFile(usd_path) as zf:
        zf.extractall(target)
    logger.info("Extracted %s → %s", usd_path, target)
    return target


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="USD or USDZ file from the PPISP exporter")
    parser.add_argument("hdr_dir", type=Path,
                        help="Directory of HDR inputs, organised as <camera>/<frame>.{npy,exr,png}")
    parser.add_argument("out_dir", type=Path,
                        help="Where to write LDR PNG outputs")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Optional list of camera (RenderProduct) names to render")
    parser.add_argument("--frames", nargs="*", type=int, default=None,
                        help="Optional list of frame indices to render")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase logging verbosity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.usd.exists():
        logger.error("USD not found: %s", args.usd)
        return 2

    usd_dir = _resolve_usd_dir(args.usd)
    if args.usd.suffix.lower() == ".usdz":
        # Find the actual default scene file inside the extracted dir.
        default_scene = next((p for p in usd_dir.glob("*.usd*") if p.suffix in (".usd", ".usda", ".usdc")),
                             None)
        if default_scene is None:
            logger.error("No top-level usd/usda/usdc inside %s", args.usd)
            return 2
        usd_path = default_scene
    else:
        usd_path = args.usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        logger.error("Failed to open USD stage at %s", usd_path)
        return 2

    products = _find_render_products(stage)
    if not products:
        logger.error("No RenderProducts found under /Render")
        return 1

    target_names = set(args.cameras) if args.cameras else None
    for rp in products:
        if target_names is not None and rp.GetName() not in target_names:
            continue
        _process_render_product(
            rp,
            usd_dir=usd_dir,
            hdr_dir=args.hdr_dir,
            out_dir=args.out_dir,
            frames=args.frames,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
