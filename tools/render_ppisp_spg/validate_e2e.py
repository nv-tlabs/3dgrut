#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PPISP SPG export + render validation.

The pipeline this script exercises:

1. Build a ``ppisp.PPISP`` module with non-trivial random weights and a
   handful of synthetic HDR frames.
2. Author a USD stage with one ``RenderProduct`` per camera, attach the
   PPISP shader chain (controller + dynamic PPISP) using the in-repo
   writer, and save the USD plus the SPG sidecars to disk.
3. Run the slangpy CLI (`render_renderproduct.py`) against the saved USD
   and the synthetic HDR frames to produce LDR PNGs through the slang
   shaders.
4. Apply the same PPISP module *in PyTorch* to the same HDR frames, save
   them as the reference LDR PNGs.
5. Compare slangpy vs PyTorch images per-frame; report PSNR / max abs
   diff. Pass / fail on a configurable PSNR threshold.

The "training" step is replaced with a perturbed PPISP module because
the validation question is "does the SPG asset reproduce the in-process
PPISP forward pass for these (camera, frame) pairs", not "is the trained
model good". A real trained checkpoint would give the same answer
because the path through both runtimes is identical.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from pxr import Gf, Sdf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut.export.usd.writers.ppisp_writer import (  # noqa: E402
    add_ppisp_to_all_render_products,
)
from threedgrut.export.usd.writers.ppisp_controller_writer import (  # noqa: E402
    get_controller_sidecars,
)
from threedgrut.export.usd.ppisp_spg import (  # noqa: E402
    get_ppisp_spg_dyn_files,
)
from ppisp import PPISP, DEFAULT_PPISP_CONFIG  # noqa: E402

logger = logging.getLogger("validate_e2e")


def _make_perturbed_ppisp(num_cameras: int, num_frames: int, seed: int) -> PPISP:
    """Build a PPISP module with non-trivial parameters for every stage."""
    torch.manual_seed(seed)
    cfg = DEFAULT_PPISP_CONFIG
    ppisp = PPISP(num_cameras=num_cameras, num_frames=num_frames, config=cfg).eval()
    with torch.no_grad():
        ppisp.exposure_params.normal_(mean=0.0, std=0.5)
        ppisp.color_params.normal_(mean=0.0, std=0.05)
        ppisp.vignetting_params.normal_(mean=0.0, std=0.02)
        # Keep CRF near identity so the comparison isn't dominated by huge
        # nonlinearities; the math is identical between paths regardless.
        ppisp.crf_params.add_(torch.randn_like(ppisp.crf_params) * 0.05)
        # Perturb every controller's weights so the per-frame override has
        # work to do during the dynamic-PPISP path.
        for controller in ppisp.controllers:
            for p in controller.parameters():
                p.normal_(mean=0.0, std=0.01)
    return ppisp


def _build_render_product(stage: Usd.Stage, cam_name: str, width: int, height: int) -> Usd.Prim:
    rp_path = f"/Render/{cam_name}"
    rp = stage.DefinePrim(rp_path, "RenderProduct")
    rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    cam_prim = stage.DefinePrim(f"/World/Cameras/{cam_name}", "Camera")
    rp.CreateRelationship("camera").SetTargets([cam_prim.GetPath()])
    hdr = stage.DefinePrim(f"{rp_path}/HdrColor", "RenderVar")
    hdr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("HdrColor")
    hdr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path("HdrColor")])
    return rp


class _SyntheticDataset:
    """Minimal stub matching what build_camera_frame_mapping reads."""

    def __init__(self, frame_to_camera: List[int], camera_names: List[str]):
        self._f2c = list(frame_to_camera)
        self._names = list(camera_names)

    def __len__(self) -> int:
        return len(self._f2c)

    def get_camera_names(self) -> List[str]:
        return list(self._names)

    def get_camera_idx(self, frame_idx: int) -> int:
        return int(self._f2c[frame_idx])


def _torch_reference_ldr(
    ppisp: PPISP, hdr_image: np.ndarray, camera_idx: int, frame_idx: int
) -> np.ndarray:
    """Apply PPISP in PyTorch with the *same* (camera, frame) state the
    slang controller path will see at runtime: the controller predicts
    exposure / color from the HDR image, while vignetting and CRF use
    the per-camera parameters."""
    h, w = hdr_image.shape[:2]
    rgb = torch.from_numpy(hdr_image).float()
    # Pixel coords like the in-process renderer: integer (x, y).
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    pixel_coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    # We want the same path as the slang shader: controller predicts the
    # frame state, PPISP applies it. PPISP.forward picks the controller
    # path when frame_idx == -1 (novel-view). Pass -1 here so the torch
    # reference exercises the controller, matching the slang path.
    ppisp_eval = ppisp.eval().to("cuda")
    rgb_cuda = rgb.to("cuda")
    pixel_coords_cuda = pixel_coords.to("cuda")
    with torch.no_grad():
        out = ppisp_eval(
            rgb_cuda,
            pixel_coords_cuda,
            resolution=(w, h),
            camera_idx=camera_idx,
            frame_idx=-1,
        )
    out = out.detach().cpu().numpy()
    # The PPISP CUDA kernel saturates internally; convert to uint8 like the
    # slang shader does (`saturate(rgb)` -> rgba8_unorm).
    ldr = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = ldr
    rgba[..., 3] = 255
    return rgba


def _save_png(path: Path, image_rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgba, mode="RGBA").save(path)


def _save_npy_hdr(path: Path, hdr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, hdr.astype(np.float32))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float((diff * diff).mean())
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _author_stage(
    out_dir: Path,
    ppisp: PPISP,
    cam_names: List[str],
    frame_to_camera: List[int],
    resolutions: Dict[str, Tuple[int, int]],
) -> Path:
    """Build and save the USD stage + ship the SPG sidecars to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_dir / "scene.usda"))
    stage.SetMetadata("upAxis", UsdGeom.Tokens.y)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/Render", "Scope")
    for cam_name, (w, h) in resolutions.items():
        _build_render_product(stage, cam_name, w, h)

    dataset = _SyntheticDataset(frame_to_camera, cam_names)
    from threedgrut.export.usd.writers.ppisp_writer import build_camera_frame_mapping
    cam_names_built, mapping = build_camera_frame_mapping(dataset)

    add_ppisp_to_all_render_products(
        stage=stage,
        ppisp=ppisp,
        camera_names=cam_names_built,
        camera_frame_mapping=mapping,
        use_controller=True,
    )
    stage.GetRootLayer().Save()

    # Sidecars: shared dyn PPISP + shared controller files.
    for s in get_ppisp_spg_dyn_files():
        (out_dir / s.filename).write_bytes(s.serialized)
    for s in get_controller_sidecars():
        (out_dir / s.filename).write_bytes(s.serialized)
    logger.info("Authored stage at %s with %d sidecars",
                out_dir, len(list(out_dir.glob("*.slang*"))))
    return out_dir / "scene.usda"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--frames-per-camera", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--psnr-threshold", type=float, default=40.0,
                        help="Per-frame minimum PSNR (dB) for pass.")
    parser.add_argument("--keep", type=Path, default=None,
                        help="Keep working dir at this path instead of a tmpdir.")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("PPISP forward requires CUDA.")

    work_dir = args.keep
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="ppisp_e2e_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    usd_dir = work_dir / "usd"
    hdr_dir = work_dir / "hdr"
    ref_dir = work_dir / "reference"
    slang_dir = work_dir / "slangpy"

    try:
        # ----------------------------------------------------------------
        # 1. Build a non-trivial PPISP and a synthetic frame plan.
        # ----------------------------------------------------------------
        cam_names = [f"cam_{i}" for i in range(args.num_cameras)]
        frame_to_camera: List[int] = []
        for cam_idx in range(args.num_cameras):
            frame_to_camera.extend([cam_idx] * args.frames_per_camera)
        num_frames = len(frame_to_camera)
        resolutions = {n: (args.width, args.height) for n in cam_names}
        ppisp = _make_perturbed_ppisp(args.num_cameras, num_frames, seed=args.seed)

        # ----------------------------------------------------------------
        # 2. Synthesise HDR inputs and the PyTorch reference LDR images.
        # ----------------------------------------------------------------
        rng = np.random.default_rng(args.seed)
        for frame_idx, cam_idx in enumerate(frame_to_camera):
            cam_name = cam_names[cam_idx]
            # Smooth HDR with a few high-frequency components so the
            # controller and the vignetting see real spatial variation.
            yy, xx = np.mgrid[0:args.height, 0:args.width].astype(np.float32)
            base = 0.4 + 0.4 * rng.random((3,), dtype=np.float32)
            hdr = (
                base[None, None, :]
                + 0.15 * np.cos((xx / args.width * 4 + frame_idx) * 2 * np.pi)[..., None]
                + 0.15 * np.sin((yy / args.height * 4 + cam_idx) * 2 * np.pi)[..., None]
            ).astype(np.float32)
            hdr += rng.normal(scale=0.02, size=hdr.shape).astype(np.float32)
            hdr = np.clip(hdr, 0.0, 1.5)
            _save_npy_hdr(hdr_dir / cam_name / f"{frame_idx}.npy", hdr)

            ref = _torch_reference_ldr(ppisp, hdr, cam_idx, frame_idx)
            _save_png(ref_dir / cam_name / f"{frame_idx}.png", ref)

        # ----------------------------------------------------------------
        # 3. Author the USD stage + sidecars on disk.
        # ----------------------------------------------------------------
        usd_path = _author_stage(
            usd_dir, ppisp, cam_names, frame_to_camera, resolutions
        )

        # ----------------------------------------------------------------
        # 4. Run the slangpy CLI against the saved USD.
        # ----------------------------------------------------------------
        cli = Path(__file__).resolve().parent / "render_renderproduct.py"
        cmd = [
            sys.executable, str(cli),
            str(usd_path), str(hdr_dir), str(slang_dir),
            "-vv",
        ]
        logger.info("Running slangpy CLI: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"render_renderproduct.py failed (exit {proc.returncode})")

        # ----------------------------------------------------------------
        # 5. Compare per-frame reference vs slangpy outputs.
        # ----------------------------------------------------------------
        worst_psnr = float("inf")
        worst_pair = None
        all_pass = True
        for frame_idx, cam_idx in enumerate(frame_to_camera):
            cam_name = cam_names[cam_idx]
            ref_img = np.asarray(Image.open(ref_dir / cam_name / f"{frame_idx}.png").convert("RGBA"))
            sl_path = slang_dir / cam_name / f"{frame_idx}.png"
            if not sl_path.exists():
                logger.error("slangpy output missing: %s", sl_path)
                all_pass = False
                continue
            sl_img = np.asarray(Image.open(sl_path).convert("RGBA"))
            psnr = _psnr(ref_img[..., :3], sl_img[..., :3])
            max_abs = int(np.max(np.abs(ref_img[..., :3].astype(int) - sl_img[..., :3].astype(int))))
            ok = psnr >= args.psnr_threshold
            print(f"  cam={cam_name} frame={frame_idx} "
                  f"PSNR={psnr:7.3f} dB  max|Δ|={max_abs}  {'OK' if ok else 'FAIL'}")
            if psnr < worst_psnr:
                worst_psnr = psnr
                worst_pair = (cam_name, frame_idx)
            if not ok:
                all_pass = False

        print()
        print(f"worst frame: {worst_pair} at {worst_psnr:.3f} dB "
              f"(threshold {args.psnr_threshold} dB)")
        return 0 if all_pass else 1

    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
