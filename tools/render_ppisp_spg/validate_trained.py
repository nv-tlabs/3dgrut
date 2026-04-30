#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation against a *trained* checkpoint.

Workflow:

1. Load the trained checkpoint (which contains the model + the trained
   PPISP module including its controllers).
2. For every val frame, run the gaussian renderer to get the pre-PPISP
   HDR image. Save it as ``hdr/<cam>/<frame>.npy``.
3. Apply PPISP in PyTorch (the same novel-view path the SPG shader will
   use, i.e. ``frame_idx=-1`` so the controller predicts the per-frame
   correction). Save it as ``reference/<cam>/<frame>.png``.
4. Author the controller-aware USD via the production exporter and ship
   the SPG sidecars to ``usd/``.
5. Run the slangpy CLI (`render_renderproduct.py`) on the USD with the
   HDR inputs from step (2) and write its outputs to ``slangpy/``.
6. Compare reference vs slangpy LDR per frame; report PSNR / max abs
   diff. Pass / fail on a configurable PSNR threshold.

This is the workflow a downstream consumer of the asset would actually
exercise: real trained PPISP, real exporter call, real slang dispatch
through the CLI.
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
    add_ppisp_to_all_render_products, build_camera_frame_mapping,
)
from threedgrut.export.usd.writers.ppisp_controller_writer import (  # noqa: E402
    get_controller_sidecars,
)
from threedgrut.export.usd.ppisp_spg import get_ppisp_spg_dyn_files  # noqa: E402
from threedgrut.render import Renderer  # noqa: E402

logger = logging.getLogger("validate_trained")


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def _save_png(path: Path, image_rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgba, mode="RGBA").save(path)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float((diff * diff).mean())
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _to_rgba8(rgb: np.ndarray) -> np.ndarray:
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = rgb.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return rgba


def _build_render_product(stage: Usd.Stage, cam_name: str, width: int, height: int) -> Usd.Prim:
    rp_path = f"/Render/{cam_name}"
    rp = stage.DefinePrim(rp_path, "RenderProduct")
    rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(int(width), int(height)))
    cam_prim = stage.DefinePrim(f"/World/Cameras/{cam_name}", "Camera")
    rp.CreateRelationship("camera").SetTargets([cam_prim.GetPath()])
    hdr = stage.DefinePrim(f"{rp_path}/HdrColor", "RenderVar")
    hdr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("HdrColor")
    hdr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path("HdrColor")])
    return rp


class _StubDataset:
    def __init__(self, frame_to_camera, names):
        self.f2c = list(frame_to_camera)
        self.names = list(names)

    def __len__(self): return len(self.f2c)

    def get_camera_names(self): return list(self.names)

    def get_camera_idx(self, i): return int(self.f2c[i])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Override the dataset path stored in the checkpoint.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Working directory (default: tmp). Outputs are kept here.")
    parser.add_argument("--psnr-threshold", type=float, default=35.0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of val frames processed.")
    parser.add_argument("--use-train", action="store_true",
                        help="Use train frames instead of val (model has overfit, "
                             "so the gaussian renderer produces non-trivial HDR even after short runs).")
    parser.add_argument("--save-hdr-png", action="store_true",
                        help="Save the HDR render as a normalised PNG for inspection.")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    work = args.out_dir or Path(tempfile.mkdtemp(prefix="ppisp_trained_"))
    work.mkdir(parents=True, exist_ok=True)
    hdr_dir = work / "hdr"
    ref_dir = work / "reference"
    usd_dir = work / "usd"
    slang_dir = work / "slangpy"

    # ------------------------------------------------------------------
    # 1. Load checkpoint via Renderer.from_checkpoint (uses val dataset).
    # ------------------------------------------------------------------
    renderer = Renderer.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        path=str(args.data_path) if args.data_path else "",
        out_dir=str(work / "_renderer_unused"),
        save_gt=False,
        computes_extra_metrics=False,
    )
    model = renderer.model
    post_processing = renderer.post_processing
    if post_processing is None or type(post_processing).__name__ != "PPISP":
        raise SystemExit("Checkpoint has no PPISP post-processing module.")
    if not getattr(post_processing.config, "use_controller", False):
        raise SystemExit("PPISP was trained without a controller; nothing to validate.")
    if args.use_train:
        # Pull the train dataloader by re-creating it (Renderer doesn't keep one).
        from threedgrut.datasets.utils import configure_dataloader_for_platform
        from threedgrut import datasets as ds
        conf_for_train = renderer.conf
        train_dataset, _ = ds.make(name=conf_for_train.dataset.type,
                                   config=conf_for_train, ray_jitter=None)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            **configure_dataloader_for_platform(
                {"num_workers": 0, "batch_size": 1, "shuffle": False, "pin_memory": True}
            ),
        )
        val_dataset = train_dataset
        val_dataloader = train_dataloader
        logger.info("Using TRAIN frames (model overfit) for validation.")
    else:
        val_dataset = renderer.dataset
        val_dataloader = renderer.dataloader

    cam_names: List[str]
    if hasattr(val_dataset, "get_camera_names"):
        cam_names = list(val_dataset.get_camera_names())
    else:
        cam_names = ["cam_0"]

    frame_to_camera: List[int] = []
    resolutions: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # 2-3. Render HDR + PyTorch reference LDR for each val frame.
    # ------------------------------------------------------------------
    from threedgrut.utils.render import apply_post_processing  # noqa: E402

    seen = 0
    for frame_idx, batch in enumerate(val_dataloader):
        if args.max_frames is not None and seen >= args.max_frames:
            break

        gpu_batch = val_dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            outputs = model(gpu_batch)
            hdr_tensor = outputs["pred_rgb"][0]  # [H, W, 3]

        # If the gaussian render is degenerate (e.g. short training, mismatched
        # camera frames), the "HDR" is all zeros and the slang/PyTorch
        # comparison becomes vacuous. Fall back to the dataset GT image so the
        # comparison still exercises the full PPISP pipeline on real-world
        # spatial / colour variation.
        if hdr_tensor.abs().max().item() < 1e-6:
            logger.warning("Gaussian render is degenerate (all zero); "
                           "substituting GT image as HDR input.")
            gt = gpu_batch.rgb_gt
            hdr_tensor = gt[0] if gt.dim() == 4 else gt

        hdr_np = hdr_tensor.detach().cpu().numpy().astype(np.float32)
        h, w = hdr_np.shape[:2]

        cam_idx = (val_dataset.get_camera_idx(frame_idx)
                   if hasattr(val_dataset, "get_camera_idx") else 0)
        cam_name = cam_names[cam_idx] if cam_idx < len(cam_names) else "cam_0"
        resolutions[cam_name] = (w, h)
        frame_to_camera.append(cam_idx)

        _save_npy(hdr_dir / cam_name / f"{frame_idx}.npy", hdr_np)

        # Reference path: same PPISP that the slang shader will execute.
        # PPISP.forward picks the controller branch when frame_idx=-1, so
        # we mirror that here for an apples-to-apples comparison.
        with torch.no_grad():
            outputs_ref = dict(outputs)
            outputs_ref["pred_rgb"] = hdr_tensor.unsqueeze(0)
            # apply_post_processing expects a batch dim.
            ref = apply_post_processing(
                post_processing, outputs_ref, gpu_batch, training=False
            )["pred_rgb"][0]
        ref_np = ref.detach().cpu().numpy().astype(np.float32)
        _save_png(ref_dir / cam_name / f"{frame_idx}.png", _to_rgba8(ref_np))
        # Save the pre-quantization float reference so we can quantify the
        # numerical drift independent of the rgba8_unorm round-trip.
        np.save(ref_dir / cam_name / f"{frame_idx}.npy", ref_np)
        seen += 1

    if not frame_to_camera:
        raise SystemExit("No validation frames found in dataset.")
    logger.info("Rendered %d val frame(s)", len(frame_to_camera))

    # ------------------------------------------------------------------
    # 4. Author the controller-aware USD + sidecars.
    # ------------------------------------------------------------------
    usd_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_dir / "scene.usda"))
    stage.SetMetadata("upAxis", UsdGeom.Tokens.y)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/Render", "Scope")
    for cam_name, (w, h) in resolutions.items():
        _build_render_product(stage, cam_name, w, h)

    dataset_stub = _StubDataset(frame_to_camera, cam_names)
    cam_names_built, mapping = build_camera_frame_mapping(dataset_stub)
    add_ppisp_to_all_render_products(
        stage=stage,
        ppisp=post_processing,
        camera_names=cam_names_built,
        camera_frame_mapping=mapping,
        use_controller=True,
    )
    stage.GetRootLayer().Save()

    for s in get_ppisp_spg_dyn_files():
        (usd_dir / s.filename).write_bytes(s.serialized)
    for s in get_controller_sidecars():
        (usd_dir / s.filename).write_bytes(s.serialized)

    # ------------------------------------------------------------------
    # 5. Run the slangpy CLI.
    # ------------------------------------------------------------------
    cli = Path(__file__).resolve().parent / "render_renderproduct.py"
    cmd = [
        sys.executable, str(cli),
        str(usd_dir / "scene.usda"), str(hdr_dir), str(slang_dir), "-vv",
    ]
    logger.info("Running slangpy CLI: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"render_renderproduct.py failed (exit {proc.returncode})")

    # ------------------------------------------------------------------
    # 5b. Probe: run the controller alone via slangpy on each saved HDR
    # and compare its 9-element output to PyTorch. This isolates whether
    # any drift in the final image originates in the controller (CNN+MLP)
    # or further downstream in the PPISP shader.
    # ------------------------------------------------------------------
    from threedgrut.export.usd.writers.ppisp_controller_writer import (
        flatten_controller_weights,
    )
    from tools.render_ppisp_spg.spg_runtime import run_controller as _run_ctrl
    print("\nController (9-float) drift, slang vs torch:")
    for frame_idx, cam_idx in enumerate(frame_to_camera):
        cam_name = cam_names[cam_idx]
        hdr_np = np.load(hdr_dir / cam_name / f"{frame_idx}.npy")
        ctrl = post_processing.controllers[cam_idx]
        # Torch reference
        rgb_t = torch.from_numpy(hdr_np).float().to("cuda")
        pe_t = torch.zeros(1, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            exposure, color = ctrl(rgb_t, pe_t)
        torch_out = np.concatenate([
            np.array([float(exposure)], dtype=np.float32),
            color.detach().cpu().numpy().astype(np.float32),
        ])
        # Slang
        weights = flatten_controller_weights(ctrl)
        slang_out = _run_ctrl(usd_dir / "ppisp_controller.slang", hdr_np, weights, prior_exposure=0.0)
        diff = np.abs(slang_out - torch_out)
        print(f"  frame={frame_idx} torch={torch_out}  slang={slang_out}  max|Δ|={diff.max():.4g}")

    # ------------------------------------------------------------------
    # 6. Compare images.
    # ------------------------------------------------------------------
    all_pass = True
    worst = (None, float("inf"))
    for frame_idx, cam_idx in enumerate(frame_to_camera):
        cam_name = cam_names[cam_idx]
        ref_path = ref_dir / cam_name / f"{frame_idx}.png"
        sl_path = slang_dir / cam_name / f"{frame_idx}.png"
        if not sl_path.exists():
            print(f"  cam={cam_name} frame={frame_idx} MISSING slangpy output")
            all_pass = False
            continue
        ref = np.asarray(Image.open(ref_path).convert("RGBA"))
        sl = np.asarray(Image.open(sl_path).convert("RGBA"))
        psnr = _psnr(ref[..., :3], sl[..., :3])
        max_abs = int(np.max(np.abs(ref[..., :3].astype(int) - sl[..., :3].astype(int))))
        ok = psnr >= args.psnr_threshold

        # Also report a float-domain diff: the slang shader writes through
        # rgba8_unorm, so its output is already quantized; we re-quantize the
        # PyTorch reference with the same rule and compare the float values
        # of the reference to that re-quantized form. This shows whether the
        # shader is matching the *post-quantization* spec exactly.
        ref_float_path = ref_dir / cam_name / f"{frame_idx}.npy"
        if ref_float_path.exists():
            ref_float = np.clip(np.load(ref_float_path), 0.0, 1.0)
            sl_float = sl[..., :3].astype(np.float32) / 255.0
            float_diff = ref_float - sl_float
            mean_abs = float(np.mean(np.abs(float_diff)))
            max_abs_f = float(np.max(np.abs(float_diff)))
            print(f"  cam={cam_name} frame={frame_idx} PSNR={psnr:7.3f} dB  "
                  f"max|Δ|_u8={max_abs}  max|Δ|_float={max_abs_f:.4f}  "
                  f"mean|Δ|_float={mean_abs:.5f}  "
                  f"{'OK' if ok else 'FAIL'}")
        else:
            print(f"  cam={cam_name} frame={frame_idx} PSNR={psnr:7.3f} dB  "
                  f"max|Δ|={max_abs}  {'OK' if ok else 'FAIL'}")
        if psnr < worst[1]:
            worst = ((cam_name, frame_idx), psnr)
        if not ok:
            all_pass = False

    print()
    print(f"worst frame: {worst[0]} @ {worst[1]:.3f} dB  (threshold {args.psnr_threshold} dB)")
    print(f"work dir: {work}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
