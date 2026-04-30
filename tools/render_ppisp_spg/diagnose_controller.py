#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triage helper for "controller-driven Omniverse render disagrees with the
optimized-params render".

Runs three independent checks in order, each isolating one of the three
hypotheses you flagged.

H1 -- *Did the controller actually learn the optimized params?*
    For a sample of frames, compare in-process
        exposure, color = controller(gaussian_render(frame), prior=0)
    against the trained per-frame parameters
        ppisp.exposure_params[frame_idx], ppisp.color_params[frame_idx].
    Pure PyTorch, no SPG/slang. If these disagree, the controller has not
    converged to the per-frame state and any SPG export will inherit that.

H2 -- *Does the slang controller match the PyTorch controller?*
    Run controller(rgb, prior) twice on the same input:
      a. PyTorch (ppisp.controllers[c]).
      b. slangpy on ppisp_controller.slang.
    These should agree to ~1e-6 (we measured 3e-7 on bonsai). A larger
    delta means the slang shader, the weight flatten, or the buffer
    upload disagrees with the trained controller.

H3 -- *Is the SPG controller -> PPISP plumbing sound?*
    Two ways to drive the dynamic PPISP shader on the same HDR:
      a. dynamic path: controller slang writes ControllerParams texture,
         ppisp_usd_spg_dyn.slang reads it.
      b. static path: feed the *same* 9 floats (taken from the PyTorch
         controller in step H1/H2) as USD attributes into the legacy
         ppisp_usd_spg.slang shader.
    These should produce byte-for-byte the same LDR image. If they
    disagree, the dynamic shader's texture binding or layout is wrong.

Usage:
    python tools/render_ppisp_spg/diagnose_controller.py \
        --checkpoint runs/<scene>/ckpt_last.pt \
        --max-frames 4
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut.export.usd.writers.ppisp_controller_writer import (  # noqa: E402
    flatten_controller_weights,
)
from threedgrut.render import Renderer  # noqa: E402
from tools.render_ppisp_spg.spg_runtime import (  # noqa: E402
    CrfParams, VignetteParams, run_controller, run_ppisp_dyn, run_ppisp_static,
)

logger = logging.getLogger("diagnose_controller")


PPISP_SPG_DIR = Path(__file__).resolve().parents[2] / "threedgrut/export/usd/ppisp_spg"
CONTROLLER_SLANG = PPISP_SPG_DIR / "ppisp_controller.slang"
PPISP_DYN_SLANG  = PPISP_SPG_DIR / "ppisp_usd_spg_dyn.slang"
PPISP_STATIC_SLANG = PPISP_SPG_DIR / "ppisp_usd_spg.slang"


def _vignette_for_camera(ppisp, camera_idx: int) -> VignetteParams:
    v = ppisp.vignetting_params[camera_idx].detach().cpu().numpy()
    p = VignetteParams()
    for ch_idx, ch in enumerate(("r", "g", "b")):
        setattr(p, f"center_{ch}", (float(v[ch_idx, 0]), float(v[ch_idx, 1])))
        setattr(p, f"alpha1_{ch}", float(v[ch_idx, 2]))
        setattr(p, f"alpha2_{ch}", float(v[ch_idx, 3]))
        setattr(p, f"alpha3_{ch}", float(v[ch_idx, 4]))
    return p


def _crf_for_camera(ppisp, camera_idx: int) -> CrfParams:
    crf = ppisp.crf_params[camera_idx].detach().cpu().numpy()
    p = CrfParams()
    for ch_idx, ch in enumerate(("r", "g", "b")):
        setattr(p, f"toe_{ch}",      float(crf[ch_idx, 0]))
        setattr(p, f"shoulder_{ch}", float(crf[ch_idx, 1]))
        setattr(p, f"gamma_{ch}",    float(crf[ch_idx, 2]))
        setattr(p, f"center_{ch}",   float(crf[ch_idx, 3]))
    return p


def _torch_controller(controller, rgb_np: np.ndarray, prior: float = 0.0) -> np.ndarray:
    rgb = torch.from_numpy(rgb_np).float().to("cuda")
    pe = torch.tensor([prior], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        e, c = controller(rgb, pe)
    return np.concatenate([
        np.array([float(e)], dtype=np.float32),
        c.detach().cpu().numpy().astype(np.float32),
    ])


def _gather_frames(renderer: Renderer, max_frames: int):
    """For each batch yield (frame_idx, camera_idx, hdr_np)."""
    out = []
    for i, batch in enumerate(renderer.dataloader):
        if i >= max_frames:
            break
        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            outputs = renderer.model(gpu_batch)
        hdr = outputs["pred_rgb"][0].detach().cpu().numpy().astype(np.float32)
        cam = (renderer.dataset.get_camera_idx(i) if hasattr(renderer.dataset, "get_camera_idx") else 0)
        out.append((i, int(cam), hdr))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--prior", type=float, default=0.0,
                        help="priorExposure value to use at inference. Match what "
                             "you pass at export time (default 0.0).")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    renderer = Renderer.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        path=str(args.data_path) if args.data_path else "",
        out_dir="/tmp/_diag_unused", save_gt=False, computes_extra_metrics=False,
    )
    pp = renderer.post_processing
    if pp is None or type(pp).__name__ != "PPISP":
        raise SystemExit("Checkpoint has no PPISP module.")
    if not pp.controllers or len(pp.controllers) == 0:
        raise SystemExit("Checkpoint PPISP has no controllers.")

    frames = _gather_frames(renderer, args.max_frames)
    if not frames:
        raise SystemExit("No frames produced by the dataloader.")

    # ------------------------------------------------------------------
    # H1: trained per-frame params vs in-process controller prediction
    # ------------------------------------------------------------------
    print("\n=== H1: PyTorch controller(rgb) vs trained per-frame params ===")
    print(f"{'frame':>5}  {'cam':>3}  {'exp_train':>10}  {'exp_pred':>10}  "
          f"{'Δexp(stops)':>12}  {'col_max|Δ|':>11}")
    h1_max_exp_diff = 0.0
    h1_max_col_diff = 0.0
    for fidx, cam, hdr in frames:
        ctrl = pp.controllers[cam]
        pred = _torch_controller(ctrl, hdr, args.prior)
        exp_pred  = float(pred[0])
        col_pred  = pred[1:]
        if fidx >= int(pp.exposure_params.shape[0]):
            print(f"  frame {fidx}: out of range for exposure_params (size {pp.exposure_params.shape[0]})")
            continue
        exp_train = float(pp.exposure_params[fidx].detach().cpu())
        col_train = pp.color_params[fidx].detach().cpu().numpy().astype(np.float32)
        d_exp = exp_pred - exp_train
        d_col = float(np.max(np.abs(col_pred - col_train)))
        h1_max_exp_diff = max(h1_max_exp_diff, abs(d_exp))
        h1_max_col_diff = max(h1_max_col_diff, d_col)
        print(f"{fidx:>5}  {cam:>3}  {exp_train:>+10.4f}  {exp_pred:>+10.4f}  "
              f"{d_exp:>+12.3f}  {d_col:>11.4f}")
    print(f"  H1 worst:  Δexposure = {h1_max_exp_diff:.3f} stops   max|Δcolor| = {h1_max_col_diff:.4f}")
    print(f"  Interpretation: if Δexposure > ~0.3 stops or Δcolor > ~0.05, the controller has")
    print(f"  not converged to the optimized per-frame state. The static-export path uses the")
    print(f"  trained values directly, so it will look 'less exposed' than the controller path.")

    # ------------------------------------------------------------------
    # H2: PyTorch controller vs slang controller
    # ------------------------------------------------------------------
    print("\n=== H2: PyTorch controller vs slang controller (same HDR) ===")
    print(f"{'frame':>5}  {'cam':>3}  {'max|Δ|':>11}")
    h2_max = 0.0
    for fidx, cam, hdr in frames:
        ctrl = pp.controllers[cam]
        torch_out = _torch_controller(ctrl, hdr, args.prior)
        weights = flatten_controller_weights(ctrl)
        slang_out = run_controller(CONTROLLER_SLANG, hdr, weights, prior_exposure=args.prior)
        d = float(np.max(np.abs(torch_out - slang_out)))
        h2_max = max(h2_max, d)
        print(f"{fidx:>5}  {cam:>3}  {d:>11.3e}")
    print(f"  H2 worst: max|Δ| = {h2_max:.3e}")
    print(f"  Interpretation: should be ~3e-7. Anything > 1e-3 means the slang shader,")
    print(f"  weight flatten, or buffer upload disagrees with the trained controller.")

    # ------------------------------------------------------------------
    # H3: dynamic shader (reads texture) vs static shader (USD attrs),
    #     both fed the same 9 floats from PyTorch.
    # ------------------------------------------------------------------
    print("\n=== H3: slang dyn (reads ControllerParams texture) vs slang static (USD attrs) ===")
    print(f"{'frame':>5}  {'cam':>3}  {'max|Δ|_u8':>11}  {'mean|Δ|_u8':>12}")
    h3_max_diff = 0
    for fidx, cam, hdr in frames:
        ctrl = pp.controllers[cam]
        ctrl_out = _torch_controller(ctrl, hdr, args.prior)  # 9-float ground truth
        vig = _vignette_for_camera(pp, cam)
        crf = _crf_for_camera(pp, cam)

        # Dynamic path: controller-output 9-float buffer fed via texture.
        ldr_dyn = run_ppisp_dyn(PPISP_DYN_SLANG, hdr, ctrl_out, vig, crf)

        # Static path: same 9 floats as USD attributes. Splits ctrl_out into
        # exposure (1) + 4x float2 colour latents in declared order.
        exposure = float(ctrl_out[0])
        color = list(ctrl_out[1:].astype(float))
        ldr_stat = run_ppisp_static(PPISP_STATIC_SLANG, hdr, exposure, color, vig, crf)

        diff = np.abs(ldr_dyn[..., :3].astype(int) - ldr_stat[..., :3].astype(int))
        max_d = int(diff.max()); mean_d = float(diff.mean())
        h3_max_diff = max(h3_max_diff, max_d)
        print(f"{fidx:>5}  {cam:>3}  {max_d:>11d}  {mean_d:>12.4f}")
    print(f"  H3 worst: max|Δ|_u8 = {h3_max_diff}")
    print(f"  Interpretation: should be 0 (or 1 from dispatch ordering). > a few means the")
    print(f"  dynamic shader's texture binding / texel layout disagrees with what the")
    print(f"  controller writes — i.e. the SPG plumbing has a bug.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    h1_bad = h1_max_exp_diff > 0.3 or h1_max_col_diff > 0.05
    h2_bad = h2_max > 1e-3
    h3_bad = h3_max_diff > 3
    verdict = []
    if h1_bad: verdict.append("H1 fails -- controller did not learn the per-frame params (training).")
    if h2_bad: verdict.append("H2 fails -- slang controller != PyTorch controller (shader / flatten).")
    if h3_bad: verdict.append("H3 fails -- dyn shader != static shader on the same 9 floats (plumbing).")
    if not verdict:
        print("  All three checks pass within thresholds. If Omniverse still disagrees")
        print("  with Python, suspect: (i) Kit applies camera exposure to HdrColor before")
        print("  the SPG dispatch, or (ii) Kit's HdrColor scale != gaussian-renderer scale,")
        print("  or (iii) priorExposure mismatch between training and the USD attribute.")
    else:
        for v in verdict:
            print(f"  - {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
