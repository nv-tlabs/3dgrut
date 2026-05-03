#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep PPISP SH-bake modes on a trained checkpoint and report aggregated metrics.

The bake fits gamma-space (display-referred) SH coefficients against the
PPISP forward output of the trained model, matching the colour space of
the no-PPISP export. Modes vary along two axes:

* ``simple`` flavours skip optimisation and write only the DC band.
* ``fit`` flavours run :func:`bake_post_processing_into_sh` -- Adam over
  features_albedo, features_specular, and (optionally) density. View
  sampling is either ``training`` (iterate the dataloader) or
  ``trajectory`` (NN+2-opt arc-length-parameterised slerp through training
  poses; useful when training views are sparse).

Per-frame validation:
  reference = full PPISP applied to reference-model render at val pose
  baked     = baked-model render (already display-referred) clipped to [0, 1]

Metrics: per-frame PSNR (+ optional SSIM / LPIPS), aggregated mean /
median / min / max across the val split. Raw per-frame numbers are
persisted to ``<out_dir>/metrics.json``.

Usage:

    python tools/ppisp_export/bake_modes_benchmark/benchmark.py \\
        --checkpoint runs/<scene>/ckpt_last.pt \\
        --out-dir /tmp/bake_modes \\
        --camera-id 0 --frame-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut.export.usd.post_processing_sh_bake import (  # noqa: E402
    MODE_PPISP_BAKE_VIGNETTING_NONE,
    PPISPPostProcessingBakeAdapter,
    bake_post_processing_into_sh,
    FixedPPISP,
)
from threedgrut.export.usd.post_processing_sh_simple_bake import simple_bake  # noqa: E402
from threedgrut.render import Renderer  # noqa: E402
from threedgrut.utils.render import apply_post_processing  # noqa: E402

logger = logging.getLogger("bake_modes_benchmark")


# ---------------------------------------------------------------------------
# Mode catalogue
# ---------------------------------------------------------------------------


@dataclass
class BakeMode:
    """One row in the sweep -- a bake configuration with a short name."""
    name: str
    description: str
    builder: Callable[..., nn.Module]


def _build_simple(*, model, ppisp, camera_id, frame_id, higher_order,
                  dataset=None, conf=None):
    del dataset, conf  # unused by the simple flavours
    baked = model.clone().eval()
    simple_bake(
        baked, ppisp,
        camera_id=camera_id, frame_id=frame_id,
        higher_order=higher_order, apply_srgb_to_linear=False,
    )
    baked.build_acc()
    return baked


def _build_fit(*, model, ppisp, dataset, conf, camera_id, frame_id,
               view_mode, view_seed, epochs, learning_rate, optimize_density: bool):
    """Run the full fit-by-bake flow with the production adapter (gamma SH,
    no vignetting). ``optimize_density=False`` ablates the density param
    group by setting its lr to zero."""
    adapter = PPISPPostProcessingBakeAdapter(
        camera_id=camera_id, frame_id=frame_id,
        vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_NONE,
    )
    return bake_post_processing_into_sh(
        model=model, post_processing=ppisp, train_dataset=dataset, conf=conf,
        adapter=adapter, epochs=epochs, learning_rate=learning_rate,
        learning_rate_density=(5.0e-2 if optimize_density else 0.0),
        view_sampling_mode=view_mode, interpolated_views_seed=view_seed,
    )


def all_modes(*, fit_epochs: int, fit_lr: float, view_seed: int) -> List[BakeMode]:
    return [
        BakeMode(
            "simple",
            "one-shot DC-only bake (no fit, gamma SH)",
            lambda **k: _build_simple(**k, higher_order=False),
        ),
        BakeMode(
            "simple-higher-order",
            "one-shot DC + Jacobian-rotated specular (no fit)",
            lambda **k: _build_simple(**k, higher_order=True),
        ),
        BakeMode(
            "fit-color-only",
            "Adam fit on features_albedo + features_specular only, training views",
            lambda **k: _build_fit(
                **k, view_mode="training", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, optimize_density=False,
            ),
        ),
        BakeMode(
            "fit",
            "Adam fit on albedo + specular + density, training views (production default)",
            lambda **k: _build_fit(
                **k, view_mode="training", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, optimize_density=True,
            ),
        ),
        BakeMode(
            "fit-trajectory",
            "Adam fit on albedo + specular + density, trajectory views (NN+2-opt slerp)",
            lambda **k: _build_fit(
                **k, view_mode="trajectory", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, optimize_density=True,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------


@dataclass
class FrameMetrics:
    psnr: List[float] = field(default_factory=list)
    ssim: List[float] = field(default_factory=list)
    lpips: List[float] = field(default_factory=list)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def _evaluate_mode(
    baked_model,
    reference_model,
    fixed_pp,
    dataset,
    dataloader,
    criteria,
    max_frames: Optional[int],
) -> FrameMetrics:
    fm = FrameMetrics()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_frames is not None and i >= max_frames:
                break
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)

            # reference: full per-frame PPISP applied to reference render
            ref_outputs = reference_model(gpu_batch)
            ref_outputs = apply_post_processing(fixed_pp, ref_outputs, gpu_batch, training=False)
            ref_rgb = ref_outputs["pred_rgb"].clip(0, 1)

            # baked: SH eval is already display-referred (gamma); just clip.
            baked_outputs = baked_model(gpu_batch)
            baked_rgb = torch.clamp(baked_outputs["pred_rgb"], 0, 1)

            fm.psnr.append(criteria["psnr"](baked_rgb, ref_rgb).item())
            if "ssim" in criteria:
                fm.ssim.append(criteria["ssim"](
                    baked_rgb.permute(0, 3, 1, 2), ref_rgb.permute(0, 3, 1, 2),
                ).item())
            if "lpips" in criteria:
                fm.lpips.append(criteria["lpips"](
                    baked_rgb.clip(0, 1).permute(0, 3, 1, 2),
                    ref_rgb.clip(0, 1).permute(0, 3, 1, 2),
                ).item())
    return fm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(rows: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Print one table per metric (PSNR / SSIM / LPIPS), sorted by mean."""
    for metric in ("psnr", "ssim", "lpips"):
        any_data = any(metric in r for r in rows.values())
        if not any_data:
            continue
        print(f"\n=== {metric.upper()} (val split, {next(iter(rows.values())).get(metric, {}).get('n', '?')} frames) ===")
        if metric == "psnr":
            print(f"{'mode':<28} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
        else:
            print(f"{'mode':<28} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
        sorted_modes = sorted(
            rows.items(),
            key=lambda kv: -kv[1].get(metric, {}).get("mean", float("-inf"))
                if metric == "psnr" or metric == "ssim"
                else kv[1].get(metric, {}).get("mean", float("inf")),
        )
        for mode_name, metrics in sorted_modes:
            s = metrics.get(metric)
            if s is None:
                continue
            fmt = "%.3f" if metric != "psnr" else "%6.3f"
            print(
                f"{mode_name:<28} "
                f"{s['mean']:>9.4f} {s['median']:>9.4f} "
                f"{s['min']:>9.4f} {s['max']:>9.4f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--fit-epochs", type=int, default=1)
    parser.add_argument("--fit-lr", type=float, default=1.0e-3)
    parser.add_argument("--view-seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit val frames for quick smoke checks.")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Subset of mode names to run (default: all).")
    parser.add_argument("--no-extra-metrics", action="store_true",
                        help="Skip SSIM/LPIPS (PSNR only).")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    renderer = Renderer.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        path=args.data_path,
        out_dir=str(args.out_dir / "_renderer"),
        save_gt=False, computes_extra_metrics=not args.no_extra_metrics,
    )
    if renderer.post_processing is None:
        raise SystemExit("Checkpoint does not contain PPISP.")
    ppisp = renderer.post_processing
    if not hasattr(ppisp, "vignetting_params"):
        raise SystemExit("Checkpoint post-processing is not PPISP-like.")

    # The bake target is PPISP-without-vignetting (matches the production
    # MODE_PPISP_BAKE_VIGNETTING_NONE adapter); both reference and baked
    # sides therefore live in the same display-referred space.
    fixed_pp = FixedPPISP(
        ppisp, args.camera_id, args.frame_id, "cuda", include_vignetting=False,
    ).eval()

    # Train dataset for the fit modes (interpolated samplers need it for poses).
    from threedgrut.export.usd.post_processing_sh_bake import _create_train_dataloader
    train_dataset = renderer.dataset.__class__  # type: ignore
    # Re-create train dataset from the loader's dataset reference: easier to
    # use renderer.conf-based factory.
    import threedgrut.datasets as datasets
    train_ds = datasets.make_train(name=renderer.conf.dataset.type, config=renderer.conf, ray_jitter=None)

    from torchmetrics import PeakSignalNoiseRatio
    criteria: Dict[str, nn.Module] = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}
    if not args.no_extra_metrics:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        criteria["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
        criteria["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda")

    catalogue = all_modes(
        fit_epochs=args.fit_epochs, fit_lr=args.fit_lr, view_seed=args.view_seed,
    )
    if args.modes is not None:
        wanted = set(args.modes)
        catalogue = [m for m in catalogue if m.name in wanted]
        if not catalogue:
            raise SystemExit(f"No modes match {sorted(wanted)}")

    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    timings: Dict[str, float] = {}

    for mode in catalogue:
        logger.info("=" * 60)
        logger.info("MODE %s -- %s", mode.name, mode.description)
        t0 = time.time()
        baked = mode.builder(
            model=renderer.model, ppisp=ppisp, dataset=train_ds, conf=renderer.conf,
            camera_id=args.camera_id, frame_id=args.frame_id,
        )
        build_time = time.time() - t0
        logger.info("  built in %.2fs", build_time)

        fm = _evaluate_mode(
            baked, renderer.model, fixed_pp,
            renderer.dataset, renderer.dataloader, criteria,
            max_frames=args.max_frames,
        )
        row = {"psnr": _stats(fm.psnr)}
        if not args.no_extra_metrics:
            row["ssim"] = _stats(fm.ssim)
            row["lpips"] = _stats(fm.lpips)
        rows[mode.name] = row
        timings[mode.name] = build_time
        logger.info(
            "  %s: PSNR mean=%.3f median=%.3f (n=%d)",
            mode.name, row["psnr"]["mean"], row["psnr"]["median"], row["psnr"]["n"],
        )

    _print_table(rows)
    print("\n=== Build time (seconds) ===")
    for name, t in sorted(timings.items(), key=lambda kv: kv[1]):
        print(f"  {name:<28} {t:>7.2f} s")

    # Persist raw per-frame numbers for offline analysis.
    serial = {
        name: {
            "build_time_s": timings[name],
            **{
                metric: rows[name][metric]
                for metric in ("psnr", "ssim", "lpips") if metric in rows[name]
            },
        }
        for name in rows
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(serial, f, indent=2)
    logger.info("metrics.json saved to %s", args.out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
