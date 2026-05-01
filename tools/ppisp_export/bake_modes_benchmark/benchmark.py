#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep PPISP SH-bake modes on a trained checkpoint and report aggregated metrics.

For each configured (bake mode, view sampling mode, init policy) tuple
the script:

1. Builds a baked model from the cloned checkpoint:
   - ``simple`` / ``simple-higher-order`` flavours run :func:`simple_bake`
     directly (with optional sRGB→linear).
   - ``fit`` flavours run :func:`bake_post_processing_into_sh` with the
     desired view sampling mode and init policy.

2. Renders every validation frame through:
   - the *reference* model + chromatic-vignette PPISP at the chosen
     (camera, frame) -- the per-frame target.
   - the *baked* model with ``linear_to_srgb`` applied to its output and
     an achromatic-vignette correction (matches the evaluator in
     ``post_processing_sh_bake_validation.py``).

3. Computes per-frame PSNR, SSIM and LPIPS, then aggregates mean /
   median / min / max across the validation split.

4. Prints a table sorted by mean PSNR and writes the raw per-frame
   numbers to ``<out_dir>/metrics.json``.

Usage:

    python tools/bake_modes_benchmark/benchmark.py \\
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
    MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
    PPISPPostProcessingBakeAdapter,
    apply_achromatic_vignetting,
    bake_post_processing_into_sh,
    FixedPPISP,
)
from threedgrut.export.usd.post_processing_sh_simple_bake import simple_bake  # noqa: E402
from threedgrut.render import Renderer  # noqa: E402
from threedgrut.utils.post_processing_linear_to_srgb import linear_to_srgb  # noqa: E402
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


def _build_simple(*, model, ppisp, camera_id, frame_id, higher_order, srgb,
                  dataset=None, conf=None):
    del dataset, conf  # unused by the simple flavours
    baked = model.clone().eval()
    simple_bake(
        baked, ppisp,
        camera_id=camera_id, frame_id=frame_id,
        higher_order=higher_order, apply_srgb_to_linear=srgb,
    )
    baked.build_acc()
    return baked


def _build_fit(*, model, ppisp, dataset, conf, camera_id, frame_id,
               vignetting_mode, view_mode, view_seed, epochs, learning_rate,
               init: str):
    """Run the full fit-by-bake flow.

    ``init`` chooses the warm-start applied before Adam takes over:
      * ``"none"``       -- patch out initialize_fit, fit from the clone.
      * ``"higher"``     -- adapter default: simple_bake(higher_order=True, srgb=True).
      * ``"dc-srgb"``    -- DC-only simple_bake with sRGB->linear (leaves
                            features_specular at the trained values).
    """
    from threedgrut.export.usd.post_processing_sh_simple_bake import simple_bake

    adapter = PPISPPostProcessingBakeAdapter(
        camera_id=camera_id, frame_id=frame_id, vignetting_mode=vignetting_mode,
    )
    if init == "none":
        adapter.initialize_fit = lambda *a, **kw: None  # type: ignore[assignment]
    elif init == "dc-srgb":
        def _dc_srgb_init(baked_model, post_processing, _cid=camera_id, _fid=frame_id):
            simple_bake(baked_model, post_processing,
                        camera_id=_cid, frame_id=_fid,
                        higher_order=False, apply_srgb_to_linear=True)
        adapter.initialize_fit = _dc_srgb_init  # type: ignore[assignment]
    elif init != "higher":
        raise ValueError(f"unknown init: {init!r}")
    return bake_post_processing_into_sh(
        model=model, post_processing=ppisp, train_dataset=dataset, conf=conf,
        adapter=adapter, epochs=epochs, learning_rate=learning_rate,
        view_sampling_mode=view_mode, interpolated_views_seed=view_seed,
    )


def all_modes(*, fit_epochs: int, fit_lr: float, view_seed: int) -> List[BakeMode]:
    return [
        BakeMode(
            "simple",
            "one-shot DC-only bake",
            lambda **k: _build_simple(**k, higher_order=False, srgb=False),
        ),
        BakeMode(
            "simple-higher-order",
            "one-shot bake with higher-order Jacobian linearisation",
            lambda **k: _build_simple(**k, higher_order=True, srgb=False),
        ),
        BakeMode(
            "simple-higher-order-srgb",
            "simple-higher-order with sRGB→linear before RGB2SH",
            lambda **k: _build_simple(**k, higher_order=True, srgb=True),
        ),
        BakeMode(
            "fit-base",
            "Adam fit, training views, no warm-start",
            lambda **k: _build_fit(
                **k, vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
                view_mode="training", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, init="none",
            ),
        ),
        BakeMode(
            "fit-base-srgb",
            "Adam fit, training views, DC-only simple-bake (sRGB->linear) warm-start",
            lambda **k: _build_fit(
                **k, vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
                view_mode="training", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, init="dc-srgb",
            ),
        ),
        BakeMode(
            "fit-base-srgb-trajectory",
            "Adam fit, DC-only sRGB warm-start + trajectory views (NN+2-opt, slerp)",
            lambda **k: _build_fit(
                **k, vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
                view_mode="trajectory", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, init="dc-srgb",
            ),
        ),
        BakeMode(
            "fit-init",
            "Adam fit, training views, higher-order simple-bake (sRGB->linear) warm-start",
            lambda **k: _build_fit(
                **k, vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
                view_mode="training", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, init="higher",
            ),
        ),
        BakeMode(
            "fit-init-trajectory",
            "Adam fit, higher-order init + trajectory views (NN+2-opt, slerp)",
            lambda **k: _build_fit(
                **k, vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
                view_mode="trajectory", view_seed=view_seed,
                epochs=fit_epochs, learning_rate=fit_lr, init="higher",
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
    vignetting_mode: str,
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

            # baked: render + achromatic-vignette + linear_to_srgb
            baked_outputs = baked_model(gpu_batch)
            baked_rgb_lin = baked_outputs["pred_rgb"]
            if vignetting_mode != "none":
                _, h, w, _ = baked_rgb_lin.shape
                baked_rgb_lin = apply_achromatic_vignetting(
                    rgb=baked_rgb_lin, ppisp=fixed_pp.ppisp,
                    camera_id=fixed_pp.camera_id,
                    pixel_coords=gpu_batch.pixel_coords,
                    resolution=(w, h),
                )
            baked_rgb = torch.clamp(linear_to_srgb(baked_rgb_lin), 0, 1)

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

    fixed_pp = FixedPPISP(
        ppisp, args.camera_id, args.frame_id, "cuda", include_vignetting=True,
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
            vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
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
