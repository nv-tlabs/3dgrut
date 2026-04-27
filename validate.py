#!/usr/bin/env python3
"""
Validation script: run 3DGUT and 3DGRT on the drums NeRF-Synthetic scene
(MCMC, 100k particles, 15k iterations) for SH and optionally NHT features,
then produce a report with PSNR, SSIM, training time, and render time.

Usage
-----
  python validate.py --data /path/to/nerf_synthetic/drums [OPTIONS]

Options
-------
  --data PATH        Path to the drums scene directory (required)
  --out-dir PATH     Root output directory (default: runs/validate)
  --nht              Also run NHT experiments (requires NHT support in the
                     codebase; silently skipped if unavailable)
  --iterations N     Training iterations per experiment (default: 15000)
  --particles N      Initial particle count (default: 100000)
  --skip-existing    Skip training if the checkpoint already exists
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# NHT availability check
# ---------------------------------------------------------------------------

def _nht_available() -> bool:
    """Return True if this codebase has NHT support compiled in."""
    try:
        from threedgrut.model.features import Features
        _ = Features.Type.NHT
        return True
    except (ImportError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def _experiments(args, nht_ok: bool):
    """Yield (name, renderer, feature_type, app_config) tuples."""
    base = [
        ("3dgut_sh", "3dgut", "sh", "apps/nerf_synthetic_3dgut_mcmc_nht"),
        ("3dgrt_sh", "3dgrt", "sh", "apps/nerf_synthetic_3dgrt_mcmc_nht"),
    ]
    nht = [
        ("3dgut_nht", "3dgut", "nht", "apps/nerf_synthetic_3dgut_mcmc_nht"),
        ("3dgrt_nht", "3dgrt", "nht", "apps/nerf_synthetic_3dgrt_mcmc_nht"),
    ]
    for entry in base:
        yield entry
    if args.nht and nht_ok:
        for entry in nht:
            yield entry


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _find_latest(directory: Path, pattern: str):
    """Return the most recently modified file matching pattern under directory, or None."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _train(name: str, app_config: str, feature_type: str, args) -> tuple[float, str]:
    """Run training and return (wall_time_seconds, checkpoint_path)."""
    exp_dir = Path(args.out_dir) / name

    # Trainer saves under exp_dir/<object>-<timestamp>/ckpt_last.pt
    if args.skip_existing:
        ckpt = _find_latest(exp_dir, "*/ckpt_last.pt")
        if ckpt is not None:
            print(f"  [skip] checkpoint already exists: {ckpt}")
            return 0.0, str(ckpt)

    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        f"--config-name={app_config}",
        f"path={args.data}",
        f"out_dir={args.out_dir}",
        f"experiment_name={name}",
        f"n_iterations={args.iterations}",
        f"initialization.num_gaussians={args.particles}",
        f"model.feature_type={feature_type}",
        # disable feature_output_half for SH (it's set to true in NHT app configs)
        f"render.feature_output_half={'true' if feature_type == 'nht' else 'false'}",
        # save checkpoint only at the end; intermediate checkpoints not needed here
        f"checkpoint.iterations=[{args.iterations}]",
        # disable GUI
        "with_gui=false",
        "with_viser_gui=false",
    ]

    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0

    ckpt = _find_latest(exp_dir, "*/ckpt_last.pt")
    if ckpt is None:
        raise FileNotFoundError(f"Expected checkpoint not found under: {exp_dir}/*/ckpt_last.pt")

    return elapsed, str(ckpt)


# ---------------------------------------------------------------------------
# Render / evaluate
# ---------------------------------------------------------------------------

def _render(name: str, ckpt: str, args) -> dict:
    """Run render.py and return the metrics dict."""
    eval_dir = Path(args.out_dir) / name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "render.py",
        "--checkpoint", ckpt,
        "--out-dir", str(eval_dir),
    ]

    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Renderer saves under eval_dir/<experiment_name>/<object>-<timestamp>/metrics.json
    metrics_path = _find_latest(eval_dir, "**/metrics.json")
    if metrics_path is None:
        raise FileNotFoundError(f"metrics.json not found after render under: {eval_dir}")

    with open(metrics_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_HEADER = (
    "| Experiment       | PSNR (dB) | SSIM   | LPIPS  | Train (min) | Render (ms/f) |\n"
    "|------------------|-----------|--------|--------|-------------|---------------|\n"
)


def _row(name: str, m: dict, train_sec: float) -> str:
    psnr  = f"{m.get('mean_psnr', float('nan')):.2f}"
    ssim  = f"{m.get('mean_ssim', float('nan')):.4f}"
    lpips = f"{m.get('mean_lpips', float('nan')):.4f}"
    t_min = f"{train_sec / 60:.1f}" if train_sec > 0 else "—"
    r_ms  = f"{m.get('mean_inference_time_ms', float('nan')):.2f}"
    return f"| {name:<16} | {psnr:>9} | {ssim:>6} | {lpips:>6} | {t_min:>11} | {r_ms:>13} |\n"


def _write_report(rows: list[tuple], args, nht_ok: bool) -> str:
    scene = Path(args.data).name
    lines = [
        f"# Validation Report: {scene}\n\n",
        f"Scene: `{args.data}`  \n",
        f"Iterations: {args.iterations}  \n",
        f"Particles: {args.particles:,}  \n",
        f"Strategy: MCMC  \n",
        f"NHT requested: {args.nht}  ",
        f"{'(supported)' if nht_ok else '(not available in this build — skipped)'}  \n\n",
        "## Results\n\n",
        _HEADER,
    ]
    for name, metrics, train_sec in rows:
        lines.append(_row(name, metrics, train_sec))

    report = "".join(lines)

    report_path = Path(args.out_dir) / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",         required=True, help="Path to the drums scene directory")
    parser.add_argument("--out-dir",      default="runs/validate", help="Root output directory")
    parser.add_argument("--nht",          action="store_true", help="Also run NHT experiments")
    parser.add_argument("--iterations",   type=int, default=15000, help="Training iterations per experiment")
    parser.add_argument("--particles",    type=int, default=100000, help="Initial particle count")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training if checkpoint exists")
    args = parser.parse_args()

    nht_ok = _nht_available()
    if args.nht and not nht_ok:
        print("WARNING: --nht requested but NHT is not available in this build; NHT experiments will be skipped.")

    rows = []
    for name, renderer, feature_type, app_config in _experiments(args, nht_ok):
        print(f"\n{'='*60}")
        print(f"Experiment: {name}  (renderer={renderer}, features={feature_type})")
        print(f"{'='*60}")

        print("\n[1/2] Training ...")
        try:
            train_sec, ckpt = _train(name, app_config, feature_type, args)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: training failed for {name}: {e}")
            rows.append((name, {}, 0.0))
            continue

        print(f"\n[2/2] Evaluating ...")
        try:
            metrics = _render(name, ckpt, args)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: render failed for {name}: {e}")
            rows.append((name, {}, train_sec))
            continue

        rows.append((name, metrics, train_sec))
        print(f"\n  PSNR={metrics.get('mean_psnr', '?'):.2f} dB  "
              f"SSIM={metrics.get('mean_ssim', '?'):.4f}  "
              f"LPIPS={metrics.get('mean_lpips', '?'):.4f}  "
              f"render={metrics.get('mean_inference_time_ms', '?'):.2f} ms/frame")

    print(f"\n{'='*60}")
    print("REPORT")
    print(f"{'='*60}")
    report = _write_report(rows, args, nht_ok)
    print(report)
    print(f"Report saved to: {Path(args.out_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
