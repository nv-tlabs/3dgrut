#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validator using the *real* ``ppisp`` package.

Builds an actual ``ppisp.PPISP`` module (one camera, one frame, default
``PPISPConfig``), runs its controller through both PyTorch and the
slangpy SPG harness, and reports the per-output abs diff. Run this after
``install_env_uv.sh`` so the full env including ``ppisp`` is available.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make the in-repo writer importable. The module path goes through
# threedgrut.export.usd.writers.ppisp_controller_writer; that import
# chain pulls heavy CUDA pieces from threedgrut/__init__.py which exist
# in the real env, so we just rely on regular imports here.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut.export.usd.writers.ppisp_controller_writer import (  # noqa: E402
    flatten_controller_weights,
)
from tools.render_ppisp_spg.spg_runtime import run_controller  # noqa: E402

logger = logging.getLogger("validate_real_ppisp")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--prior", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--num-cameras", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    from ppisp import PPISP, DEFAULT_PPISP_CONFIG

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ppisp = PPISP(num_cameras=args.num_cameras, num_frames=args.num_frames,
                  config=DEFAULT_PPISP_CONFIG).eval()
    if not ppisp.controllers or len(ppisp.controllers) == 0:
        raise SystemExit("PPISP has no controllers — config.use_controller must be True.")
    controller = ppisp.controllers[0]

    # Perturb the controller so the output is non-trivial (PPISP initialises
    # everything to zero; without weights, the slang/torch outputs would both
    # be zero and the validation would be vacuous).
    with torch.no_grad():
        for p in controller.parameters():
            p.normal_(0.0, 0.01)

    hdr = (rng.random((args.height, args.width, 3), dtype=np.float32) * 0.6 + 0.2)

    rgb_t = torch.from_numpy(hdr).float().to(controller.exposure_head.weight.device)
    pe_t = torch.tensor([args.prior], dtype=torch.float32, device=rgb_t.device)
    with torch.no_grad():
        exposure, color = controller(rgb_t, pe_t)
    expected = np.concatenate([
        np.array([float(exposure)], dtype=np.float32),
        color.detach().cpu().numpy().astype(np.float32),
    ])

    weights = flatten_controller_weights(controller)
    slang_path = Path(__file__).resolve().parents[2] / (
        "threedgrut/export/usd/ppisp_spg/ppisp_controller.slang"
    )
    actual = run_controller(slang_path, hdr, weights, prior_exposure=args.prior)

    diff = np.abs(actual - expected)
    print(f"reference: {expected}")
    print(f"slangpy:   {actual}")
    print(f"abs diff:  {diff}")
    print(f"max abs diff: {diff.max():.6g} (tol={args.tol})")

    return 0 if diff.max() <= args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
