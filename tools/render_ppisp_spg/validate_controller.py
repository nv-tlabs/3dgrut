#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical sanity check: generate a controller slang for a torch
``_PPISPController`` with known weights, dispatch it via slangpy, and
compare its 9-element output to the PyTorch forward pass.

This script does not require the full 3DGRUT environment — only
``torch``, ``numpy``, ``slangpy`` and the in-repo writer module. It
fabricates a controller (without needing a ``ppisp.PPISP`` checkpoint)
by reproducing ``ppisp._PPISPController`` from the public
architecture description.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import the writer directly (bypass threedgrut.__init__ which depends on
# heavy CUDA-only packages we don't need here).
import importlib.util as _ilu  # noqa: E402
import types as _types  # noqa: E402

# Stub the threedgrut packages so we don't trigger their __init__.
for _pkg in (
    "threedgrut",
    "threedgrut.export",
    "threedgrut.export.usd",
    "threedgrut.export.usd.writers",
):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _types.ModuleType(_pkg)

# Stub stage_utils so the writer can import NamedSerialized.
_stage_utils_stub = _types.ModuleType("threedgrut.export.usd.stage_utils")
import dataclasses as _dc


@_dc.dataclass
class _NamedSerialized:
    filename: str
    serialized: bytes


_stage_utils_stub.NamedSerialized = _NamedSerialized
sys.modules["threedgrut.export.usd.stage_utils"] = _stage_utils_stub

# Stub ppisp_spg package so get_controller_sidecars() can resolve _SPG_DIR.
_ppisp_spg_stub = _types.ModuleType("threedgrut.export.usd.ppisp_spg")
_ppisp_spg_stub._SPG_DIR = (
    Path(__file__).resolve().parents[2] / "threedgrut/export/usd/ppisp_spg"
)
sys.modules["threedgrut.export.usd.ppisp_spg"] = _ppisp_spg_stub

_writer_path = (
    Path(__file__).resolve().parents[2]
    / "threedgrut/export/usd/writers/ppisp_controller_writer.py"
)
_spec = _ilu.spec_from_file_location(
    "threedgrut.export.usd.writers.ppisp_controller_writer", str(_writer_path)
)
_writer_mod = _ilu.module_from_spec(_spec)
sys.modules["threedgrut.export.usd.writers.ppisp_controller_writer"] = _writer_mod
_spec.loader.exec_module(_writer_mod)
EXPECTED_SIZES = _writer_mod.EXPECTED_SIZES
flatten_controller_weights = _writer_mod.flatten_controller_weights

from tools.render_ppisp_spg.spg_runtime import run_controller  # noqa: E402


logger = logging.getLogger("validate_controller")


def _make_test_controller(seed: int = 0):
    """Build a torch module with the same architecture as
    ``ppisp._PPISPController``. Importing the real one is preferred but
    we duplicate it here so the validator runs without the ppisp package."""
    import torch
    from torch import nn

    class _Controller(nn.Module):
        def __init__(self):
            super().__init__()
            cfd = EXPECTED_SIZES["cnn_feature_dim"]
            grid = (EXPECTED_SIZES["pool_grid_h"], EXPECTED_SIZES["pool_grid_w"])
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1),
                nn.MaxPool2d(EXPECTED_SIZES["input_downsampling"],
                             stride=EXPECTED_SIZES["input_downsampling"]),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, cfd, kernel_size=1),
                nn.AdaptiveAvgPool2d(grid),
                nn.Flatten(),
            )
            in_dim = cfd * grid[0] * grid[1] + 1
            hd = EXPECTED_SIZES["mlp_hidden_dim"]
            self.mlp_trunk = nn.Sequential(
                nn.Linear(in_dim, hd), nn.ReLU(inplace=True),
                nn.Linear(hd, hd),     nn.ReLU(inplace=True),
                nn.Linear(hd, hd),     nn.ReLU(inplace=True),
            )
            self.exposure_head = nn.Linear(hd, 1)
            self.color_head = nn.Linear(hd, EXPECTED_SIZES["color_params_per_frame"])

        def forward(self, rgb: torch.Tensor, prior_exposure: torch.Tensor):
            features = self.cnn_encoder(rgb.permute(2, 0, 1).unsqueeze(0).detach())
            features = torch.cat([features.squeeze(0), prior_exposure], dim=0)
            hidden = self.mlp_trunk(features)
            return self.exposure_head(hidden).squeeze(-1), self.color_head(hidden)

    torch.manual_seed(seed)
    ctrl = _Controller().eval()
    # Mostly-zero weights with a tiny perturbation so outputs are non-trivial.
    with torch.no_grad():
        for p in ctrl.parameters():
            p.normal_(0.0, 0.01)
    return ctrl


def _torch_reference(ctrl, hdr_image: np.ndarray, prior_exposure: float) -> np.ndarray:
    import torch
    rgb = torch.from_numpy(hdr_image).float()
    pe = torch.tensor([prior_exposure], dtype=torch.float32)
    with torch.no_grad():
        exposure, color = ctrl(rgb, pe)
    return np.concatenate([
        np.array([float(exposure)], dtype=np.float32),
        color.cpu().numpy().astype(np.float32),
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1.0e-3,
                        help="abs tol per output element")
    parser.add_argument("--keep", type=Path, default=None,
                        help="Where to write the generated slang file (defaults to a tmp dir)")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ctrl = _make_test_controller(args.seed)
    rng = np.random.default_rng(args.seed)
    hdr = (rng.random((args.height, args.width, 3), dtype=np.float32) * 0.8 + 0.1)

    expected = _torch_reference(ctrl, hdr, args.prior)

    weights = flatten_controller_weights(ctrl)
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
