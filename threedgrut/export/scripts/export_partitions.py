#!/usr/bin/env python3
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

"""
Command-line script for exporting a 3DGRUT model as spatial volume partitions.

Slices the scene into axis-aligned partitions, each holding at most
``--max-particles-per-field`` particles, and writes one ParticleField3DGaussianSplat
prim per partition (USD) and/or one PLY point cloud per partition.

When ``--max-particles-per-field`` is at least the scene's particle count, no
partitioning happens and the output is identical to a regular (geometry-only) export.

Usage:
    python -m threedgrut.export.scripts.export_partitions \
        --checkpoint path/to/checkpoint.pt --output out/scene \
        --max-particles-per-field 200000 --format both
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.export.formats import export_partitions as export_ply_partitions
from threedgrut.export.partition import partition_scene
from threedgrut.export.scripts.export_usd import load_model_from_checkpoint
from threedgrut.export.usd.partition_exporter import VolumePartitionUSDExporter
from threedgrut.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a 3DGRUT Gaussian model as spatial volume partitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the 3DGRUT checkpoint (.pt)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output base path (extension is derived per format; e.g. out/scene -> out/scene_partition_000.ply)",
    )
    parser.add_argument(
        "--max-particles-per-field",
        dest="max_per_volume",
        type=int,
        required=True,
        help="Maximum number of particles per ParticleField. Partitioning only happens when this "
        "is smaller than the scene's particle count.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["usd", "ply", "both"],
        default="both",
        help="Output format(s) to write. Default: both.",
    )
    parser.add_argument(
        "--usd-format",
        type=str,
        choices=["usdz", "usda", "usd", "usdc"],
        default="usdz",
        help="USD container to write when --format includes usd. Default: usdz.",
    )

    # Writer precision passthrough (USD)
    parser.add_argument("--half", action="store_true", help="Use half precision for geometry and features (USD).")
    parser.add_argument("--half-geometry", action="store_true", help="Half precision for positions/orientations/scales.")
    parser.add_argument("--half-features", action="store_true", help="Half precision for opacities and SH coefficients.")
    parser.add_argument("--no-usd-validate", action="store_true", help="Skip OpenUSD stage validation after USD export.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    try:
        model, conf, _background, _post_processing = load_model_from_checkpoint(str(checkpoint_path))
    except Exception as e:  # noqa: BLE001 - surface load failures cleanly to the CLI user
        logger.error(f"Failed to load checkpoint: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    accessor = GaussianExportAccessor(model, conf)
    num_gaussians = accessor.get_num_gaussians()
    max_sh_degree = accessor.get_max_sh_degree()
    logger.info(f"Loaded model with {num_gaussians} Gaussians (max SH degree {max_sh_degree})")

    result = partition_scene(model, max_per_volume=args.max_per_volume, conf=conf)

    if result.is_partitioned:
        m = result.metrics
        overlap = f"{m['overlap_ratio']:.4f}" if m.get("overlap_ratio") is not None else "n/a (too many partitions)"
        logger.info(
            "Partition summary: %d partitions | counts min=%d max=%d mean=%.0f std=%.0f | "
            "split-added=%d | AABB overlap ratio=%s",
            m["num_partitions"],
            m["count_min"],
            m["count_max"],
            m["count_mean"],
            m["count_std"],
            m["num_split_added"],
            overlap,
        )
    else:
        logger.info("Scene fits within one volume; writing a single unpartitioned export")

    output_base = Path(args.output)
    half_geometry = args.half_geometry or args.half
    half_features = args.half_features or args.half

    try:
        if args.format in ("ply", "both"):
            written = export_ply_partitions(result, output_base.with_suffix(".ply"))
            logger.info(f"Wrote {len(written)} PLY file(s)")

        if args.format in ("usd", "both"):
            usd_path = output_base.with_suffix(f".{args.usd_format}")
            exporter = VolumePartitionUSDExporter(half_geometry=half_geometry, half_features=half_features)
            written_usd = exporter.export(
                model, result, usd_path, conf=conf, validate_usd=not args.no_usd_validate
            )
            logger.info(f"Wrote USD: {written_usd}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Export failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    logger.info("Export successful")


if __name__ == "__main__":
    main()
