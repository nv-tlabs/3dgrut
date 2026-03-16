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
Command-line script for exporting 3DGRUT models to USD format.

Usage:
    python -m threedgrut.export.scripts.export_usd --checkpoint path/to/checkpoint.pt --output output.usdz

    # Export with NuRec format (Omniverse compatibility)
    python -m threedgrut.export.scripts.export_usd --checkpoint path/to/checkpoint.pt --output output.usdz --format nurec

    # Export without cameras/background
    python -m threedgrut.export.scripts.export_usd --checkpoint path/to/checkpoint.pt --output output.usdz --no-cameras --no-background
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from threedgrut.export import USDExporter, NuRecExporter
from threedgrut.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export 3DGRUT Gaussian model to USD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic export to USDZ (default)
    python -m threedgrut.export.scripts.export_usd -c checkpoint.pt -o output.usdz

    # Export with NuRec format for Omniverse
    python -m threedgrut.export.scripts.export_usd -c checkpoint.pt -o output.usdz --format nurec

    # Export to plain USDA (human-readable)
    python -m threedgrut.export.scripts.export_usd -c checkpoint.pt -o output.usda

    # Export with half precision for smaller files
    python -m threedgrut.export.scripts.export_usd -c checkpoint.pt -o output.usdz --half

    # Skip camera and background export
    python -m threedgrut.export.scripts.export_usd -c checkpoint.pt -o output.usdz --no-cameras --no-background
""",
    )

    # Required arguments
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Path to the 3DGRUT checkpoint file (.pt)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path (.usdz, .usda, or .usd)",
    )

    # Format options
    parser.add_argument(
        "--format",
        type=str,
        choices=["standard", "nurec"],
        default="standard",
        help="USD format to use: 'standard' (ParticleField3DGaussianSplat), 'nurec' (Omniverse)",
    )

    # Export options
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision for both geometry and features (same as --half-geometry --half-features).",
    )
    parser.add_argument(
        "--half-geometry",
        action="store_true",
        help="Use half precision for positions, orientations, scales (LightField).",
    )
    parser.add_argument(
        "--half-features",
        action="store_true",
        help="Use half precision for opacities and SH coefficients (LightField).",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip camera export",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Skip background/environment export",
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Skip normalizing transform",
    )
    parser.add_argument(
        "--linear-srgb",
        action="store_true",
        help="Set prim color space to lin_rec709_scene (linear). Default is srgb_rec709_display.",
    )

    # Dataset path (optional, overrides checkpoint's dataset path)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset (overrides path from checkpoint). Required for camera export if not in checkpoint.",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str):
    """Load a 3DGRUT model from checkpoint."""
    from threedgrut.model.model import MixtureOfGaussians
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # weights_only=False needed for checkpoints containing numpy arrays (PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get configuration from checkpoint
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config' key")
    
    conf = checkpoint["config"]
    
    # Create model from configuration
    model = MixtureOfGaussians(conf, scene_extent=checkpoint.get("scene_extent"))
    
    # Load model parameters from checkpoint (without setting up optimizer)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.eval()
    
    return model, conf, model.background


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_path = Path(args.output)

    # Load model from checkpoint
    try:
        model, conf, background = load_model_from_checkpoint(str(checkpoint_path))
        logger.info(f"Loaded model with {model.get_positions().shape[0]} Gaussians")
    except ImportError:
        logger.error("Failed to import model class. Is 3DGRUT properly installed?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Load dataset for camera export
    dataset = None
    if not args.no_cameras:
        try:
            import threedgrut.datasets as datasets
            
            # Override dataset path if provided via CLI
            if args.dataset:
                conf.path = args.dataset
                logger.info(f"Using dataset path from CLI: {args.dataset}")
            
            # Check if dataset path exists in config
            if not hasattr(conf, 'path') or not conf.path:
                logger.warning("No dataset path in checkpoint. Use --dataset to specify path for camera export.")
            elif not hasattr(conf, 'dataset') or not hasattr(conf.dataset, 'type'):
                logger.warning("No dataset type in checkpoint config. Cannot load dataset for camera export.")
            else:
                dataset = datasets.make_test(name=conf.dataset.type, config=conf)
                split = getattr(dataset, "split", "unknown")
                logger.info(f"Loaded dataset with {len(dataset)} frames for camera export (split={split})")
        except Exception as e:
            logger.warning(f"Failed to load dataset for camera export: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Create exporter based on format
    if args.format == "nurec":
        exporter = NuRecExporter()
        logger.info("Using NuRec format (Omniverse compatible)")
    else:
        half_geometry = args.half_geometry or args.half
        half_features = args.half_features or args.half
        exporter = USDExporter(
            half_geometry=half_geometry,
            half_features=half_features,
            export_cameras=not args.no_cameras,
            export_background=not args.no_background,
            apply_normalizing_transform=not args.no_transform,
            linear_srgb=args.linear_srgb,
        )
        logger.info("Using ParticleField3DGaussianSplat schema (standard)")

    # Export
    try:
        exporter.export(
            model=model,
            output_path=output_path,
            dataset=dataset,
            conf=conf,
            background=background,
        )
        logger.info(f"Export successful: {output_path}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
