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
    python -m threedgrut.export.scripts.export_usd --checkpoint path/to/checkpoint.pt \
        --output output.usdz --format nurec

    # Export without cameras/background
    python -m threedgrut.export.scripts.export_usd --checkpoint path/to/checkpoint.pt \
        --output output.usdz --no-cameras --no-background
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from threedgrut.export import NuRecExporter, USDExporter
from threedgrut.export.usd.particle_field_hints import (
    DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
    PARTICLE_FIELD_SORTING_MODE_HINTS,
)
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
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the 3DGRUT checkpoint file (.pt)",
    )
    parser.add_argument(
        "-o",
        "--output",
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
    parser.add_argument(
        "--sorting-mode-hint",
        type=str,
        choices=PARTICLE_FIELD_SORTING_MODE_HINTS,
        default=None,
        help=(
            "ParticleField sortingModeHint for standard USD export. "
            "Use rayHitDistance for ray-tracing renderers that support ray-hit sorting."
        ),
    )
    post_processing_group = parser.add_mutually_exclusive_group()
    post_processing_group.add_argument(
        "--export-post-processing",
        dest="export_post_processing",
        action="store_true",
        default=None,
        help="Export post-processing effects when the checkpoint contains a supported post-processing module.",
    )
    post_processing_group.add_argument(
        "--no-export-post-processing",
        dest="export_post_processing",
        action="store_false",
        help="Skip post-processing export even when the checkpoint contains a supported post-processing module.",
    )
    parser.add_argument(
        "--post-processing-export-mode",
        type=str,
        choices=["baked-sh", "omni-native"],
        default=None,
        help="Post-processing export mode. 'omni-native' uses PPISP SPG and Omniverse material authoring.",
    )
    parser.add_argument(
        "--post-processing-export-camera-id",
        type=int,
        default=None,
        help="Optional PPISP camera id to use for every RenderProduct in omni-native export.",
    )
    parser.add_argument(
        "--post-processing-export-frame-id",
        type=int,
        default=None,
        help="Optional PPISP frame id to write as static omni-native shader inputs instead of animation.",
    )
    parser.add_argument(
        "--ignore-ppisp-controller",
        action="store_true",
        help=(
            "If the checkpoint contains trained PPISP controllers, ignore them and "
            "export the optimized per-frame exposure/color parameters as time-sampled "
            "USD attributes instead. Has no effect when the checkpoint has no controllers."
        ),
    )
    parser.add_argument(
        "--post-processing-bake-epochs",
        type=int,
        default=None,
        help="Number of sequential passes over the train/reference set for post-processing baked-SH export.",
    )
    parser.add_argument(
        "--post-processing-bake-learning-rate",
        type=float,
        default=None,
        help="Adam learning rate for post-processing baked-SH export.",
    )
    parser.add_argument(
        "--post-processing-bake-camera-id",
        type=int,
        default=None,
        help="Camera id used by the fixed post-processing baked-SH export.",
    )
    parser.add_argument(
        "--post-processing-bake-frame-id",
        type=int,
        default=None,
        help="Frame id used by the fixed post-processing baked-SH export.",
    )
    parser.add_argument(
        "--ppisp-bake-vignetting-mode",
        type=str,
        choices=["none", "achromatic-fit"],
        default=None,
        help=(
            "Vignetting handling for PPISP baked-SH fitting. 'none' disables PPISP vignetting; "
            "'achromatic-fit' uses chromatic PPISP reference and an achromatic fit-only vignette."
        ),
    )
    parser.add_argument(
        "--post-processing-bake-view-mode",
        type=str,
        choices=["training", "random-pair-slerp", "trajectory"],
        default=None,
        help=(
            "Which views the bake fit sees per step. 'training' (default) iterates the train "
            "dataloader. 'random-pair-slerp' picks two random training views and slerps. "
            "'trajectory' orders views along an NN+2-opt camera path and samples random t in [0,1]."
        ),
    )
    parser.add_argument(
        "--post-processing-bake-view-seed",
        type=int,
        default=None,
        help="Optional RNG seed for the interpolation samplers (None = non-deterministic).",
    )
    parser.add_argument(
        "--post-processing-bake-trajectory-weight-position",
        type=float,
        default=None,
        help="Trajectory mode only: weight on the (mean-normalised) position term in pose distance.",
    )
    parser.add_argument(
        "--post-processing-bake-trajectory-weight-rotation",
        type=float,
        default=None,
        help="Trajectory mode only: weight on the (1 - cos(angle)) rotation term in pose distance.",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-usd-validate",
        action="store_true",
        help="Skip OpenUSD stage validation after standard (ParticleField) export",
    )

    return parser.parse_args()


def _load_ppisp_from_checkpoint(checkpoint, conf):
    """Load trained PPISP state for USD export when available."""
    post_conf = getattr(conf, "post_processing", None)
    if "post_processing" not in checkpoint or post_conf is None or getattr(post_conf, "method", None) != "ppisp":
        return None

    try:
        from ppisp import PPISP, PPISPConfig
    except ImportError:
        logger.warning("Checkpoint contains PPISP state, but ppisp is not available; skipping PPISP USD export")
        return None

    use_controller = post_conf.get("use_controller", True)
    n_distillation_steps = post_conf.get("n_distillation_steps", 5000)
    if use_controller and n_distillation_steps > 0:
        main_training_steps = conf.n_iterations - n_distillation_steps
        controller_activation_ratio = main_training_steps / conf.n_iterations
        controller_distillation = True
    elif use_controller:
        controller_activation_ratio = 0.8
        controller_distillation = False
    else:
        controller_activation_ratio = 0.0
        controller_distillation = False

    ppisp_config = PPISPConfig(
        use_controller=use_controller,
        controller_distillation=controller_distillation,
        controller_activation_ratio=controller_activation_ratio,
    )
    post_processing = PPISP.from_state_dict(checkpoint["post_processing"]["module"], config=ppisp_config)
    post_processing = post_processing.to("cpu")
    logger.info("Loaded PPISP post-processing state for USD export")
    return post_processing


def _get_export_conf_value(export_conf, dashed_name: str, attr_name: str, default):
    if hasattr(export_conf, "get"):
        return export_conf.get(dashed_name, getattr(export_conf, attr_name, default))
    return getattr(export_conf, attr_name, default)


def _get_export_post_processing_default(export_conf):
    if hasattr(export_conf, "get"):
        return export_conf.get(
            "export-post-processing",
            getattr(export_conf, "export_post_processing", True),
        )
    return getattr(export_conf, "export_post_processing", True)


def _arg_or_conf(cli_value, export_conf, dashed_name: str, attr_name: str, default):
    if cli_value is not None:
        return cli_value
    return _get_export_conf_value(export_conf, dashed_name, attr_name, default)


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

    post_processing = _load_ppisp_from_checkpoint(checkpoint, conf)
    return model, conf, model.background, post_processing


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_path = Path(args.output)

    # Load model from checkpoint
    try:
        model, conf, background, post_processing = load_model_from_checkpoint(str(checkpoint_path))
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

    export_conf = getattr(conf, "export_usd", None) or conf
    if args.export_post_processing is not None:
        export_post_processing = args.export_post_processing
    elif post_processing is not None:
        export_post_processing = True
    else:
        export_post_processing = bool(_get_export_post_processing_default(export_conf))
    post_processing_export_mode = _arg_or_conf(
        args.post_processing_export_mode,
        export_conf,
        "post-processing-export-mode",
        "post_processing_export_mode",
        "baked-sh",
    )
    # Load dataset for camera export and for train-split post-processing SH baking.
    dataset = None
    needs_dataset = not args.no_cameras or (post_processing is not None and export_post_processing)
    if needs_dataset:
        try:
            import threedgrut.datasets as datasets

            # Override dataset path if provided via CLI
            if args.dataset:
                conf.path = args.dataset
                logger.info(f"Using dataset path from CLI: {args.dataset}")

            # Check if dataset path exists in config
            if not hasattr(conf, "path") or not conf.path:
                logger.warning("No dataset path in checkpoint. Use --dataset to specify path for camera export.")
            elif not hasattr(conf, "dataset") or not hasattr(conf.dataset, "type"):
                logger.warning("No dataset type in checkpoint config. Cannot load dataset for camera export.")
            else:
                dataset = datasets.make_train(name=conf.dataset.type, config=conf, ray_jitter=None)
                split = getattr(dataset, "split", "unknown")
                logger.info(f"Loaded dataset with {len(dataset)} frames for camera export (split={split})")
        except Exception as e:
            logger.error(f"Failed to load dataset for camera export: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)

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
            sorting_mode_hint=_arg_or_conf(
                args.sorting_mode_hint,
                export_conf,
                "sorting-mode-hint",
                "sorting_mode_hint",
                DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
            ),
            linear_srgb=args.linear_srgb or getattr(export_conf, "linear_srgb", False),
            export_post_processing=export_post_processing,
            post_processing_export_mode=post_processing_export_mode,
            post_processing_export_camera_id=_arg_or_conf(
                args.post_processing_export_camera_id,
                export_conf,
                "post-processing-export-camera-id",
                "post_processing_export_camera_id",
                None,
            ),
            post_processing_export_frame_id=_arg_or_conf(
                args.post_processing_export_frame_id,
                export_conf,
                "post-processing-export-frame-id",
                "post_processing_export_frame_id",
                None,
            ),
            ignore_ppisp_controller=args.ignore_ppisp_controller,
            post_processing_bake_epochs=_arg_or_conf(
                args.post_processing_bake_epochs,
                export_conf,
                "post-processing-bake-epochs",
                "post_processing_bake_epochs",
                1,
            ),
            post_processing_bake_learning_rate=_arg_or_conf(
                args.post_processing_bake_learning_rate,
                export_conf,
                "post-processing-bake-learning-rate",
                "post_processing_bake_learning_rate",
                1.0e-3,
            ),
            post_processing_bake_camera_id=_arg_or_conf(
                args.post_processing_bake_camera_id,
                export_conf,
                "post-processing-bake-camera-id",
                "post_processing_bake_camera_id",
                0,
            ),
            post_processing_bake_frame_id=_arg_or_conf(
                args.post_processing_bake_frame_id,
                export_conf,
                "post-processing-bake-frame-id",
                "post_processing_bake_frame_id",
                0,
            ),
            ppisp_bake_vignetting_mode=_arg_or_conf(
                args.ppisp_bake_vignetting_mode,
                export_conf,
                "ppisp-bake-vignetting-mode",
                "ppisp_bake_vignetting_mode",
                "achromatic-fit",
            ),
            post_processing_bake_view_mode=_arg_or_conf(
                args.post_processing_bake_view_mode,
                export_conf,
                "post-processing-bake-view-mode",
                "post_processing_bake_view_mode",
                "training",
            ),
            post_processing_bake_view_seed=_arg_or_conf(
                args.post_processing_bake_view_seed,
                export_conf,
                "post-processing-bake-view-seed",
                "post_processing_bake_view_seed",
                None,
            ),
            post_processing_bake_trajectory_weight_position=_arg_or_conf(
                args.post_processing_bake_trajectory_weight_position,
                export_conf,
                "post-processing-bake-trajectory-weight-position",
                "post_processing_bake_trajectory_weight_position",
                1.0,
            ),
            post_processing_bake_trajectory_weight_rotation=_arg_or_conf(
                args.post_processing_bake_trajectory_weight_rotation,
                export_conf,
                "post-processing-bake-trajectory-weight-rotation",
                "post_processing_bake_trajectory_weight_rotation",
                0.5,
            ),
            frames_per_second=getattr(export_conf, "frames_per_second", 1.0),
        )
        logger.info("Using ParticleField3DGaussianSplat schema (standard)")

    # Export
    try:
        export_kw = {}
        if args.format == "standard":
            export_kw["validate_usd"] = not args.no_usd_validate
        exporter.export(
            model=model,
            output_path=output_path,
            dataset=dataset,
            conf=conf,
            background=background,
            post_processing=post_processing,
            **export_kw,
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
