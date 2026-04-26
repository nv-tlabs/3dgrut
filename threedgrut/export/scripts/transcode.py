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
Transcode script for converting between Gaussian splatting export formats.

Supports conversions between:
- PLY (pre-activation)
- USD LightField (post-activation)
- NuRec USD/USDZ (Omniverse format; UsdVol::Volume + .nurec payload)

Usage:
    python -m threedgrut.export.scripts.transcode input.ply -o output.usdz --format lightfield
    python -m threedgrut.export.scripts.transcode input.usdz -o output.ply
    python -m threedgrut.export.scripts.transcode nurec.usd -o lightfield.usdz --format lightfield

USD/USDZ → LightField: source /World prims (e.g. rig_trajectories) merge into default.usda at the
same paths; referenced layers are bundled unchanged (preserves camera animation curves).
/World/Gaussians is skipped by default; use --copy-source-include-gaussians to merge it too.
Use --no-copy-source-prims to disable.
"""

import argparse
import logging
import sys
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

from threedgrut.export.adapter import AttributesExportAdapter
from threedgrut.export.base import ModelExporter
from threedgrut.export.formats import PLYExporter
from threedgrut.export.importers import (
    FormatImporter,
    NuRecUSDImporter,
    PLYImporter,
    USDImporter,
)
from threedgrut.export.usd.camera_copy import usd_stage_path_context_for_camera_copy
from threedgrut.export.usd.exporter import USDExporter
from threedgrut.export.usd.nurec.exporter import NuRecExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Supported output formats
OUTPUT_FORMATS = {
    "ply": "PLY point cloud format (pre-activation values)",
    "lightfield": "USD ParticleField3DGaussianSplat schema (post-activation values)",
    "nurec": "NuRec USDZ format for Omniverse",
}


def _is_nurec_stage(stage_path: Path) -> bool:
    """Return True if the USD stage contains a NuRec Volume prim."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if not stage:
        return False
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Volume":
            continue
        attr = prim.GetAttribute("omni:nurec:isNuRecVolume")
        if attr.IsValid() and attr.Get():
            return True
    return False


def detect_input_format(path: Path) -> str:
    """Detect input format from file extension and, for USD, from stage content.

    Args:
        path: Input file path

    Returns:
        Format string: 'ply', 'nurec', or 'lightfield'
    """
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return "ply"
    elif suffix in [".usd", ".usda", ".usdc", ".usdz"]:
        # Refine to NuRec vs Lightfield by inspecting the stage
        if suffix == ".usdz":
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmpdir_path)
                usd_files = list(tmpdir_path.glob("*.usd*"))
                root_file = None
                for f in usd_files:
                    if f.stem == "default":
                        root_file = f
                        break
                if root_file is None and usd_files:
                    root_file = usd_files[0]
                if root_file is not None and _is_nurec_stage(root_file):
                    return "nurec"
        else:
            if _is_nurec_stage(path):
                return "nurec"
        return "lightfield"
    else:
        raise ValueError(f"Unknown input format for extension: {suffix}")


def get_importer(format_name: str, max_sh_degree: int = 3) -> FormatImporter:
    """Get importer for the specified format.

    Args:
        format_name: Format name ('ply', 'lightfield', 'nurec')
        max_sh_degree: Maximum SH degree for PLY importer

    Returns:
        FormatImporter instance
    """
    if format_name == "ply":
        return PLYImporter(max_sh_degree=max_sh_degree)
    elif format_name == "lightfield":
        return USDImporter()
    elif format_name == "nurec":
        return NuRecUSDImporter()
    else:
        raise ValueError(f"Unknown input format: {format_name}")


def get_exporter(
    format_name: str,
    half_precision: bool = False,
    half_geometry: bool = False,
    half_features: bool = False,
    render_order_hint: Optional[str] = None,
    linear_srgb: bool = False,
) -> Tuple[ModelExporter, bool]:
    """Get exporter for the specified format.

    Args:
        format_name: Format name ('ply', 'lightfield', 'nurec')
        half_precision: If True, use half for both geometry and features (LightField). Backward compat.
        half_geometry: Use half precision for positions, orientations, scales (LightField only).
        half_features: Use half precision for opacities and SH coefficients (LightField only).
        render_order_hint: If set, force sortingModeHint for lightfield. Ignored for other formats.
        linear_srgb: If True, set prim color space to lin_rec709_scene (lightfield only).

    Returns:
        Tuple of (ModelExporter instance, expects_preactivation)
    """
    if half_precision:
        half_geometry = True
        half_features = True
    if format_name == "ply":
        return PLYExporter(), True
    elif format_name == "lightfield":
        return (
            USDExporter(
                half_geometry=half_geometry,
                half_features=half_features,
                export_cameras=False,
                export_background=False,
                apply_normalizing_transform=False,
                sorting_mode_hint=render_order_hint if render_order_hint is not None else "cameraDistance",
                linear_srgb=linear_srgb,
            ),
            False,
        )
    elif format_name == "nurec":
        return NuRecExporter(), True
    else:
        raise ValueError(f"Unknown output format: {format_name}")


def infer_output_format(output_path: Path) -> Optional[str]:
    """Infer output format from file extension.

    Args:
        output_path: Output file path

    Returns:
        Format string or None if cannot be inferred
    """
    suffix = output_path.suffix.lower()
    if suffix == ".ply":
        return "ply"
    elif suffix in [".usd", ".usda", ".usdc", ".usdz"]:
        return "lightfield"  # Default USD to lightfield
    return None


def transcode(
    input_path: Path,
    output_path: Path,
    output_format: str,
    max_sh_degree: int = 3,
    half_precision: bool = False,
    half_geometry: bool = False,
    half_features: bool = False,
    apply_coordinate_transform: bool = False,
    render_order_hint: Optional[str] = None,
    linear_srgb: bool = False,
    copy_cameras_source: Optional[Tuple[Path, Path]] = None,
    copy_source_skip_subtrees: Optional[Tuple] = None,
    validate_usd: bool = True,
) -> None:
    """Transcode between Gaussian splatting formats.

    Args:
        input_path: Path to input file
        output_path: Path for output file
        output_format: Target format name
        max_sh_degree: Maximum SH degree for PLY import
        half_precision: If True, use half for both geometry and features (LightField). Backward compat.
        half_geometry: Use half for positions, orientations, scales (LightField only).
        half_features: Use half for opacities and SH coefficients (LightField only).
        apply_coordinate_transform: Apply 3DGRUT-to-USDZ transform (for both lightfield and nurec)
        render_order_hint: If set, force sortingModeHint for lightfield only; ignored for other formats (warning logged).
        linear_srgb: If True, set prim color space to lin_rec709_scene (lightfield only).
        copy_cameras_source: If set, (root_usd_path, asset_resolution_dir) to copy source /World prims from.
        copy_source_skip_subtrees: Optional tuple of Sdf.Path roots to skip under /World (None = default skip Gaussians).
        validate_usd: If True and output is lightfield, run OpenUSD stage validation after export.
    """
    if render_order_hint is not None and output_format != "lightfield":
        logger.warning(
            "--render-order-hint is only applied for lightfield format; ignoring for format '%s'",
            output_format,
        )
    # Detect input format
    input_format = detect_input_format(input_path)
    logger.info(f"Input format: {input_format}")
    logger.info(f"Output format: {output_format}")

    # Get importer and load data
    importer = get_importer(input_format, max_sh_degree)
    attrs, caps = importer.load(input_path)
    source_is_preactivation = importer.stores_preactivation

    logger.info(f"Loaded {attrs.num_gaussians} Gaussians (preactivation={source_is_preactivation})")

    # Get exporter
    exporter, target_expects_preactivation = get_exporter(
        output_format,
        half_precision=half_precision,
        half_geometry=half_geometry,
        half_features=half_features,
        render_order_hint=render_order_hint if output_format == "lightfield" else None,
        linear_srgb=linear_srgb if output_format == "lightfield" else False,
    )

    # Create adapter with correct activation state
    # The adapter needs to know the source data state
    adapter = AttributesExportAdapter(
        attrs=attrs,
        caps=caps,
        is_preactivation=source_is_preactivation,
    )

    # NuRec export always produces USDZ (zip). Require .usdz extension.
    if output_format == "nurec" and output_path.suffix.lower() != ".usdz":
        raise ValueError(
            f"NuRec format requires output extension .usdz, got '{output_path.suffix}'. "
            f"Use e.g. -o {output_path.with_suffix('.usdz')}"
        )

    # Export
    logger.info(f"Exporting to {output_path}...")
    exporter.export(
        adapter,
        output_path,
        apply_coordinate_transform=apply_coordinate_transform,
        copy_cameras_source=copy_cameras_source,
        copy_source_skip_subtrees=copy_source_skip_subtrees,
        validate_usd=validate_usd if output_format == "lightfield" else False,
    )

    logger.info(f"Transcode complete: {input_path} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcode between Gaussian splatting export formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  Input:  ply, usd/usda/usdc/usdz (auto-detected: LightField vs NuRec)
  Output: ply, lightfield, nurec

Examples:
  # Convert PLY to USD LightField
  python -m threedgrut.export.scripts.transcode model.ply -o model.usdz --format lightfield

  # Convert NuRec USD to LightField
  python -m threedgrut.export.scripts.transcode nurec.usd -o lightfield.usdz --format lightfield

  # Convert USD to PLY
  python -m threedgrut.export.scripts.transcode model.usdz -o model.ply

  # Convert PLY to NuRec (Omniverse)
  python -m threedgrut.export.scripts.transcode model.ply -o model.usdz --format nurec
""",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input file path (ply, usd, usda, usdc, usdz)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=list(OUTPUT_FORMATS.keys()),
        default=None,
        help=f"Output format. If not specified, inferred from output extension. "
        f"Choices: {', '.join(OUTPUT_FORMATS.keys())}",
    )
    parser.add_argument(
        "--max-sh-degree",
        type=int,
        default=3,
        help="Maximum SH degree for PLY import (default: 3)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision for both geometry and features (LightField). Same as --half-geometry --half-features.",
    )
    parser.add_argument(
        "--half-geometry",
        action="store_true",
        help="Use half precision for positions, orientations, scales (LightField only).",
    )
    parser.add_argument(
        "--half-features",
        action="store_true",
        help="Use half precision for opacities and SH coefficients (LightField only).",
    )
    parser.add_argument(
        "--apply-coordinate-transform",
        action="store_true",
        help="Apply 3DGRUT-to-USDZ coordinate transform (Omniverse convention). Use for both lightfield and nurec.",
    )
    parser.add_argument(
        "--render-order-hint",
        type=str,
        default=None,
        metavar="MODE",
        help="Force sortingModeHint for lightfield export (e.g. cameraDistance, zDepth). Ignored with --format ply/nurec (warning only).",
    )
    parser.add_argument(
        "--linear-srgb",
        action="store_true",
        help="Set prim color space to lin_rec709_scene (lightfield only). Default is srgb_rec709_display.",
    )
    parser.add_argument(
        "--no-copy-source-prims",
        action="store_true",
        dest="no_copy_source_prims",
        help="When input is USD/USDZ and output is LightField, do not merge source /World prims into default.usda.",
    )
    parser.add_argument(
        "--no-copy-source-cameras",
        action="store_true",
        dest="no_copy_source_prims",
        help="Deprecated alias for --no-copy-source-prims.",
    )
    parser.add_argument(
        "--copy-source-include-gaussians",
        action="store_true",
        help="Also copy /World/Gaussians from the source (duplicates old LightField data; can be very large).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-usd-validate",
        action="store_true",
        help="Skip OpenUSD stage validation after lightfield (.usd/.usdz) export",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input exists
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    # Determine output format
    output_format = args.format
    if output_format is None:
        output_format = infer_output_format(output_path)
        if output_format is None:
            logger.error(
                f"Cannot infer output format from extension '{output_path.suffix}'. "
                f"Please specify --format explicitly."
            )
            sys.exit(1)
        logger.info(f"Inferred output format: {output_format}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix_in = input_path.suffix.lower()
    use_camera_copy_ctx = (
        output_format == "lightfield"
        and suffix_in in (".usd", ".usda", ".usdc", ".usdz")
        and not args.no_copy_source_prims
    )
    camera_ctx = usd_stage_path_context_for_camera_copy(input_path) if use_camera_copy_ctx else nullcontext(None)

    try:
        with camera_ctx as copy_cameras_source:
            skip_subtrees = () if args.copy_source_include_gaussians else None
            transcode(
                input_path=input_path,
                output_path=output_path,
                output_format=output_format,
                max_sh_degree=args.max_sh_degree,
                half_precision=args.half,
                half_geometry=args.half_geometry,
                half_features=args.half_features,
                apply_coordinate_transform=args.apply_coordinate_transform,
                render_order_hint=args.render_order_hint,
                linear_srgb=args.linear_srgb,
                copy_cameras_source=copy_cameras_source,
                copy_source_skip_subtrees=skip_subtrees,
                validate_usd=not args.no_usd_validate,
            )
    except Exception as e:
        logger.error(f"Transcode failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
