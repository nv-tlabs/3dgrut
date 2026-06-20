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

USD/USDZ → LightField: source /World prims (e.g. rig_trajectories) and /Render
merge into default.usda at the same paths; referenced layers are bundled unchanged
(preserves camera animation curves and authored render products).
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
from threedgrut.export.usd.particle_field_hints import (
    DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
    PARTICLE_FIELD_SORTING_MODE_HINTS,
)

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
) -> Tuple[ModelExporter, bool]:
    """Get exporter for the specified format.

    Args:
        format_name: Format name ('ply', 'lightfield', 'nurec')
        half_precision: If True, use half for both geometry and features (LightField). Backward compat.
        half_geometry: Use half precision for positions, orientations, scales (LightField only).
        half_features: Use half precision for opacities and SH coefficients (LightField only).
        render_order_hint: If set, force sortingModeHint for lightfield. Ignored for other formats.

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
                sorting_mode_hint=(
                    render_order_hint if render_order_hint is not None else DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT
                ),
            ),
            False,
        )
    elif format_name == "nurec":
        # Generic transcode has Gaussian attributes but no training dataset.
        # Source USD cameras / RenderProducts are copied separately when the
        # input is USD, so do not ask NuRecExporter to regenerate them.
        return NuRecExporter(export_cameras=False, export_post_processing=False), True
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
    copy_cameras_source: Optional[Tuple[Path, Path]] = None,
    copy_source_skip_subtrees: Optional[Tuple] = None,
    validate_usd: bool = True,
    max_per_volume: Optional[int] = None,
    split_large_gaussians: bool = False,
    split_target_size: Optional[float] = None,
    split_target_fraction: float = 0.5,
    max_splits: int = 4,
) -> None:
    """Transcode a single input file between Gaussian splatting formats.

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
        copy_cameras_source: If set, (root_usd_path, asset_resolution_dir) to copy source /World prims from.
        copy_source_skip_subtrees: Optional tuple of Sdf.Path roots to skip under /World (None = default skip Gaussians).
        validate_usd: If True and output is lightfield, run OpenUSD stage validation after export.
        max_per_volume: If set, each source prim/group whose own Gaussian count exceeds this is
            subdivided into several partitions; prims within budget are kept as-is. Applies to
            ply / lightfield outputs only.
        split_large_gaussians: Like the exporter's option, split boundary-straddling Gaussians in
            oversized prims before subdividing them (reduces inter-partition overlap).
        split_target_size: Footprint threshold for the split pass (default: fraction of cell edge).
        split_target_fraction: Fraction of the cell edge for the default split threshold.
        max_splits: Maximum split iterations.
    """
    if render_order_hint is not None and output_format != "lightfield":
        logger.warning(
            "--render-order-hint is only applied for lightfield format; ignoring for format '%s'",
            output_format,
        )
    input_format = detect_input_format(input_path)
    logger.info(f"Input format: {input_format}")
    logger.info(f"Output format: {output_format}")

    importer = get_importer(input_format, max_sh_degree)
    fields, source_transforms = _load_sources(importer, input_path)
    source_is_preactivation = importer.stores_preactivation
    logger.info(
        f"Loaded {len(fields)} source field(s), {sum(a.num_gaussians for a, _ in fields)} Gaussians "
        f"(preactivation={source_is_preactivation})"
    )

    _transcode_core(
        fields=fields,
        source_transforms=source_transforms,
        source_is_preactivation=source_is_preactivation,
        output_path=output_path,
        output_format=output_format,
        half_precision=half_precision,
        half_geometry=half_geometry,
        half_features=half_features,
        apply_coordinate_transform=apply_coordinate_transform,
        render_order_hint=render_order_hint,
        copy_cameras_source=copy_cameras_source,
        copy_source_skip_subtrees=copy_source_skip_subtrees,
        validate_usd=validate_usd,
        max_per_volume=max_per_volume,
        split_options=_split_options(split_large_gaussians, split_target_size, split_target_fraction, max_splits),
    )
    logger.info(f"Transcode complete: {input_path} -> {output_path}")


def _split_options(
    split_large_gaussians: bool,
    split_target_size: Optional[float],
    split_target_fraction: float,
    max_splits: int,
) -> dict:
    """Bundle the Gaussian-split options forwarded to partition_scene."""
    return {
        "split": split_large_gaussians,
        "split_target_size": split_target_size,
        "split_target_fraction": split_target_fraction,
        "max_splits": max_splits,
    }


def _load_sources(importer, path: Path):
    """Load a file as a list of ``(attrs, caps)`` fields plus their source transforms.

    Each ParticleField prim / source is kept separate — never concatenated. PLY/NuRec yield a
    single field; a multi-prim USD yields one field per prim.
    """
    if hasattr(importer, "load_fields"):
        fields = importer.load_fields(path)
        transforms = getattr(importer, "source_gaussian_transforms", None)
        if not transforms or len(transforms) != len(fields):
            transforms = [getattr(importer, "source_gaussian_transform", None)] * len(fields)
        return fields, transforms
    attrs, caps = importer.load(path)
    return [(attrs, caps)], [getattr(importer, "source_gaussian_transform", None)]


def transcode_files(
    input_paths,
    output_path: Path,
    output_format: str,
    max_sh_degree: int = 3,
    half_precision: bool = False,
    half_geometry: bool = False,
    half_features: bool = False,
    render_order_hint: Optional[str] = None,
    validate_usd: bool = True,
    max_per_volume: Optional[int] = None,
    split_large_gaussians: bool = False,
    split_target_size: Optional[float] = None,
    split_target_fraction: float = 0.5,
    max_splits: int = 4,
) -> None:
    """Combine several inputs into one asset with one ParticleField prim / volume per input.

    Each input field becomes its own output partition; a field whose Gaussian count exceeds
    ``max_per_volume`` is subdivided into several (optionally splitting oversized Gaussians first).
    Inputs are never merged into one array. Output must be ``lightfield`` or ``nurec``; all inputs
    must agree on activation convention (e.g. all PLY).
    """
    input_paths = [Path(p) for p in input_paths]
    if len(input_paths) == 1:
        transcode(
            input_paths[0],
            output_path,
            output_format,
            max_sh_degree=max_sh_degree,
            half_precision=half_precision,
            half_geometry=half_geometry,
            half_features=half_features,
            render_order_hint=render_order_hint,
            validate_usd=validate_usd,
            max_per_volume=max_per_volume,
            split_large_gaussians=split_large_gaussians,
            split_target_size=split_target_size,
            split_target_fraction=split_target_fraction,
            max_splits=max_splits,
        )
        return

    if output_format not in ("lightfield", "nurec"):
        raise ValueError(
            f"Combining multiple inputs into one asset requires lightfield or nurec output, got '{output_format}'."
        )

    fields = []
    transforms = []
    preactivation_flags = set()
    for p in input_paths:
        fmt = detect_input_format(p)
        importer = get_importer(fmt, max_sh_degree)
        fs, ts = _load_sources(importer, p)
        fields.extend(fs)
        transforms.extend(ts)
        preactivation_flags.add(importer.stores_preactivation)
        logger.info(f"Loaded {len(fs)} field(s) from {p} ({fmt})")

    if len(preactivation_flags) != 1:
        raise ValueError("All inputs must share the same activation convention (e.g. all PLY).")
    source_is_preactivation = preactivation_flags.pop()
    logger.info(f"Combining {len(input_paths)} inputs ({len(fields)} fields) into {output_path}")

    _transcode_core(
        fields=fields,
        source_transforms=transforms,
        source_is_preactivation=source_is_preactivation,
        output_path=output_path,
        output_format=output_format,
        half_precision=half_precision,
        half_geometry=half_geometry,
        half_features=half_features,
        apply_coordinate_transform=False,
        render_order_hint=render_order_hint,
        copy_cameras_source=None,
        copy_source_skip_subtrees=None,
        validate_usd=validate_usd,
        max_per_volume=max_per_volume,
        split_options=_split_options(split_large_gaussians, split_target_size, split_target_fraction, max_splits),
    )
    logger.info(f"Transcode complete: {len(input_paths)} inputs -> {output_path}")


def _transcode_core(
    *,
    fields,
    source_transforms,
    source_is_preactivation: bool,
    output_path: Path,
    output_format: str,
    half_precision: bool,
    half_geometry: bool,
    half_features: bool,
    apply_coordinate_transform: bool,
    render_order_hint: Optional[str],
    copy_cameras_source,
    copy_source_skip_subtrees,
    validate_usd: bool,
    max_per_volume: Optional[int],
    split_options: Optional[dict] = None,
) -> None:
    """Partition each source independently and author the union of partitions.

    Each field becomes its own :class:`PartitionResult` (one per source, never merged); a source
    over budget subdivides into several partitions. The writers author one ParticleField prim / PLY
    file / NuRec volume per partition.
    """
    from threedgrut.export.partition import partition_scene

    if half_precision:
        half_geometry = True
        half_features = True

    if output_format == "nurec" and output_path.suffix.lower() != ".usdz":
        raise ValueError(
            f"NuRec format requires output extension .usdz, got '{output_path.suffix}'. "
            f"Use e.g. -o {output_path.with_suffix('.usdz')}"
        )

    adapters = [
        AttributesExportAdapter(attrs=a, caps=c, is_preactivation=source_is_preactivation) for a, c in fields
    ]
    results = [partition_scene(adapter, max_per_volume, **(split_options or {})) for adapter in adapters]
    total_partitions = sum(r.num_partitions for r in results)
    partitioned = total_partitions > 1

    # PLY: one file per partition (handles the single-partition case too).
    if output_format == "ply":
        from threedgrut.export.formats import export_partitions

        written = export_partitions(results, output_path)
        logger.info("Wrote %d PLY file(s)", len(written))
        return

    # USD / NuRec: author one prim / volume per partition. The exporter's source-prim merge still
    # copies cameras, RenderProducts and every other prim into the target as-is.
    exporter, _target_expects_preactivation = get_exporter(
        output_format,
        half_precision=half_precision,
        half_geometry=half_geometry,
        half_features=half_features,
        render_order_hint=render_order_hint if output_format == "lightfield" else None,
    )
    export_kwargs = dict(
        apply_coordinate_transform=apply_coordinate_transform,
        copy_cameras_source=copy_cameras_source,
        copy_source_skip_subtrees=copy_source_skip_subtrees,
        source_gaussian_transform=source_transforms[0] if source_transforms else None,
        validate_usd=validate_usd if output_format == "lightfield" else False,
        partition=results if partitioned else None,
    )
    logger.info(f"Exporting to {output_path}...")
    exporter.export(adapters[0], output_path, **export_kwargs)


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
        nargs="+",
        help=(
            "Input file path(s) (ply, usd, usda, usdc, usdz). Pass several files to combine them "
            "into one lightfield USD with one ParticleField prim per input."
        ),
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
        choices=PARTICLE_FIELD_SORTING_MODE_HINTS,
        default=None,
        metavar="MODE",
        help=(
            "Force sortingModeHint for lightfield export "
            "(zDepth, cameraDistance, rayHitDistance). Ignored with --format ply/nurec (warning only)."
        ),
    )
    parser.add_argument(
        "--no-copy-source-prims",
        action="store_true",
        dest="no_copy_source_prims",
        help="When input and output are USD flavors, do not merge source /World and /Render prims into the target.",
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
        "--max-particles-per-field",
        dest="max_per_volume",
        type=int,
        default=None,
        help=(
            "Subdivide any ParticleField prim / input whose own particle count exceeds this into "
            "several partitions; prims within budget are kept as-is (ply and lightfield outputs only)."
        ),
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

    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)

    # Validate inputs exist
    for p in input_paths:
        if not p.exists():
            logger.error(f"Input file does not exist: {p}")
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

    try:
        if len(input_paths) > 1:
            # Combine several inputs into one multi-ParticleField USD (one prim per input).
            transcode_files(
                input_paths=input_paths,
                output_path=output_path,
                output_format=output_format,
                max_sh_degree=args.max_sh_degree,
                half_precision=args.half,
                half_geometry=args.half_geometry,
                half_features=args.half_features,
                render_order_hint=args.render_order_hint,
                validate_usd=not args.no_usd_validate,
                max_per_volume=args.max_per_volume,
            )
        else:
            input_path = input_paths[0]
            suffix_in = input_path.suffix.lower()
            use_camera_copy_ctx = (
                output_format in {"lightfield", "nurec"}
                and suffix_in in (".usd", ".usda", ".usdc", ".usdz")
                and not args.no_copy_source_prims
            )
            camera_ctx = (
                usd_stage_path_context_for_camera_copy(input_path) if use_camera_copy_ctx else nullcontext(None)
            )
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
                    copy_cameras_source=copy_cameras_source,
                    copy_source_skip_subtrees=skip_subtrees,
                    validate_usd=not args.no_usd_validate,
                    max_per_volume=args.max_per_volume,
                )
    except Exception as e:
        logger.error(f"Transcode failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
