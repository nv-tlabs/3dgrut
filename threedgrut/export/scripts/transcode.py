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
- NuRec USDZ (Omniverse format)

Usage:
    python -m threedgrut.export.scripts.transcode input.ply -o output.usdz --format lightfield
    python -m threedgrut.export.scripts.transcode input.usdz -o output.ply
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from threedgrut.export.adapter import AttributesExportAdapter
from threedgrut.export.importers import PLYImporter, USDImporter, FormatImporter
from threedgrut.export.formats import PLYExporter
from threedgrut.export.usd.exporter import USDExporter
from threedgrut.export.usd.nurec.exporter import NuRecExporter
from threedgrut.export.base import ModelExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Supported output formats
OUTPUT_FORMATS = {
    "ply": "PLY point cloud format (pre-activation values)",
    "lightfield": "USD ParticleField3DGaussianSplat schema (post-activation values)",
    "nurec": "NuRec USDZ format for Omniverse",
}


def detect_input_format(path: Path) -> str:
    """Detect input format from file extension.

    Args:
        path: Input file path

    Returns:
        Format string: 'ply' or 'usd'
    """
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return "ply"
    elif suffix in [".usd", ".usda", ".usdc", ".usdz"]:
        return "usd"
    else:
        raise ValueError(f"Unknown input format for extension: {suffix}")


def get_importer(format_name: str, max_sh_degree: int = 3) -> FormatImporter:
    """Get importer for the specified format.

    Args:
        format_name: Format name ('ply', 'usd')
        max_sh_degree: Maximum SH degree for PLY importer

    Returns:
        FormatImporter instance
    """
    if format_name == "ply":
        return PLYImporter(max_sh_degree=max_sh_degree)
    elif format_name == "usd":
        return USDImporter()
    else:
        raise ValueError(f"Unknown input format: {format_name}")


def get_exporter(format_name: str, half_precision: bool = False) -> Tuple[ModelExporter, bool]:
    """Get exporter for the specified format.

    Args:
        format_name: Format name ('ply', 'lightfield', 'nurec')
        half_precision: Use half precision for supported formats

    Returns:
        Tuple of (ModelExporter instance, expects_preactivation)
    """
    if format_name == "ply":
        return PLYExporter(), True
    elif format_name == "lightfield":
        return USDExporter(
            half_precision=half_precision,
            export_cameras=False,
            export_background=False,
            apply_normalizing_transform=False,
        ), False
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
) -> None:
    """Transcode between Gaussian splatting formats.

    Args:
        input_path: Path to input file
        output_path: Path for output file
        output_format: Target format name
        max_sh_degree: Maximum SH degree for PLY import
        half_precision: Use half precision for USD exports
    """
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
    exporter, target_expects_preactivation = get_exporter(output_format, half_precision)

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
    exporter.export(adapter, output_path)

    logger.info(f"Transcode complete: {input_path} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcode between Gaussian splatting export formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  Input:  ply, usd/usda/usdc/usdz
  Output: ply, lightfield, nurec

Examples:
  # Convert PLY to USD LightField
  python -m threedgrut.export.scripts.transcode model.ply -o model.usdz --format lightfield

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
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-f", "--format",
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
        help="Use half-precision (float16) for supported formats",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    try:
        transcode(
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            max_sh_degree=args.max_sh_degree,
            half_precision=args.half,
        )
    except Exception as e:
        logger.error(f"Transcode failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
