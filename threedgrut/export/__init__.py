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
Export module for 3DGRUT Gaussian Splatting models.

Provides exporters and importers for various formats:
- PLY: Standard point cloud format
- USD: OpenUSD with ParticleField3DGaussianSplat schema (new default)
- NuRec: Omniverse-compatible USDZ format (legacy)

Transcoding:
    Use the transcode script to convert between formats:
    >>> python -m threedgrut.export.scripts.transcode input.ply -o output.usdz

Programmatic usage:
    >>> from threedgrut.export import PLYImporter, USDExporter, AttributesExportAdapter
    >>> importer = PLYImporter()
    >>> attrs, caps = importer.load("input.ply")
    >>> adapter = AttributesExportAdapter(attrs, caps, is_preactivation=True)
    >>> exporter = USDExporter()
    >>> exporter.export(adapter, "output.usdz")
"""

# Data accessor and filtering
from threedgrut.export.accessor import (
    ExportFilterSettings,
    GaussianAttributes,
    GaussianExportAccessor,
    ModelCapabilities,
    filter_gaussians,
)

# Export adapter for transcoding
from threedgrut.export.adapter import AttributesExportAdapter

# Core interfaces
from threedgrut.export.base import ExportableModel, ModelExporter

# Format-specific exporters
from threedgrut.export.formats.ply import PLYExporter

# Format importers
from threedgrut.export.importers.base import FormatImporter
from threedgrut.export.importers.ply import PLYImporter
from threedgrut.export.importers.usd import USDImporter

# Visibility filtering
from threedgrut.export.scripts.filter_visibility import (
    compute_average_visibility,
    compute_visibility_and_filter,
)

# Transform utilities
from threedgrut.export.transforms import estimate_normalizing_transform

# USD exporters
from threedgrut.export.usd.exporter import USDExporter
from threedgrut.export.usd.nurec.exporter import NuRecExporter

__all__ = [
    # Core interfaces
    "ExportableModel",
    "ModelExporter",
    # Data accessor and filtering
    "GaussianExportAccessor",
    "GaussianAttributes",
    "ModelCapabilities",
    "ExportFilterSettings",
    "filter_gaussians",
    # Transforms
    "estimate_normalizing_transform",
    # Format exporters
    "PLYExporter",
    "USDExporter",  # ParticleField3DGaussianSplat schema
    "NuRecExporter",  # Omniverse-compatible format
    # Format importers
    "FormatImporter",
    "PLYImporter",
    "USDImporter",
    # Transcoding adapter
    "AttributesExportAdapter",
    # Visibility filtering
    "compute_average_visibility",
    "compute_visibility_and_filter",
]
