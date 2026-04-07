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
Format importers for Gaussian splatting data.

Provides importers that load various formats into the intermediate
GaussianAttributes representation for transcoding.
"""

from threedgrut.export.importers.base import FormatImporter
from threedgrut.export.importers.ply import PLYImporter

__all__ = [
    "FormatImporter",
    "PLYImporter",
]

try:
    from threedgrut.export.importers.nurec_usd import NuRecUSDImporter
    from threedgrut.export.importers.usd import USDImporter

    __all__ += [
        "NuRecUSDImporter",
        "USDImporter",
    ]
except ImportError as e:
    import warnings

    warnings.warn(f"USD importers unavailable: {e}", ImportWarning, stacklevel=2)
