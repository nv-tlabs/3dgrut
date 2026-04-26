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
PPISP SPG shader assets for USD RenderProduct post-processing.

Provides loader for the three SPG sidecar files (Slang shader, Lua launcher,
USDA definition) that must be packaged alongside the exported USDZ.
"""

import logging
from pathlib import Path
from typing import List

from threedgrut.export.usd.stage_utils import NamedSerialized

log = logging.getLogger(__name__)

_SPG_DIR = Path(__file__).parent
_SPG_FILES = [
    "ppisp_usd_spg.slang",
    "ppisp_usd_spg.slang.lua",
    "ppisp_usd_spg.slang.usda",
]


def get_ppisp_spg_files() -> List[NamedSerialized]:
    """Load all PPISP SPG sidecar files as serialized data for USDZ packaging.

    Returns:
        List of NamedSerialized for each SPG file (slang, lua, usda).
    """
    result: List[NamedSerialized] = []
    for filename in _SPG_FILES:
        path = _SPG_DIR / filename
        if path.exists():
            result.append(NamedSerialized(filename=filename, serialized=path.read_bytes()))
            log.debug(f"Loaded PPISP SPG sidecar: {filename}")
        else:
            log.warning(f"PPISP SPG sidecar not found: {path}")
    return result
