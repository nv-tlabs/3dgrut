# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import gui

UNKNOWN_VERSION = "0.0.0+unknown"


def _read_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject_path.exists():
        with pyproject_path.open("rb") as f:
            return tomllib.load(f)["project"]["version"]

    try:
        return version("threedgrut")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = _read_version()

__all__ = ["gui", "__version__"]
