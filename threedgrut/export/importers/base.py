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

"""Base class for format importers."""

import abc
from pathlib import Path
from typing import Tuple

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities


class FormatImporter(abc.ABC):
    """Abstract base class for format importers.

    Importers load Gaussian splatting data from various file formats
    into the intermediate GaussianAttributes representation.
    """

    @abc.abstractmethod
    def load(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load format into intermediate representation.

        Args:
            path: Path to the input file

        Returns:
            Tuple of (GaussianAttributes, ModelCapabilities)
        """
        pass

    @property
    @abc.abstractmethod
    def stores_preactivation(self) -> bool:
        """Whether this format stores pre-activation values.

        Pre-activation: raw parameter values (PLY)
        Post-activation: activated values with sigmoid/exp applied (LightField)
        """
        pass
