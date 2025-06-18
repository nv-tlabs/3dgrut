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

import abc
from pathlib import Path

import torch


class ExportableModel(abc.ABC):
    """Abstract base class defining the interface for models that can be exported.

    This class defines the required getter methods that a model must implement
    to be compatible with the export system.
    """

    @abc.abstractmethod
    def get_positions(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_max_n_features(self) -> int:
        pass

    @abc.abstractmethod
    def get_n_active_features(self) -> int:
        pass

    @abc.abstractmethod
    def get_scale(self, preactivation: bool = False) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_rotation(self, preactivation: bool = False) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_density(self, preactivation: bool = False) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_features_albedo(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_features_specular(self) -> torch.Tensor:
        pass


class ModelExporter(abc.ABC):
    """Abstract base class for model exporters.

    This class defines the interface that specific exporters (PLY, INGP, USDZ, etc.)
    must implement to export models to their respective formats.
    """

    @abc.abstractmethod
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, **kwargs) -> None:
        pass
