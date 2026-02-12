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
Export adapter for GaussianAttributes.

Provides an adapter that wraps GaussianAttributes as an ExportableModel,
allowing existing exporters to work with imported data.
"""

import numpy as np
import torch

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.base import ExportableModel


class AttributesExportAdapter(ExportableModel):
    """Wraps GaussianAttributes as ExportableModel for use with existing exporters.

    This adapter enables transcoding by providing the ExportableModel interface
    expected by exporters, backed by GaussianAttributes data from importers.

    Args:
        attrs: GaussianAttributes containing Gaussian data
        caps: ModelCapabilities describing the data
        is_preactivation: Whether the data is in pre-activation format.
            Pre-activation: raw parameters (PLY export)
            Post-activation: sigmoid/exp applied (LightField export)
        device: Torch device for tensor operations (default: cuda if available)

    Note on activation handling:
        - When is_preactivation=True, get_density(preactivation=False) applies sigmoid
        - When is_preactivation=False, get_density(preactivation=True) applies inverse sigmoid
        - Same logic applies to scale with exp/log
    """

    def __init__(
        self,
        attrs: GaussianAttributes,
        caps: ModelCapabilities,
        is_preactivation: bool = True,
        device: str = None,
    ):
        self._attrs = attrs
        self._caps = caps
        self._is_preactivation = is_preactivation
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to torch tensors
        self._positions = torch.from_numpy(attrs.positions).to(self._device)
        self._rotations = torch.from_numpy(attrs.rotations).to(self._device)
        self._scales = torch.from_numpy(attrs.scales).to(self._device)
        self._densities = torch.from_numpy(attrs.densities).to(self._device)
        self._albedo = torch.from_numpy(attrs.albedo).to(self._device)
        self._specular = torch.from_numpy(attrs.specular).to(self._device)

    def get_positions(self) -> torch.Tensor:
        """Return Gaussian center positions [N, 3]."""
        return self._positions

    def get_max_n_features(self) -> int:
        """Return maximum SH degree."""
        return self._caps.sh_degree

    def get_n_active_features(self) -> int:
        """Return active SH degree."""
        return self._caps.sh_degree

    def get_scale(self, preactivation: bool = False) -> torch.Tensor:
        """Return Gaussian scales [N, 3].

        Args:
            preactivation: If True, return raw scale parameters.
                          If False, return activated scales (exp applied).
        """
        if self._is_preactivation:
            # Source data is pre-activation
            if preactivation:
                return self._scales
            else:
                # Apply exp activation
                return torch.exp(self._scales)
        else:
            # Source data is post-activation
            if preactivation:
                # Apply inverse (log) to get raw parameters
                return torch.log(torch.clamp(self._scales, min=1e-8))
            else:
                return self._scales

    def get_rotation(self, preactivation: bool = False) -> torch.Tensor:
        """Return Gaussian rotations as quaternions [N, 4] in wxyz order.

        Rotations are typically stored normalized, so preactivation flag
        doesn't affect them significantly (just normalization).
        """
        if preactivation:
            return self._rotations
        else:
            # Normalize quaternions
            return torch.nn.functional.normalize(self._rotations, dim=1)

    def get_density(self, preactivation: bool = False) -> torch.Tensor:
        """Return Gaussian densities/opacities [N, 1].

        Args:
            preactivation: If True, return raw density parameters (logit).
                          If False, return activated densities (sigmoid applied).
        """
        if self._is_preactivation:
            # Source data is pre-activation (logit values)
            if preactivation:
                return self._densities
            else:
                # Apply sigmoid activation
                return torch.sigmoid(self._densities)
        else:
            # Source data is post-activation (probability values in [0,1])
            if preactivation:
                # Apply inverse sigmoid (logit) to get raw parameters
                clamped = torch.clamp(self._densities, min=1e-7, max=1.0 - 1e-7)
                return torch.log(clamped / (1.0 - clamped))
            else:
                return self._densities

    def get_features_albedo(self) -> torch.Tensor:
        """Return SH DC coefficients (albedo) [N, 3]."""
        return self._albedo

    def get_features_specular(self) -> torch.Tensor:
        """Return higher-order SH coefficients [N, M*3]."""
        return self._specular

    @property
    def is_preactivation(self) -> bool:
        """Whether the underlying data is in pre-activation format."""
        return self._is_preactivation

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return self._caps
