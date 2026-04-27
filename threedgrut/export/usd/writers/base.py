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
Abstract base class for USD Gaussian prim writers.

Provides a schema-agnostic interface for writing Gaussian splatting data to USD.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pxr import Gf, Usd, Vt

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities

logger = logging.getLogger(__name__)


class GaussianUSDWriter(ABC):
    """Abstract base class for USD Gaussian prim writers.

    Writers are responsible for creating and populating USD prims with Gaussian
    splatting data. Different implementations can use different USD schemas while
    maintaining a common interface.

    Args:
        stage: USD stage to write prims to
        capabilities: Model capabilities descriptor
        content_root_path: Root path for content in USD stage
    """

    def __init__(
        self,
        stage: Usd.Stage,
        capabilities: ModelCapabilities,
        content_root_path: str = "/World/Gaussians",
        linear_srgb: bool = False,
        omni_usd: bool = False,
        has_post_processing: bool = False,
    ):
        self.stage = stage
        self.capabilities = capabilities
        self.content_root_path = content_root_path
        self.linear_srgb = linear_srgb
        self.omni_usd = omni_usd
        self.has_post_processing = has_post_processing
        self.prim: Optional[Usd.Prim] = None

    def apply_color_space_to_prim(self, prim: Usd.Prim) -> None:
        """Apply ColorSpaceAPI and set color space based on linear_srgb flag.

        Per USD color space conventions:
        - lin_rec709_scene: Linear Rec.709 (post-processed/linear RGB data)
        - srgb_rec709_display: sRGB Rec.709 (gamma-encoded data)
        """
        color_space = "lin_rec709_scene" if self.linear_srgb else "srgb_rec709_display"
        color_space_api = Usd.ColorSpaceAPI.Apply(prim)
        color_space_api.CreateColorSpaceNameAttr().Set(color_space)

    @abstractmethod
    def create_prim(self, num_gaussians: int) -> Usd.Prim:
        """Create the USD prim for Gaussian data.

        Args:
            num_gaussians: Number of gaussians to create prim for

        Returns:
            Created USD prim
        """
        pass

    @abstractmethod
    def write_attributes(
        self,
        attributes: GaussianAttributes,
        force_sh_0: bool = False,
    ) -> None:
        """Write Gaussian attributes to the prim.

        Args:
            attributes: Gaussian attributes to write
            force_sh_0: Force SH degree to 0 (skip f_rest coefficients)
        """
        pass

    @abstractmethod
    def finalize(self, positions: np.ndarray) -> None:
        """Finalize the prim after all attributes are written.

        Args:
            positions: Position array for extent computation
        """
        pass

    def compute_extent(self, positions: np.ndarray) -> Vt.Vec3fArray:
        """Compute bounding box extent from positions.

        Args:
            positions: Position array [N, 3]

        Returns:
            Vt.Vec3fArray with [min, max] bounds
        """
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        return Vt.Vec3fArray(
            [
                Gf.Vec3f(float(min_bounds[0]), float(min_bounds[1]), float(min_bounds[2])),
                Gf.Vec3f(float(max_bounds[0]), float(max_bounds[1]), float(max_bounds[2])),
            ]
        )


def create_gaussian_writer(
    stage: Usd.Stage,
    capabilities: ModelCapabilities,
    content_root_path: str = "/World/Gaussians",
    half_geometry: bool = False,
    half_features: bool = False,
    sorting_mode_hint: str = "cameraDistance",
    linear_srgb: bool = False,
    omni_usd: bool = False,
    has_post_processing: bool = False,
) -> GaussianUSDWriter:
    """Factory function to create USD Gaussian writer.

    Args:
        stage: USD stage to write to
        capabilities: Model capabilities descriptor
        content_root_path: Root path for content
        half_geometry: Use half precision for positions, orientations, scales (LightField)
        half_features: Use half precision for opacities and SH coefficients (LightField)
        sorting_mode_hint: Sorting mode hint for LightField schema
        linear_srgb: If True, set prim color space to lin_rec709_scene; else srgb_rec709_display
        omni_usd: If True, author Omniverse-specific USD features.
        has_post_processing: If True, configure Omniverse material for external post-processing.

    Returns:
        Configured GaussianUSDWriter instance (LightField schema)
    """
    from threedgrut.export.usd.writers.lightfield import GaussianLightFieldWriter

    return GaussianLightFieldWriter(
        stage=stage,
        capabilities=capabilities,
        content_root_path=content_root_path,
        half_geometry=half_geometry,
        half_features=half_features,
        sorting_mode_hint=sorting_mode_hint,
        linear_srgb=linear_srgb,
        omni_usd=omni_usd,
        has_post_processing=has_post_processing,
    )
