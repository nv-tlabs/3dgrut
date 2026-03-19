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
Gaussian Export Accessor

Provides a unified interface for extracting Gaussian splatting data from models
for export to various formats (PLY, USD, etc.).

This module defines:
- GaussianAttributes: Dataclass holding all Gaussian attribute arrays
- ModelCapabilities: Dataclass describing model features
- GaussianExportAccessor: Unified accessor for ExportableModel interface
- ExportFilterSettings: Configuration for export filtering
- filter_gaussians: Utility to apply filters and log results
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from threedgrut.export.base import ExportableModel

logger = logging.getLogger(__name__)


@dataclass
class GaussianAttributes:
    """
    Gaussian attribute data ready for export.

    All arrays are NumPy arrays on CPU. Rotations are stored as wxyz quaternions.

    Attributes:
        positions: Gaussian centers [N, 3]
        rotations: Orientation quaternions [N, 4] in wxyz order
        scales: Anisotropic scales [N, 3]
        densities: Opacity/density values [N, 1]
        albedo: SH DC term (degree 0) [N, 3]
        specular: Higher-order SH coefficients [N, M] where M = (degree+1)^2 - 1) * 3
    """

    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    densities: np.ndarray
    albedo: np.ndarray
    specular: np.ndarray

    def __post_init__(self):
        """Validate array shapes."""
        n = self.positions.shape[0]
        assert self.positions.shape == (n, 3), f"positions shape mismatch: {self.positions.shape}"
        assert self.rotations.shape == (n, 4), f"rotations shape mismatch: {self.rotations.shape}"
        assert self.scales.shape == (n, 3), f"scales shape mismatch: {self.scales.shape}"
        assert self.densities.shape[0] == n, f"densities count mismatch: {self.densities.shape[0]} vs {n}"
        assert self.albedo.shape == (n, 3), f"albedo shape mismatch: {self.albedo.shape}"
        assert self.specular.shape[0] == n, f"specular count mismatch: {self.specular.shape[0]} vs {n}"

    @property
    def num_gaussians(self) -> int:
        """Return the number of Gaussians."""
        return self.positions.shape[0]

    def filter_by_mask(self, mask: np.ndarray) -> "GaussianAttributes":
        """
        Filter Gaussians by a boolean mask.

        Args:
            mask: Boolean array [N] where True means keep the Gaussian

        Returns:
            New GaussianAttributes with filtered data
        """
        return GaussianAttributes(
            positions=self.positions[mask],
            rotations=self.rotations[mask],
            scales=self.scales[mask],
            densities=self.densities[mask],
            albedo=self.albedo[mask],
            specular=self.specular[mask],
        )

    def get_valid_mask(self) -> np.ndarray:
        """
        Get a mask of Gaussians with valid (non-NaN/Inf) values.

        Returns:
            Boolean mask [N] where True means the Gaussian is valid
        """
        valid_positions = np.all(np.isfinite(self.positions), axis=1)
        valid_rotations = np.all(np.isfinite(self.rotations), axis=1)
        valid_scales = np.all(np.isfinite(self.scales), axis=1)
        valid_densities = np.all(np.isfinite(self.densities.reshape(-1, 1)), axis=1)
        valid_albedo = np.all(np.isfinite(self.albedo), axis=1)
        valid_specular = np.all(np.isfinite(self.specular), axis=1)

        return valid_positions & valid_rotations & valid_scales & valid_densities & valid_albedo & valid_specular

    def get_low_opacity_mask(self, threshold: float = 1e-6) -> np.ndarray:
        """
        Get a mask of Gaussians with opacity above threshold.

        Args:
            threshold: Minimum opacity value (after activation). Default 1e-6.

        Returns:
            Boolean mask [N] where True means opacity >= threshold
        """
        opacities = self.densities.flatten()
        return opacities >= threshold

    def get_visibility_mask(self, visibility: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Get a mask of Gaussians with visibility above threshold.

        Args:
            visibility: Per-Gaussian visibility values [N] (average visibility across views)
            threshold: Minimum visibility value. Default 0.0 (no filtering).

        Returns:
            Boolean mask [N] where True means visibility >= threshold
        """
        return visibility >= threshold


@dataclass
class ExportFilterSettings:
    """
    Configuration for Gaussian filtering during export.

    Attributes:
        filter_low_opacity: Enable low opacity filtering
        opacity_threshold: Minimum opacity value (after activation)
        filter_invalid: Enable filtering of NaN/Inf values
        visibility_threshold: Minimum average visibility (0.0 = disabled)
    """

    filter_low_opacity: bool = True
    opacity_threshold: float = 1e-6

    filter_invalid: bool = True

    visibility_threshold: float = 0.0


def filter_gaussians(
    attrs: GaussianAttributes,
    settings: Optional[ExportFilterSettings] = None,
    visibility: Optional[np.ndarray] = None,
) -> Tuple[GaussianAttributes, dict]:
    """
    Apply export filters to Gaussian attributes.

    Args:
        attrs: Input Gaussian attributes
        settings: Filter settings (uses defaults if None)
        visibility: Per-Gaussian average visibility [N] (optional, needed for visibility filtering)

    Returns:
        Tuple of (filtered GaussianAttributes, stats dict with filter counts)
    """
    if settings is None:
        settings = ExportFilterSettings()

    initial_count = attrs.num_gaussians
    stats = {"initial": initial_count}

    # Start with all True mask
    combined_mask = np.ones(initial_count, dtype=bool)

    # Filter invalid values (NaN/Inf)
    if settings.filter_invalid:
        valid_mask = attrs.get_valid_mask()
        invalid_count = initial_count - np.sum(valid_mask)
        if invalid_count > 0:
            logger.info(f"Filtered {invalid_count} invalid (NaN/Inf) Gaussians")
        stats["invalid"] = invalid_count
        combined_mask &= valid_mask

    # Filter low opacity
    if settings.filter_low_opacity:
        opacity_mask = attrs.get_low_opacity_mask(settings.opacity_threshold)
        low_opacity_count = np.sum(combined_mask) - np.sum(combined_mask & opacity_mask)
        if low_opacity_count > 0:
            logger.info(
                f"Filtered {low_opacity_count} low-opacity Gaussians " f"(threshold={settings.opacity_threshold:.1e})"
            )
        stats["low_opacity"] = low_opacity_count
        combined_mask &= opacity_mask

    # Filter by visibility threshold (only if threshold > 0 and visibility provided)
    if settings.visibility_threshold > 0.0 and visibility is not None:
        if len(visibility) != initial_count:
            logger.warning(
                f"Visibility array length ({len(visibility)}) does not match "
                f"number of Gaussians ({initial_count}). Skipping visibility filtering."
            )
            stats["low_visibility"] = 0
        else:
            visibility_mask = attrs.get_visibility_mask(visibility, settings.visibility_threshold)
            low_visibility_count = np.sum(combined_mask) - np.sum(combined_mask & visibility_mask)
            if low_visibility_count > 0:
                logger.info(
                    f"Filtered {low_visibility_count} low-visibility Gaussians "
                    f"(threshold={settings.visibility_threshold:.4f})"
                )
            stats["low_visibility"] = low_visibility_count
            combined_mask &= visibility_mask
    else:
        stats["low_visibility"] = 0

    # Apply combined mask
    filtered_attrs = attrs.filter_by_mask(combined_mask)
    final_count = filtered_attrs.num_gaussians
    total_filtered = initial_count - final_count

    stats["final"] = final_count
    stats["total_filtered"] = total_filtered

    if total_filtered > 0:
        logger.info(
            f"Export filtering: {initial_count} -> {final_count} Gaussians "
            f"({total_filtered} removed, {100 * total_filtered / initial_count:.2f}%)"
        )

    return filtered_attrs, stats


@dataclass
class ModelCapabilities:
    """
    Model feature flags for exporters.

    Describes what features the model supports, allowing exporters
    to adapt their output accordingly.

    Attributes:
        has_spherical_harmonics: Whether model uses SH for view-dependent color
        sh_degree: Maximum SH degree (0-3 typically)
        num_gaussians: Total number of Gaussians in the model
        is_surfel: Whether model uses 2D Gaussian surfels (trisurfel primitives)
        density_activation: Name of density activation function
        scale_activation: Name of scale activation function
    """

    has_spherical_harmonics: bool
    sh_degree: int
    num_gaussians: int
    is_surfel: bool = False
    density_activation: str = "sigmoid"
    scale_activation: str = "exp"


class GaussianExportAccessor:
    """
    Unified accessor for extracting Gaussian data from ExportableModel.

    Provides a clean interface for all exporters to access model data
    without duplicating extraction logic.

    Example:
        >>> accessor = GaussianExportAccessor(model, conf)
        >>> attrs = accessor.get_attributes(preactivation=True)
        >>> print(f"Exporting {attrs.num_gaussians} Gaussians")
    """

    def __init__(self, model: ExportableModel, conf=None):
        """
        Initialize the accessor.

        Args:
            model: Model implementing ExportableModel interface
            conf: Optional configuration object (for activation functions, etc.)
        """
        self.model = model
        self.conf = conf

    def get_capabilities(self) -> ModelCapabilities:
        """
        Get model capabilities.

        Returns:
            ModelCapabilities describing model features
        """
        # Get SH degree from model
        sh_degree = self.model.get_max_n_features()
        n_active = self.model.get_n_active_features()

        # Get activation function names from config if available
        density_activation = "sigmoid"
        scale_activation = "exp"
        is_surfel = False
        if self.conf is not None:
            density_activation = getattr(self.conf.model, "density_activation", "sigmoid")
            scale_activation = getattr(self.conf.model, "scale_activation", "exp")
            primitive_type = getattr(self.conf.render, "primitive_type", "")
            is_surfel = primitive_type == "trisurfel"

        return ModelCapabilities(
            has_spherical_harmonics=True,  # 3DGRUT always uses SH
            sh_degree=n_active,
            num_gaussians=self.model.get_positions().shape[0],
            is_surfel=is_surfel,
            density_activation=density_activation,
            scale_activation=scale_activation,
        )

    @torch.no_grad()
    def get_attributes(self, preactivation: bool = False) -> GaussianAttributes:
        """
        Extract all Gaussian attributes from the model.

        Args:
            preactivation: If True, return pre-activation values for density/scale.
                          If False, return post-activation values.

        Returns:
            GaussianAttributes containing all Gaussian data as NumPy arrays
        """
        # Extract tensors from model
        positions = self.model.get_positions().detach().cpu().numpy()
        rotations = self.model.get_rotation(preactivation=preactivation).detach().cpu().numpy()
        scales = self.model.get_scale(preactivation=preactivation).detach().cpu().numpy()
        densities = self.model.get_density(preactivation=preactivation).detach().cpu().numpy()
        albedo = self.model.get_features_albedo().detach().cpu().numpy()
        specular = self.model.get_features_specular().detach().cpu().numpy()

        # Ensure densities has correct shape [N, 1]
        if densities.ndim == 1:
            densities = densities[:, np.newaxis]

        return GaussianAttributes(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            albedo=albedo,
            specular=specular,
        )

    def get_num_gaussians(self) -> int:
        """Get the number of Gaussians in the model."""
        return self.model.get_positions().shape[0]

    def get_sh_degree(self) -> int:
        """Get the active SH degree."""
        return self.model.get_n_active_features()

    def get_max_sh_degree(self) -> int:
        """Get the maximum SH degree."""
        return self.model.get_max_n_features()
