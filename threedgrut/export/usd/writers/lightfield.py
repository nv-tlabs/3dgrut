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
USD Gaussian writer using UsdVol ParticleField schema.

Uses UsdVol.ParticleField3DGaussianSplat for 3DGS and UsdVol.ParticleField with
applied API schemas for 2DGS/surfels. Requires USD 26+ with ParticleField schema support.
"""

import logging
from typing import Optional

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol, Vt

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.usd.particle_field_hints import (
    DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
    normalize_particle_field_sorting_mode_hint,
)
from threedgrut.export.usd.writers.base import GaussianUSDWriter

logger = logging.getLogger(__name__)


class GaussianLightFieldWriter(GaussianUSDWriter):
    """USD Gaussian writer using ParticleField schemas.

    Uses either ParticleField3DGaussianSplat (3D ellipsoid) or ParticleField
    with ParticleFieldKernelGaussianSurfletAPI (2D surfel) based on model capabilities.

    Reference: https://github.com/PixarAnimationStudios/OpenUSD/blob/dev/pxr/usd/usdVol/schema.usda
    """

    def __init__(
        self,
        stage: Usd.Stage,
        capabilities: ModelCapabilities,
        content_root_path: str = "/World/Gaussians",
        half_geometry: bool = False,
        half_features: bool = False,
        projection_mode_hint: str = "perspective",
        sorting_mode_hint: str = DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
        linear_srgb: bool = False,
        omni_usd: bool = False,
        has_post_processing: bool = False,
    ) -> None:
        super().__init__(
            stage,
            capabilities,
            content_root_path,
            linear_srgb=linear_srgb,
            omni_usd=omni_usd,
            has_post_processing=has_post_processing,
        )
        self.half_geometry = half_geometry
        self.half_features = half_features
        self.projection_mode_hint = projection_mode_hint
        self.sorting_mode_hint = normalize_particle_field_sorting_mode_hint(sorting_mode_hint)

        # Use surflet kernel for surfel models, ellipsoid for 3DGS
        self.use_surflet_kernel = capabilities.is_surfel

        # Attribute handles (set in _create_attributes)
        self.positions_attr: Optional[Usd.Attribute] = None
        self.orientations_attr: Optional[Usd.Attribute] = None
        self.scales_attr: Optional[Usd.Attribute] = None
        self.opacities_attr: Optional[Usd.Attribute] = None
        self.sh_coeffs_attr: Optional[Usd.Attribute] = None
        self.sh_degree_attr: Optional[Usd.Attribute] = None
        # Schema wrapper for 3DGS (ParticleField3DGaussianSplat); None for surfel
        self._schema: Optional[UsdVol.ParticleField3DGaussianSplat] = None

    def create_prim(self, num_gaussians: int) -> Usd.Prim:
        """Create ParticleField prim with appropriate kernel schema."""
        prim_path = f"{self.content_root_path}/gaussians"

        if self.use_surflet_kernel:
            self.prim = UsdVol.ParticleField.Define(self.stage, prim_path).GetPrim()
            self._apply_surflet_kernel_schemas()
            logger.info(f"Created ParticleField with GaussianSurfletAPI (2DGS/surfel) at {prim_path}")
        else:
            self.prim = UsdVol.ParticleField3DGaussianSplat.Define(self.stage, prim_path).GetPrim()
            self._schema = UsdVol.ParticleField3DGaussianSplat(self.prim)
            logger.info(f"Created ParticleField3DGaussianSplat at {prim_path}")

        # Create attributes
        self._create_attributes()

        # Set rendering hints
        self._set_rendering_hints()

        self.apply_color_space_to_prim(self.prim)
        if self.omni_usd:
            from threedgrut.export.usd.writers.omni_material import bind_particlefield_emissive_material

            bind_particlefield_emissive_material(
                stage=self.stage,
                prim=self.prim,
                has_post_processing=self.has_post_processing,
            )
        return self.prim

    def _apply_surflet_kernel_schemas(self) -> None:
        """Apply API schemas for 2DGS/surfel particles via UsdVol schema types."""
        for api_schema in (
            UsdVol.ParticleFieldPositionAttributeAPI,
            UsdVol.ParticleFieldOrientationAttributeAPI,
            UsdVol.ParticleFieldScaleAttributeAPI,
            UsdVol.ParticleFieldOpacityAttributeAPI,
            UsdVol.ParticleFieldKernelGaussianSurfletAPI,
            UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI,
        ):
            self.prim.ApplyAPI(api_schema)

    def _create_attributes(self) -> None:
        """Create particle field attributes via UsdVol schema API.

        half_geometry: positions, orientations, scales use *h (half) attributes.
        half_features: opacities and SH coefficients use *h (half) attributes.
        """
        if self._schema is not None:
            # 3DGS: ParticleField3DGaussianSplat has all attributes
            self.positions_attr = (
                self._schema.CreatePositionshAttr() if self.half_geometry else self._schema.CreatePositionsAttr()
            )
            self.orientations_attr = (
                self._schema.CreateOrientationshAttr() if self.half_geometry else self._schema.CreateOrientationsAttr()
            )
            self.scales_attr = (
                self._schema.CreateScaleshAttr() if self.half_geometry else self._schema.CreateScalesAttr()
            )
            self.opacities_attr = (
                self._schema.CreateOpacitieshAttr() if self.half_features else self._schema.CreateOpacitiesAttr()
            )
            if self.capabilities.has_spherical_harmonics:
                self.sh_degree_attr = self._schema.CreateRadianceSphericalHarmonicsDegreeAttr()
                self.sh_coeffs_attr = (
                    self._schema.CreateRadianceSphericalHarmonicsCoefficientshAttr()
                    if self.half_features
                    else self._schema.CreateRadianceSphericalHarmonicsCoefficientsAttr()
                )
        else:
            # Surfel: use applied API schemas
            pos_api = UsdVol.ParticleFieldPositionAttributeAPI(self.prim)
            orient_api = UsdVol.ParticleFieldOrientationAttributeAPI(self.prim)
            scale_api = UsdVol.ParticleFieldScaleAttributeAPI(self.prim)
            opacity_api = UsdVol.ParticleFieldOpacityAttributeAPI(self.prim)
            self.positions_attr = (
                pos_api.CreatePositionshAttr() if self.half_geometry else pos_api.CreatePositionsAttr()
            )
            self.orientations_attr = (
                orient_api.CreateOrientationshAttr() if self.half_geometry else orient_api.CreateOrientationsAttr()
            )
            self.scales_attr = scale_api.CreateScaleshAttr() if self.half_geometry else scale_api.CreateScalesAttr()
            self.opacities_attr = (
                opacity_api.CreateOpacitieshAttr() if self.half_features else opacity_api.CreateOpacitiesAttr()
            )
            if self.capabilities.has_spherical_harmonics:
                rad_api = UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI(self.prim)
                self.sh_degree_attr = rad_api.CreateRadianceSphericalHarmonicsDegreeAttr()
                self.sh_coeffs_attr = (
                    rad_api.CreateRadianceSphericalHarmonicsCoefficientshAttr()
                    if self.half_features
                    else rad_api.CreateRadianceSphericalHarmonicsCoefficientsAttr()
                )
        if self.half_geometry or self.half_features:
            logger.info(
                "LightField precision: geometry=%s, features=%s",
                "half" if self.half_geometry else "float",
                "half" if self.half_features else "float",
            )

    def _set_rendering_hints(self) -> None:
        """Set rendering hints via schema API."""
        if self._schema is not None:
            self._schema.CreateProjectionModeHintAttr().Set(self.projection_mode_hint)
            self._schema.CreateSortingModeHintAttr().Set(self.sorting_mode_hint)
        else:
            # Surfel: prim is ParticleField; use base ParticleField schema for hint attrs
            field_schema = UsdVol.ParticleField(self.prim)
            field_schema.CreateProjectionModeHintAttr().Set(self.projection_mode_hint)
            field_schema.CreateSortingModeHintAttr().Set(self.sorting_mode_hint)

    def write_attributes(
        self,
        attributes: GaussianAttributes,
        force_sh_0: bool = False,
    ) -> None:
        """Write Gaussian attributes to ParticleField3DGaussianSplat."""
        if self.prim is None:
            raise RuntimeError("create_prim must be called before write_attributes")

        num_gaussians = attributes.num_gaussians

        # Positions (geometry)
        if self.half_geometry:
            self.positions_attr.Set(Vt.Vec3hArray.FromNumpy(attributes.positions.astype(np.float16)))
        else:
            self.positions_attr.Set(Vt.Vec3fArray.FromNumpy(attributes.positions.astype(np.float32)))

        # Orientations (geometry)
        if self.half_geometry:
            quats = [Gf.Quath(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in attributes.rotations]
            self.orientations_attr.Set(Vt.QuathArray(quats))
        else:
            quats = [Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in attributes.rotations]
            self.orientations_attr.Set(Vt.QuatfArray(quats))

        # Scales (geometry)
        if self.half_geometry:
            self.scales_attr.Set(Vt.Vec3hArray.FromNumpy(attributes.scales.astype(np.float16)))
        else:
            self.scales_attr.Set(Vt.Vec3fArray.FromNumpy(attributes.scales.astype(np.float32)))

        # Opacities (features)
        densities_clamped = np.clip(attributes.densities.flatten(), 0.0, 1.0)
        if self.half_features:
            self.opacities_attr.Set(Vt.HalfArray.FromNumpy(densities_clamped.astype(np.float16)))
        else:
            self.opacities_attr.Set(Vt.FloatArray.FromNumpy(densities_clamped.astype(np.float32)))

        # SH coefficients (features)
        if self.capabilities.has_spherical_harmonics and self.sh_coeffs_attr is not None:
            sh_degree = 0 if force_sh_0 else self.capabilities.sh_degree
            if self.sh_degree_attr is not None:
                self.sh_degree_attr.Set(sh_degree)

            if force_sh_0 or sh_degree == 0:
                all_coeffs_flat = attributes.albedo.reshape(-1, 3)
                num_sh_coeffs = 1
            else:
                num_sh_coeffs = (sh_degree + 1) ** 2
                num_rest_coeffs = num_sh_coeffs - 1
                specular_reshaped = attributes.specular.reshape((num_gaussians, num_rest_coeffs, 3))
                albedo_expanded = attributes.albedo.reshape((num_gaussians, 1, 3))
                all_coeffs = np.concatenate([albedo_expanded, specular_reshaped], axis=1)
                all_coeffs_flat = all_coeffs.reshape(-1, 3)

            if self.half_features:
                self.sh_coeffs_attr.Set(Vt.Vec3hArray.FromNumpy(all_coeffs_flat.astype(np.float16)))
            else:
                self.sh_coeffs_attr.Set(Vt.Vec3fArray.FromNumpy(all_coeffs_flat.astype(np.float32)))

            self.sh_coeffs_attr.SetMetadata("elementSize", num_sh_coeffs)

    def finalize(self, positions: np.ndarray) -> None:
        """Finalize prim with extent (UsdGeomBoundable API; ParticleField inherits from Boundable)."""
        if self.prim is None:
            raise RuntimeError("create_prim must be called before finalize")
        extent_range = self.compute_extent(positions)
        if self._schema is not None:
            self._schema.CreateExtentAttr().Set(extent_range)
        else:
            UsdGeom.Boundable(self.prim).CreateExtentAttr().Set(extent_range)
