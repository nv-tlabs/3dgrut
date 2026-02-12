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
USD Gaussian writer using ParticleField3DGaussianSplat schema.

This is the standard OpenUSD schema for Gaussian splatting representation.
Reference: https://github.com/PixarAnimationStudios/OpenUSD/blob/dev/pxr/usd/usdVol/schema.usda
"""

import logging
from typing import Optional

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.usd.writers.base import GaussianUSDWriter

logger = logging.getLogger(__name__)

# Material constants for ParticleField
USD_LOOKS_PATH = "/World/Looks"
USD_PARTICLEFIELD_MATERIAL_PATH = USD_LOOKS_PATH + "/ParticleFieldEmissive"
USD_PARTICLEFIELD_SHADER_PATH = USD_PARTICLEFIELD_MATERIAL_PATH + "/Shader"
PARTICLEFIELD_MATERIAL_MDL_FILE = "ParticleFieldEmissive.mdl"
PARTICLEFIELD_MATERIAL_NAME = "ParticleFieldEmissive"


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
        half_precision: bool = False,
        projection_mode_hint: str = "perspective",
        sorting_mode_hint: str = "cameraDistance",
    ) -> None:
        super().__init__(stage, capabilities, content_root_path)
        self.half_precision = half_precision
        self.projection_mode_hint = projection_mode_hint
        self.sorting_mode_hint = sorting_mode_hint

        # Use surflet kernel for surfel models, ellipsoid for 3DGS
        self.use_surflet_kernel = capabilities.is_surfel

        # Attribute handles
        self.positions_attr: Optional[Usd.Attribute] = None
        self.orientations_attr: Optional[Usd.Attribute] = None
        self.scales_attr: Optional[Usd.Attribute] = None
        self.opacities_attr: Optional[Usd.Attribute] = None
        self.sh_coeffs_attr: Optional[Usd.Attribute] = None
        self.sh_degree_attr: Optional[Usd.Attribute] = None

    def create_prim(self, num_gaussians: int) -> Usd.Prim:
        """Create ParticleField prim with appropriate kernel schema."""
        prim_path = f"{self.content_root_path}/gaussians"

        if self.use_surflet_kernel:
            # For 2DGS/surfels: use ParticleField with GaussianSurfletAPI
            self.prim = self.stage.DefinePrim(prim_path, "ParticleField")
            self._apply_surflet_kernel_schemas()
            logger.info(f"Created ParticleField with GaussianSurfletAPI (2DGS/surfel) at {prim_path}")
        else:
            # For 3DGS: use ParticleField3DGaussianSplat (auto-applies EllipsoidAPI)
            self.prim = self.stage.DefinePrim(prim_path, "ParticleField3DGaussianSplat")
            logger.info(f"Created ParticleField3DGaussianSplat at {prim_path}")

        # Create and bind material
        self._create_and_bind_material()

        # Create attributes
        self._create_attributes()

        # Set rendering hints
        self._set_rendering_hints()

        return self.prim

    def _apply_surflet_kernel_schemas(self) -> None:
        """Apply API schemas for 2DGS/surfel particles.

        First tries to use ApplyAPI (if USD has the schemas registered).
        Falls back to manually setting apiSchemas metadata if not available.
        """
        api_schemas = [
            "ParticleFieldPositionAttributeAPI",
            "ParticleFieldOrientationAttributeAPI",
            "ParticleFieldScaleAttributeAPI",
            "ParticleFieldOpacityAttributeAPI",
            "ParticleFieldKernelGaussianSurfletAPI",  # Surflet kernel instead of Ellipsoid
            "ParticleFieldSphericalHarmonicsAttributeAPI",
        ]

        # Try to apply schemas via ApplyAPI (works if USD has them registered)
        try:
            for schema_name in api_schemas:
                self.prim.ApplyAPI(schema_name)
            logger.info("Applied ParticleField API schemas via ApplyAPI")
        except Exception:
            # Fallback: manually set apiSchemas metadata
            self.prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(api_schemas))
            logger.info("Set ParticleField API schemas via metadata (USD schemas not registered)")

    def _create_and_bind_material(self) -> None:
        """Create and bind ParticleFieldEmissive material."""
        material_prim = self._create_particlefield_material()
        material = UsdShade.Material(material_prim)
        binding_api = UsdShade.MaterialBindingAPI(self.prim)
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)

    def _create_particlefield_material(self) -> Usd.Prim:
        """Create ParticleFieldEmissive material for LightField schema."""
        looks_prim = self.stage.GetPrimAtPath(USD_LOOKS_PATH)
        if not looks_prim.IsValid():
            self.stage.DefinePrim(USD_LOOKS_PATH, "Scope")

        material_prim = self.stage.DefinePrim(USD_PARTICLEFIELD_MATERIAL_PATH, "Material")
        shader_prim = self.stage.DefinePrim(USD_PARTICLEFIELD_SHADER_PATH, "Shader")

        shader_prim.CreateAttribute(
            "info:implementationSource", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        ).Set("sourceAsset")
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False, variability=Sdf.VariabilityUniform
        ).Set(Sdf.AssetPath(PARTICLEFIELD_MATERIAL_MDL_FILE))
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier",
            Sdf.ValueTypeNames.Token,
            custom=False,
            variability=Sdf.VariabilityUniform,
        ).Set(PARTICLEFIELD_MATERIAL_NAME)

        outputs_out = shader_prim.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token)
        outputs_out.SetMetadata("renderType", "material")

        material = UsdShade.Material(material_prim)
        shader = UsdShade.Shader(shader_prim)
        for output_name in ["mdl:displacement", "mdl:surface", "mdl:volume"]:
            output = material.CreateOutput(output_name, Sdf.ValueTypeNames.Token)
            output.ConnectToSource(shader.GetOutput("out"))

        return material_prim

    def _create_attributes(self) -> None:
        """Create particle field attributes with appropriate precision."""
        if self.half_precision:
            self.positions_attr = self.prim.CreateAttribute("positionsh", Sdf.ValueTypeNames.Point3hArray, custom=False)
            self.orientations_attr = self.prim.CreateAttribute("orientationsh", Sdf.ValueTypeNames.QuathArray, custom=False)
            self.scales_attr = self.prim.CreateAttribute("scalesh", Sdf.ValueTypeNames.Half3Array, custom=False)
            self.opacities_attr = self.prim.CreateAttribute("opacitiesh", Sdf.ValueTypeNames.HalfArray, custom=False)
            logger.info("Using half-precision (float16) attributes")
        else:
            self.positions_attr = self.prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray, custom=False)
            self.orientations_attr = self.prim.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray, custom=False)
            self.scales_attr = self.prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array, custom=False)
            self.opacities_attr = self.prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray, custom=False)

        if self.capabilities.has_spherical_harmonics:
            self.sh_degree_attr = self.prim.CreateAttribute(
                "radiance:sphericalHarmonicsDegree",
                Sdf.ValueTypeNames.Int,
                custom=False,
                variability=Sdf.VariabilityUniform,
            )
            if self.half_precision:
                self.sh_coeffs_attr = self.prim.CreateAttribute(
                    "radiance:sphericalHarmonicsCoefficientsh", Sdf.ValueTypeNames.Half3Array, custom=False
                )
            else:
                self.sh_coeffs_attr = self.prim.CreateAttribute(
                    "radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array, custom=False
                )

    def _set_rendering_hints(self) -> None:
        """Set rendering hints for the ParticleField schema."""
        projection_attr = self.prim.CreateAttribute(
            "projectionModeHint", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        )
        projection_attr.Set(self.projection_mode_hint)

        sorting_attr = self.prim.CreateAttribute(
            "sortingModeHint", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        )
        sorting_attr.Set(self.sorting_mode_hint)

    def write_attributes(
        self,
        attributes: GaussianAttributes,
        force_sh_0: bool = False,
    ) -> None:
        """Write Gaussian attributes to ParticleField3DGaussianSplat."""
        if self.prim is None:
            raise RuntimeError("create_prim must be called before write_attributes")

        num_gaussians = attributes.num_gaussians
        dtype = np.float16 if self.half_precision else np.float32

        # Positions
        if self.half_precision:
            self.positions_attr.Set(Vt.Vec3hArray.FromNumpy(attributes.positions.astype(np.float16)))
        else:
            self.positions_attr.Set(Vt.Vec3fArray.FromNumpy(attributes.positions.astype(np.float32)))

        # Orientations (quaternions)
        if self.half_precision:
            quats = [Gf.Quath(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in attributes.rotations]
            self.orientations_attr.Set(Vt.QuathArray(quats))
        else:
            quats = [Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in attributes.rotations]
            self.orientations_attr.Set(Vt.QuatfArray(quats))

        # Scales
        if self.half_precision:
            self.scales_attr.Set(Vt.Vec3hArray.FromNumpy(attributes.scales.astype(np.float16)))
        else:
            self.scales_attr.Set(Vt.Vec3fArray.FromNumpy(attributes.scales.astype(np.float32)))

        # Opacities
        densities_clamped = np.clip(attributes.densities.flatten(), 0.0, 1.0)
        if self.half_precision:
            self.opacities_attr.Set(Vt.HalfArray.FromNumpy(densities_clamped.astype(np.float16)))
        else:
            self.opacities_attr.Set(Vt.FloatArray.FromNumpy(densities_clamped.astype(np.float32)))

        # SH coefficients
        if self.capabilities.has_spherical_harmonics and self.sh_coeffs_attr is not None:
            sh_degree = 0 if force_sh_0 else self.capabilities.sh_degree
            if self.sh_degree_attr is not None:
                self.sh_degree_attr.Set(sh_degree)

            if force_sh_0:
                # Only DC term (albedo)
                all_coeffs_flat = attributes.albedo.reshape(-1, 3)
                num_sh_coeffs = 1
            else:
                # Full SH: combine albedo (DC) + specular (higher orders)
                num_sh_coeffs = (sh_degree + 1) ** 2
                num_rest_coeffs = num_sh_coeffs - 1

                # Reshape specular from [N, M*3] to [N, M, 3]
                specular_reshaped = attributes.specular.reshape((num_gaussians, num_rest_coeffs, 3))
                albedo_expanded = attributes.albedo.reshape((num_gaussians, 1, 3))

                # Combine DC + higher order
                all_coeffs = np.concatenate([albedo_expanded, specular_reshaped], axis=1)
                all_coeffs_flat = all_coeffs.reshape(-1, 3)

            if self.half_precision:
                self.sh_coeffs_attr.Set(Vt.Vec3hArray.FromNumpy(all_coeffs_flat.astype(np.float16)))
            else:
                self.sh_coeffs_attr.Set(Vt.Vec3fArray.FromNumpy(all_coeffs_flat.astype(np.float32)))

            self.sh_coeffs_attr.SetMetadata("elementSize", num_sh_coeffs)

    def finalize(self, positions: np.ndarray) -> None:
        """Finalize prim with extent."""
        if self.prim is None:
            raise RuntimeError("create_prim must be called before finalize")

        extent_attr = self.prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array, custom=False)
        extent_range = self.compute_extent(positions)
        extent_attr.Set(extent_range)
