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

"""Omniverse-specific USD material authoring for Gaussian ParticleFields."""

from pxr import Sdf, Usd, UsdShade

USD_LOOKS_PATH = "/World/Looks"
USD_PARTICLEFIELD_MATERIAL_PATH = f"{USD_LOOKS_PATH}/ParticleFieldEmissive"
USD_PARTICLEFIELD_SHADER_PATH = f"{USD_PARTICLEFIELD_MATERIAL_PATH}/Shader"
PARTICLEFIELD_MATERIAL_MDL_FILE = "ParticleFieldEmissive.mdl"
PARTICLEFIELD_MATERIAL_NAME = "ParticleFieldEmissive"


def bind_particlefield_emissive_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    has_post_processing: bool = False,
) -> None:
    """Bind Kit's ParticleFieldEmissive MDL material to a Gaussian ParticleField."""
    looks_prim = stage.GetPrimAtPath(USD_LOOKS_PATH)
    if not looks_prim.IsValid():
        stage.DefinePrim(USD_LOOKS_PATH, "Scope")

    material_prim = stage.DefinePrim(USD_PARTICLEFIELD_MATERIAL_PATH, "Material")
    shader_prim = stage.DefinePrim(USD_PARTICLEFIELD_SHADER_PATH, "Shader")
    shader_prim.CreateAttribute(
        "info:implementationSource",
        Sdf.ValueTypeNames.Token,
        custom=False,
        variability=Sdf.VariabilityUniform,
    ).Set("sourceAsset")
    shader_prim.CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
        custom=False,
        variability=Sdf.VariabilityUniform,
    ).Set(Sdf.AssetPath(PARTICLEFIELD_MATERIAL_MDL_FILE))
    shader_prim.CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier",
        Sdf.ValueTypeNames.Token,
        custom=False,
        variability=Sdf.VariabilityUniform,
    ).Set(PARTICLEFIELD_MATERIAL_NAME)

    if has_post_processing:
        shader_prim.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool).Set(False)
        shader_prim.CreateAttribute("inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool).Set(False)

    output_attr = shader_prim.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token)
    output_attr.SetMetadata("renderType", "material")

    material = UsdShade.Material(material_prim)
    shader = UsdShade.Shader(shader_prim)
    for output_name in ("mdl:displacement", "mdl:surface", "mdl:volume"):
        output = material.CreateOutput(output_name, Sdf.ValueTypeNames.Token)
        output.ConnectToSource(shader.GetOutput("out"))

    binding_api = UsdShade.MaterialBindingAPI(prim)
    binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)
