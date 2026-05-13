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
USD RenderProduct writer.

Creates a /Render Scope with one RenderProduct per camera, each holding the
requested RenderVars and the camera relationship required by downstream
renderers.
"""

import logging
from typing import Dict, Sequence, Tuple

from pxr import Gf, Sdf, Usd

log = logging.getLogger(__name__)

_LDR_COLOR_VAR = "LdrColor"
_RENDER_SCOPE_PATH = "/Render"


def create_render_products(
    stage: Usd.Stage,
    camera_entries: Dict[str, Tuple[str, int, int]],
    render_scope_path: str = _RENDER_SCOPE_PATH,
    render_vars: Sequence[str] = (_LDR_COLOR_VAR,),
) -> None:
    """Create a /Render Scope with one RenderProduct per camera.

    Each RenderProduct is named after its camera and contains:
    - ``camera`` relationship pointing to the USD camera prim.
    - ``resolution`` attribute.
    - ``orderedVars`` relationship to child RenderVars.
    - Child ``RenderVar`` prims with matching ``sourceName`` values.

    Args:
        stage: USD stage that already contains the camera prims.
        camera_entries: Mapping ``{camera_name: (usd_camera_path, width, height)}``.
            The camera_name is used as the RenderProduct prim name (after USD
            identifier sanitization to match what export_cameras_to_usd produced).
        render_scope_path: Root path for the Render scope (default ``/Render``).
        render_vars: RenderVar source names to author. Defaults to ``LdrColor``.
    """
    from threedgrut.export.usd.writers.camera import _make_usd_prim_name

    render_vars = tuple(render_vars)
    if not render_vars:
        raise ValueError("create_render_products requires at least one RenderVar")

    stage.DefinePrim(render_scope_path, "Scope")

    created_count = 0
    for camera_name, (camera_path, width, height) in camera_entries.items():
        prim_name = _make_usd_prim_name(camera_name)
        product_path = f"{render_scope_path}/{prim_name}"

        existing_product = stage.GetPrimAtPath(product_path)
        if existing_product.IsValid():
            log.info("Keeping existing RenderProduct at %s; not overwriting", product_path)
            continue

        product_prim = stage.DefinePrim(product_path, "RenderProduct")

        # Resolution
        product_prim.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(
            Gf.Vec2i(int(width), int(height))
        )

        # Camera relationship
        camera_rel = product_prim.CreateRelationship("camera")
        camera_rel.SetTargets([Sdf.Path(camera_path)])

        ordered_var_targets = []
        for render_var_name in render_vars:
            var_path = f"{product_path}/{render_var_name}"
            var_prim = stage.DefinePrim(var_path, "RenderVar")
            var_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(render_var_name)
            var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
            ordered_var_targets.append(Sdf.Path(var_path))

        # orderedVars relationship
        ordered_vars_rel = product_prim.CreateRelationship("orderedVars")
        ordered_vars_rel.SetTargets(ordered_var_targets)

        log.debug(f"Created RenderProduct at {product_path} → camera {camera_path} ({width}×{height})")
        created_count += 1

    log.info(f"Created {created_count} RenderProduct(s) under {render_scope_path}")
