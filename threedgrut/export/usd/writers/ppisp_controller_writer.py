# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
PPISP Controller USD writer.

Writes the per-camera PPISP controller as a UsdShade Shader prim that
references the shared ``ppisp_controller.slang`` SPG asset. The trained
controller weights are flattened into a single ``float[] inputs:weights``
attribute on the Shader prim — the Slang shader picks them up as a
``StructuredBuffer<float>`` at dispatch time.

The flatten layout must match ``ppisp_controller.slang``'s ``OFF_*``
constants:

    conv1_weight  (16 x 3)        |  conv1_bias  (16)
    conv2_weight  (32 x 16)       |  conv2_bias  (32)
    conv3_weight  (64 x 32)       |  conv3_bias  (64)
    trunk0_weight (128 x 1601)    |  trunk0_bias (128)
    trunk1_weight (128 x 128)     |  trunk1_bias (128)
    trunk2_weight (128 x 128)     |  trunk2_bias (128)
    exposure_head_weight (128)    |  exposure_head_bias (1)
    color_head_weight    (8 x 128)|  color_head_bias    (8)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Sequence

import numpy as np

from pxr import Sdf, Usd, UsdShade, Vt

from threedgrut.export.usd.stage_utils import NamedSerialized

if TYPE_CHECKING:
    import torch.nn as nn  # noqa: F401

log = logging.getLogger(__name__)


# Names must match ppisp_controller.slang's bindings and ppisp_controller.slang.usda.
CONTROLLER_INPUT_RENDER_VAR = "HdrColor"
CONTROLLER_OUTPUT_NAME = "ControllerParams"
PRIOR_EXPOSURE_INPUT = "priorExposure"
WEIGHTS_INPUT = "weights"

CONTROLLER_USDA_FILE = "ppisp_controller.slang.usda"
CONTROLLER_SLANG_FILE = "ppisp_controller.slang"

# Architecture sizes (mirror ppisp._PPISPController defaults / shader constants).
EXPECTED_SIZES = {
    "cnn_feature_dim": 64,
    "pool_grid_h": 5,
    "pool_grid_w": 5,
    "mlp_hidden_dim": 128,
    "color_params_per_frame": 8,
    "input_downsampling": 3,
}

# Total weight count. This *must* match ppisp_controller.slang::TOTAL_WEIGHTS.
EXPECTED_WEIGHTS_LEN = (
    16 * 3 + 16 +
    32 * 16 + 32 +
    64 * 32 + 64 +
    128 * 1601 + 128 +
    128 * 128 + 128 +
    128 * 128 + 128 +
    128 + 1 +
    8 * 128 + 8
)


# ---------------------------------------------------------------------------
# Weight extraction and validation
# ---------------------------------------------------------------------------


def _validate_controller_shape(controller) -> None:
    """Sanity-check a ``_PPISPController`` matches the shader's hard-coded sizes."""
    cnn_encoder = controller.cnn_encoder
    conv1 = cnn_encoder[0]
    conv2 = cnn_encoder[3]
    conv3 = cnn_encoder[5]
    maxpool = cnn_encoder[1]
    avgpool = cnn_encoder[6]

    if conv1.in_channels != 3 or conv1.out_channels != 16:
        raise ValueError(f"controller conv1 must be 3->16, got {conv1.in_channels}->{conv1.out_channels}")
    if conv1.kernel_size != (1, 1):
        raise ValueError(f"controller conv1 kernel must be 1x1, got {conv1.kernel_size}")
    if conv2.in_channels != 16 or conv2.out_channels != 32:
        raise ValueError(f"controller conv2 must be 16->32, got {conv2.in_channels}->{conv2.out_channels}")
    if conv3.in_channels != 32 or conv3.out_channels != EXPECTED_SIZES["cnn_feature_dim"]:
        raise ValueError(
            f"controller conv3 out_channels must be {EXPECTED_SIZES['cnn_feature_dim']}, got {conv3.out_channels}"
        )
    if maxpool.kernel_size != EXPECTED_SIZES["input_downsampling"]:
        raise ValueError(
            f"controller maxpool kernel must be {EXPECTED_SIZES['input_downsampling']}, got {maxpool.kernel_size}"
        )
    if maxpool.stride != EXPECTED_SIZES["input_downsampling"]:
        raise ValueError(
            f"controller maxpool stride must be {EXPECTED_SIZES['input_downsampling']}, got {maxpool.stride}"
        )

    expected_grid = (EXPECTED_SIZES["pool_grid_h"], EXPECTED_SIZES["pool_grid_w"])
    if tuple(avgpool.output_size) != expected_grid:
        raise ValueError(f"controller AdaptiveAvgPool2d must be {expected_grid}, got {tuple(avgpool.output_size)}")

    trunk = controller.mlp_trunk
    linear_layers = [m for m in trunk if hasattr(m, "weight") and m.weight.dim() == 2]
    if len(linear_layers) != 3:
        raise ValueError(f"controller MLP trunk must have 3 Linear layers, got {len(linear_layers)}")

    expected_input_dim = (
        EXPECTED_SIZES["pool_grid_h"]
        * EXPECTED_SIZES["pool_grid_w"]
        * EXPECTED_SIZES["cnn_feature_dim"]
        + 1
    )
    if linear_layers[0].in_features != expected_input_dim:
        raise ValueError(
            f"controller trunk[0].in_features must be {expected_input_dim}, got {linear_layers[0].in_features}"
        )
    for idx, layer in enumerate(linear_layers):
        if layer.out_features != EXPECTED_SIZES["mlp_hidden_dim"]:
            raise ValueError(
                f"controller trunk[{idx}].out_features must be {EXPECTED_SIZES['mlp_hidden_dim']}, "
                f"got {layer.out_features}"
            )

    if controller.exposure_head.out_features != 1:
        raise ValueError("controller exposure_head must produce one output")
    if controller.color_head.out_features != EXPECTED_SIZES["color_params_per_frame"]:
        raise ValueError(
            f"controller color_head must produce {EXPECTED_SIZES['color_params_per_frame']} outputs"
        )


def _to_np(t) -> np.ndarray:
    import torch
    return t.detach().cpu().to(dtype=torch.float32).numpy()


def flatten_controller_weights(controller) -> np.ndarray:
    """Concatenate all controller weights into one float32 buffer.

    The order must match ``ppisp_controller.slang``'s ``OFF_*`` offsets.
    Returns a 1-D ``np.float32`` array of length :data:`EXPECTED_WEIGHTS_LEN`.
    """
    _validate_controller_shape(controller)

    cnn_encoder = controller.cnn_encoder
    conv1 = cnn_encoder[0]
    conv2 = cnn_encoder[3]
    conv3 = cnn_encoder[5]

    trunk = controller.mlp_trunk
    linear_layers = [m for m in trunk if hasattr(m, "weight") and m.weight.dim() == 2]

    def conv_w(layer) -> np.ndarray:
        # PyTorch Conv2d weight: [out, in, kH, kW]. With 1x1 kernels we
        # emit row-major [out * in].
        return _to_np(layer.weight).reshape(layer.out_channels, layer.in_channels).reshape(-1)

    parts: List[np.ndarray] = [
        conv_w(conv1), _to_np(conv1.bias).reshape(-1),
        conv_w(conv2), _to_np(conv2.bias).reshape(-1),
        conv_w(conv3), _to_np(conv3.bias).reshape(-1),
        _to_np(linear_layers[0].weight).reshape(-1), _to_np(linear_layers[0].bias).reshape(-1),
        _to_np(linear_layers[1].weight).reshape(-1), _to_np(linear_layers[1].bias).reshape(-1),
        _to_np(linear_layers[2].weight).reshape(-1), _to_np(linear_layers[2].bias).reshape(-1),
        _to_np(controller.exposure_head.weight).reshape(-1), _to_np(controller.exposure_head.bias).reshape(-1),
        _to_np(controller.color_head.weight).reshape(-1), _to_np(controller.color_head.bias).reshape(-1),
    ]

    flat = np.concatenate(parts).astype(np.float32, copy=False)
    if flat.size != EXPECTED_WEIGHTS_LEN:
        raise RuntimeError(
            f"flatten_controller_weights produced {flat.size} floats; expected {EXPECTED_WEIGHTS_LEN}. "
            "Did the controller architecture change?"
        )
    if not np.all(np.isfinite(flat)):
        raise RuntimeError(
            "controller weights contain NaN/Inf; refusing to export. "
            "Investigate the trained checkpoint before retrying."
        )
    return flat


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------


def add_controller_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    controller,
    *,
    prior_exposure: float | None = None,
) -> UsdShade.Shader:
    """Author the controller Shader prim and connect ``HdrColor`` → ``ControllerParams``.

    Returns the created Shader so the caller can wire its output into the
    PPISP shader. The PPISP shader is responsible for *consuming* the
    output via its dynamic-controller binding.
    """
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    # Mark HdrColor RenderVar input as an opaque AOV (no connection needed here).
    input_var_path = f"{render_product_path}/{CONTROLLER_INPUT_RENDER_VAR}"
    input_var_prim = stage.GetPrimAtPath(input_var_path)
    if input_var_prim.IsValid():
        input_var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)

    shader_prim_name = f"PPISPController_{camera_index}"
    shader_path = f"{render_product_path}/{shader_prim_name}"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.GetPrim().GetReferences().AddReference(CONTROLLER_USDA_FILE)
    shader.GetPrim().CreateAttribute(
        "info:implementationSource", Sdf.ValueTypeNames.Token, custom=False
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False
    ).Set(Sdf.AssetPath(CONTROLLER_SLANG_FILE))
    shader.GetPrim().CreateAttribute(
        "info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False
    ).Set("controllerProcess")

    hdr_input = shader.CreateInput(CONTROLLER_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{CONTROLLER_INPUT_RENDER_VAR}.omni:rtx:aov")])

    shader.CreateOutput(CONTROLLER_OUTPUT_NAME, Sdf.ValueTypeNames.Opaque)

    prior_input = shader.CreateInput(PRIOR_EXPOSURE_INPUT, Sdf.ValueTypeNames.Float)
    prior_input.Set(float(prior_exposure or 0.0))

    weights = flatten_controller_weights(controller)
    weights_input = shader.CreateInput(WEIGHTS_INPUT, Sdf.ValueTypeNames.FloatArray)
    weights_input.Set(Vt.FloatArray.FromNumpy(weights))

    # Route the controller output through a RenderVar with omni:rtx:aov, so
    # SPG resolves it the same way it resolves HdrColor / LdrColor. Direct
    # Shader -> Shader connections work in slangpy but Kit's runtime walks
    # AOV connections, not arbitrary UsdShade outputs.
    var_path = f"{render_product_path}/{CONTROLLER_OUTPUT_NAME}"
    render_var = stage.DefinePrim(var_path, "RenderVar")
    render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(CONTROLLER_OUTPUT_NAME)
    aov_attr = render_var.CreateAttribute(
        "omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False
    )
    aov_attr.SetConnections([
        shader.GetPath().AppendProperty(f"outputs:{CONTROLLER_OUTPUT_NAME}")
    ])

    # Add the intermediate var to RenderProduct.orderedVars so SPG discovers it.
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        targets = list(ordered_vars_rel.GetTargets())
        path = Sdf.Path(CONTROLLER_OUTPUT_NAME)
        if path not in targets:
            targets.append(path)
            ordered_vars_rel.SetTargets(targets)

    log.debug(
        "Authored PPISP controller shader at %s (camera %d, %d weights), "
        "AOV RenderVar at %s",
        shader_path, camera_index, weights.size, var_path,
    )
    return shader


# ---------------------------------------------------------------------------
# Sidecar packaging
# ---------------------------------------------------------------------------


def get_controller_sidecars() -> List[NamedSerialized]:
    """Load the shared controller SPG sidecar files.

    Unlike the dynamic PPISP path, the controller does not need per-camera
    sidecar generation: the weights live in USD attributes, so the slang /
    lua / usda assets are identical for every camera.
    """
    from threedgrut.export.usd.ppisp_spg import _SPG_DIR
    filenames = [CONTROLLER_SLANG_FILE, CONTROLLER_SLANG_FILE + ".lua", CONTROLLER_USDA_FILE]
    out: List[NamedSerialized] = []
    for name in filenames:
        path = _SPG_DIR / name
        if path.exists():
            out.append(NamedSerialized(filename=name, serialized=path.read_bytes()))
    return out
