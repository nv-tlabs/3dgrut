# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPISP automatic-parameter USD writer (EXPERIMENTAL).

.. warning::

   This module authors USD shader prims for the **EXPERIMENTAL** PPISP
   controller export path. The controller is the default mode used
   during PPISP training to produce realistic novel views, but its
   USD export is disabled by default in the gaussian USD asset
   writer:

   * Omniverse SPG does not currently expose first-class neural-network
     weight bindings. The export path generates a per-camera CUDA
     sidecar with trained weights embedded as device constants, so the
     runtime does not upload a large weight asset every frame.
   * External renderers consuming the asset may not resolve the
     controller CUDA sidecars at all. For production USD assets,
     prefer the SH-optimized integration mode (PPISP folded into
     Gaussian SH coefficients) or the ``spg-runtime`` mode with a fixed
     reference ``(camera_id, frame_id)`` pair.
   * The controller export path should only be used for research
     and experimental workflows that knowingly accept the runtime
     compatibility risk.

This module is the UsdShade authoring half of the controller
export. It consumes:

* :func:`threedgrut.export.usd.writers.ppisp_controller_weights.flatten_controller_weights`
  to flatten per-camera CNN/MLP weights into generated CUDA source
  files whose layout is encoded by :file:`ppisp_controller.cu`'s
  ``OFF_*`` offsets.
* :func:`threedgrut.export.usd.writers.ppisp_controller_weights.select_camera_controller`
  to resolve a validated PPISP camera id into the corresponding
  ``_PPISPController`` instance.

Authoring contract: per RenderProduct, the writer creates **three**
shader prims and two intermediate RenderVars:

1. ``<rp>/PPISPControllerPool_<cam>`` -- consumes ``HdrColor`` AOV
   and produces ``ControllerFeatures`` AOV via the generated
   ``ppisp_controller_<cam>.cu`` sidecar.
2. ``<rp>/ControllerFeatures`` -- intermediate RenderVar with
   ``omni:rtx:aov`` connection to the pool shader's
   ``outputs:ControllerFeatures``.
3. ``<rp>/PPISPController_<cam>`` -- consumes ``ControllerFeatures``
   AOV and produces ``ControllerParams`` AOV via the same generated
   ``ppisp_controller_<cam>.cu`` sidecar. Authors
   ``inputs:priorExposure`` only.
4. ``<rp>/ControllerParams`` -- intermediate RenderVar with
   ``omni:rtx:aov`` connection to the controller's
   ``outputs:ControllerParams``. Routing the controller outputs
   through RenderVars (rather than direct UsdShade Shader ->
   Shader connections) is required because Kit's runtime walks
   AOV connections, not arbitrary UsdShade outputs.
5. ``<rp>/PPISPAuto`` -- references
   :file:`ppisp_usd_spg_auto.usda`. Authors
   ``inputs:responsivity`` and per-camera vignetting / CRF;
   consumes ``HdrColor`` and ``ControllerParams`` AOVs; produces
   ``PPISPColor`` AOV. ``LdrColor`` is wired to the
   ``PPISPColor`` output so display-AOV consumers see the
   processed result.

The combined :func:`add_ppisp_auto_shader_to_render_product`
authors the entire controller + auto-PPISP graph in one call. There is no
public API for authoring just the controller or just the
auto-PPISP shader: splitting them would expose the AOV-graph
wiring (the most fragile part of the port) to call sites, and
nothing in the export pipeline needs the prims separately.

Sidecar packaging is split between
:func:`get_ppisp_embedded_controller_spg_files` (the per-camera
generated ``ppisp_controller_<cam>.cu*`` files), and
:func:`get_ppisp_auto_spg_files` (the three
``ppisp_usd_spg_auto.cu*`` files). The two sets are
complementary: the controller emits the
``ControllerParams`` AOV that the auto-PPISP shader reads, so a
consumer that ships one without the other will produce a runtime
error. Higher-level packaging code is expected to call both when
``enable_ppisp_controller_export=True``.

"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import torch.nn as nn
from pxr import Gf, Sdf, Usd, UsdShade, Vt

from threedgrut.export.usd.stage_utils import NamedSerialized
from threedgrut.export.usd.writers.ppisp_controller_weights import (
    EXPECTED_CONTROLLER_WEIGHTS_LEN,
    flatten_controller_weights,
    select_camera_controller,
)

if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]


log = logging.getLogger(__name__)


# =============================================================================
# Sidecar filenames and prim / AOV / input names
# =============================================================================


PPISP_SPG_DIR = Path(__file__).parent.parent / "ppisp_spg"

# Controller template files used to generate per-camera embedded-weight
# sidecars.
CONTROLLER_CU_FILE = "ppisp_controller.cu"
CONTROLLER_LUA_FILE = "ppisp_controller.cu.lua"

# Each camera receives a generated CUDA source file whose file-scope device
# constant contains that camera's flattened controller weights; the Lua
# launcher has no weights input and the authored USD graph therefore does not
# bind a weight asset.
CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE = "ppisp_controller_{camera_index}.cu"
CONTROLLER_EMBEDDED_LUA_FILE_TEMPLATE = "ppisp_controller_{camera_index}.cu.lua"
EMBEDDED_CONTROLLER_WEIGHTS_SYMBOL = "kControllerWeights"
EMBEDDED_CONTROLLER_WEIGHTS_MARKER = "// __PPISP_CONTROLLER_EMBEDDED_WEIGHTS__"

# Automatic-parameter PPISP sidecar set.
PPISP_AUTO_CU_FILE = "ppisp_usd_spg_auto.cu"
PPISP_AUTO_LUA_FILE = "ppisp_usd_spg_auto.cu.lua"
PPISP_AUTO_USDA_FILE = "ppisp_usd_spg_auto.usda"

# Sub-identifiers (must match the entry point names in the corresponding
# controller CUDA and auto-PPISP CUDA files).
CONTROLLER_POOL_SUB_IDENTIFIER = "controllerPoolProcess"
CONTROLLER_SUB_IDENTIFIER = "controllerProcess"
PPISP_AUTO_SUB_IDENTIFIER = "ppispProcessAuto"

# RenderVar / AOV names. ``HdrColor`` and ``LdrColor`` mirror the
# static-PPISP writer's conventions so the two paths are
# indistinguishable to display-AOV consumers.
HDR_COLOR_RENDER_VAR = "HdrColor"
CONTROLLER_FEATURES_RENDER_VAR = "ControllerFeatures"
CONTROLLER_PARAMS_RENDER_VAR = "ControllerParams"
PPISP_AUTO_OUTPUT_RENDER_VAR = "PPISPColor"
LDR_COLOR_RENDER_VAR = "LdrColor"

# Prim name templates within a RenderProduct scope.
CONTROLLER_POOL_PRIM_NAME_TEMPLATE = "PPISPControllerPool_{camera_index}"
CONTROLLER_PRIM_NAME_TEMPLATE = "PPISPController_{camera_index}"
PPISP_AUTO_PRIM_NAME = "PPISPAuto"

# Input-attribute names mirrored by the CUDA Lua launchers.
CONTROLLER_PRIOR_EXPOSURE_INPUT = "priorExposure"
CONTROLLER_HDR_INPUT = HDR_COLOR_RENDER_VAR
CONTROLLER_FEATURES_INPUT = CONTROLLER_FEATURES_RENDER_VAR
CONTROLLER_POOL_OUTPUT = CONTROLLER_FEATURES_RENDER_VAR
CONTROLLER_OUTPUT = CONTROLLER_PARAMS_RENDER_VAR

PPISP_AUTO_RESPONSIVITY_INPUT = "responsivity"
PPISP_AUTO_HDR_INPUT = HDR_COLOR_RENDER_VAR
PPISP_AUTO_CONTROLLER_INPUT = CONTROLLER_PARAMS_RENDER_VAR
PPISP_AUTO_OUTPUT = PPISP_AUTO_OUTPUT_RENDER_VAR

NUM_CHANNELS = 3
CHANNEL_SUFFIXES = ("R", "G", "B")

DEFAULT_PRIOR_EXPOSURE = 0.0
DEFAULT_RESPONSIVITY = 1.0

# Namespace for source-of-truth PPISP attributes authored on the
# ``<cam>_ppisp`` camera. Kept local to avoid a circular import with
# ``ppisp_writer``.
PPISP_ATTR_NAMESPACE = "ppisp:"
# Controller weights are baked into the per-camera CUDA sidecar. They are also
# mirrored onto the camera for documentation/downstream tools; runtime does not
# read this attribute.
CONTROLLER_WEIGHTS_CAMERA_ATTR = "controllerWeights"

# Tiled-RenderProduct grid. Tiling is a RenderProduct property, so the grid is
# authored as literal inputs on each shader node, not as camera attributes.
PPISP_TILE_COUNT_INPUT_NAMES = ("tileCountX", "tileCountY")
DEFAULT_PPISP_TILE_COUNT = 1


# =============================================================================
# Public API
# =============================================================================


def add_ppisp_auto_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: "PPISP",
    controller: nn.Module,
    *,
    responsivity: float = DEFAULT_RESPONSIVITY,
    prior_exposure: float = DEFAULT_PRIOR_EXPOSURE,
    ppisp_camera_path: Optional[Sdf.Path] = None,
) -> Usd.Prim:
    """Author the controller + automatic-parameter PPISP graph on a RenderProduct.

    Authors the two shader prims (controller + auto-PPISP) plus the
    intermediate ``ControllerParams`` RenderVar that wires them
    together, and the ``LdrColor`` RenderVar that publishes the
    auto-PPISP output as the display AOV.

    The combined function has no separately callable counterparts on
    purpose: the AOV-graph wiring is the most fragile part of the
    Omniverse SPG port, and exposing it to call sites would multiply
    the integration risk surface.

    Args:
        stage: USD stage containing the RenderProduct.
        render_product_path: Path to the RenderProduct prim under
            which the shader prims will be authored.
        camera_index: PPISP camera index. Used both to resolve
            ``ppisp.controllers[camera_index]`` (validated against the
            supplied ``controller`` argument as a defensive check)
            and to author the per-camera vignetting / CRF inputs from
            ``ppisp.vignetting_params`` / ``ppisp.crf_params``.
        ppisp: Trained PPISP module exposing ``num_cameras``,
            ``vignetting_params`` (``[num_cameras, 3, 5]``) and
            ``crf_params`` (``[num_cameras, 3, 4]``).
        controller: Per-camera controller (typically obtained via
            :func:`threedgrut.export.usd.writers.ppisp_controller_weights.select_camera_controller`).
            Validated and embedded into generated CUDA sidecars by the
            exporter.
        responsivity: Achromatic HDR multiplier authored on the
            auto-PPISP shader's ``inputs:responsivity`` attribute.
            Must be a strictly positive finite number.
        prior_exposure: EXIF-derived prior exposure scalar authored
            on the controller shader's ``inputs:priorExposure``
            attribute. Defaults to 0.0 (matching training-time
            inference when no prior is wired).
        ppisp_camera_path: Optional path to the ``<cam>_ppisp`` camera
            that holds source-of-truth PPISP parameters as ``ppisp:*``
            attributes. When set, responsivity, vignetting, CRF, and a
            documentation mirror of the embedded controller weights are
            authored on that camera. Exposure/color stay controller-driven
            via the ``ControllerParams`` AOV.
    Returns:
        The auto-PPISP shader prim, mirroring the return type of
        :func:`threedgrut.export.usd.writers.ppisp_writer.add_ppisp_shader_to_render_product`.

    Raises:
        TypeError: ``camera_index`` is not an int, or
            ``responsivity`` / ``prior_exposure`` is not numeric.
        ValueError: ``render_product_path`` does not resolve to a
            valid prim, ``camera_index`` is out of range, or any
            argument fails its fail-loud validation.
    """
    log.warning(
        "PPISP controller export is EXPERIMENTAL: the authored CUDA controller "
        "graph is runtime-specific and may not be resolved by external renderers. "
        "Prefer the sh-optimized integration mode or the spg-runtime mode with a "
        "fixed (camera_id, frame_id) pair for production USD assets."
    )

    _validate_camera_index(camera_index)
    _validate_finite_positive("responsivity", responsivity, allow_zero=False)
    _validate_finite("prior_exposure", prior_exposure)

    render_product = _require_render_product(stage, render_product_path)
    _validate_camera_index_against_ppisp(camera_index, ppisp)

    weight_count = _validate_controller_weights(controller)
    controller_source_file = _controller_embedded_cuda_filename(camera_index)

    _mark_render_var_as_aov(stage, render_product_path, HDR_COLOR_RENDER_VAR)

    controller_pool_shader = _create_controller_pool_shader(
        stage=stage,
        render_product_path=render_product_path,
        camera_index=camera_index,
        controller_source_file=controller_source_file,
    )

    _create_controller_features_render_var(
        stage=stage,
        render_product=render_product,
        render_product_path=render_product_path,
        controller_pool_shader=controller_pool_shader,
    )

    controller_shader = _create_controller_shader(
        stage=stage,
        render_product_path=render_product_path,
        camera_index=camera_index,
        controller_source_file=controller_source_file,
        prior_exposure=float(prior_exposure),
    )

    _create_controller_render_var(
        stage=stage,
        render_product=render_product,
        render_product_path=render_product_path,
        controller_shader=controller_shader,
    )

    if ppisp_camera_path is not None:
        _author_auto_camera_attributes(
            stage=stage,
            camera_path=ppisp_camera_path,
            ppisp=ppisp,
            camera_index=camera_index,
            responsivity=float(responsivity),
            controller=controller,
        )

    auto_shader = _create_auto_ppisp_shader(
        stage=stage,
        render_product_path=render_product_path,
        ppisp=ppisp,
        camera_index=camera_index,
        responsivity=float(responsivity),
        ppisp_camera_path=ppisp_camera_path,
    )

    _create_ldr_color_render_var(
        stage=stage,
        render_product=render_product,
        render_product_path=render_product_path,
        auto_shader=auto_shader,
    )

    log.info(
        "Authored PPISP auto-parameter graph at %s (camera %d, %d embedded controller weights)",
        render_product_path,
        camera_index,
        weight_count,
    )
    return auto_shader.GetPrim()


def get_ppisp_embedded_controller_spg_files(ppisp: "PPISP", camera_indices: Sequence[int]) -> List[NamedSerialized]:
    """Generate embedded-weight CUDA controller sidecars for camera indices.

    Args:
        ppisp: Trained PPISP module exposing ``controllers``.
        camera_indices: Camera indices to package. Each index is
            validated by :func:`select_camera_controller`.

    Returns:
        ``NamedSerialized`` entries for ``ppisp_controller_<cam>.cu`` and
        ``ppisp_controller_<cam>.cu.lua`` for each camera. The CUDA source
        contains the flattened controller weights as a file-scope device
        constant, so no separate binary weight asset is needed.
    """
    out: List[NamedSerialized] = []
    seen_filenames: set[str] = set()
    lua_source = render_embedded_controller_lua_source().encode("utf-8")
    for camera_index in camera_indices:
        controller = select_camera_controller(ppisp, camera_index)
        entries = [
            NamedSerialized(
                filename=_controller_embedded_cuda_filename(camera_index),
                serialized=render_embedded_controller_cuda_source(controller).encode("utf-8"),
            ),
            NamedSerialized(
                filename=_controller_embedded_lua_filename(camera_index),
                serialized=lua_source,
            ),
        ]
        for entry in entries:
            if entry.filename in seen_filenames:
                raise ValueError(f"duplicate PPISP controller sidecar filename: {entry.filename}")
            seen_filenames.add(entry.filename)
            out.append(entry)
    return out


def render_embedded_controller_cuda_source(controller: nn.Module) -> str:
    """Render a CUDA controller source file with ``controller`` weights embedded."""
    flat_weights = flatten_controller_weights(controller)
    return _render_embedded_controller_cuda_source_from_flat_weights(flat_weights)


def render_embedded_controller_lua_source() -> str:
    """Render the Lua launcher for embedded-weight controller CUDA sidecars."""
    source = (PPISP_SPG_DIR / CONTROLLER_LUA_FILE).read_text(encoding="utf-8")
    if 'inputs["weights"]' in source or "cuda.array(weights" in source:
        raise RuntimeError("embedded controller Lua template unexpectedly binds controller weights")
    return source


def get_ppisp_auto_spg_files() -> List[NamedSerialized]:
    """Load the three automatic-parameter PPISP SPG sidecar files.

    Returns:
        ``NamedSerialized`` entries for ``ppisp_usd_spg_auto.cu``,
        ``ppisp_usd_spg_auto.cu.lua``, and
        ``ppisp_usd_spg_auto.usda``. Same skip-with-warning
        behavior on missing files as the static PPISP sidecar loader.
    """
    return _load_sidecars([PPISP_AUTO_CU_FILE, PPISP_AUTO_LUA_FILE, PPISP_AUTO_USDA_FILE])


# =============================================================================
# Internals: argument validation
# =============================================================================


def _validate_camera_index(camera_index: object) -> None:
    if isinstance(camera_index, bool) or not isinstance(camera_index, int):
        raise TypeError(f"camera_index must be int, got {type(camera_index).__name__}")
    if camera_index < 0:
        raise ValueError(f"camera_index must be non-negative, got {camera_index}")


def _validate_camera_index_against_ppisp(camera_index: int, ppisp: "PPISP") -> None:
    num_cameras = int(ppisp.num_cameras)  # type: ignore[attr-defined]
    if camera_index >= num_cameras:
        raise ValueError(f"camera_index {camera_index} out of range for PPISP with num_cameras={num_cameras}")


def _validate_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number (int or float), got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _validate_finite_positive(name: str, value: object, *, allow_zero: bool) -> None:
    _validate_finite(name, value)
    threshold = 0.0
    if allow_zero:
        if float(value) < threshold:  # type: ignore[arg-type]
            raise ValueError(f"{name} must be non-negative, got {value!r}")
    else:
        if float(value) <= threshold:  # type: ignore[arg-type]
            raise ValueError(f"{name} must be strictly positive, got {value!r}")


def _require_render_product(stage: Usd.Stage, render_product_path: str) -> Usd.Prim:
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")
    return render_product


def _validate_controller_weights(controller: nn.Module) -> int:
    """Validate ``controller`` and return the flattened weight count."""
    flat_weights = flatten_controller_weights(controller)
    if flat_weights.shape != (EXPECTED_CONTROLLER_WEIGHTS_LEN,):
        raise ValueError(
            "flatten_controller_weights produced "
            f"{flat_weights.shape[0]} floats; expected {EXPECTED_CONTROLLER_WEIGHTS_LEN}."
        )
    return int(flat_weights.shape[0])


def _controller_embedded_cuda_filename(camera_index: int) -> str:
    return CONTROLLER_EMBEDDED_CU_FILE_TEMPLATE.format(camera_index=camera_index)


def _controller_embedded_lua_filename(camera_index: int) -> str:
    return CONTROLLER_EMBEDDED_LUA_FILE_TEMPLATE.format(camera_index=camera_index)


def _format_cuda_float_literal(value: np.float32) -> str:
    return (
        np.format_float_scientific(
            np.float32(value),
            unique=False,
            precision=9,
            trim="k",
        )
        + "f"
    )


def _format_cuda_float_array(values: np.ndarray) -> str:
    lines: List[str] = []
    for start in range(0, values.shape[0], 8):
        chunk = values[start : start + 8]
        lines.append("    " + ", ".join(_format_cuda_float_literal(value) for value in chunk))
    return ",\n".join(lines)


def _render_embedded_controller_cuda_source_from_flat_weights(flat_weights: np.ndarray) -> str:
    if flat_weights.shape != (EXPECTED_CONTROLLER_WEIGHTS_LEN,):
        raise ValueError(
            "embedded controller source expected "
            f"{EXPECTED_CONTROLLER_WEIGHTS_LEN} weights, got {flat_weights.shape[0]}"
        )
    if not np.isfinite(flat_weights).all():
        raise ValueError("embedded controller weights contain NaN/Inf")

    source = (PPISP_SPG_DIR / CONTROLLER_CU_FILE).read_text(encoding="utf-8")
    weight_array = (
        f"static_assert(TOTAL_WEIGHTS == {EXPECTED_CONTROLLER_WEIGHTS_LEN}, "
        '"embedded PPISP controller weight count mismatch");\n'
        f"static __device__ const float {EMBEDDED_CONTROLLER_WEIGHTS_SYMBOL}[TOTAL_WEIGHTS] = {{\n"
        f"{_format_cuda_float_array(flat_weights.astype(np.float32, copy=False))}\n"
        "};"
    )
    source = source.replace(
        EMBEDDED_CONTROLLER_WEIGHTS_MARKER,
        weight_array,
    )
    if EMBEDDED_CONTROLLER_WEIGHTS_SYMBOL not in source:
        raise RuntimeError("failed to embed controller weights into CUDA source")
    if EMBEDDED_CONTROLLER_WEIGHTS_MARKER in source:
        raise RuntimeError("failed to replace embedded controller weight marker")
    if (
        'extern "C" __global__ void controllerPoolProcess(\n'
        "    int inW,\n"
        "    int inH,\n"
        "    cudaTextureObject_t inHdrColor,\n"
        "    const float* __restrict__ weights" in source
    ):
        raise RuntimeError("failed to remove controllerPoolProcess weights argument")
    if (
        'extern "C" __global__ void controllerProcess(\n'
        "    const float* __restrict__ controllerFeatures,\n"
        "    const float* __restrict__ weights" in source
    ):
        raise RuntimeError("failed to remove controllerProcess weights argument")
    return source


# =============================================================================
# Internals: USD authoring helpers
# =============================================================================


def _mark_render_var_as_aov(stage: Usd.Stage, render_product_path: str, render_var_name: str) -> None:
    """Mark an existing input RenderVar's ``omni:rtx:aov`` as opaque.

    No-op when the RenderVar prim does not exist; the caller may not
    have populated ``orderedVars`` with an HdrColor entry. This
    matches the static PPISP writer's defensive behavior.
    """
    var_path = f"{render_product_path}/{render_var_name}"
    var_prim = stage.GetPrimAtPath(var_path)
    if var_prim.IsValid():
        var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)


def _create_controller_pool_shader(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    controller_source_file: str,
) -> UsdShade.Shader:
    """Author the per-camera PPISP controller CNN/pool Shader prim.

    ``gridDimY`` is authored as ``1`` for the untiled fallback. The Lua
    launcher expands the launch grid to ``25 x tileCountX*tileCountY`` from
    the tile-count inputs.
    """
    shader_prim_name = CONTROLLER_POOL_PRIM_NAME_TEMPLATE.format(camera_index=camera_index)
    shader_path = f"{render_product_path}/{shader_prim_name}"
    shader = UsdShade.Shader.Define(stage, shader_path)
    _author_source_asset(shader, controller_source_file, CONTROLLER_POOL_SUB_IDENTIFIER)
    _author_cuda_launch_dimensions(shader, block=(256, 1, 1), grid=(25, 1, 1))

    hdr_input = shader.CreateInput(CONTROLLER_HDR_INPUT, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{HDR_COLOR_RENDER_VAR}.omni:rtx:aov")])

    shader.CreateOutput(CONTROLLER_POOL_OUTPUT, Sdf.ValueTypeNames.Opaque)

    _author_tile_counts(shader)

    return shader


def _create_controller_shader(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    controller_source_file: str,
    prior_exposure: float,
) -> UsdShade.Shader:
    """Author the per-camera PPISP controller Shader prim.

    ``gridDimX`` is authored as ``1`` for the untiled fallback. The Lua
    launcher expands the launch grid to one MLP block per tile from the
    tile-count inputs.
    """
    shader_prim_name = CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=camera_index)
    shader_path = f"{render_product_path}/{shader_prim_name}"
    shader = UsdShade.Shader.Define(stage, shader_path)
    _author_source_asset(shader, controller_source_file, CONTROLLER_SUB_IDENTIFIER)
    _author_cuda_launch_dimensions(shader, block=(128, 1, 1), grid=(1, 1, 1))

    features_input = shader.CreateInput(CONTROLLER_FEATURES_INPUT, Sdf.ValueTypeNames.Opaque)
    features_input.GetAttr().SetConnections([Sdf.Path(f"../{CONTROLLER_FEATURES_RENDER_VAR}.omni:rtx:aov")])

    shader.CreateOutput(CONTROLLER_OUTPUT, Sdf.ValueTypeNames.Opaque)

    shader.CreateInput(CONTROLLER_PRIOR_EXPOSURE_INPUT, Sdf.ValueTypeNames.Float).Set(float(prior_exposure))

    _author_tile_counts(shader)

    return shader


def _create_controller_features_render_var(
    stage: Usd.Stage,
    render_product: Usd.Prim,
    render_product_path: str,
    controller_pool_shader: UsdShade.Shader,
) -> Usd.Prim:
    """Author the intermediate pooled controller-feature RenderVar."""
    var_path = f"{render_product_path}/{CONTROLLER_FEATURES_RENDER_VAR}"
    render_var = stage.DefinePrim(var_path, "RenderVar")
    render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(CONTROLLER_FEATURES_RENDER_VAR)
    aov_attr = render_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    aov_attr.SetConnections([controller_pool_shader.GetPath().AppendProperty(f"outputs:{CONTROLLER_POOL_OUTPUT}")])
    _append_to_ordered_vars(render_product, var_path)
    return render_var


def _create_controller_render_var(
    stage: Usd.Stage,
    render_product: Usd.Prim,
    render_product_path: str,
    controller_shader: UsdShade.Shader,
) -> Usd.Prim:
    """Author the intermediate ``ControllerParams`` RenderVar.

    Routes the controller's ``outputs:ControllerParams`` through an
    AOV-style RenderVar so Kit's runtime can resolve the connection
    via the same mechanism it uses for ``HdrColor`` / ``LdrColor``.
    Direct UsdShade Shader -> Shader connections are not enough here:
    Kit's runtime walks AOV connections, not arbitrary UsdShade
    outputs.
    """
    var_path = f"{render_product_path}/{CONTROLLER_PARAMS_RENDER_VAR}"
    render_var = stage.DefinePrim(var_path, "RenderVar")
    render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(CONTROLLER_PARAMS_RENDER_VAR)
    aov_attr = render_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    aov_attr.SetConnections([controller_shader.GetPath().AppendProperty(f"outputs:{CONTROLLER_OUTPUT}")])
    _append_to_ordered_vars(render_product, var_path)
    return render_var


def _create_auto_ppisp_shader(
    stage: Usd.Stage,
    render_product_path: str,
    ppisp: "PPISP",
    camera_index: int,
    responsivity: float,
    ppisp_camera_path: Optional[Sdf.Path] = None,
) -> UsdShade.Shader:
    """Author the automatic-parameter PPISP Shader prim.

    Responsivity / vignetting / CRF are authored as literal shader inputs.
    When a camera path is provided, the same values are also mirrored onto the
    ``<cam>_ppisp`` camera by :func:`_author_auto_camera_attributes`.
    """
    shader_path = f"{render_product_path}/{PPISP_AUTO_PRIM_NAME}"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.GetPrim().GetReferences().AddReference(PPISP_AUTO_USDA_FILE)
    _author_source_asset(shader, PPISP_AUTO_CU_FILE, PPISP_AUTO_SUB_IDENTIFIER)

    hdr_input = shader.CreateInput(PPISP_AUTO_HDR_INPUT, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{HDR_COLOR_RENDER_VAR}.omni:rtx:aov")])

    controller_input = shader.CreateInput(PPISP_AUTO_CONTROLLER_INPUT, Sdf.ValueTypeNames.Opaque)
    controller_input.GetAttr().SetConnections([Sdf.Path(f"../{CONTROLLER_PARAMS_RENDER_VAR}.omni:rtx:aov")])

    shader.CreateOutput(PPISP_AUTO_OUTPUT, Sdf.ValueTypeNames.Opaque)

    shader.CreateInput(PPISP_AUTO_RESPONSIVITY_INPUT, Sdf.ValueTypeNames.Float).Set(float(responsivity))
    _author_per_camera_vignetting(shader, ppisp, camera_index)
    _author_per_camera_crf(shader, ppisp, camera_index)
    if ppisp_camera_path is not None:
        # TODO(kit): connect these inputs to the <cam>_ppisp ppisp:* attrs
        # once SPG can resolve dynamic params from the bound RenderProduct
        # camera. The camera attrs are still authored as source-of-truth
        # metadata for downstream workflows.
        # _connect_input_to_camera(shader, PPISP_AUTO_RESPONSIVITY_INPUT, Sdf.ValueTypeNames.Float, ppisp_camera_path)
        # _connect_per_camera_vignetting_crf_to_camera(shader, ppisp_camera_path)
        pass

    _author_tile_counts(shader)

    return shader


def _create_ldr_color_render_var(
    stage: Usd.Stage,
    render_product: Usd.Prim,
    render_product_path: str,
    auto_shader: UsdShade.Shader,
) -> Usd.Prim:
    """Wire the ``LdrColor`` display AOV to the auto-PPISP output.

    Mirrors the static PPISP writer's ``LdrColor`` wiring so display
    consumers see the post-PPISP frame regardless of whether the
    static or automatic-parameter variant was authored.
    """
    ldr_var_path = f"{render_product_path}/{LDR_COLOR_RENDER_VAR}"
    ldr_var = stage.DefinePrim(ldr_var_path, "RenderVar")
    ldr_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(LDR_COLOR_RENDER_VAR)
    aov_attr = ldr_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    aov_attr.SetConnections([auto_shader.GetPath().AppendProperty(f"outputs:{PPISP_AUTO_OUTPUT}")])
    _append_to_ordered_vars(render_product, ldr_var_path)
    return ldr_var


def _author_source_asset(shader: UsdShade.Shader, source_file: str, sub_identifier: str) -> None:
    """Duplicate SPG sourceAsset metadata on the shader prim instance.

    Some Kit SPG / Fabric paths do not resolve referenced shader
    metadata when opening packaged USDZ files; authoring on the
    instance defends against that case (mirrors the static PPISP
    writer's policy).
    """
    prim = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set("sourceAsset")
    prim.CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(Sdf.AssetPath(source_file))
    prim.CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        sub_identifier
    )


def _author_cuda_launch_dimensions(
    shader: UsdShade.Shader,
    *,
    block: tuple[int, int, int],
    grid: tuple[int, int, int],
) -> None:
    """Author CUDA block/grid dimensions directly on a shader prim."""
    values = {
        "blockDimX": block[0],
        "blockDimY": block[1],
        "blockDimZ": block[2],
        "gridDimX": grid[0],
        "gridDimY": grid[1],
        "gridDimZ": grid[2],
    }
    for name, value in values.items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Int).Set(value)


def _author_per_camera_vignetting(shader: UsdShade.Shader, ppisp: "PPISP", camera_index: int) -> None:
    """Author per-camera vignetting inputs from ``ppisp.vignetting_params[camera_index]``."""
    vig = ppisp.vignetting_params[camera_index].detach().cpu().numpy()  # type: ignore[attr-defined]
    for ch in range(NUM_CHANNELS):
        suffix = CHANNEL_SUFFIXES[ch]
        shader.CreateInput(f"vignettingCenter{suffix}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(vig[ch, 0]), float(vig[ch, 1]))
        )
        shader.CreateInput(f"vignettingAlpha1{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 2]))
        shader.CreateInput(f"vignettingAlpha2{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 3]))
        shader.CreateInput(f"vignettingAlpha3{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 4]))


def _author_per_camera_crf(shader: UsdShade.Shader, ppisp: "PPISP", camera_index: int) -> None:
    """Author per-camera CRF inputs from ``ppisp.crf_params[camera_index]``."""
    crf = ppisp.crf_params[camera_index].detach().cpu().numpy()  # type: ignore[attr-defined]
    for ch in range(NUM_CHANNELS):
        suffix = CHANNEL_SUFFIXES[ch]
        shader.CreateInput(f"crfToe{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 0]))
        shader.CreateInput(f"crfShoulder{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 1]))
        shader.CreateInput(f"crfGamma{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 2]))
        shader.CreateInput(f"crfCenter{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 3]))


def _ppisp_camera_attr(camera_prim: Usd.Prim, input_name: str, value_type: Sdf.ValueTypeName) -> Usd.Attribute:
    """Create or fetch a namespaced ``ppisp:<input_name>`` custom attribute."""
    return camera_prim.CreateAttribute(f"{PPISP_ATTR_NAMESPACE}{input_name}", value_type, custom=True)


def _author_auto_camera_attributes(
    stage: Usd.Stage,
    camera_path: Sdf.Path,
    ppisp: "PPISP",
    camera_index: int,
    responsivity: float,
    controller: nn.Module,
) -> None:
    """Author auto-path PPISP source-of-truth attributes on ``<cam>_ppisp``."""
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise ValueError(f"PPISP camera prim not found at path: {camera_path}")

    _ppisp_camera_attr(camera_prim, PPISP_AUTO_RESPONSIVITY_INPUT, Sdf.ValueTypeNames.Float).Set(float(responsivity))

    vig = ppisp.vignetting_params[camera_index].detach().cpu().numpy()  # type: ignore[attr-defined]
    crf = ppisp.crf_params[camera_index].detach().cpu().numpy()  # type: ignore[attr-defined]
    for ch in range(NUM_CHANNELS):
        suffix = CHANNEL_SUFFIXES[ch]
        _ppisp_camera_attr(camera_prim, f"vignettingCenter{suffix}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(vig[ch, 0]), float(vig[ch, 1]))
        )
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha1{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 2]))
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha2{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 3]))
        _ppisp_camera_attr(camera_prim, f"vignettingAlpha3{suffix}", Sdf.ValueTypeNames.Float).Set(float(vig[ch, 4]))
        _ppisp_camera_attr(camera_prim, f"crfToe{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 0]))
        _ppisp_camera_attr(camera_prim, f"crfShoulder{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 1]))
        _ppisp_camera_attr(camera_prim, f"crfGamma{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 2]))
        _ppisp_camera_attr(camera_prim, f"crfCenter{suffix}", Sdf.ValueTypeNames.Float).Set(float(crf[ch, 3]))

    flat_weights = flatten_controller_weights(controller).astype(np.float32, copy=False)
    _ppisp_camera_attr(camera_prim, CONTROLLER_WEIGHTS_CAMERA_ATTR, Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray.FromNumpy(flat_weights)
    )


def _connect_input_to_camera(
    shader: UsdShade.Shader,
    input_name: str,
    value_type: Sdf.ValueTypeName,
    camera_path: Sdf.Path,
) -> None:
    """Declare a shader input and connect it to ``<camera>.ppisp:<input_name>``."""
    shader_input = shader.CreateInput(input_name, value_type)
    source = camera_path.AppendProperty(f"{PPISP_ATTR_NAMESPACE}{input_name}")
    shader_input.GetAttr().SetConnections([source])


def _author_tile_counts(shader: UsdShade.Shader) -> None:
    """Author untiled-default ``inputs:tileCount{X,Y}`` = 1 on a shader."""
    for name in PPISP_TILE_COUNT_INPUT_NAMES:
        shader.CreateInput(name, Sdf.ValueTypeNames.Int).Set(DEFAULT_PPISP_TILE_COUNT)


def _connect_per_camera_vignetting_crf_to_camera(shader: UsdShade.Shader, camera_path: Sdf.Path) -> None:
    """Connect auto-PPISP vignetting / CRF inputs to camera ``ppisp:*`` attrs."""
    for ch in range(NUM_CHANNELS):
        suffix = CHANNEL_SUFFIXES[ch]
        _connect_input_to_camera(shader, f"vignettingCenter{suffix}", Sdf.ValueTypeNames.Float2, camera_path)
        _connect_input_to_camera(shader, f"vignettingAlpha1{suffix}", Sdf.ValueTypeNames.Float, camera_path)
        _connect_input_to_camera(shader, f"vignettingAlpha2{suffix}", Sdf.ValueTypeNames.Float, camera_path)
        _connect_input_to_camera(shader, f"vignettingAlpha3{suffix}", Sdf.ValueTypeNames.Float, camera_path)
        _connect_input_to_camera(shader, f"crfToe{suffix}", Sdf.ValueTypeNames.Float, camera_path)
        _connect_input_to_camera(shader, f"crfShoulder{suffix}", Sdf.ValueTypeNames.Float, camera_path)
        _connect_input_to_camera(shader, f"crfGamma{suffix}", Sdf.ValueTypeNames.Float, camera_path)
        _connect_input_to_camera(shader, f"crfCenter{suffix}", Sdf.ValueTypeNames.Float, camera_path)


def _append_to_ordered_vars(render_product: Usd.Prim, var_path: str) -> None:
    """Append ``var_path`` to ``RenderProduct.orderedVars`` if not already present."""
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if not ordered_vars_rel:
        return
    targets = list(ordered_vars_rel.GetTargets())
    sdf_path = Sdf.Path(var_path)
    if sdf_path not in targets:
        targets.append(sdf_path)
        ordered_vars_rel.SetTargets(targets)


# =============================================================================
# Internals: sidecar packaging
# =============================================================================


def _load_sidecars(filenames: List[str]) -> List[NamedSerialized]:
    out: List[NamedSerialized] = []
    for filename in filenames:
        path = PPISP_SPG_DIR / filename
        if path.exists():
            with open(path, "rb") as fh:
                payload = fh.read()
            out.append(NamedSerialized(filename=filename, serialized=payload))
            log.info("Loaded PPISP controller-export SPG file: %s", filename)
        else:
            log.warning("PPISP controller-export SPG file not found: %s", path)
    return out
