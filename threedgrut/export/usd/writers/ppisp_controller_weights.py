# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPISP controller weight flattening for USD export (EXPERIMENTAL).

.. warning::

   The PPISP controller is the default mode used during training to
   produce realistic novel views, but its USD export path is
   **EXPERIMENTAL** and disabled by default.

   * Omniverse SPG does not currently expose first-class neural-network
     weight bindings. The controller export path flattens trained
     weights so the writer can embed them directly into generated
     per-camera CUDA sources.
   * For production USD assets you should prefer either the
     ``sh-optimized`` integration mode (PPISP folded into Gaussian SH
     coefficients) or the ``spg-runtime`` mode with a fixed reference
     ``(camera_id, frame_id)`` pair (time-sampled or single-frame
     authoring.
   * The controller export path should only be used for research and
     experimental workflows that knowingly accept the runtime
     compatibility risk.

This module is the pure-Python (CPU/numpy/torch) half of the
controller export. It is responsible for:

1. **Locking the architecture contract.**
   :data:`EXPECTED_CONTROLLER_ARCHITECTURE` is the single source of
   truth for the per-camera CNN/MLP shape that the runtime SPG
   shader hard-codes via its ``OFF_*`` / ``TOTAL_WEIGHTS`` constants.
   Any change to the PPISP controller defaults requires a
   matching update to the SPG shader and to this constant; a
   mismatch is caught by :func:`validate_controller_architecture`
   before any USD authoring runs.

2. **Validating trained weights.**
   :func:`validate_controller_architecture` fails loudly on any
   shape mismatch and :func:`validate_controller_weights_finite`
   rejects any non-finite parameter so a corrupt checkpoint never
   reaches USD.

3. **Flattening to the SPG-readable layout.**
   :func:`flatten_controller_weights` returns a 1-D ``np.float32``
   buffer in the exact concatenation order the SPG CUDA controller's
   ``OFF_*`` offsets expect:

   ===========================  ==========================  ================
   Layer                         Shape (per element)         Element count
   ===========================  ==========================  ================
   ``cnn_encoder[0]`` weight     ``(16, 3, 1, 1)``           ``16 * 3 = 48``
   ``cnn_encoder[0]`` bias       ``(16,)``                   ``16``
   ``cnn_encoder[3]`` weight     ``(32, 16, 1, 1)``          ``32 * 16 = 512``
   ``cnn_encoder[3]`` bias       ``(32,)``                   ``32``
   ``cnn_encoder[5]`` weight     ``(64, 32, 1, 1)``          ``64 * 32 = 2048``
   ``cnn_encoder[5]`` bias       ``(64,)``                   ``64``
   ``mlp_trunk[0]`` weight       ``(128, 1601)``             ``128 * 1601 = 204928``
   ``mlp_trunk[0]`` bias         ``(128,)``                  ``128``
   ``mlp_trunk[1]`` weight       ``(128, 128)``              ``128 * 128 = 16384``
   ``mlp_trunk[1]`` bias         ``(128,)``                  ``128``
   ``mlp_trunk[2]`` weight       ``(128, 128)``              ``128 * 128 = 16384``
   ``mlp_trunk[2]`` bias         ``(128,)``                  ``128``
   ``exposure_head`` weight      ``(1, 128)``                ``128``
   ``exposure_head`` bias        ``(1,)``                    ``1``
   ``color_head`` weight         ``(8, 128)``                ``8 * 128 = 1024``
   ``color_head`` bias           ``(8,)``                    ``8``
   **Total**                                                 **241,961**
   ===========================  ==========================  ================

   The MLP trunk layer count is found by scanning ``mlp_trunk`` for
   ``nn.Linear`` modules in declaration order; the layer indices
   into ``cnn_encoder`` are positional (matching the construction
   order of the default ``ppisp._PPISPController`` ``nn.Sequential``).
   The ``128 * 1601 = 204928`` multiplier reflects
   ``cnn_feature_dim * pool_grid_h * pool_grid_w + 1`` (the trailing
   ``+1`` is the per-frame prior-exposure scalar concatenated to
   the CNN features before the trunk).

4. **Selecting the per-camera controller.**
   :func:`select_camera_controller` resolves a validated camera id
   into the corresponding ``_PPISPController`` instance. This is
   the boundary between the PPISP-module-level reference resolution
   in the PPISP module and the
   per-controller flattening here.

This module is intentionally pure Python -- no USD, no Omniverse,
no SPG asset references. The UsdShade authoring side (Shader prim
creation, AOV wiring, RenderVar wiring) lives in
:mod:`threedgrut.export.usd.writers.ppisp_controller_writer`, which consumes
this module's flat buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn


if TYPE_CHECKING:
    from ppisp import PPISP  # type: ignore[import-not-found]


_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


# =============================================================================
# Architecture contract
# =============================================================================


@dataclass(frozen=True)
class ControllerArchitectureSpec:
    """Hard-coded controller architecture sizes.

    These mirror the default ``ppisp._PPISPController`` architecture
    *and* the runtime SPG shader's ``OFF_*`` / ``TOTAL_WEIGHTS``
    constants. The two must stay in lockstep; this dataclass exists
    to make the contract explicit on the Python side and to centralize
    the size used by :func:`validate_controller_architecture`.

    Attributes:
        input_downsampling: Stride/kernel of the CNN ``MaxPool2d``
            applied right after the first 1x1 convolution. Used to
            cross-check the controller's pooling layer.
        cnn_in_channels: Number of input channels (RGB radiance image).
        cnn_layer_1_channels: Output channel count of the first 1x1
            conv. Locked to 16.
        cnn_layer_2_channels: Output channel count of the second 1x1
            conv. Locked to 32.
        cnn_feature_dim: Output channel count of the third 1x1 conv,
            i.e. the per-spatial-location feature dimension. Locked
            to 64.
        pool_grid_size: Output spatial size of the
            ``AdaptiveAvgPool2d`` that follows the CNN. Locked to
            ``(5, 5)``.
        mlp_hidden_dim: Trunk hidden size and width of every trunk
            ``Linear`` layer. Locked to 128.
        num_mlp_trunk_layers: Number of ``Linear`` layers in the
            trunk (each followed by a ReLU). Locked to 3.
        color_params_per_frame: Output dimension of the color head;
            mirrors ``ppisp.COLOR_PARAMS_PER_FRAME``. Locked to 8.
        prior_exposure_dim: The trunk receives the flattened CNN
            features concatenated with a per-frame prior exposure
            scalar. This is always ``1``.
    """

    input_downsampling: int = 3
    cnn_in_channels: int = 3
    cnn_layer_1_channels: int = 16
    cnn_layer_2_channels: int = 32
    cnn_feature_dim: int = 64
    pool_grid_size: Tuple[int, int] = (5, 5)
    mlp_hidden_dim: int = 128
    num_mlp_trunk_layers: int = 3
    color_params_per_frame: int = 8
    prior_exposure_dim: int = 1

    @property
    def trunk_input_dim(self) -> int:
        """Flattened CNN feature length plus the prior-exposure scalar."""
        pool_h, pool_w = self.pool_grid_size
        return self.cnn_feature_dim * pool_h * pool_w + self.prior_exposure_dim

    @property
    def expected_weights_len(self) -> int:
        """Total number of float32 elements in the flattened weight buffer.

        Computed from the architecture sizes; locks the SPG shader's
        ``TOTAL_WEIGHTS`` constant. With the current defaults this
        evaluates to 241,961.
        """
        # Three 1x1 conv layers (weight + bias).
        conv1 = self.cnn_layer_1_channels * self.cnn_in_channels + self.cnn_layer_1_channels
        conv2 = self.cnn_layer_2_channels * self.cnn_layer_1_channels + self.cnn_layer_2_channels
        conv3 = self.cnn_feature_dim * self.cnn_layer_2_channels + self.cnn_feature_dim

        # MLP trunk: first layer reads trunk_input_dim, subsequent
        # layers map mlp_hidden_dim -> mlp_hidden_dim.
        trunk_first = self.mlp_hidden_dim * self.trunk_input_dim + self.mlp_hidden_dim
        trunk_rest = (self.num_mlp_trunk_layers - 1) * (self.mlp_hidden_dim * self.mlp_hidden_dim + self.mlp_hidden_dim)

        # Two output heads.
        exposure_head = 1 * self.mlp_hidden_dim + 1
        color_head = self.color_params_per_frame * self.mlp_hidden_dim + self.color_params_per_frame

        return conv1 + conv2 + conv3 + trunk_first + trunk_rest + exposure_head + color_head


EXPECTED_CONTROLLER_ARCHITECTURE = ControllerArchitectureSpec()
"""Single source of truth for the controller export architecture.

Locks the byte layout that the runtime SPG CUDA controller shader
(``ppisp_controller.cu``) reads back.
"""


# Convenience aliases for tests and the writer-side module. Computing
# these here keeps the expected length out of any caller's hands.
EXPECTED_CONTROLLER_WEIGHTS_LEN: int = EXPECTED_CONTROLLER_ARCHITECTURE.expected_weights_len


# =============================================================================
# Public API
# =============================================================================


def select_camera_controller(ppisp_module: "PPISP", camera_id: int) -> nn.Module:
    """Resolve ``ppisp_module.controllers[camera_id]`` with fail-loud guards.

    The boundary between PPISP-module-level reference resolution
    (see PPISP reference selection in the USD exporter)
    and per-controller weight flattening (this module). Returns
    the underlying ``_PPISPController`` ``nn.Module`` so the caller
    can pass it to :func:`flatten_controller_weights`.

    Args:
        ppisp_module: Trained PPISP module exposing a ``controllers``
            ``nn.ModuleList`` populated when the controller was
            enabled at construction time.
        camera_id: Index into ``ppisp_module.controllers``. Must
            satisfy ``0 <= camera_id < len(controllers)``.

    Returns:
        The per-camera ``_PPISPController`` instance.

    Raises:
        TypeError: ``camera_id`` is not an ``int`` (``bool`` is
            rejected explicitly), or ``ppisp_module`` does not
            expose a ``controllers`` attribute.
        ValueError: ``ppisp_module.controllers`` is empty (the
            controller was disabled at construction time, i.e.
            ``PPISPConfig.use_controller=False``), or ``camera_id``
            is out of range.
    """
    if isinstance(camera_id, bool) or not isinstance(camera_id, int):
        raise TypeError(f"camera_id must be int, got {type(camera_id).__name__}")
    if not hasattr(ppisp_module, "controllers"):
        raise TypeError(
            "ppisp_module is missing required attribute 'controllers'. "
            "Controller export requires a PPISP module trained with "
            "PPISPConfig(use_controller=True)."
        )
    controllers = ppisp_module.controllers  # type: ignore[attr-defined]
    num_controllers = len(controllers)
    if num_controllers == 0:
        raise ValueError(
            "ppisp_module.controllers is empty. Controller export requires a PPISP "
            "module trained with PPISPConfig(use_controller=True); the configured "
            "module was trained with use_controller=False."
        )
    if camera_id < 0 or camera_id >= num_controllers:
        raise ValueError(
            f"camera_id must be in [0, {num_controllers - 1}], got {camera_id} "
            f"(len(ppisp_module.controllers) = {num_controllers})"
        )
    return controllers[camera_id]


def validate_controller_architecture(
    controller: nn.Module,
    spec: ControllerArchitectureSpec = EXPECTED_CONTROLLER_ARCHITECTURE,
) -> None:
    """Fail loudly if ``controller`` does not match the expected architecture.

    Each constraint raises a distinct :class:`ValueError` with a
    precise message so users can diagnose architecture drift quickly.
    The runtime SPG shader hard-codes the expected sizes via its
    ``OFF_*`` constants; any mismatch here would cause silent
    corruption at runtime, hence the fail-loud policy.

    The check is structural (duck-typed): it does not require
    ``controller`` to be the private ``ppisp._PPISPController``
    class, only that it expose ``cnn_encoder`` (an
    ``nn.Sequential`` whose elements at indices 0/1/3/5/6 are the
    three 1x1 convs, the MaxPool, and the AdaptiveAvgPool2d in the
    same construction order as the upstream implementation),
    ``mlp_trunk`` (a module containing the trunk Linear layers in
    declaration order, discoverable via iteration), ``exposure_head``
    (a Linear) and ``color_head`` (a Linear).

    Args:
        controller: Per-camera controller to validate.
        spec: Architecture spec to validate against. Defaults to the
            module-global :data:`EXPECTED_CONTROLLER_ARCHITECTURE`.

    Raises:
        TypeError: ``controller`` is missing one of the required
            top-level attributes (``cnn_encoder``, ``mlp_trunk``,
            ``exposure_head``, ``color_head``) or one of those
            attributes is the wrong kind of module.
        ValueError: One of the architecture sizes does not match the
            spec.
    """
    cnn_encoder = _require_attr(controller, "cnn_encoder", nn.Sequential)
    mlp_trunk = _require_attr(controller, "mlp_trunk", nn.Module)
    exposure_head = _require_attr(controller, "exposure_head", nn.Linear)
    color_head = _require_attr(controller, "color_head", nn.Linear)

    conv1 = _require_sequential_layer(cnn_encoder, "cnn_encoder", 0, nn.Conv2d)
    maxpool = _require_sequential_layer(cnn_encoder, "cnn_encoder", 1, nn.MaxPool2d)
    conv2 = _require_sequential_layer(cnn_encoder, "cnn_encoder", 3, nn.Conv2d)
    conv3 = _require_sequential_layer(cnn_encoder, "cnn_encoder", 5, nn.Conv2d)
    avgpool = _require_sequential_layer(cnn_encoder, "cnn_encoder", 6, nn.AdaptiveAvgPool2d)
    # Layout depends on conv biases being present; required since the
    # SPG shader's OFF_*_BIAS offsets are non-skippable.
    _require_layer_bias(conv1, "cnn_encoder[0]")
    _require_layer_bias(conv2, "cnn_encoder[3]")
    _require_layer_bias(conv3, "cnn_encoder[5]")
    _require_layer_bias(exposure_head, "exposure_head")
    _require_layer_bias(color_head, "color_head")

    _check_conv(conv1, "cnn_encoder[0]", spec.cnn_in_channels, spec.cnn_layer_1_channels)
    _check_conv(conv2, "cnn_encoder[3]", spec.cnn_layer_1_channels, spec.cnn_layer_2_channels)
    _check_conv(conv3, "cnn_encoder[5]", spec.cnn_layer_2_channels, spec.cnn_feature_dim)

    # ``nn.MaxPool2d`` stores ``kernel_size`` / ``stride`` exactly as passed in
    # (no ``_pair`` normalization), so a controller built with the tuple form
    # ``MaxPool2d(kernel_size=(k, k), stride=(k, k))`` would compare unequal to
    # the scalar spec value even though the architecture is correct. Normalize
    # both sides to ``(h, w)`` before comparing; keep the original messages and
    # show the raw value in the "got" for easier debugging.
    expected_downsampling_pair = (spec.input_downsampling, spec.input_downsampling)
    if _normalize_pool_int_pair(maxpool.kernel_size, "MaxPool2d kernel_size") != expected_downsampling_pair:
        raise ValueError(
            f"cnn_encoder[1] (MaxPool2d) kernel_size must be {spec.input_downsampling}, got {maxpool.kernel_size}"
        )
    if _normalize_pool_int_pair(maxpool.stride, "MaxPool2d stride") != expected_downsampling_pair:
        raise ValueError(f"cnn_encoder[1] (MaxPool2d) stride must be {spec.input_downsampling}, got {maxpool.stride}")

    expected_grid = spec.pool_grid_size
    avgpool_grid = _normalize_avgpool_output_size(avgpool.output_size)
    if avgpool_grid != expected_grid:
        raise ValueError(f"cnn_encoder[6] (AdaptiveAvgPool2d) output_size must be {expected_grid}, got {avgpool_grid}")

    trunk_linear_layers = _collect_trunk_linear_layers(mlp_trunk)
    if len(trunk_linear_layers) != spec.num_mlp_trunk_layers:
        raise ValueError(
            f"mlp_trunk must contain exactly {spec.num_mlp_trunk_layers} Linear layers, got {len(trunk_linear_layers)}"
        )
    if trunk_linear_layers[0].in_features != spec.trunk_input_dim:
        raise ValueError(
            f"mlp_trunk[0].in_features must be {spec.trunk_input_dim} "
            f"(cnn_feature_dim * pool_h * pool_w + 1), got "
            f"{trunk_linear_layers[0].in_features}"
        )
    for idx, layer in enumerate(trunk_linear_layers):
        if layer.out_features != spec.mlp_hidden_dim:
            raise ValueError(f"mlp_trunk[{idx}].out_features must be {spec.mlp_hidden_dim}, got {layer.out_features}")
        if idx > 0 and layer.in_features != spec.mlp_hidden_dim:
            raise ValueError(f"mlp_trunk[{idx}].in_features must be {spec.mlp_hidden_dim}, got {layer.in_features}")

    if exposure_head.in_features != spec.mlp_hidden_dim:
        raise ValueError(f"exposure_head.in_features must be {spec.mlp_hidden_dim}, got {exposure_head.in_features}")
    if exposure_head.out_features != 1:
        raise ValueError(f"exposure_head.out_features must be 1, got {exposure_head.out_features}")
    if color_head.in_features != spec.mlp_hidden_dim:
        raise ValueError(f"color_head.in_features must be {spec.mlp_hidden_dim}, got {color_head.in_features}")
    if color_head.out_features != spec.color_params_per_frame:
        raise ValueError(
            f"color_head.out_features must be {spec.color_params_per_frame}, got {color_head.out_features}"
        )


def validate_controller_weights_finite(controller: nn.Module) -> None:
    """Reject any non-finite parameter in ``controller``.

    A NaN or Inf in the trained weights silently corrupts the
    runtime SPG shader's predictions; we refuse to export rather
    than ship a broken USD asset. Iterates over every parameter
    (weights *and* biases) under ``controller``; the controller is
    a small CNN so the cost is negligible.

    Args:
        controller: Per-camera controller whose parameters will be
            scanned.

    Raises:
        ValueError: At least one parameter contains a NaN or Inf
            value. The error message names the offending parameter.
    """
    for name, param in controller.named_parameters():
        if not torch.isfinite(param).all():
            raise ValueError(
                f"controller parameter {name!r} contains NaN/Inf values; "
                "refusing to export. Investigate the trained PPISP "
                "checkpoint before retrying."
            )


def flatten_controller_weights(
    controller: nn.Module,
    spec: ControllerArchitectureSpec = EXPECTED_CONTROLLER_ARCHITECTURE,
) -> np.ndarray:
    """Concatenate trained controller weights into one float32 array.

    The flattened layout matches the runtime SPG controller shader's
    ``OFF_*`` offsets (see this module's docstring for the full
    table). Validates the architecture and the finiteness of every
    parameter before flattening; both checks are fail-loud.

    Args:
        controller: Per-camera controller to flatten.
        spec: Architecture spec to validate against. Defaults to the
            module-global :data:`EXPECTED_CONTROLLER_ARCHITECTURE`.

    Returns:
        A 1-D ``np.float32`` array whose length is
        :attr:`ControllerArchitectureSpec.expected_weights_len`.

    Raises:
        TypeError: ``controller`` is missing one of the required
            top-level attributes (delegated from
            :func:`validate_controller_architecture`).
        ValueError: Architecture mismatch, or a non-finite parameter,
            or (defensive) the concatenated buffer length does not
            match the spec.
    """
    validate_controller_architecture(controller, spec)
    validate_controller_weights_finite(controller)

    # Re-resolve handles via the same typed helpers used by validation
    # so the navigation is mypy-narrowed end-to-end. The duplicated
    # walks are negligible for a controller of this size.
    cnn_encoder = _require_attr(controller, "cnn_encoder", nn.Sequential)
    mlp_trunk = _require_attr(controller, "mlp_trunk", nn.Module)
    exposure_head = _require_attr(controller, "exposure_head", nn.Linear)
    color_head = _require_attr(controller, "color_head", nn.Linear)

    conv1 = _require_sequential_layer(cnn_encoder, "cnn_encoder", 0, nn.Conv2d)
    conv2 = _require_sequential_layer(cnn_encoder, "cnn_encoder", 3, nn.Conv2d)
    conv3 = _require_sequential_layer(cnn_encoder, "cnn_encoder", 5, nn.Conv2d)
    trunk_linear_layers = _collect_trunk_linear_layers(mlp_trunk)

    parts: List[np.ndarray] = [
        _conv_weight_to_np(conv1),
        _to_numpy_flat(_require_layer_bias(conv1, "cnn_encoder[0]")),
        _conv_weight_to_np(conv2),
        _to_numpy_flat(_require_layer_bias(conv2, "cnn_encoder[3]")),
        _conv_weight_to_np(conv3),
        _to_numpy_flat(_require_layer_bias(conv3, "cnn_encoder[5]")),
        _to_numpy_flat(trunk_linear_layers[0].weight),
        _to_numpy_flat(_require_layer_bias(trunk_linear_layers[0], "mlp_trunk[0]")),
        _to_numpy_flat(trunk_linear_layers[1].weight),
        _to_numpy_flat(_require_layer_bias(trunk_linear_layers[1], "mlp_trunk[1]")),
        _to_numpy_flat(trunk_linear_layers[2].weight),
        _to_numpy_flat(_require_layer_bias(trunk_linear_layers[2], "mlp_trunk[2]")),
        _to_numpy_flat(exposure_head.weight),
        _to_numpy_flat(_require_layer_bias(exposure_head, "exposure_head")),
        _to_numpy_flat(color_head.weight),
        _to_numpy_flat(_require_layer_bias(color_head, "color_head")),
    ]
    flat = np.concatenate(parts).astype(np.float32, copy=False)

    expected = spec.expected_weights_len
    if flat.size != expected:
        # Defensive: validate_controller_architecture should have
        # already prevented this, but a layout/spec drift is the most
        # damaging failure mode for this module.
        raise ValueError(
            f"flatten_controller_weights produced {flat.size} floats; expected "
            f"{expected}. The architecture spec and the actual controller layout "
            "have diverged."
        )
    if not np.all(np.isfinite(flat)):
        # Defensive: validate_controller_weights_finite should have
        # already caught this; the post-check guards against any
        # numerical loss introduced by the float32 cast.
        raise ValueError("flattened controller weights contain NaN/Inf after float32 cast; refusing to export.")

    return flat


# =============================================================================
# Internals
# =============================================================================


def _require_attr(obj: object, name: str, kind: type[_ModuleT]) -> _ModuleT:
    """Return ``getattr(obj, name)`` narrowed to the requested ``kind``.

    Both the missing-attribute and wrong-type cases raise
    :class:`TypeError` so callers can rely on a single exception
    class for structural failures.
    """
    if not hasattr(obj, name):
        raise TypeError(
            f"controller is missing required attribute {name!r}. Expected a {kind.__name__} on the controller."
        )
    value = getattr(obj, name)
    if not isinstance(value, kind):
        raise TypeError(f"controller.{name} must be a {kind.__name__}, got {type(value).__name__}")
    return value


def _require_sequential_layer(
    sequential: nn.Sequential, container_name: str, index: int, kind: type[_ModuleT]
) -> _ModuleT:
    """Index into ``sequential`` and require the right module type at ``index``."""
    if index >= len(sequential):
        raise ValueError(f"controller.{container_name} must have at least {index + 1} children, got {len(sequential)}")
    layer = sequential[index]
    if not isinstance(layer, kind):
        raise ValueError(f"controller.{container_name}[{index}] must be a {kind.__name__}, got {type(layer).__name__}")
    return layer


def _require_layer_bias(layer: nn.Module, name: str) -> torch.Tensor:
    """Return ``layer.bias`` and fail loudly if it is ``None``.

    The flatten layout has no bias-skip slot; a layer constructed
    with ``bias=False`` would silently shift the byte layout in the
    SPG shader, so we refuse to export.
    """
    bias = getattr(layer, "bias", None)
    if not isinstance(bias, torch.Tensor):
        raise ValueError(f"{name} must have a bias parameter; got {type(bias).__name__}")
    return bias


def _check_conv(conv: nn.Conv2d, name: str, expected_in: int, expected_out: int) -> None:
    """Assert that ``conv`` is a 1x1 ``Conv2d`` with the expected channels."""
    if conv.in_channels != expected_in or conv.out_channels != expected_out:
        raise ValueError(
            f"{name} must be Conv2d({expected_in} -> {expected_out}), got "
            f"Conv2d({conv.in_channels} -> {conv.out_channels})"
        )
    if conv.kernel_size != (1, 1):
        raise ValueError(f"{name} kernel_size must be (1, 1), got {conv.kernel_size}")


def _collect_trunk_linear_layers(mlp_trunk: nn.Module) -> List[nn.Linear]:
    """Return the trunk's ``nn.Linear`` children in declaration order."""
    layers: List[nn.Linear] = [m for m in mlp_trunk.modules() if isinstance(m, nn.Linear)]
    return layers


def _normalize_avgpool_output_size(value: object) -> Tuple[int, int]:
    """Normalize ``AdaptiveAvgPool2d.output_size`` to a 2-tuple of ints.

    PyTorch accepts either a scalar ``int`` (interpreted as a square
    output) or a 2-tuple. ``None`` (or a tuple containing ``None``)
    is rejected because the controller export needs a fixed
    spatial size to size the trunk input.
    """
    if isinstance(value, bool):
        raise ValueError(f"AdaptiveAvgPool2d output_size must be int or (int, int), got {value!r}")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        h, w = value
        if isinstance(h, int) and isinstance(w, int):
            return (h, w)
    raise ValueError(f"AdaptiveAvgPool2d output_size must be int or (int, int), got {value!r}")


def _normalize_pool_int_pair(value: object, kind: str) -> Tuple[int, int]:
    """Normalize an ``nn.MaxPool2d`` ``kernel_size`` / ``stride`` to ``(h, w)``.

    Unlike ``nn.Conv2d``, ``nn.MaxPool2d`` stores these attributes exactly as
    the caller passed them in (no ``_pair`` normalization), so equality
    comparisons against a scalar spec value require us to canonicalize first.
    ``kind`` is folded into the error message so callers don't need their
    own wrapping (e.g. ``"MaxPool2d kernel_size"``).
    """
    if isinstance(value, bool):
        raise ValueError(f"{kind} must be int or (int, int), got {value!r}")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        h, w = value
        if isinstance(h, int) and isinstance(w, int):
            return (h, w)
    raise ValueError(f"{kind} must be int or (int, int), got {value!r}")


def _to_numpy_flat(tensor: torch.Tensor) -> np.ndarray:
    """Detach, cast to float32, move to CPU, return a flat ``np.ndarray``."""
    return tensor.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)


def _conv_weight_to_np(conv: nn.Conv2d) -> np.ndarray:
    """Flatten a 1x1 conv weight ``[out, in, 1, 1]`` row-major to ``[out * in]``.

    The runtime SPG shader interprets the flattened block as a
    ``out_channels x in_channels`` matrix, so the reshape order
    (``[out, in]``) is part of the export contract.
    """
    weight = conv.weight
    return (
        weight.detach().to(dtype=torch.float32).cpu().numpy().reshape(conv.out_channels, conv.in_channels).reshape(-1)
    )
