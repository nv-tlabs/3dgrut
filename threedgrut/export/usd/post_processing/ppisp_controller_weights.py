# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PPISP controller weight flattening for USD export (EXPERIMENTAL).

Pure-Python (CPU/numpy/torch) half of the controller export. Provides:

1. Architecture contract.
   :data:`EXPECTED_CONTROLLER_ARCHITECTURE` defines the per-camera
   CNN/MLP shape that the runtime SPG shader hard-codes via its
   ``OFF_*`` / ``TOTAL_WEIGHTS`` constants.

2. Weight validation.
   :func:`validate_controller_architecture` raises on any shape
   mismatch; :func:`validate_controller_weights_finite` raises on any
   non-finite parameter.

3. Flattening to the SPG-readable layout.
   :func:`flatten_controller_weights` returns a 1-D ``np.float32``
   buffer in the concatenation order the SPG CUDA controller's
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

   The trunk Linear layers are found by scanning ``mlp_trunk`` in
   declaration order; ``cnn_encoder`` layers are accessed positionally.
   The ``1601`` trunk input width is
   ``cnn_feature_dim * pool_grid_h * pool_grid_w + 1``, where the
   trailing ``+1`` is the per-frame prior-exposure scalar concatenated
   to the CNN features before the trunk.

4. Per-camera controller selection.
   :func:`select_camera_controller` resolves a camera id into its
   ``_PPISPController`` instance.

The UsdShade authoring side lives in
:mod:`threedgrut.export.usd.post_processing.ppisp_controller_writer`,
which consumes this module's flat buffer.
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
    """Controller architecture sizes.

    Attributes:
        input_downsampling: Stride/kernel of the CNN ``MaxPool2d``
            applied after the first 1x1 convolution.
        cnn_in_channels: Number of input channels (RGB radiance image).
        cnn_layer_1_channels: Output channel count of the first 1x1 conv.
        cnn_layer_2_channels: Output channel count of the second 1x1 conv.
        cnn_feature_dim: Output channel count of the third 1x1 conv,
            i.e. the per-spatial-location feature dimension.
        pool_grid_size: Output spatial size of the
            ``AdaptiveAvgPool2d`` that follows the CNN.
        mlp_hidden_dim: Trunk hidden size and width of every trunk
            ``Linear`` layer.
        num_mlp_trunk_layers: Number of ``Linear`` layers in the trunk,
            each followed by a ReLU.
        color_params_per_frame: Output dimension of the color head.
        prior_exposure_dim: Per-frame prior exposure scalar
            concatenated to the flattened CNN features at the trunk
            input. Always ``1``.
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

        Computed from the architecture sizes; 241,961 with the
        defaults.
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
"""Controller export architecture; matches the byte layout the runtime
SPG CUDA controller shader (``ppisp_controller.cu``) reads back."""


EXPECTED_CONTROLLER_WEIGHTS_LEN: int = EXPECTED_CONTROLLER_ARCHITECTURE.expected_weights_len


# =============================================================================
# Public API
# =============================================================================


def select_camera_controller(ppisp_module: "PPISP", camera_id: int) -> nn.Module:
    """Return the per-camera ``_PPISPController`` at ``ppisp_module.controllers[camera_id]``.

    Args:
        ppisp_module: Trained PPISP module exposing a ``controllers``
            ``nn.ModuleList``.
        camera_id: Index into ``ppisp_module.controllers``. Must
            satisfy ``0 <= camera_id < len(controllers)``.

    Returns:
        The per-camera ``_PPISPController`` instance.

    Raises:
        TypeError: ``camera_id`` is not an ``int`` (``bool`` is
            rejected explicitly), or ``ppisp_module`` does not expose
            a ``controllers`` attribute.
        ValueError: ``ppisp_module.controllers`` is empty, or
            ``camera_id`` is out of range.
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
    """Raise if ``controller`` does not match the expected architecture.

    The check is structural (duck-typed): ``controller`` must expose
    ``cnn_encoder`` (an ``nn.Sequential`` whose elements at indices
    0/1/3/5/6 are the three 1x1 convs, the MaxPool, and the
    AdaptiveAvgPool2d), ``mlp_trunk`` (a module containing the trunk
    Linear layers in declaration order), ``exposure_head`` (a Linear)
    and ``color_head`` (a Linear).

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
    # All conv/head biases must be present; the flattened layout has no
    # bias-skip slots.
    _require_layer_bias(conv1, "cnn_encoder[0]")
    _require_layer_bias(conv2, "cnn_encoder[3]")
    _require_layer_bias(conv3, "cnn_encoder[5]")
    _require_layer_bias(exposure_head, "exposure_head")
    _require_layer_bias(color_head, "color_head")

    _check_conv(conv1, "cnn_encoder[0]", spec.cnn_in_channels, spec.cnn_layer_1_channels)
    _check_conv(conv2, "cnn_encoder[3]", spec.cnn_layer_1_channels, spec.cnn_layer_2_channels)
    _check_conv(conv3, "cnn_encoder[5]", spec.cnn_layer_2_channels, spec.cnn_feature_dim)

    # Normalize MaxPool2d kernel_size/stride to (h, w) before comparing;
    # error messages show the raw value.
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
    """Raise if any parameter in ``controller`` is non-finite.

    Iterates over every parameter (weights and biases) under
    ``controller``.

    Args:
        controller: Per-camera controller whose parameters are scanned.

    Raises:
        ValueError: At least one parameter contains a NaN or Inf value.
            The error message names the offending parameter.
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
    ``OFF_*`` offsets (see this module's docstring for the full table).
    Validates the architecture and the finiteness of every parameter
    before flattening.

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

    # Re-resolve layer handles via the typed helpers.
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
        raise ValueError(
            f"flatten_controller_weights produced {flat.size} floats; expected "
            f"{expected}. The architecture spec and the actual controller layout "
            "have diverged."
        )
    if not np.all(np.isfinite(flat)):
        # Re-check after the float32 cast.
        raise ValueError("flattened controller weights contain NaN/Inf after float32 cast; refusing to export.")

    return flat


# =============================================================================
# Internals
# =============================================================================


def _require_attr(obj: object, name: str, kind: type[_ModuleT]) -> _ModuleT:
    """Return ``getattr(obj, name)`` narrowed to ``kind``.

    Raises:
        TypeError: The attribute is missing or is not an instance of
            ``kind``.
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
    """Return ``layer.bias``; raise ``ValueError`` if it is not a tensor."""
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

    A scalar ``int`` maps to ``(value, value)``. ``None`` or any
    non-int output size is rejected with ``ValueError``.
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

    A scalar ``int`` maps to ``(value, value)``; any other type raises
    ``ValueError``. ``kind`` is folded into the error message
    (e.g. ``"MaxPool2d kernel_size"``).
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

    The flattened block is ordered as an ``out_channels x in_channels``
    matrix.
    """
    weight = conv.weight
    return (
        weight.detach().to(dtype=torch.float32).cpu().numpy().reshape(conv.out_channels, conv.in_channels).reshape(-1)
    )
