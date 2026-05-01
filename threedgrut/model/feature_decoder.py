# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.nn as nn
import tinycudann as tcnn


class FeatureDecoder(nn.Module):
    """Transforms N-dimensional feature maps to RGB radiance using tiny-cuda-nn.

    Takes rendered feature maps and ray directions, encodes directions, concatenates
    with features, and decodes to RGB via a tiny-cuda-nn MLP.
    """

    def __init__(
        self,
        ray_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dir_encoding: str = "SphericalHarmonics",
        dir_encoding_degree: int = 3,
        sh_scale: float = 1.0,
        output_activation: str = "Sigmoid",
        ema_decay: float = 0.0,
        ema_start_step: int = 0,
        unpremultiply_alpha: bool = False,
    ):
        """Initialize the feature decoder.

        Args:
            ray_feature_dim: Per-ray feature dimension (rendered features input to the decoder MLP)
            hidden_dim: Hidden layer dimension for MLP decoder (default 128)
            num_layers: Number of hidden layers in the MLP (default 4)
            dir_encoding: Direction encoding type ("SphericalHarmonics" or "Frequency")
            dir_encoding_degree: Degree for direction encoding (SH degree or frequency bands; default 3)
            sh_scale: Scale applied to ray directions before encoding: (v*sh_scale+1)/2 maps to [0,1].
                      sh_scale=1 is standard unit sphere coverage; sh_scale=3 extends coverage for
                      directions beyond the unit sphere.
            output_activation: Output layer activation ("Sigmoid" for [0,1] RGB, or "ReLU")
            ema_decay: If > 0, keep EMA shadow of parameters (decay factor). 0 = no EMA.
            ema_start_step: Global step at which to start updating EMA.
            unpremultiply_alpha: If True, decode features / alpha then multiply RGB by alpha.
        """
        super().__init__()
        self.ray_feature_dim = ray_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sh_scale = sh_scale
        self.output_activation = output_activation
        self.unpremultiply_alpha = unpremultiply_alpha
        self._ema_decay = ema_decay
        self._ema_start_step = ema_start_step
        self._ema_shadow: dict[str, torch.Tensor] = {}
        self._ema_backup: dict[str, torch.Tensor] = {}

        if dir_encoding == "SphericalHarmonics":
            dir_enc = {"otype": "SphericalHarmonics", "degree": dir_encoding_degree, "n_dims_to_encode": 3}
        elif dir_encoding == "Frequency":
            dir_enc = {"otype": "Frequency", "n_frequencies": dir_encoding_degree, "n_dims_to_encode": 3}
        else:
            raise ValueError(f"Unknown dir_encoding: {dir_encoding}")

        composite_encoding_config = {
            "otype": "Composite",
            "nested": [
                {"otype": "Identity", "n_dims_to_encode": ray_feature_dim},
                dir_enc,
            ],
        }

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": output_activation,
            "n_neurons": hidden_dim,
            "n_hidden_layers": num_layers,
        }

        self.network = tcnn.NetworkWithInputEncoding(
            n_input_dims=ray_feature_dim + 3,
            n_output_dims=3,
            encoding_config=composite_encoding_config,
            network_config=network_config,
        )

        if self._ema_decay > 0:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self._ema_shadow[name] = param.data.clone()

    def ema_update(self, global_step: int) -> None:
        """Update EMA shadow when global_step >= ema_start_step. No-op if ema_decay <= 0."""
        if self._ema_decay <= 0 or global_step < self._ema_start_step:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self._ema_shadow:
                    if self._ema_shadow[name].device != param.device:
                        self._ema_shadow[name] = self._ema_shadow[name].to(param.device)
                    self._ema_shadow[name].lerp_(param.data, 1.0 - self._ema_decay)

    def apply_ema_shadow(self) -> None:
        """Use EMA weights for inference (e.g. validation). No-op if no EMA."""
        if not self._ema_shadow:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self._ema_shadow:
                    self._ema_backup[name] = param.data.clone()
                    param.data.copy_(self._ema_shadow[name])

    def restore_ema(self) -> None:
        """Restore training weights after inference. No-op if no EMA."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self._ema_backup:
                    param.data.copy_(self._ema_backup[name])
        self._ema_backup.clear()

    def ema_state_dict(self) -> dict:
        """State dict of EMA shadow for checkpoint. Empty if no EMA."""
        return {k: v.clone() for k, v in self._ema_shadow.items()}

    def load_ema_state_dict(self, state_dict: dict) -> None:
        """Load EMA shadow from checkpoint."""
        self._ema_shadow = {k: v.clone() for k, v in state_dict.items()}

    def forward(
        self,
        features: torch.Tensor,
        ray_directions: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform features and ray directions to RGB.

        Args:
            features: Input features of shape [H*W, N] or [B, H, W, N] (alpha-blended).
            ray_directions: Ray directions of shape [H*W, 3] or [B, H, W, 3].
            alpha: Optional opacity used only when unpremultiply_alpha is enabled.

        Returns:
            RGB tensor of shape [H*W, 3] or [B, H, W, 3]
        """
        features_shape = features.shape
        ray_dirs_shape = ray_directions.shape

        if len(features_shape) == 4:  # [B, H, W, N]
            B, H, W, N = features_shape
            assert ray_dirs_shape == (B, H, W, 3), f"Ray directions shape mismatch: expected {(B, H, W, 3)}, got {ray_dirs_shape}"
            assert N == self.ray_feature_dim, f"Expected {self.ray_feature_dim} features, got {N}"

            features_flat = features.reshape(B * H * W, N)
            ray_dirs_flat = ray_directions.reshape(B * H * W, 3)
            alpha_flat = alpha.reshape(B * H * W, 1) if alpha is not None else None

            rgb_flat = self._process(features_flat, ray_dirs_flat, alpha_flat)
            return rgb_flat.reshape(B, H, W, 3)

        elif len(features_shape) == 2:  # [H*W, N]
            HW, N = features_shape
            assert ray_dirs_shape == (HW, 3), f"Ray directions shape mismatch: expected {(HW, 3)}, got {ray_dirs_shape}"
            assert N == self.ray_feature_dim, f"Expected {self.ray_feature_dim} features, got {N}"
            alpha_flat = alpha.reshape(HW, 1) if alpha is not None else None
            return self._process(features, ray_directions, alpha_flat)
        else:
            raise ValueError(f"Expected input shape [B, H, W, N] or [H*W, N], got {features_shape}")

    def _process(
        self,
        features: torch.Tensor,
        ray_directions: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.unpremultiply_alpha and alpha is not None:
            alpha_safe = alpha.clamp(min=1e-8)
            features = features / alpha_safe

        # tcnn SH/Frequency encoding expects inputs in [0,1]: (v*sh_scale+1)/2
        dirs_unit_cube = (ray_directions * self.sh_scale + 1.0) * 0.5
        full_input = torch.cat([features, dirs_unit_cube], dim=-1)
        rgb = self.network(full_input)

        if self.unpremultiply_alpha and alpha is not None:
            rgb = rgb * alpha_safe

        return rgb.float()

    def regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss on decoder weights."""
        loss = torch.tensor(0.0, device=self.network.params.device)
        loss = loss + torch.sum(self.network.params**2)
        return loss

    def extra_repr(self) -> str:
        return (
            f"ray_feature_dim={self.ray_feature_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"sh_scale={self.sh_scale}, "
            f"output_activation={self.output_activation}, "
            f"unpremultiply_alpha={self.unpremultiply_alpha}"
        )
