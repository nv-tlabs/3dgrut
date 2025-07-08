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

import os
import imageio
import imageio.plugins.freeimage as fi
import numpy as np
import torch
from typing import Optional


class Environment:
    """Manages environment maps & background color for the 3DGRUT engine.

    Handles loading, tonemapping, and processing of HDR environment maps.
    Supports multiple tonemapping algorithms and exposure control.

    Attributes:
        folder (str): Path to folder containing .hdr files
        envmap (torch.Tensor): Current processed environment map, ready to use with renderer
        _hdr_data (np.ndarray): Unprocessed HDR data of currently picked  environment map
        current_name (str): Name of currently loaded environment map
        tonemapper (str): Current tonemapping algorithm
        exposure (float): Current exposure value
        saturation (float): Color saturation value
        scale (float): Scale factor for tonemapping
    """
    FIXED_ENVMAP_OPTIONS =  ['Model-Background', 'Black', 'White']
    TONEMAPPER_OPTIONS = ['None', 'Reinhard', 'Filmic']

    def __init__(self, folder: Optional[str] = None, device: Optional[torch.device] = None):

        self.device = device

        """ Name of last hdr loaded from disk, or None if no envmap is currently loaded. """
        self.current_name = 'Model-Background'

        # Tone-map settings
        """ Tonemapping algorithm used """
        self.tonemapper = 'None'
        """ Image Based Lighting Intensity (linearly scales HDR data) """
        self.ibl_intensity = 1.0
        """ Exposure value - applies 2**exposure to the image just before tone mapping """
        self.exposure = 0.0

        # HDR map fields
        """ Contains the last hdr map data loaded, before tone mapping """
        self._hdr_data = None
        """ Contains processed (tone mapped) envmap, ready to load into the renderer """
        self.envmap = None
        """ Rotation of envmap along spherical (theta, phi) axes """
        self.envmap_offset = [0.0, 0.0]

        # IO
        """ Path to a folder containing .hdr files """
        self.folder = folder
        """ List of available .hdr file names"""
        self.available_envmaps = [option for option in self.FIXED_ENVMAP_OPTIONS]

        if folder is not None:
            self.available_envmaps += [f for f in os.listdir(folder) if f.lower().endswith('.hdr')]

        self._last_update_details = dict()

    def _load_hdr(self, envmap_name: Optional[str] = None) -> Optional[torch.Tensor]:
        """Load an hdr environment map from the folder.
        The loaded environment map updates the `envmap` field of this class,
        and returns it.

        Args:
            envmap_name: Optional name of specific envmap to load. If None, No envmap is set.
        """
        if not self.available_envmaps or envmap_name in self.FIXED_ENVMAP_OPTIONS:
            self.envmap = None
            return

        # If no specific envmap requested, use first one
        if envmap_name not in self.available_envmaps:
            raise ValueError(f"Environment map {self.folder}{os.path.sep}{envmap_name} not found.")

        # Only reload HDR if it's a different file
        if envmap_name != self.current_name:
            envmap_path = os.path.join(self.folder, envmap_name)

            try:
                self._hdr_data = imageio.v2.imread(envmap_path, format='HDR-FI')
            except RuntimeError:
                # HDR loading requires a plugin: download the FreeImage library if missing
                fi.download()
                self._hdr_data = imageio.v2.imread(envmap_path, format='HDR-FI')
            self._update()   # Updates self.envmap in place

        return self.envmap

    def set_env(self, env_name: Optional[str] = None) -> None:
        if env_name == 'Model-Background':
            self._hdr_data = None
            self.envmap = torch.zeros([512, 512, 4], dtype=torch.float32, device=self.device)
        elif env_name == 'Black':
            self._hdr_data = None
            self.envmap = torch.zeros([512, 512, 4], dtype=torch.float32, device=self.device)
        elif env_name == 'White':
            self._hdr_data = None
            self.envmap = torch.ones([512, 512, 4], dtype=torch.float32, device=self.device)
        else:
            self.envmap = self._load_hdr(env_name)
        self.current_name = env_name

    def _update(self) -> None:
        """Update the processed environment map based on current settings."""
        if self._hdr_data is None:
            return
        # Apply exposure
        hdr = np.maximum(0, self._hdr_data * self.ibl_intensity)
        # Pad to a 4 channel texture because CUDA does not support 3 channels textures
        envmap = torch.tensor(hdr, device=self.device, dtype=torch.float32)
        pad = envmap.new_ones(envmap.shape[0], envmap.shape[1], 1)
        self.envmap = torch.cat([envmap, pad], dim=-1)

    def tonemap(self, hdr):

        hdr *= 2**self.exposure

        if self.tonemapper == 'Filmic':
            # Apply filmic tonemapping, assumed to be used by polyhaven
            A, B, C, D, E, F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30

            def filmic_curve(x):
                return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

            white_scale = filmic_curve(11.2)
            ldr = filmic_curve(hdr) / white_scale

        elif self.tonemapper == 'Reinhard':
            epsilon = 1e-6
            key = 0.18
            luminance = 0.2126 * hdr[:, :, 0] + 0.7152 * hdr[:, :, 1] + 0.0722 * hdr[:, :, 2]
            log_average_luminance = torch.exp(torch.mean(torch.log(epsilon + luminance)))
            scaled_luminance = key / log_average_luminance * luminance
            tone_mapped_luminance = scaled_luminance / (1.0 + scaled_luminance)
            ldr = hdr * (tone_mapped_luminance / (luminance + epsilon)).unsqueeze(2)
        elif self.tonemapper == 'None' or self.tonemapper is None:
            ldr = hdr
        else:
            raise ValueError(f"Invalid tonemapper. Must be one of {self.TONEMAPPER_OPTIONS}")

        # Clean up any NaN values
        ldr = torch.nan_to_num(ldr)
        return ldr

    def get_envmap(self) -> torch.Tensor:
        if self._is_dirty():
            self._update()
            self._cache_last_update_details()
        return self.envmap

    def get_envmap_offset(self) -> torch.Tensor:
        return torch.tensor(self.envmap_offset)

    def _cache_last_update_details(self) -> None:
        if self.envmap is None:
            self._last_update_details = dict()
        else:
            self._last_update_details = dict(
                current_name=self.current_name,
                tonemapper=self.tonemapper,
                ibl_intensity=self.ibl_intensity,
                exposure=self.exposure,
                envmap_offset=self.envmap_offset
            )

    def _is_dirty(self):
        return (
            self._last_update_details.get('current_name') != self.current_name or
            self._last_update_details.get('tonemapper') != self.tonemapper or
            self._last_update_details.get('ibl_intensity') != self.exposure or
            self._last_update_details.get('exposure') != self.exposure or
            self._last_update_details.get('envmap_offset') != self.envmap_offset
        )

    def is_ignore_envmap(self):
        return self.current_name == 'Model-Background'