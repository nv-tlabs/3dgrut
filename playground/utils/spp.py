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

from __future__ import annotations
import torch
from playground.utils.rng import rng_torch_low_discrepancy
from playground.utils.antialiasing import StratifiedRayJitter


####################################
##    --- Samples per Pixel ---   ##
###################################

class SPP:
    def __init__(self, mode: str = 'msaa', spp=4, batch_size=1, device='cuda'):
        """ Sampling mode -
            - none: sample center of each pixel
            - independent random: straightforward torch's random, each number is IID
            - msaa: Fixed pattern borrowed from DirectX's multisampling antialiasing
            - low discrepancy sequences: uses Owen's scrambling of Sobol sequence, converges faster for accumulated spp
        """
        mode = mode.lower()
        assert mode in ('none', 'independent_random', 'low_discrepancy_seq', 'msaa')
        assert mode == 'msaa' and spp in (1, 2, 4, 8, 16), 'MSAA supports only power of 2 spp within [2,16]'

        self.spp = spp
        self.mode = mode
        self.device = device
        self.batch_size = 1
        # Running index which is updated for any accumulated frame
        self.spp_accumulated_for_frame = 1

        if mode == 'msaa':
            self.msaa = StratifiedRayJitter(fixed_pattern=True, num_samples=spp, device=device)

    def reset_accumulation(self):
        self.spp_accumulated_for_frame = self.batch_size

    def has_more_to_accumulate(self):
        return self.spp_accumulated_for_frame <= self.spp

    def __call__(self, img_h, img_w):
        # Returns a per-pixel jitter in range [-0.5, 0.5]
        if self.mode == 'none':
            jitter = torch.zeros([img_h, img_w, 2], device=self.device)
        elif self.mode == 'msaa':
            jitter = 0.5 - self.msaa([img_h, img_w])
        elif self.mode == 'low_discrepancy_seq':
            pixel_x = torch.arange(img_w, device=self.device).unsqueeze(0).expand(img_h, img_w)
            pixel_y = torch.arange(img_h, device=self.device).unsqueeze(1).expand(img_h, img_w)
            ray_count = img_h * img_w
            seed = (pixel_x.long() * 19349663 + pixel_y.long() * 96925573).reshape(ray_count) & 0xFFFFFFFF
            index = (seed.new_ones(ray_count) * self.spp_accumulated_for_frame).long()
            jitter = rng_torch_low_discrepancy(index, seed)
            jitter = torch.stack(jitter, dim=1) - 0.5
            jitter = jitter.reshape(img_h, img_w, 2)
        else:
            raise ValueError('Unknown spp mode')

        self.spp_accumulated_for_frame += 1

        return jitter
