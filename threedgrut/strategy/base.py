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

from typing import Optional

from threedgrut.model.model import MixtureOfGaussians


class BaseStrategy:
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        self.conf = config
        self.model = model

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        """Callback function to initialize the densification buffers."""
        pass

    def pre_backward(
        self,
        step: int,
    ) -> None:
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def post_backward(self, **kwargs) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""
        pass

    def update_gradient_buffer(self, **kwargs):
        """Callback function to update the gradient buffers during each training iteration."""
        pass
