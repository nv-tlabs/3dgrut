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
from threedgrut.model.model import MixtureOfGaussians


def join_gaussians(*gaussians):
    """ Concatenates multiple MixtureOfGaussians models into a single new MixtureOfGaussians object.
    For simplicity, non-concatenable attributes will be picked from the first MixtureOfGaussians arg,
    i.e.: max_sh_degree, n_active_features and scene_extent.
    """
    main_gaussian = gaussians[0]
    composition = MixtureOfGaussians(conf=main_gaussian.conf, scene_extent=main_gaussian.scene_extent)
    fields = ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']
    concatenated_fields = {}

    for field in fields:
        concatenated_fields[field] = []
    for gaussian in gaussians:
        for field in fields:
            field_value = getattr(gaussian, field)
            concatenated_fields[field].append(field_value)
    for field in fields:
        concatenated_tensor = torch.cat(concatenated_fields[field], dim=0)
        setattr(composition, field, torch.nn.Parameter(concatenated_tensor))

    composition.max_sh_degree = main_gaussian.max_sh_degree
    composition.n_active_features = main_gaussian.n_active_features
    composition.scene_extent = main_gaussian.scene_extent
    return composition
