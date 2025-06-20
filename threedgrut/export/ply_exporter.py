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

from pathlib import Path

import numpy as np
import torch

from plyfile import PlyData, PlyElement

from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.utils.logger import logger


class PLYExporter(ModelExporter):
    """Exporter for PLY format files.

    Implements export functionality for Gaussian models
    in the PLY file format.
    """

    @staticmethod
    def _construct_list_of_attributes(features_albedo, features_specular, scale, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_albedo.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_specular.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, **kwargs) -> None:
        """Export the model to a PLY file.

        Args:
            model: The model to export (must implement ExportableModel)
            output_path: Path where the PLY file will be saved
            dataset: Optional dataset (not used for PLY export)
            conf: Optional configuration (not used for PLY export)
            **kwargs: Additional parameters (not used for PLY export)
        """
        logger.info(f"exporting ply file to {output_path}...")
        positions = model.get_positions().detach().cpu().numpy()
        num_gaussians = positions.shape[0]
        mogt_nrm = np.repeat(
            np.array([[0, 0, 1]], dtype=np.float32), repeats=num_gaussians, axis=0)
        mogt_albedo = model.get_features_albedo().detach().cpu().numpy()
        num_speculars = (model.get_max_n_features() + 1) ** 2 - 1
        mogt_specular = model.get_features_specular().detach(
        ).cpu().numpy().reshape((num_gaussians, num_speculars, 3))
        mogt_specular = mogt_specular.transpose(
            0, 2, 1).reshape((num_gaussians, num_speculars*3))
        mogt_densities = model.get_density(
            preactivation=True).detach().cpu().numpy()
        mogt_scales = model.get_scale(
            preactivation=True).detach().cpu().numpy()
        mogt_rotation = model.get_rotation(
            preactivation=True).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in PLYExporter._construct_list_of_attributes(
            mogt_albedo, mogt_specular, mogt_scales, mogt_rotation)]

        elements = np.empty(num_gaussians, dtype=dtype_full)
        attributes = np.concatenate((positions, mogt_nrm, mogt_albedo,
                                    mogt_specular, mogt_densities, mogt_scales, mogt_rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
