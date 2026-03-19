# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""PLY format exporter for Gaussian splatting models."""

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.utils.logger import logger


class PLYExporter(ModelExporter):
    """Exporter for PLY format files.

    Implements export functionality for Gaussian models
    in the PLY file format.
    """

    @staticmethod
    def _construct_list_of_attributes(features_albedo, features_specular, scale, rotation):
        """Construct the list of PLY attribute names."""
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        # DC coefficients (albedo)
        for i in range(features_albedo.shape[1]):
            attrs.append(f"f_dc_{i}")
        # Higher-order SH coefficients (specular)
        for i in range(features_specular.shape[1]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(scale.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(rotation.shape[1]):
            attrs.append(f"rot_{i}")
        return attrs

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

        # Use accessor to get attributes
        accessor = GaussianExportAccessor(model, conf)
        attrs = accessor.get_attributes(preactivation=True)

        num_gaussians = attrs.num_gaussians

        # Create normal vectors (placeholder, pointing up)
        mogt_nrm = np.repeat(np.array([[0, 0, 1]], dtype=np.float32), repeats=num_gaussians, axis=0)

        # Reshape specular coefficients for PLY format (channel-major layout)
        # From [N, M*3] to [N, M, 3] to [N, 3, M] to [N, M*3] (channel-major)
        num_speculars = (accessor.get_max_sh_degree() + 1) ** 2 - 1
        mogt_specular = attrs.specular.reshape((num_gaussians, num_speculars, 3))
        mogt_specular = mogt_specular.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))

        # Build PLY dtype
        dtype_full = [
            (attribute, "f4")
            for attribute in self._construct_list_of_attributes(
                attrs.albedo, mogt_specular, attrs.scales, attrs.rotations
            )
        ]

        # Create PLY element
        elements = np.empty(num_gaussians, dtype=dtype_full)
        attributes = np.concatenate(
            (attrs.positions, mogt_nrm, attrs.albedo, mogt_specular, attrs.densities, attrs.scales, attrs.rotations),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(output_path)
