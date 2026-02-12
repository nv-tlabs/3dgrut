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

"""PLY format importer for Gaussian splatting data."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from plyfile import PlyData

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.importers.base import FormatImporter

logger = logging.getLogger(__name__)


class PLYImporter(FormatImporter):
    """Importer for PLY format files.

    PLY files store pre-activation values (raw parameters).
    """

    def __init__(self, max_sh_degree: int = 3):
        """Initialize PLY importer.

        Args:
            max_sh_degree: Maximum SH degree to expect/load (default 3)
        """
        self.max_sh_degree = max_sh_degree

    @property
    def stores_preactivation(self) -> bool:
        return True

    def load(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load PLY file into GaussianAttributes.

        Args:
            path: Path to PLY file

        Returns:
            Tuple of (GaussianAttributes, ModelCapabilities)
        """
        logger.info(f"Loading PLY file: {path}")
        plydata = PlyData.read(str(path))

        # Extract positions
        positions = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        ).astype(np.float32)

        num_gaussians = positions.shape[0]

        # Extract density/opacity
        densities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        # Extract albedo (DC coefficients)
        albedo = np.zeros((num_gaussians, 3), dtype=np.float32)
        albedo[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        albedo[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        albedo[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        # Extract specular (higher-order SH coefficients)
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

        num_speculars = (self.max_sh_degree + 1) ** 2 - 1
        expected_extra_f_count = 3 * num_speculars

        # Determine actual SH degree from file
        if len(extra_f_names) > 0:
            actual_sh_degree = int(np.sqrt(len(extra_f_names) // 3 + 1)) - 1
        else:
            actual_sh_degree = 0

        specular = np.zeros((num_gaussians, num_speculars * 3), dtype=np.float32)
        if len(extra_f_names) == expected_extra_f_count:
            # Full spherical harmonics data available
            for idx, attr_name in enumerate(extra_f_names):
                specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Convert from channel-major to feature-major layout
            specular = specular.reshape((num_gaussians, 3, num_speculars))
            specular = specular.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))
        elif len(extra_f_names) == 0:
            logger.info("PLY file only contains DC components, higher-order SH set to zero")
        elif len(extra_f_names) < expected_extra_f_count:
            # Partial SH - load what's available
            actual_speculars = len(extra_f_names) // 3
            temp_specular = np.zeros((num_gaussians, len(extra_f_names)), dtype=np.float32)
            for idx, attr_name in enumerate(extra_f_names):
                temp_specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Convert layout and pad
            temp_specular = temp_specular.reshape((num_gaussians, 3, actual_speculars))
            temp_specular = temp_specular.transpose(0, 2, 1).reshape((num_gaussians, actual_speculars * 3))
            specular[:, : actual_speculars * 3] = temp_specular
            logger.info(f"PLY file has SH degree {actual_sh_degree}, padding to {self.max_sh_degree}")
        else:
            raise ValueError(
                f"Unexpected number of f_rest_ properties: found {len(extra_f_names)}, "
                f"expected {expected_extra_f_count} or fewer"
            )

        # Extract scales
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((num_gaussians, len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Extract rotations
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rotations = np.zeros((num_gaussians, len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rotations[:, idx] = np.asarray(plydata.elements[0][attr_name])

        attrs = GaussianAttributes(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            albedo=albedo,
            specular=specular,
        )

        # Report actual SH degree from file (0 if only DC coefficients present)
        caps = ModelCapabilities(
            has_spherical_harmonics=True,
            sh_degree=actual_sh_degree,
            num_gaussians=num_gaussians,
            is_surfel=scales.shape[1] == 2,
            density_activation="sigmoid",
            scale_activation="exp",
        )

        logger.info(f"Loaded {num_gaussians} Gaussians, SH degree {caps.sh_degree}")
        return attrs, caps
