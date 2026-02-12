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

"""USD format importer for Gaussian splatting data."""

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pxr import Usd

from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
from threedgrut.export.importers.base import FormatImporter

logger = logging.getLogger(__name__)


class USDImporter(FormatImporter):
    """Importer for USD/USDZ format files.

    Supports LightField schema (ParticleField3DGaussianSplat) which stores post-activation values.
    """

    @property
    def stores_preactivation(self) -> bool:
        # LightField stores post-activation values
        return False

    def load(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load USD/USDZ file into GaussianAttributes.

        Args:
            path: Path to USD/USDZ file

        Returns:
            Tuple of (GaussianAttributes, ModelCapabilities)
        """
        logger.info(f"Loading USD file: {path}")

        # Handle USDZ by extracting to temp dir
        if path.suffix.lower() == ".usdz":
            return self._load_usdz(path)
        else:
            return self._load_usd_stage(path)

    def _load_usdz(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load USDZ file (extract and load root stage)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Extract USDZ
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir_path)

            # Find the root USD file (usually default.usda or first .usd*)
            usd_files = list(tmpdir_path.glob("*.usd*"))
            if not usd_files:
                raise ValueError(f"No USD files found in USDZ: {path}")

            # Prefer default.usda
            root_file = None
            for f in usd_files:
                if f.stem == "default":
                    root_file = f
                    break
            if root_file is None:
                root_file = usd_files[0]

            return self._load_usd_stage(root_file)

    def _load_usd_stage(self, path: Path) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load USD stage and extract Gaussian data."""
        stage = Usd.Stage.Open(str(path))
        if not stage:
            raise ValueError(f"Failed to open USD stage: {path}")

        # Find Gaussian prim (LightField schema only)
        gaussian_prim = self._find_gaussian_prim(stage)
        if gaussian_prim is None:
            raise ValueError(f"No Gaussian prim found in USD file: {path}")

        prim_type = gaussian_prim.GetTypeName()
        logger.info(f"Found Gaussian prim: {gaussian_prim.GetPath()} (type: {prim_type})")

        # Load LightField schema
        if prim_type == "ParticleField3DGaussianSplat" or prim_type == "ParticleField":
            return self._load_lightfield(gaussian_prim)
        else:
            raise ValueError(f"Unknown Gaussian prim type: {prim_type}")

    def _find_gaussian_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        """Find the Gaussian data prim in the stage."""
        # Search for LightField prim types
        for prim in stage.Traverse():
            prim_type = prim.GetTypeName()
            if prim_type in ["ParticleField3DGaussianSplat", "ParticleField"]:
                return prim
        return None

    def _load_lightfield(self, prim: Usd.Prim) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load from ParticleField3DGaussianSplat schema."""
        prim_path = str(prim.GetPath())
        
        # Try both regular and half-precision attribute names
        positions = self._get_vec3_array(prim, "positions", "positionsh")
        if positions is None:
            raise ValueError(f"No positions attribute found in prim {prim_path}")

        num_gaussians = len(positions)

        # Orientations (quaternions)
        rotations = self._get_quat_array(prim, "orientations", "orientationsh")
        if rotations is None:
            logger.warning("No orientations attribute found, using identity quaternions")
            rotations = np.tile([1.0, 0.0, 0.0, 0.0], (num_gaussians, 1)).astype(np.float32)

        # Scales
        scales = self._get_vec3_array(prim, "scales", "scalesh")
        if scales is None:
            logger.warning("No scales attribute found, using unit scales")
            scales = np.ones((num_gaussians, 3), dtype=np.float32)

        # Opacities
        densities = self._get_float_array(prim, "opacities", "opacitiesh")
        if densities is None:
            logger.warning("No opacities attribute found, using full opacity")
            densities = np.ones(num_gaussians, dtype=np.float32)
        densities = densities.reshape(-1, 1)

        # SH coefficients
        sh_degree_attr = prim.GetAttribute("radiance:sphericalHarmonicsDegree")
        sh_degree = sh_degree_attr.Get() if sh_degree_attr.IsValid() else 0

        sh_coeffs = self._get_vec3_array(
            prim, "radiance:sphericalHarmonicsCoefficients", "radiance:sphericalHarmonicsCoefficientsh"
        )

        if sh_coeffs is not None and len(sh_coeffs) > 0:
            # SH coeffs are stored as [N * num_sh_coeffs, 3]
            num_sh_coeffs = (sh_degree + 1) ** 2
            sh_coeffs = sh_coeffs.reshape(num_gaussians, num_sh_coeffs, 3)
            albedo = sh_coeffs[:, 0, :]  # DC term
            if num_sh_coeffs > 1:
                specular = sh_coeffs[:, 1:, :].reshape(num_gaussians, -1)
            else:
                specular = np.zeros((num_gaussians, 0), dtype=np.float32)
        else:
            raise ValueError(f"No SH coefficients found in prim {prim_path}")
            
        # Pad specular to max SH degree 3 if needed
        max_specular_size = ((3 + 1) ** 2 - 1) * 3  # degree 3
        if specular.shape[1] < max_specular_size:
            padded_specular = np.zeros((num_gaussians, max_specular_size), dtype=np.float32)
            padded_specular[:, : specular.shape[1]] = specular
            specular = padded_specular

        attrs = GaussianAttributes(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            albedo=albedo,
            specular=specular,
        )

        caps = ModelCapabilities(
            has_spherical_harmonics=True,
            sh_degree=sh_degree,
            num_gaussians=num_gaussians,
            is_surfel=False,
            density_activation="sigmoid",
            scale_activation="exp",
        )

        logger.info(f"Loaded {num_gaussians} Gaussians from LightField schema, SH degree {caps.sh_degree}")
        return attrs, caps

    def _get_vec3_array(
        self, prim: Usd.Prim, attr_name: str, half_attr_name: str
    ) -> Optional[np.ndarray]:
        """Get Vec3 array from prim, trying both regular and half precision names."""
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid() and attr.Get() is not None:
            return np.array(attr.Get(), dtype=np.float32)

        attr = prim.GetAttribute(half_attr_name)
        if attr.IsValid() and attr.Get() is not None:
            return np.array(attr.Get(), dtype=np.float32)

        return None

    def _get_quat_array(
        self, prim: Usd.Prim, attr_name: str, half_attr_name: str
    ) -> Optional[np.ndarray]:
        """Get quaternion array from prim."""
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            attr = prim.GetAttribute(half_attr_name)

        if attr.IsValid() and attr.Get() is not None:
            quats = attr.Get()
            # Convert from Gf.Quat* to numpy array [w, x, y, z]
            result = np.zeros((len(quats), 4), dtype=np.float32)
            for i, q in enumerate(quats):
                result[i] = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
            return result

        return None

    def _get_float_array(
        self, prim: Usd.Prim, attr_name: str, half_attr_name: str
    ) -> Optional[np.ndarray]:
        """Get float array from prim."""
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid() and attr.Get() is not None:
            return np.array(attr.Get(), dtype=np.float32)

        attr = prim.GetAttribute(half_attr_name)
        if attr.IsValid() and attr.Get() is not None:
            return np.array(attr.Get(), dtype=np.float32)

        return None
