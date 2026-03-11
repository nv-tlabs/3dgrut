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

"""USD format importer for Gaussian splatting data.

Reads ParticleField3DGaussianSplat / ParticleField via UsdVol schema API
(GetPositionsAttr, GetOrientationsAttr, etc.). Requires USD 26+ with ParticleField support.
"""

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pxr import Usd, UsdVol

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
        gaussian_prim = self._find_particlefield_prim(stage)
        if gaussian_prim is None:
            prim_types = sorted({p.GetTypeName() for p in stage.Traverse()})
            raise ValueError(
                f"No Gaussian prim found in USD file: {path}. "
                f"Expected UsdVol.ParticleField or ParticleField3DGaussianSplat. "
                f"Prim types in stage: {prim_types}. "
                f"NuRec (UsdVol::Volume) and other formats are not supported for import."
            )

        logger.info(f"Found Gaussian prim: {gaussian_prim.GetPath()} (type: {gaussian_prim.GetTypeName()})")

        # Load via UsdVol schema API (ParticleField is base; ParticleField3DGaussianSplat is derived)
        if gaussian_prim.IsA(UsdVol.ParticleField):
            return self._load_lightfield(gaussian_prim)
        raise ValueError(f"Prim is not a UsdVol ParticleField: {gaussian_prim.GetTypeName()}")

    def _find_particlefield_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        """Find the Gaussian data prim in the stage (UsdVol.ParticleField or derived)."""
        for prim in stage.Traverse():
            if prim.IsA(UsdVol.ParticleField):
                return prim
        return None

    def _load_lightfield(self, prim: Usd.Prim) -> Tuple[GaussianAttributes, ModelCapabilities]:
        """Load from ParticleField3DGaussianSplat or ParticleField via UsdVol schema API."""
        prim_path = str(prim.GetPath())

        if prim.IsA(UsdVol.ParticleField3DGaussianSplat):
            schema = UsdVol.ParticleField3DGaussianSplat(prim)
            positions = _get_vec3_from_schema(schema.GetPositionsAttr(), schema.GetPositionshAttr())
            rotations = _get_quat_from_schema(schema.GetOrientationsAttr(), schema.GetOrientationshAttr())
            scales = _get_vec3_from_schema(schema.GetScalesAttr(), schema.GetScaleshAttr())
            densities = _get_float_from_schema(schema.GetOpacitiesAttr(), schema.GetOpacitieshAttr())
            sh_degree_attr = schema.GetRadianceSphericalHarmonicsDegreeAttr()
            sh_coeffs = _get_vec3_from_schema(
                schema.GetRadianceSphericalHarmonicsCoefficientsAttr(),
                schema.GetRadianceSphericalHarmonicsCoefficientshAttr(),
            )
        else:
            # Base ParticleField with applied attribute APIs (e.g. surfel/2DGS)
            assert prim.IsA(UsdVol.ParticleField)
            pos_api = UsdVol.ParticleFieldPositionAttributeAPI(prim)
            orient_api = UsdVol.ParticleFieldOrientationAttributeAPI(prim)
            scale_api = UsdVol.ParticleFieldScaleAttributeAPI(prim)
            opacity_api = UsdVol.ParticleFieldOpacityAttributeAPI(prim)
            rad_api = UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI(prim)
            positions = _get_vec3_from_schema(pos_api.GetPositionsAttr(), pos_api.GetPositionshAttr())
            rotations = _get_quat_from_schema(orient_api.GetOrientationsAttr(), orient_api.GetOrientationshAttr())
            scales = _get_vec3_from_schema(scale_api.GetScalesAttr(), scale_api.GetScaleshAttr())
            densities = _get_float_from_schema(opacity_api.GetOpacitiesAttr(), opacity_api.GetOpacitieshAttr())
            sh_degree_attr = rad_api.GetRadianceSphericalHarmonicsDegreeAttr()
            sh_coeffs = _get_vec3_from_schema(
                rad_api.GetRadianceSphericalHarmonicsCoefficientsAttr(),
                rad_api.GetRadianceSphericalHarmonicsCoefficientshAttr(),
            )

        if positions is None:
            raise ValueError(f"No positions attribute found in prim {prim_path}")
        num_gaussians = len(positions)

        if rotations is None:
            logger.warning("No orientations attribute found, using identity quaternions")
            rotations = np.tile([1.0, 0.0, 0.0, 0.0], (num_gaussians, 1)).astype(np.float32)
        if scales is None:
            logger.warning("No scales attribute found, using unit scales")
            scales = np.ones((num_gaussians, 3), dtype=np.float32)
        if densities is None:
            logger.warning("No opacities attribute found, using full opacity")
            densities = np.ones(num_gaussians, dtype=np.float32)
        densities = densities.reshape(-1, 1)

        sh_degree = sh_degree_attr.Get() if sh_degree_attr.IsValid() else 0

        if sh_coeffs is None or len(sh_coeffs) == 0:
            raise ValueError(f"No SH coefficients found in prim {prim_path}")
        num_sh_coeffs = (sh_degree + 1) ** 2
        sh_coeffs = sh_coeffs.reshape(num_gaussians, num_sh_coeffs, 3)
        albedo = sh_coeffs[:, 0, :]
        specular = (
            sh_coeffs[:, 1:, :].reshape(num_gaussians, -1)
            if num_sh_coeffs > 1
            else np.zeros((num_gaussians, 0), dtype=np.float32)
        )
        max_specular_size = ((3 + 1) ** 2 - 1) * 3
        if specular.shape[1] < max_specular_size:
            padded = np.zeros((num_gaussians, max_specular_size), dtype=np.float32)
            padded[:, : specular.shape[1]] = specular
            specular = padded

        attrs = GaussianAttributes(
            positions=positions,
            rotations=rotations,
            scales=scales,
            densities=densities,
            albedo=albedo,
            specular=specular,
        )
        is_surfel = prim.HasAPI(UsdVol.ParticleFieldKernelGaussianSurfletAPI)
        caps = ModelCapabilities(
            has_spherical_harmonics=True,
            sh_degree=sh_degree,
            num_gaussians=num_gaussians,
            is_surfel=is_surfel,
            density_activation="sigmoid",
            scale_activation="exp",
        )
        logger.info(f"Loaded {num_gaussians} Gaussians from LightField schema, SH degree {caps.sh_degree}")
        return attrs, caps


def _get_vec3_from_schema(attr, attr_half) -> Optional[np.ndarray]:
    """Get Vec3 array from schema attr; prefer float over half."""
    for a in (attr, attr_half):
        if a.IsValid() and a.Get() is not None:
            return np.array(a.Get(), dtype=np.float32)
    return None


def _get_quat_from_schema(attr, attr_half) -> Optional[np.ndarray]:
    """Get quaternion array [w,x,y,z] from schema attr."""
    a = attr if attr.IsValid() else attr_half
    if not a.IsValid() or a.Get() is None:
        return None
    quats = a.Get()
    out = np.zeros((len(quats), 4), dtype=np.float32)
    for i, q in enumerate(quats):
        im = q.GetImaginary()
        out[i] = [q.GetReal(), im[0], im[1], im[2]]
    return out


def _get_float_from_schema(attr, attr_half) -> Optional[np.ndarray]:
    """Get float array from schema attr."""
    for a in (attr, attr_half):
        if a.IsValid() and a.Get() is not None:
            return np.array(a.Get(), dtype=np.float32)
    return None
