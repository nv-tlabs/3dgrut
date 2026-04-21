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

from enum import IntEnum


class Features:
    """Enums and conf-driven getters for feature type, activation, and interpolation."""

    class Type(IntEnum):
        """Feature representation mode: integer value used directly in C preprocessor defines."""
        SH  = 0  # Spherical harmonics
        NHT = 1  # Neural harmonic texture

        @classmethod
        def from_string(cls, value: str) -> "Features.Type":
            value_lower = value.lower()
            for member in cls:
                if member.name.lower() == value_lower:
                    return member
            raise ValueError(f"Invalid feature_type: '{value}'. Must be one of {[m.name.lower() for m in cls]}")

    class ActivationType(IntEnum):
        NONE   = 0
        SIREN  = 1
        SINCOS = 2
        RELU   = 3

    class InterpolationType(IntEnum):
        CENTER = 0
        BEZIER = 1

    class InterpolationSupport(IntEnum):
        CENTER     = 0
        TETRAHEDRA = 1
        TRIANGLE   = 2

    def __init__(self, conf):
        self._conf = conf

    @property
    def transform_type(self):
        """SH or NHT — integer value used directly in C preprocessor defines."""
        return Features.Type.from_string(self._conf.model.feature_type)

    @property
    def activation_type(self):
        """Feature activation type from nht_features.activation.type."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type != "nht":
            return Features.ActivationType.NONE
        v = getattr(self._conf.model.nht_features, "activation", None)
        if v is None:
            return Features.ActivationType.NONE
        t = getattr(v, "type", "none")
        if isinstance(t, str):
            t = t.lower()
        if t == "none":
            return Features.ActivationType.NONE
        if t == "siren":
            return Features.ActivationType.SIREN
        if t == "sincos":
            return Features.ActivationType.SINCOS
        if t == "relu":
            return Features.ActivationType.RELU
        raise ValueError(f"Unknown nht_features.activation.type: {t}")

    @property
    def activation_num_frequencies(self):
        """Number of frequency bands. 1 when activation is none or relu."""
        if self.activation_type in (Features.ActivationType.NONE, Features.ActivationType.RELU):
            return 1
        v = getattr(self._conf.model.nht_features, "activation", None)
        return int(getattr(v, "num_frequencies", 1)) if v else 1

    @property
    def interpolation_type(self):
        """CENTER (none/barycentric) or BEZIER."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type != "nht":
            return Features.InterpolationType.CENTER
        v = getattr(self._conf.model.nht_features, "interpolation_type", "none").lower()
        if v == "none":
            return Features.InterpolationType.CENTER
        if v == "barycentric":
            return Features.InterpolationType.CENTER
        if v == "bezier":
            return Features.InterpolationType.BEZIER
        raise ValueError(f"Unknown nht_features.interpolation_type: {v}")

    @property
    def interpolation_support(self):
        """CENTER, TETRAHEDRA (gaussian), or TRIANGLE (trisurfel)."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type != "nht":
            return Features.InterpolationSupport.CENTER
        v = getattr(self._conf.model.nht_features, "interpolation_type", "none").lower()
        if v == "none":
            return Features.InterpolationSupport.CENTER
        if v == "barycentric":
            primitive = getattr(self._conf.render, "primitive_type", "instances")
            return Features.InterpolationSupport.TRIANGLE if primitive == "trisurfel" else Features.InterpolationSupport.TETRAHEDRA
        raise ValueError(f"Unknown nht_features.interpolation_type: {v}")

    @property
    def num_interpolation_points(self):
        """1 for center support, 4 for barycentric (tetrahedra or trisurfel)."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type != "nht":
            return 1
        if self.interpolation_support == Features.InterpolationSupport.CENTER:
            return 1
        return 4  # barycentric: tetrahedra (4 verts) or trisurfel (2 coplanar triangles, 4 verts)

    @property
    def particle_feature_dim(self):
        """Total feature dim per particle (buffer stride). For NHT = nht_features.dim."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type == "sh":
            sh_degree = self._conf.model.progressive_training.max_n_features
            return 3 * ((sh_degree + 1) ** 2)
        elif feature_type == "nht":
            return self._conf.model.nht_features.dim
        raise ValueError(f"Unknown feature_type: {feature_type}")

    @property
    def interp_point_feature_dim(self):
        """Per-interpolation-point feature dim before activation."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type != "nht":
            return 3
        return self._conf.model.nht_features.dim // self.num_interpolation_points

    @property
    def ray_feature_dim(self):
        """Per-ray feature dim (decoder input): interp_point_dim * num_frequencies."""
        feature_type = self._conf.model.feature_type.lower()
        if feature_type == "sh":
            return 3  # RGB output
        elif feature_type == "nht":
            return self.interp_point_feature_dim * self.activation_num_frequencies
        raise ValueError(f"Unknown feature_type: {feature_type}")

    @property
    def feature_defines(self):
        """C preprocessor defines for all feature-related kernel parameters."""
        return [
            f"-DPARTICLE_FEATURE_DIM={self.particle_feature_dim}",
            f"-DRAY_FEATURE_DIM={self.ray_feature_dim}",
            f"-DFEATURE_TRANSFORM_TYPE={self.transform_type}",
            f"-DFEATURE_INTERPOLATION_TYPE={self.interpolation_type}",
            f"-DFEATURE_INTERPOLATION_SUPPORT={self.interpolation_support}",
            f"-DFEATURE_ACTIVATION_TYPE={self.activation_type}",
            f"-DFEATURE_ACTIVATION_NUM_FREQUENCIES={self.activation_num_frequencies}",
            f"-DINTERP_POINT_FEATURE_DIM={self.interp_point_feature_dim}",
        ]
