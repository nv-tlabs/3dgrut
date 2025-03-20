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
from typing import Union

import numpy as np
import torch


PI = torch.pi if hasattr(torch, 'pi') else np.pi  # Fallback for older torch versions


class ObjectTransform:
    """ObjectTransform represents the local transformations that convert the object from local space coordinates
    to world coordinates.
    """

    def __init__(self, device=None, dtype=None):
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        self._translation = torch.zeros(3, device=self.device, dtype=self.dtype)
        self._rotation = torch.zeros(3, device=self.device, dtype=self.dtype)
        self._scale = torch.ones(4, device=self.device, dtype=self.dtype)
        self._permutation = torch.eye(4, device=self.device, dtype=self.dtype)

    def reset(self):
        """ Restore object transform to unit scale, at the origin, with zero orientation. """
        self._translation = torch.zeros(3, device=self.device, dtype=self.dtype)
        self._rotation = torch.zeros(3, device=self.device, dtype=self.dtype)
        self._scale = torch.ones(4, device=self.device, dtype=self.dtype)
        self._permutation = torch.eye(4, device=self.device, dtype=self.dtype)

    def translate(self, translation: torch.Tensor):
        """
        Args:
            translation (torch.Tensor):
                Amount of translation in world space coordinates, as a (3,) shaped tensor
        """
        self._translation += translation

    def rotate(self, rotation: torch.Tensor):
        """
        Args:
            rotation (torch.Tensor):
                Amount of rotation as euler angles, as a (3,) shaped tensor
        """
        self._rotation += rotation

    def scale(self, scale: Union[float, torch.Tensor]):
        """
        Args:
            scale (float or torch.Tensor):
                Amount of scale in world space coordinates.
                For non uniform scale, pass a (3,) shaped tensor.
                For uniform scale, pass a float.
        """
        self._scale[:3] *= scale

    def permute(self, permutation):
        """
        Args:
            permutation (list[int]): The permutation of the axis.
                                     For example, [1, 0, 2] will swap the x and y axis.
        """
        permutation = torch.tensor(permutation, device=self.device, dtype=torch.long)
        if permutation.shape[0] != 3:
            raise Exception("Permutation only supports 3 axis.")
        if torch.any(permutation < 0):
            raise Exception("Permutation axis cannot be negative.")
        if torch.any(permutation > 2):
            raise Exception("Permutation axis out of bounds.")
        self._permutation.fill_(0)
        self._permutation[(0,1,2), permutation] = 1
        self._permutation[3, 3] = 1

    def _translation_mat(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): The translation component of the transform
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the translation component of the transform
        """
        mat = torch.eye(4, device=self.device, dtype=self.dtype)
        mat[:3, -1] = t
        return mat

    def _rotation_mat_x(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angle (torch.Tensor): The x axis euler angle component of the transform, in radians
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the rotation x component of the transform
        """
        mat = torch.eye(4, device=self.device, dtype=self.dtype)
        mat[1, 1] = torch.cos(angle)
        mat[1, 2] = torch.sin(angle)
        mat[2, 1] = -torch.sin(angle)
        mat[2, 2] = torch.cos(angle)
        return mat

    def _rotation_mat_y(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angle (torch.Tensor): The y axis euler angle component of the transform, in radians
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the rotation y component of the transform
        """
        mat = torch.eye(4, device=self.device, dtype=self.dtype)
        mat[0, 0] = torch.cos(angle)
        mat[0, 2] = -torch.sin(angle)
        mat[2, 0] = torch.sin(angle)
        mat[2, 2] = torch.cos(angle)
        return mat

    def _rotation_mat_z(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angle (torch.Tensor): The z axis euler angle component of the transform, in radians
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the rotation z component of the transform
        """
        mat = torch.eye(4, device=self.device, dtype=self.dtype)
        mat[0, 0] = torch.cos(angle)
        mat[0, 1] = -torch.sin(angle)
        mat[1, 0] = torch.sin(angle)
        mat[1, 1] = torch.cos(angle)
        return mat

    def _rotation_mat(self, rx, ry, rz) -> torch.Tensor:
        """
        Args:
            rx (torch.Tensor): The x axis euler angle component of the transform, in radians
            ry (torch.Tensor): The y axis euler angle component of the transform, in radians
            rz (torch.Tensor): The z axis euler angle component of the transform, in radians
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the complete rotation component of the transform
        """
        Rx = self._rotation_mat_x(rx)
        Ry = self._rotation_mat_y(ry)
        Rz = self._rotation_mat_z(rz)
        return Rz @ Ry @ Rx

    def _scale_mat(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s (torch.Tensor): The scale component of the transform
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the scale component of the transform
        """
        return torch.diag(s).to(device=self.device, dtype=self.dtype)

    def _inv_translation_mat(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): The translation component of the transform
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the inverse translation component of the transform
        """
        mat = torch.eye(4, device=self.device, dtype=self.dtype)
        mat[:3, -1] = -t
        return mat

    def _inv_rotation_mat(self, rx, ry, rz) -> torch.Tensor:
        """
        Args:
            rx (torch.Tensor): The x axis euler angle component of the transform, in radians
            ry (torch.Tensor): The y axis euler angle component of the transform, in radians
            rz (torch.Tensor): The z axis euler angle component of the transform, in radians
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the complete inverse rotation component of the transform
        """
        Rx = self._rotation_mat_x(-rx)
        Ry = self._rotation_mat_y(-ry)
        Rz = self._rotation_mat_z(-rz)
        return Rx @ Ry @ Rz

    def _inv_scale_mat(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s (torch.Tensor): The scale component of the transform
        Returns:
            (torch.Tensor): a 4x4 matrix which represents the inverse scale component of the transform
        """
        return torch.diag(1.0 / s).to(device=self.device, dtype=self.dtype)

    def model_matrix(self):
        """
        Returns:
            (torch.Tensor): a 4x4 model matrix which encodes the complete transform of the object from local
            coordinates to world coordinates.
        """
        # Transformations applied in order of
        # Translation @ Rotation @ Scale
        scale_mat = self._scale_mat(self._scale)
        rotation_rads = self._rotation.div(180.0).mul(PI)
        rotation_mat = self._rotation_mat(*rotation_rads)
        translation_mat = self._translation_mat(self._translation)
        model_mat = translation_mat @ rotation_mat @ scale_mat @ self._permutation
        return model_mat

    def inv_model_matrix(self):
        """
        Returns:
            (torch.Tensor): a 4x4 model matrix which encodes the complete transform of the object from world
            coordinates to local coordinates.
            This version can be useful, for example, for ray traced pipelines (where the rays are inverse transformed,
            rather than the object).
        """
        # Transformations applied in order of
        # Scale^(-1) @ Rotation^(-1) @ Translation^(-1)
        scale_mat = self._inv_scale_mat(self._scale)
        rotation_rads = self._rotation.div(180.0).mul(PI)
        rotation_mat = self._inv_rotation_mat(*rotation_rads)
        translation_mat = self._inv_translation_mat(self._translation)
        inv_mat = self._permutation @ scale_mat @ rotation_mat @ translation_mat
        return inv_mat

    def rotation_matrix(self):
        rotation_rads = self._rotation.div(180.0).mul(PI)
        rotation_mat = self._rotation_mat(*rotation_rads)
        return rotation_mat @ self._permutation

    def inv_rotation_matrix(self):
        rotation_rads = self._rotation.div(180.0).mul(PI)
        rotation_mat = self._inv_rotation_mat(*rotation_rads)
        return self._permutation @ rotation_mat

    def to(self, *args, **kwargs) -> ObjectTransform:
        """ Shifts the transform to a different device / dtype.
        seealso::func:`torch.Tensor.to` for an elaborate explanation of using this method.

        Returns:
            The object transform tensors will be ast to a different device / dtype
        """
        _translation = self._translation.to(*args, **kwargs)
        _rotation = self._rotation.to(*args, **kwargs)
        _scale = self._scale.to(*args, **kwargs)
        _permutation = self._permutation.to(*args, **kwargs)
        if _translation is not self._translation or \
           _rotation is not self._rotation or \
           _scale is not self._scale or \
           _permutation is not self._permutation:
            transform = ObjectTransform(device=_translation.device, dtype=_translation.dtype)
            transform._translation = _translation
            transform._rotation = _rotation
            transform._scale = _scale
            transform._permutation = _permutation
            return transform
        else:
            return self

    @property
    def tx(self):
        """
        Returns:
            The x-axis translation component tensor.
        """
        return self._translation[0]

    @tx.setter
    def tx(self, value):
        """ Sets the x-axis translation component tensor.
        Args:
            value: New translation-x value to set
        """
        self._translation[0] = value

    @property
    def ty(self):
        """
        Returns:
            The y-axis translation component tensor, shaped as (tx, ty, tz).
        """
        return self._translation[1]

    @ty.setter
    def ty(self, value):
        """ Sets the y-axis translation component tensor.
        Args:
            value: New translation-y value to set
        """
        self._translation[1] = value

    @property
    def tz(self):
        """
        Returns:
            The z-axis translation component tensor, shaped as (tx, ty, tz).
        """
        return self._translation[2]

    @tz.setter
    def tz(self, value):
        """ Sets the z-axis translation component tensor.
        Args:
            value: New translation-z value to set
        """
        self._translation[2] = value

    @property
    def rx(self):
        """
        Returns:
            The euler angle of the rotation component around the x axis, in degrees
        """
        return self._rotation[0]

    @rx.setter
    def rx(self, value):
        """ Sets the euler angle of the rotation component around the x axis
        Args:
            value: New angle-x value to set, in degrees
        """
        self._rotation[0] = value

    @property
    def ry(self):
        """
        Returns:
            The euler angle of the rotation component around the y axis, in degrees
        """
        return self._rotation[1]

    @ry.setter
    def ry(self, value):
        """ Sets the euler angle of the rotation component around the y axis
        Args:
            value: New angle-y value to set, in degrees
        """
        self._rotation[1] = value

    @property
    def rz(self):
        """
        Returns:
            The euler angle of the rotation component around the z axis, in degrees
        """
        return self._rotation[2]

    @rz.setter
    def rz(self, value):
        """ Sets the euler angle of the rotation component around the z axis
        Args:
            value: New angle-z value to set, in degrees
        """
        self._rotation[2] = value

    @property
    def sx(self):
        """
        Returns:
            The scale component for the x axis.
        """
        return self._scale[0]

    @sx.setter
    def sx(self, value):
        """ Sets the x-axis scale component tensor.
        Args:
            value: New scale-x value to set
        """
        self._scale[0] = value

    @property
    def sy(self):
        """
        Returns:
            The scale component for the y axis.
        """
        return self._scale[1]

    @sy.setter
    def sy(self, value):
        """ Sets the y-axis scale component tensor.
        Args:
            value: New scale-y value to set
        """
        self._scale[1] = value

    @property
    def sz(self):
        """
        Returns:
            The scale component for the z axis.
        """
        return self._scale[2]

    @sz.setter
    def sz(self, value):
        """ Sets the z-axis scale component tensor.
        Args:
            value: New scale-z value to set
        """
        self._scale[2] = value
