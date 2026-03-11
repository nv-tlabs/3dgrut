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

from dataclasses import dataclass
from enum import IntEnum, auto, unique

import dataclasses_json
import numpy as np
import torch


## Data classes representing stored data types
@unique
class ShutterType(IntEnum):
    """Enumerates different possible camera imager shutter types"""

    ROLLING_TOP_TO_BOTTOM = auto()  #: Rolling shutter from top to bottom of the imager
    ROLLING_LEFT_TO_RIGHT = auto()  #: Rolling shutter from left to right of the imager
    ROLLING_BOTTOM_TO_TOP = auto()  #: Rolling shutter from bottom to top of the imager
    ROLLING_RIGHT_TO_LEFT = auto()  #: Rolling shutter from right to left of the imager
    GLOBAL = auto()  #: Instantaneous global shutter (no rolling shutter)


@dataclass
class CameraModelParameters:
    """Represents parameters common to all camera models"""

    resolution: np.ndarray  #: Width and height of the image in pixels (int64, [2,])
    shutter_type: ShutterType  #: Shutter type of the camera's imaging sensor

    def __post_init__(self):
        # Sanity checks
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == np.dtype("int64")
        assert self.resolution[0] > 0 and self.resolution[1] > 0


@dataclass
class OpenCVPinholeCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    """Represents Pinhole-specific (OpenCV-like) camera model parameters"""

    #: U and v coordinate of the principal point, following the :ref:`image coordinate conventions <image_coordinate_conventions>` (float32, [2,])
    principal_point: np.ndarray
    #: Focal lengths in u and v direction, resp., mapping (distorted) normalized camera coordinates to image coordinates relative to the principal point (float32, [2,])
    focal_length: np.ndarray
    #: Radial distortion coefficients ``[k1,k2,k3,k4,k5,k6]`` parameterizing the rational radial distortion factor :math:`\frac{1 + k_1r^2 + k_2r^4 + k_3r^6}{1 + k_4r^2 + k_5r^4 + k_6r^6}` for squared norms :math:`r^2` of normalized camera coordinates (float32, [6,])
    radial_coeffs: np.ndarray
    #: Tangential distortion coefficients ``[p1,p2]`` parameterizing the tangential distortion components :math:`\begin{bmatrix} 2p_1x'y' + p_2 \left(r^2 + 2{x'}^2 \right) \\ p_1 \left(r^2 + 2{y'}^2 \right) + 2p_2x'y' \end{bmatrix}` for normalized camera coordinates :math:`\begin{bmatrix} x' \\ y' \end{bmatrix}` (float32, [2,])
    tangential_coeffs: np.ndarray
    #: Thins prism distortion coefficients ``[s1,s2,s3,s4]`` parameterizing the thin prism distortion components :math:`\begin{bmatrix} s_1r^2 + s_2r^4 \\ s_3r^2 + s_4r^4 \end{bmatrix}` for squared norms :math:`r^2` of normalized camera coordinates (float32, [4,]
    thin_prism_coeffs: np.ndarray

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == np.dtype("float32")
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0

        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == np.dtype("float32")
        assert self.focal_length[0] > 0.0 and self.focal_length[1] > 0.0

        assert self.radial_coeffs.shape == (6,)
        assert self.radial_coeffs.dtype == np.dtype("float32")

        assert self.tangential_coeffs.shape == (2,)
        assert self.tangential_coeffs.dtype == np.dtype("float32")

        assert self.thin_prism_coeffs.shape == (4,)
        assert self.thin_prism_coeffs.dtype == np.dtype("float32")


@dataclass
class OpenCVFisheyeCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    """Represents Fisheye-specific (OpenCV-like) camera model parameters"""

    #: U and v coordinate of the principal point, following the :ref:`image coordinate conventions <image_coordinate_conventions>` (float32, [2,])
    principal_point: np.ndarray
    #: Focal lengths in u and v direction, resp., mapping (distorted) normalized camera coordinates to image coordinates relative to the principal point (float32, [2,])
    focal_length: np.ndarray
    #: Radial distortion coefficients `radial_coeffs` represent OpenCV-like ``[k1,k2,k3,k4]`` coefficients to parameterize the
    #  fisheye distortion polynomial as :math:`\theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)`
    #  for extrinsic camera ray angles :math:`\theta` with the principal direction (float32, [4,])
    radial_coeffs: np.ndarray
    #: Maximal extrinsic ray angle [rad] with the principal direction (float32)
    max_angle: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == np.dtype("float32")
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0
        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == np.dtype("float32")
        assert self.focal_length[0] > 0.0 and self.focal_length[1] > 0.0
        assert self.radial_coeffs.shape == (4,)
        assert self.radial_coeffs.dtype == np.dtype("float32")
        assert self.max_angle > 0.0


def _eval_poly_horner(poly_coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluates a polynomial y=f(x) (given by poly_coefficients) at points x using
    numerically stable Horner scheme"""
    y = torch.zeros_like(x)
    for fi in torch.flip(poly_coefficients, dims=(0,)):
        y = y * x + fi
    return y


def _eval_poly_inverse_horner_newton(
    poly_coefficients: torch.Tensor,
    poly_derivative_coefficients: torch.Tensor,
    inverse_poly_approximation_coefficients: torch.Tensor,
    newton_iterations: int,
    y: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x) (given by poly_coefficients) at points y
    using numerically stable Horner scheme and Newton iterations starting from an approximate solution \\hat{x} = \\hat{f}^{-1}(y)
    (given by inverse_poly_approximation_coefficients) and the polynomials derivative df/dx (given by poly_derivative_coefficients)
    """
    x = _eval_poly_horner(
        inverse_poly_approximation_coefficients, y
    )  # approximation / starting points - also returned for zero iterations
    assert newton_iterations >= 0, "Newton-iteration number needs to be non-negative"
    # Buffers of intermediate results to allow differentiation
    x_iter = [torch.zeros_like(x) for _ in range(newton_iterations + 1)]
    x_iter[0] = x
    for i in range(newton_iterations):
        # Evaluate single Newton step
        dfdx = _eval_poly_horner(poly_derivative_coefficients, x_iter[i])
        residuals = _eval_poly_horner(poly_coefficients, x_iter[i]) - y
        x_iter[i + 1] = x_iter[i] - residuals / dfdx
    return x_iter[newton_iterations]


def image_points_to_camera_rays(
    camera_model_parameters,
    image_points,
    newton_iterations: int = 3,
    min_2d_norm: float = 1e-6,
    device: str = "cpu",
):
    """
    Computes the camera ray for each image point, performing an iterative undistortion of the nonlinear distortion model
    """

    dtype: torch.dtype = torch.float32

    principal_point = torch.tensor(camera_model_parameters.principal_point, dtype=dtype, device=device)
    focal_length = torch.tensor(camera_model_parameters.focal_length, dtype=dtype, device=device)
    resolution = torch.tensor(camera_model_parameters.resolution.astype(np.int32), device=device)
    max_angle = float(camera_model_parameters.max_angle)
    newton_iterations = newton_iterations

    # 2D pixel-distance threshold
    assert min_2d_norm > 0, "require positive minimum norm threshold"
    min_2d_norm = torch.tensor(min_2d_norm, dtype=dtype, device=device)

    assert principal_point.shape == (2,)
    assert principal_point.dtype == dtype
    assert focal_length.shape == (2,)
    assert focal_length.dtype == dtype
    assert resolution.shape == (2,)
    assert resolution.dtype == torch.int32

    k1, k2, k3, k4 = camera_model_parameters.radial_coeffs[:]
    # ninth-degree forward polynomial (mapping angles to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    forward_poly = torch.tensor([0, 1, 0, k1, 0, k2, 0, k3, 0, k4], dtype=dtype, device=device)
    # eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^8 + 9*k4*theta^8
    dforward_poly = torch.tensor([1, 0, 3 * k1, 0, 5 * k2, 0, 7 * k3, 0, 9 * k4], dtype=dtype, device=device)

    # approximate backward poly (mapping normalized distances to angles) *very crudely* by linear interpolation / equidistant angle model (also assuming image-centered principal point)
    max_normalized_dist = np.max(camera_model_parameters.resolution / 2 / camera_model_parameters.focal_length)
    approx_backward_poly = torch.tensor([0, max_angle / max_normalized_dist], dtype=dtype, device=device)

    assert image_points.is_floating_point(), "[CameraModel]: image_points must be floating point values"
    image_points = image_points.to(dtype)

    normalized_image_points = (image_points - principal_point) / focal_length
    deltas = torch.linalg.norm(normalized_image_points, axis=1, keepdims=True)

    # Evaluate backward polynomial as the inverse of the forward one
    thetas = _eval_poly_inverse_horner_newton(
        forward_poly, dforward_poly, approx_backward_poly, newton_iterations, deltas
    )

    # Compute the camera rays and set the ones at the image center to [0,0,1]
    cam_rays = torch.hstack(
        (
            torch.sin(thetas) * normalized_image_points / torch.maximum(deltas, min_2d_norm),
            torch.cos(thetas),
        )
    )
    cam_rays[deltas.flatten() < min_2d_norm, :] = torch.tensor([[0, 0, 1]]).to(normalized_image_points)

    return cam_rays


def pixels_to_image_points(pixel_idxs) -> torch.Tensor:
    """Given integer-based pixels indices, computes corresponding continuous image point coordinates representing the *center* of each pixel."""
    assert isinstance(pixel_idxs, torch.Tensor), "[CameraModel]: Pixel indices must be a torch tensor"
    assert not pixel_idxs.is_floating_point(), "[CameraModel]: Pixel indices must be integers"
    # Compute the image point coordinates representing the center of each pixel (shift from top left corner to the center)
    return pixel_idxs.to(torch.float32) + 0.5


def opencv_pinhole_image_points_to_camera_rays(
    camera_model_parameters: OpenCVPinholeCameraModelParameters,
    image_points: torch.Tensor,
    newton_iterations: int = 10,
    min_2d_norm: float = 1e-6,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes camera rays from image points by performing iterative undistortion
    of the OpenCV pinhole distortion model.

    The OpenCV distortion model is:
        x_distorted = x' * radial + tangential_x
        y_distorted = y' * radial + tangential_y

    where:
        radial = (1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶)
        r² = x'² + y'²

    Args:
        camera_model_parameters: OpenCVPinholeCameraModelParameters with distortion coefficients
        image_points: (N, 2) tensor of image point coordinates
        newton_iterations: Number of Newton-Raphson iterations for undistortion
        min_2d_norm: Minimum norm threshold to avoid division by zero
        device: Device to perform computation on

    Returns:
        cam_rays: (N, 3) tensor of normalized ray directions in camera coordinates
    """
    dtype = torch.float32

    principal_point = torch.tensor(camera_model_parameters.principal_point, dtype=dtype, device=device)
    focal_length = torch.tensor(camera_model_parameters.focal_length, dtype=dtype, device=device)
    k1, k2, k3, k4, k5, k6 = camera_model_parameters.radial_coeffs
    p1, p2 = camera_model_parameters.tangential_coeffs

    # Convert to torch tensors
    k1 = torch.tensor(k1, dtype=dtype, device=device)
    k2 = torch.tensor(k2, dtype=dtype, device=device)
    k3 = torch.tensor(k3, dtype=dtype, device=device)
    k4 = torch.tensor(k4, dtype=dtype, device=device)
    k5 = torch.tensor(k5, dtype=dtype, device=device)
    k6 = torch.tensor(k6, dtype=dtype, device=device)
    p1 = torch.tensor(p1, dtype=dtype, device=device)
    p2 = torch.tensor(p2, dtype=dtype, device=device)

    # Normalize image points to get distorted normalized coordinates
    image_points = image_points.to(dtype).to(device)
    x_distorted = (image_points[:, 0] - principal_point[0]) / focal_length[0]
    y_distorted = (image_points[:, 1] - principal_point[1]) / focal_length[1]

    # Initial guess: distorted coordinates
    x_undist = x_distorted.clone()
    y_undist = y_distorted.clone()

    # Newton-Raphson iteration to find undistorted coordinates
    for _ in range(newton_iterations):
        r2 = x_undist * x_undist + y_undist * y_undist
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion factor
        radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = radial_num / radial_denom

        # Tangential distortion
        xy = x_undist * y_undist
        tangential_x = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x_undist * x_undist)
        tangential_y = p1 * (r2 + 2.0 * y_undist * y_undist) + 2.0 * p2 * xy

        # Predicted distorted coordinates
        x_pred = x_undist * radial + tangential_x
        y_pred = y_undist * radial + tangential_y

        # Error
        error_x = x_pred - x_distorted
        error_y = y_pred - y_distorted

        # Update (simplified Newton step - approximate Jacobian as radial factor)
        # For small distortions, the Jacobian is approximately diagonal with radial on diagonal
        x_undist = x_undist - error_x / (radial + 1e-8)
        y_undist = y_undist - error_y / (radial + 1e-8)

    # Construct ray directions (pinhole model: rays pass through origin)
    cam_rays = torch.stack([x_undist, y_undist, torch.ones_like(x_undist)], dim=-1)

    # Normalize ray directions
    cam_rays = torch.nn.functional.normalize(cam_rays, dim=-1)

    return cam_rays
