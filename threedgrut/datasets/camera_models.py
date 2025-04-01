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

from enum import IntEnum, auto, unique
from dataclasses import dataclass, field

import numpy as np
import dataclasses_json

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


@dataclass
class FThetaCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    """Represents FTheta-specific camera model parameters"""

    @unique
    class PolynomialType(IntEnum):
        """Enumerates different possible polynomial types"""

        PIXELDIST_TO_ANGLE = auto()  #: Polynomial mapping pixeldistances-to-angles (also known as "backward" polynomial)
        ANGLE_TO_PIXELDIST = auto()  #: Polynomial mapping angles-to-pixeldistances (also known as "forward" polynomial)

    principal_point: np.ndarray #: U and v coordinate of the principal point, following the NVIDIA default convention for FTheta camera models in which the pixel indices represent the center of the pixel (not the top-left corners). Principal point coordinates will be adapted internally in camera model APIs to reflect the :ref:`image coordinate conventions <image_coordinate_conventions>`
    reference_poly: PolynomialType #: Indicating which of the two stored polynomials is the model's *reference* polynomial (the other polynomial is only an approximation)
    pixeldist_to_angle_poly: np.ndarray #: Coefficients of the pixeldistances-to-angles polynomial (float32, [6,])
    angle_to_pixeldist_poly: np.ndarray #: Coefficients of the angles-to-pixeldistances polynomial (float32, [6,])
    max_angle: float = 0.0  #: Maximal extrinsic ray angle [rad] with the principal direction (float32)
    linear_cde: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)) #: Coefficients of the constrained linear term [c,d;e,1] transforming between sensor coordinates (in mm) to image coordinates (in px) (float32, [3,])

    @staticmethod
    def type() -> str:
        """Returns a string-identifier of the camera model"""
        return "ftheta"

    @property
    def bw_poly(self) -> np.ndarray:
        """Alias for the pixeldistances-to-angles polynomial"""
        return self.pixeldist_to_angle_poly

    @property
    def fw_poly(self) -> np.ndarray:
        """Alias for the angles-to-pixeldistances polynomial"""
        return self.angle_to_pixeldist_poly

    POLYNOMIAL_DEGREE = 6

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == np.dtype("float32")
        assert self.principal_point[0] >= 0.0 and self.principal_point[1] >= 0.0

        assert self.reference_poly in FThetaCameraModelParameters.PolynomialType.__members__.values()

        assert self.pixeldist_to_angle_poly.ndim == 1
        assert len(self.pixeldist_to_angle_poly) <= self.POLYNOMIAL_DEGREE
        assert self.pixeldist_to_angle_poly.dtype == np.dtype("float32")

        assert self.angle_to_pixeldist_poly.ndim == 1
        assert len(self.angle_to_pixeldist_poly) <= self.POLYNOMIAL_DEGREE
        assert self.angle_to_pixeldist_poly.dtype == np.dtype("float32")

        # pad polynomials to full size
        self.pixeldist_to_angle_poly = np.pad(
            self.pixeldist_to_angle_poly,
            (0, self.POLYNOMIAL_DEGREE - len(self.pixeldist_to_angle_poly)),
            mode="constant",
            constant_values=0.0,
        )
        self.angle_to_pixeldist_poly = np.pad(
            self.angle_to_pixeldist_poly,
            (0, self.POLYNOMIAL_DEGREE - len(self.angle_to_pixeldist_poly)),
            mode="constant",
            constant_values=0.0,
        )

        assert self.max_angle > 0.0

        assert self.linear_cde.shape == (3,)
        assert self.linear_cde.dtype == np.dtype("float32")

        # some datasets might store _invalid_ linear terms (all zero) - workaround by setting these to default linear term
        if np.allclose(self.linear_cde, 0.0):
            self.linear_cde = np.array([1.0, 0.0, 0.0], dtype=np.float32)


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


def image_points_to_camera_rays_opencv_fisheye(
    camera_model_parameters: OpenCVFisheyeCameraModelParameters,
    image_points: torch.Tensor,
    newton_iterations: int = 3,
    min_2d_norm: float = 1e-6,
    device: str = "cpu",
):
    """
    Computes the camera ray for each image point, performing an iterative undistortion of the nonlinear distortion model
    """

    assert isinstance(camera_model_parameters, OpenCVFisheyeCameraModelParameters), "[CameraModel]: camera_model_parameters must be of type OpenCVFisheyeCameraModelParameters"

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
        (torch.sin(thetas) * normalized_image_points / torch.maximum(deltas, min_2d_norm), torch.cos(thetas))
    )
    cam_rays[deltas.flatten() < min_2d_norm, :] = torch.tensor([[0, 0, 1]]).to(normalized_image_points)

    return cam_rays


def image_points_to_camera_rays_ftheta(
    camera_model_parameters: FThetaCameraModelParameters,
    image_points: torch.Tensor,
    newton_iterations: int = 3,
    min_2d_norm: float = 1e-6,
    device: str = "cpu",
):
    dtype: torch.dtype = torch.float32

    reference_poly = camera_model_parameters.reference_poly

    # FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
    # the center of the first pixel. To conform to the CameraModel specification (having the image
    # coordinate origin aligned with the top-left corner of the first pixel) we therefore need to
    # offset the principal point by half a pixel.
    # Please see documentation for more information.
    principal_point = torch.tensor(camera_model_parameters.principal_point, device=device, dtype=dtype) + 0.5

    fw_poly = torch.tensor(camera_model_parameters.fw_poly, device=device, dtype=dtype)
    bw_poly = torch.tensor(camera_model_parameters.bw_poly, device=device, dtype=dtype)

    # Linear term A = [c,d;e;1], A^-1 = 1/(c-e*d)*[1,-d;-e,c]
    c, d, e = camera_model_parameters.linear_cde
    A = torch.tensor(
        [
            [c, d],
            [e, 1],
        ],
        dtype=dtype,
        device=device,
    )
    Ainv = torch.tensor(
        [
            [1, -d],
            [-e, c],
        ],
        dtype=dtype,
        device=device,
    ) / (c - e * d)

    # Initialize first derivative of polynomials for Newton iteration-based inversions
    dfw_poly = torch.tensor(
        # coefficient of first derivative of the forward polynomial
        [i * c for i, c in enumerate(camera_model_parameters.fw_poly[1:], start=1)],
        dtype=dtype,
        device=device,
    )
    dbw_poly = torch.tensor(
        # coefficient of first derivative of the backwards polynomial
        [i * c for i, c in enumerate(camera_model_parameters.bw_poly[1:], start=1)],
        dtype=dtype,
        device=device,
    )

    # max_angle = float(camera_model_parameters.max_angle)

    # 2D pixel-distance threshold
    assert min_2d_norm > 0, "require positive minimum norm threshold"
    min_2d_norm = torch.tensor(min_2d_norm, dtype=dtype, device=device)

    assert principal_point.shape == (2,)
    assert principal_point.dtype == dtype
    assert fw_poly.shape == (6,)
    assert fw_poly.dtype == dtype
    assert dfw_poly.shape == (5,)
    assert dfw_poly.dtype == dtype
    assert bw_poly.shape == (6,)
    assert bw_poly.dtype == dtype
    assert dbw_poly.shape == (5,)
    assert dbw_poly.dtype == dtype
    assert A.shape == (2, 2)
    assert A.dtype == dtype
    assert Ainv.shape == (2, 2)
    assert Ainv.dtype == dtype

    assert image_points.is_floating_point(), "[CameraModel]: image_points must be floating point values"
    image_points = image_points.to(dtype)

    # Get f(theta)-weighted normalized 2d vectors (undoing linear term)
    image_points_dist = torch.einsum("ij,nj->ni", Ainv, image_points - principal_point)
    rdist = torch.linalg.norm(image_points_dist, axis=1, keepdims=True)

    # Evaluate backward polynomial to get theta = f^-1(rdist) factors
    if reference_poly == FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
        thetas = _eval_poly_horner(bw_poly, rdist)  # bw is reference, evaluate it directly
    else:
        # fw is reference, evaluate its inverse via newton-based inversion
        thetas = _eval_poly_inverse_horner_newton(
            fw_poly, dfw_poly, bw_poly, newton_iterations, rdist
        )

    # Compute the camera rays and set the ones at the image center to [0,0,1]
    cam_rays = torch.hstack(
        (torch.sin(thetas) * image_points_dist / torch.maximum(rdist, min_2d_norm), torch.cos(thetas))
    )
    cam_rays[rdist.flatten() < min_2d_norm, :] = torch.tensor(
        [[0, 0, 1]], device=device, dtype=dtype
    )

    return cam_rays


def pixels_to_image_points(pixel_idxs) -> torch.Tensor:
    """Given integer-based pixels indices, computes corresponding continuous image point coordinates representing the *center* of each pixel."""
    assert isinstance(pixel_idxs, torch.Tensor), "[CameraModel]: Pixel indices must be a torch tensor"
    assert not pixel_idxs.is_floating_point(), "[CameraModel]: Pixel indices must be integers"
    # Compute the image point coordinates representing the center of each pixel (shift from top left corner to the center)
    return pixel_idxs.to(torch.float32) + 0.5
