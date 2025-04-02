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

from typing import List
from scipy.special import comb
from kaolin.render.camera import Camera
import numpy as np

"""
This module is to be included in next version of kaolin 0.18.0.
As of March 26, 2025 the latest public release is kaolin 0.17.0, hence it's included here independently.
"""


__all__ = [
    'interpolate_camera_on_polynomial_path',
    'interpolate_camera_on_spline_path',
    'infinite_loop_camera_path_generator',
    'camera_path_generator'
]


def _smoothstep(
    x: float,
    x_min: float = 0.0,
    x_max: float = 1,
    N: int = 3
):
    """
    Generalized smoothstep polynomials function, used for smooth interpolation of point x in [x_min, x_max].
    N determines the order polynomial, where the exact order is 2N + 1.
    """
    # See: https://en.wikipedia.org/wiki/Smoothstep#Generalization_to_higher-order_equations
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    return result


def _catmull_rom(
    t: float,
    p0: float,
    p1: float,
    p2: float,
    p3: float,
):
    """
    Interpolates a point on a Catmull-Rom spline, defined by control points p0, p1, p2, p3.
    t determines the location of the point.
    Catmull-Rom splines are guaranteed to pass through all control points.
    """

    q_t = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
            (-p0 + 3*p1- 3*p2 + p3) * t**3
    )
    return q_t


def interpolate_camera_on_polynomial_path(
    trajectory: List[Camera],
    timestep: int,
    frames_between_cameras: int = 60,
    N: int = 3
):
    r"""Interpolates a camera from a smoothed path formed by a trajectory of cameras.
    The trajectory is assumed to have a list of at least 2 cameras, where the first and last cameras form a
    looped path.
    The interpolation is done using a generalized polynomial function.

    Args:
        trajectory (List[kaolin.render.camera.Camera]): A trajectory of camera nodes, used to form a continuous path.
        timestep (int): Timestep used to interpolate a point on the smoothed trajectory.
            `timestep` can take any integer value to support continuous animations.
            e.g. if `timestep > len(trajectory) X frames_between_cameras`, the timestep will fold over using a
            modulus.
        frames_between_cameras (int): Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.
        N (int): determines the order polynomial, where the exact order is 2N + 1.

    Returns:
        (kaolin.render.camera.Camera):
            An interpolated camera object formed by the cameras trajectory.
    """
    traj_idx = (timestep // frames_between_cameras) % len(trajectory)

    cam1 = trajectory[traj_idx]
    cam2 = trajectory[traj_idx + 1]

    eye_1, target_1, up_1 = cam1.cam_pos(), cam1.cam_forward(), cam1.cam_up()
    eye_2, target_2, up_2 = cam2.cam_pos(), cam2.cam_forward(), cam2.cam_up()
    width_1, height_1 = cam1.width, cam1.height
    width_2, height_2 = cam2.width, cam2.height

    intrinsics = dict()
    if cam1.lens_type == 'pinhole' and cam2.lens_type == 'pinhole':
        intrinsics['fov'] = (cam1.fov(in_degrees=False), cam2.fov(in_degrees=False))
    elif cam1.lens_type == 'ortho' and cam2.lens_type == 'ortho':
        intrinsics['fov_distance'] = (cam1.fov_distance(), cam2.fov_distance())

    Xs = _smoothstep(np.linspace(0.0, 1.0, frames_between_cameras), N=N)
    x = Xs[timestep % frames_between_cameras]

    eye = eye_1 * (1 - x) + eye_2 * x
    target = target_1 * (1 - x) + target_2 * x
    up = up_1 * (1 - x) + up_2 * x
    width = round(width_1 * (1 - x) + width_2 * x)
    height = round(height_1 * (1 - x) + height_2 * x)
    intrinsics = {intr_name: intr_val[0] * (1 - x) + intr_val[1] * x for intr_name, intr_val in intrinsics.items()}

    cam = Camera.from_args(
        eye=eye,
        at=eye - target,
        up=up,
        width=width, height=height,
        device=cam1.device,
        **intrinsics
    )

    return cam


def interpolate_camera_on_spline_path(
    trajectory: List[Camera],
    timestep: int,
    frames_between_cameras: int = 60
):
    r"""Interpolates a camera from a linear path formed by a trajectory of cameras.
    The trajectory is assumed to have a list of at least 4 cameras.
    The interpolation is done using Catmull-Rom Splines.

    Args:
        trajectory (List[kaolin.render.camera.Camera]): A trajectory of camera nodes, used to form a continuous path.
        timestep (int): Timestep used to interpolate a point on the smoothed trajectory.
            `timestep` can take any integer value to support continuous animations.
            e.g. if `timestep > len(trajectory) X frames_between_cameras`, the timestep will fold over using a
            modulus.
        frames_between_cameras (int): Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.

    Returns:
        (kaolin.render.camera.Camera):
            An interpolated camera object formed by the cameras trajectory.
    """
    traj_idx = (timestep // frames_between_cameras) % len(trajectory)

    traj_idx = min(max(traj_idx, 0), len(trajectory) - 3)

    cam1 = trajectory[traj_idx - 1]
    cam2 = trajectory[traj_idx]
    cam3 = trajectory[traj_idx + 1]
    cam4 = trajectory[traj_idx + 2]

    eye_1, target_1, up_1 = cam1.cam_pos(), cam1.cam_forward(), cam1.cam_up()
    eye_2, target_2, up_2 = cam2.cam_pos(), cam2.cam_forward(), cam2.cam_up()
    eye_3, target_3, up_3 = cam3.cam_pos(), cam3.cam_forward(), cam3.cam_up()
    eye_4, target_4, up_4 = cam4.cam_pos(), cam4.cam_forward(), cam4.cam_up()
    width_1, height_1 = cam1.width, cam1.height
    width_2, height_2 = cam2.width, cam2.height
    width_3, height_3 = cam3.width, cam3.height
    width_4, height_4 = cam4.width, cam4.height

    intrinsics = dict()
    if cam1.lens_type == 'pinhole' and cam2.lens_type == 'pinhole':
        intrinsics['fov'] = (
            cam1.fov(in_degrees=False),
            cam2.fov(in_degrees=False),
            cam3.fov(in_degrees=False),
            cam4.fov(in_degrees=False)
        )
    elif cam1.lens_type == 'ortho' and cam2.lens_type == 'ortho':
        intrinsics['fov_distance'] = \
            (cam1.fov_distance(), cam2.fov_distance(), cam3.fov_distance(), cam4.fov_distance())

    Xs = np.linspace(0.0, 1.0, frames_between_cameras)
    t = Xs[timestep % frames_between_cameras]

    eye = _catmull_rom(t, eye_1, eye_2, eye_3, eye_4)
    target = _catmull_rom(t, target_1, target_2, target_3, target_4)
    up = _catmull_rom(t, up_1, up_2, up_3, up_4)
    width = round(_catmull_rom(t, width_1, width_2, width_3, width_4))
    height = round(_catmull_rom(t, height_1, height_2, height_3, height_4))
    intrinsics = {intr_name: _catmull_rom(t, intr_val[0], intr_val[1], intr_val[2], intr_val[3])
                  for intr_name, intr_val in intrinsics.items()}

    cam = Camera.from_args(
        eye=eye,
        at=eye - target,
        up=up,
        width=width, height=height,
        device=cam1.device,
        **intrinsics
    )

    return cam


def infinite_loop_camera_path_generator(
    trajectory: List[Camera],
    frames_between_cameras: int = 60,
    interpolation: str = 'polynomial',
):
    r"""A generator function for returning continuous camera objects an o smoothed path interpolated
    from a trajectory of cameras.
    The trajectory is assumed to have a list of at least 2 cameras, where the first and last cameras form a
    looped path.
    This generator is therefore never exhausted, and can be invoked infinitely to generate continuous camera motion.

    Args:
        trajectory (List[kaolin.render.camera.Camera]): A trajectory of camera nodes, used to form a continuous path.
        frames_between_cameras (int): Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.
        interpolation (str): Type of interpolation function used:
            'polynomial' uses a smoothstep polynomial function which tends to overshoot around the keyframes.
                This interpolator is fitting for paths orbiting an object of interest.
            'catmull_rom' uses a spline defined by 4 control points, guaranteed to pass precisely through the keyframes.

    Returns:
        (kaolin.render.camera.Camera):
            An interpolated camera object formed by the cameras trajectory.
    """
    if interpolation == 'polynomial':
        interpolator =  interpolate_camera_on_polynomial_path
    elif interpolation == 'catmull_rom':
        interpolator =  interpolate_camera_on_spline_path
    else:
        raise ValueError("Unknown interpolation function specified. Valid options: 'polynomial', 'catmull_rom'.")

    timestep = 0
    while True:
        yield interpolator(trajectory, timestep, frames_between_cameras)
        timestep += 1


def camera_path_generator(
    trajectory: List[Camera],
    frames_between_cameras: int = 60,
    interpolation: str = 'catmull_rom'
):
    r"""A generator function for returning continuous camera objects an o path interpolated on a spline
    from a trajectory of cameras.
    The trajectory is assumed to have a list of at least 4 cameras.
    This generator is exhausted after it returns the last point on the path.

    Args:
        trajectory (List[kaolin.render.camera.Camera]): A trajectory of camera nodes, used to form a continuous path.
        frames_between_cameras (int): Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.
        interpolation (str): Type of interpolation function used:
            'polynomial' uses a smoothstep polynomial function which tends to overshoot around the keyframes.
                This interpolator is fitting for paths orbiting an object of interest.
            'catmull_rom' uses a spline defined by 4 control points, guaranteed to pass precisely through the keyframes.

    Returns:
        (kaolin.render.camera.Camera):
            An interpolated camera object formed by the cameras trajectory.
    """
    if interpolation == 'polynomial':
        interpolator =  interpolate_camera_on_polynomial_path
    elif interpolation == 'catmull_rom':
        interpolator =  interpolate_camera_on_spline_path
    else:
        raise ValueError("Unknown interpolation function specified. Valid options: 'polynomial', 'catmull_rom'.")

    _trajectory = [trajectory[0]] + trajectory + [trajectory[-1], trajectory[-1]]
    timestep = frames_between_cameras

    while True:
        yield interpolator(_trajectory, timestep, frames_between_cameras)
        timestep += 1

        traj_idx = (timestep // frames_between_cameras) % len(_trajectory)
        if traj_idx == len(_trajectory) - 3:
            break
