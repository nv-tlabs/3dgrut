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

"""
Camera USD writer for exporting camera poses and intrinsics.

Exports camera poses with full intrinsics support for OpenCVPinhole and OpenCVFisheye
camera models, following the pattern established in NRE's rig_trajectories.py.
"""

import logging
from typing import List, Optional

import numpy as np
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from threedgrut.export.transforms import column_vector_4x4_to_usd_matrix

logger = logging.getLogger(__name__)

# Default clipping range for cameras
DEFAULT_NEAR_CLIP = 0.001
DEFAULT_FAR_CLIP = 10000000.0


def _add_opencv_pinhole_camera_intrinsics(
    camera_prim: Usd.Prim,
    params: OpenCVPinholeCameraModelParameters,
) -> None:
    """Add OpenCV pinhole camera intrinsics to USD camera prim."""
    # Camera projection type
    camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(Vt.Token("pinholeOpenCV"))

    # Resolution
    resolution_list = params.resolution.tolist()
    camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(float(resolution_list[0]))
    camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(float(resolution_list[1]))

    # Principal point
    principal_point_list = params.principal_point.tolist()
    camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[0]))
    camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[1]))

    # Focal length
    focal_length_list = params.focal_length.tolist()
    camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[0]))
    camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[1]))

    # Radial distortion coefficients [k1,k2,k3,k4,k5,k6]
    radial_coeffs_list = params.radial_coeffs.tolist()
    camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[0]))
    camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[1]))
    camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[2]))
    camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[3]))
    camera_prim.CreateAttribute("fthetaPolyE", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[4]))
    camera_prim.CreateAttribute("fthetaPolyF", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[5]))

    # Tangential distortion coefficients [p1,p2]
    tangential_coeffs_list = params.tangential_coeffs.tolist()
    camera_prim.CreateAttribute("p0", Sdf.ValueTypeNames.Float).Set(float(tangential_coeffs_list[0]))
    camera_prim.CreateAttribute("p1", Sdf.ValueTypeNames.Float).Set(float(tangential_coeffs_list[1]))

    # Thin prism distortion coefficients [s1,s2,s3,s4]
    thin_prism_coeffs_list = params.thin_prism_coeffs.tolist()
    camera_prim.CreateAttribute("s0", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[0]))
    camera_prim.CreateAttribute("s1", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[1]))
    camera_prim.CreateAttribute("s2", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[2]))
    camera_prim.CreateAttribute("s3", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[3]))


def _add_opencv_fisheye_camera_intrinsics(
    camera_prim: Usd.Prim,
    params: OpenCVFisheyeCameraModelParameters,
) -> None:
    """Add OpenCV fisheye camera intrinsics to USD camera prim."""
    # Camera projection type
    camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(Vt.Token("fisheyeOpenCV"))

    # Resolution
    resolution_list = params.resolution.tolist()
    camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(float(resolution_list[0]))
    camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(float(resolution_list[1]))

    # Principal point
    principal_point_list = params.principal_point.tolist()
    camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[0]))
    camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[1]))

    # Focal length
    focal_length_list = params.focal_length.tolist()
    camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[0]))
    camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[1]))

    # Radial distortion coefficients [k1,k2,k3,k4]
    radial_coeffs_list = params.radial_coeffs.tolist()
    camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[0]))
    camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[1]))
    camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[2]))
    camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[3]))

    # Max FoV (convert from radians to degrees, x2 for full FoV)
    camera_prim.CreateAttribute("fthetaMaxFov", Sdf.ValueTypeNames.Float).Set(float(2.0 * np.rad2deg(params.max_angle)))


def _add_simple_pinhole_intrinsics(
    camera_prim: Usd.Prim,
    intrinsics: List[float],
    resolution: List[int],
) -> None:
    """Add simple pinhole intrinsics [fx, fy, cx, cy] without distortion."""
    fx, fy, cx, cy = intrinsics

    # Use standard USD pinhole camera attributes
    # Compute horizontal aperture from resolution and focal length
    # USD uses mm for aperture, assuming sensor is 36mm (full-frame)
    sensor_width_mm = 36.0
    focal_length_mm = (fx / resolution[0]) * sensor_width_mm

    camera_prim.GetFocalLengthAttr().Set(focal_length_mm)
    camera_prim.GetHorizontalApertureAttr().Set(sensor_width_mm)
    camera_prim.GetVerticalApertureAttr().Set(sensor_width_mm * resolution[1] / resolution[0])

    # Principal point offset from center
    horizontal_offset = ((cx / resolution[0]) - 0.5) * sensor_width_mm
    vertical_offset = ((cy / resolution[1]) - 0.5) * (sensor_width_mm * resolution[1] / resolution[0])
    camera_prim.GetHorizontalApertureOffsetAttr().Set(horizontal_offset)
    camera_prim.GetVerticalApertureOffsetAttr().Set(vertical_offset)


def export_cameras_to_usd(
    stage: Usd.Stage,
    poses: np.ndarray,
    intrinsics: Optional[List] = None,
    camera_params: Optional[List] = None,
    resolutions: Optional[List[np.ndarray]] = None,
    root_path: str = "/World/Cameras",
    camera_prefix: str = "camera",
    visible: bool = False,
) -> str:
    """
    Export camera poses with intrinsics to USD stage.

    Supports multiple camera model types:
    - OpenCVPinholeCameraModelParameters: Full pinhole with distortion
    - OpenCVFisheyeCameraModelParameters: Fisheye with distortion
    - Simple intrinsics: [fx, fy, cx, cy] list for basic pinhole

    Args:
        stage: USD stage to export to
        poses: Camera poses [N, 4, 4] in 3DGRUT convention (right-down-front)
        intrinsics: Optional list of [fx, fy, cx, cy] for simple pinhole
        camera_params: Optional list of camera model parameters (OpenCVPinhole/Fisheye)
        resolutions: Optional list of resolutions [[w, h], ...] for simple intrinsics
        root_path: USD path for camera root xform
        camera_prefix: Prefix for camera names
        visible: Whether cameras should be visible in viewport

    Returns:
        Root path of the cameras
    """
    num_cameras = poses.shape[0]

    # Create root xform for cameras
    UsdGeom.Xform.Define(stage, root_path)

    # Coordinate transform from 3DGRUT (right-down-front) to USD camera (right-up-back)
    # 3DGRUT: X=right, Y=down, Z=front
    # USD:    X=right, Y=up, Z=back
    camera_coord_flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

    for i in range(num_cameras):
        camera_name = f"{camera_prefix}_{i:04d}"
        camera_path = f"{root_path}/{camera_name}"

        # Define camera prim
        camera_prim = stage.DefinePrim(camera_path, "Camera")
        camera = UsdGeom.Camera(camera_prim)

        # Set clipping range
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(DEFAULT_NEAR_CLIP, DEFAULT_FAR_CLIP))

        # Add intrinsics based on available data
        if camera_params is not None and i < len(camera_params) and camera_params[i] is not None:
            params = camera_params[i]
            if isinstance(params, OpenCVPinholeCameraModelParameters):
                _add_opencv_pinhole_camera_intrinsics(camera_prim, params)
            elif isinstance(params, OpenCVFisheyeCameraModelParameters):
                _add_opencv_fisheye_camera_intrinsics(camera_prim, params)
            else:
                # Fallback to default focal length
                camera.GetFocalLengthAttr().Set(24.0)
                logger.warning(f"Unsupported camera model for camera {i}, using default intrinsics")
        elif intrinsics is not None and resolutions is not None:
            # Simple pinhole from intrinsics list
            if i < len(resolutions):
                resolution = resolutions[i].tolist() if isinstance(resolutions[i], np.ndarray) else resolutions[i]
            else:
                resolution = resolutions[0].tolist() if isinstance(resolutions[0], np.ndarray) else resolutions[0]
            _add_simple_pinhole_intrinsics(camera_prim, intrinsics, resolution)
        else:
            # Fallback to default focal length
            camera.GetFocalLengthAttr().Set(24.0)

        # Set camera transform (pose)
        # Apply coordinate system transform: 3DGRUT -> USD camera, then build USD matrix via Gf API
        pose = poses[i]
        usd_pose = pose @ camera_coord_flip
        usd_matrix = column_vector_4x4_to_usd_matrix(usd_pose)

        xformable = UsdGeom.Xformable(camera_prim)
        transform_op = xformable.AddTransformOp()
        transform_op.Set(usd_matrix)

        # Set visibility
        imageable = UsdGeom.Imageable(camera_prim)
        visibility = "inherited" if visible else "invisible"
        imageable.CreateVisibilityAttr().Set(visibility)

    logger.info(f"Exported {num_cameras} cameras to {root_path}")
    return root_path


def export_camera_rig_with_timestamps(
    stage: Usd.Stage,
    poses: np.ndarray,
    timestamps_us: Optional[np.ndarray] = None,
    camera_params: Optional[List] = None,
    root_path: str = "/World/sensor_rig",
    camera_name: str = "camera",
    timestamp_offset_us: int = 0,
    visible: bool = False,
) -> str:
    """
    Export a camera rig with time-sampled poses.

    Similar to NRE's rig trajectories export but simplified for 3DGRUT
    (single camera, no rig hierarchy).

    Args:
        stage: USD stage to export to
        poses: Camera poses [N, 4, 4]
        timestamps_us: Optional timestamps in microseconds [N]
        camera_params: Optional camera model parameters
        root_path: USD path for rig root
        camera_name: Name for the camera
        timestamp_offset_us: Offset to apply to timestamps
        visible: Whether camera should be visible

    Returns:
        Root path of the rig
    """
    num_frames = poses.shape[0]

    # Create rig xform
    rig_prim = stage.DefinePrim(root_path, "Xform")
    rig_xform = UsdGeom.Xformable(rig_prim)

    # Coordinate transform
    camera_coord_flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

    # USD time code setup
    usd_time_code_per_second = stage.GetTimeCodesPerSecond()
    usd_timestamp_scale = usd_time_code_per_second * 1e-06  # microseconds to time codes

    # Create transform op for rig
    rig_transform_op = rig_xform.AddTransformOp()

    usd_start_time_code = float("inf")
    usd_end_time_code = 0.0

    # Add time-sampled transforms
    for i in range(num_frames):
        pose = poses[i]
        usd_pose = pose @ camera_coord_flip
        usd_matrix = column_vector_4x4_to_usd_matrix(usd_pose)

        if timestamps_us is not None:
            timestamp = timestamps_us[i]
            usd_time_code = usd_timestamp_scale * (timestamp - timestamp_offset_us)
            usd_start_time_code = min(usd_start_time_code, usd_time_code)
            usd_end_time_code = max(usd_end_time_code, usd_time_code)
        else:
            usd_time_code = float(i)
            usd_start_time_code = min(usd_start_time_code, usd_time_code)
            usd_end_time_code = max(usd_end_time_code, usd_time_code)

        rig_transform_op.Set(usd_matrix, usd_time_code)

    # Set time metadata
    if usd_start_time_code <= usd_end_time_code:
        stage.SetMetadata("startTimeCode", usd_start_time_code)
        stage.SetMetadata("endTimeCode", usd_end_time_code)

    if timestamps_us is not None:
        stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", timestamp_offset_us)

    # Create camera prim under rig (static relative to rig)
    camera_path = f"{root_path}/{camera_name}"
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    camera = UsdGeom.Camera(camera_prim)

    # Set default clipping range
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(DEFAULT_NEAR_CLIP, DEFAULT_FAR_CLIP))

    # Add intrinsics if provided
    if camera_params is not None and len(camera_params) > 0:
        params = camera_params[0]
        if isinstance(params, OpenCVPinholeCameraModelParameters):
            _add_opencv_pinhole_camera_intrinsics(camera_prim, params)
        elif isinstance(params, OpenCVFisheyeCameraModelParameters):
            _add_opencv_fisheye_camera_intrinsics(camera_prim, params)
        else:
            camera.GetFocalLengthAttr().Set(24.0)
    else:
        camera.GetFocalLengthAttr().Set(24.0)

    # Camera is at identity transform relative to rig (transform is on rig itself)
    xformable = UsdGeom.Xformable(camera_prim)
    transform_op = xformable.AddTransformOp()
    transform_op.Set(Gf.Matrix4d(1.0))

    # Set visibility
    imageable = UsdGeom.Imageable(camera_prim)
    visibility = "inherited" if visible else "invisible"
    imageable.CreateVisibilityAttr().Set(visibility)

    logger.info(f"Exported camera rig with {num_frames} frames to {root_path}")
    return root_path
