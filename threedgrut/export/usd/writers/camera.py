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

Exports one Camera prim per physical camera with time-sampled transforms
and static intrinsics, following the pattern established in NRE's
rig_trajectories.py.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)
from pxr import Gf, Sdf, Tf, Usd, UsdGeom

from threedgrut.export.transforms import column_vector_4x4_to_usd_matrix

logger = logging.getLogger(__name__)

DEFAULT_NEAR_CLIP = 0.001
DEFAULT_FAR_CLIP = 10000000.0

# Coordinate transform from 3DGRUT (right-down-front) to USD camera (right-up-back)
_CAMERA_COORD_FLIP = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


def _make_usd_prim_name(name: str) -> str:
    """Convert an arbitrary string to a valid USD prim identifier."""
    return Tf.MakeValidIdentifier(name)


def _add_opencv_pinhole_camera_intrinsics(
    camera_prim: Usd.Prim,
    params: OpenCVPinholeCameraModelParameters,
) -> None:
    camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("pinholeOpenCV")

    resolution_list = params.resolution.tolist()
    camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(float(resolution_list[0]))
    camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(float(resolution_list[1]))

    principal_point_list = params.principal_point.tolist()
    camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[0]))
    camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[1]))

    focal_length_list = params.focal_length.tolist()
    camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[0]))
    camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[1]))

    radial_coeffs_list = params.radial_coeffs.tolist()
    camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[0]))
    camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[1]))
    camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[2]))
    camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[3]))
    camera_prim.CreateAttribute("fthetaPolyE", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[4]))
    camera_prim.CreateAttribute("fthetaPolyF", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[5]))

    tangential_coeffs_list = params.tangential_coeffs.tolist()
    camera_prim.CreateAttribute("p0", Sdf.ValueTypeNames.Float).Set(float(tangential_coeffs_list[0]))
    camera_prim.CreateAttribute("p1", Sdf.ValueTypeNames.Float).Set(float(tangential_coeffs_list[1]))

    thin_prism_coeffs_list = params.thin_prism_coeffs.tolist()
    camera_prim.CreateAttribute("s0", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[0]))
    camera_prim.CreateAttribute("s1", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[1]))
    camera_prim.CreateAttribute("s2", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[2]))
    camera_prim.CreateAttribute("s3", Sdf.ValueTypeNames.Float).Set(float(thin_prism_coeffs_list[3]))


def _add_opencv_fisheye_camera_intrinsics(
    camera_prim: Usd.Prim,
    params: OpenCVFisheyeCameraModelParameters,
) -> None:
    camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("fisheyeOpenCV")

    resolution_list = params.resolution.tolist()
    camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(float(resolution_list[0]))
    camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(float(resolution_list[1]))

    principal_point_list = params.principal_point.tolist()
    camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[0]))
    camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(float(principal_point_list[1]))

    focal_length_list = params.focal_length.tolist()
    camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[0]))
    camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(float(focal_length_list[1]))

    radial_coeffs_list = params.radial_coeffs.tolist()
    camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[0]))
    camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[1]))
    camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[2]))
    camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(float(radial_coeffs_list[3]))

    camera_prim.CreateAttribute("fthetaMaxFov", Sdf.ValueTypeNames.Float).Set(
        float(2.0 * np.rad2deg(params.max_angle))
    )


def export_cameras_to_usd(
    stage: Usd.Stage,
    poses: np.ndarray,
    camera_names: List[str],
    frame_to_camera: List[int],
    camera_params: Optional[List] = None,
    root_path: str = "/World/Cameras",
    visible: bool = False,
) -> Dict[str, str]:
    """
    Export camera poses with intrinsics to a USD stage.

    Creates one Camera prim per physical camera with time-sampled transforms
    and static intrinsics. The time code for frame i is float(i), so
    stage.GetTimeCodesPerSecond() controls real-time playback speed.

    Args:
        stage: USD stage to export to.
        poses: Camera-to-world transforms [N_frames, 4, 4] in 3DGRUT convention
            (right-down-front).
        camera_names: Logical name for each physical camera, indexed by camera_idx.
        frame_to_camera: Per-frame camera index mapping, length N_frames.
        camera_params: Per-frame CameraModelParameters (OpenCVPinhole / Fisheye).
            Intrinsics are taken from the first frame of each camera.
        root_path: USD path for the camera root Xform.
        visible: Whether camera prims should be visible in the viewport.

    Returns:
        Mapping {camera_name: usd_prim_path} for every exported camera.
    """
    num_cameras = len(camera_names)

    # Group frame indices by camera
    camera_frames: Dict[int, List[int]] = {i: [] for i in range(num_cameras)}
    for frame_idx, cam_idx in enumerate(frame_to_camera):
        if 0 <= cam_idx < num_cameras:
            camera_frames[cam_idx].append(frame_idx)

    UsdGeom.Xform.Define(stage, root_path)

    result: Dict[str, str] = {}

    for cam_idx, cam_name in enumerate(camera_names):
        frame_indices = camera_frames[cam_idx]
        if not frame_indices:
            logger.warning(f"Camera '{cam_name}' (idx {cam_idx}) has no frames, skipping")
            continue

        prim_name = _make_usd_prim_name(cam_name)
        camera_path = f"{root_path}/{prim_name}"

        camera_prim = stage.DefinePrim(camera_path, "Camera")
        camera = UsdGeom.Camera(camera_prim)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(DEFAULT_NEAR_CLIP, DEFAULT_FAR_CLIP))

        # Static intrinsics from first frame of this camera
        first_frame = frame_indices[0]
        if camera_params is not None and first_frame < len(camera_params) and camera_params[first_frame] is not None:
            params = camera_params[first_frame]
            if isinstance(params, OpenCVPinholeCameraModelParameters):
                _add_opencv_pinhole_camera_intrinsics(camera_prim, params)
            elif isinstance(params, OpenCVFisheyeCameraModelParameters):
                _add_opencv_fisheye_camera_intrinsics(camera_prim, params)
            else:
                camera.GetFocalLengthAttr().Set(24.0)
                logger.warning(f"Unsupported camera model for '{cam_name}', using default focal length")
        else:
            camera.GetFocalLengthAttr().Set(24.0)

        # Time-sampled transforms — one sample per frame belonging to this camera
        xformable = UsdGeom.Xformable(camera_prim)
        transform_op = xformable.AddTransformOp()
        for frame_idx in frame_indices:
            usd_pose = poses[frame_idx] @ _CAMERA_COORD_FLIP
            transform_op.Set(column_vector_4x4_to_usd_matrix(usd_pose), float(frame_idx))

        imageable = UsdGeom.Imageable(camera_prim)
        imageable.CreateVisibilityAttr().Set("inherited" if visible else "invisible")

        result[cam_name] = camera_path

    logger.info(
        f"Exported {len(result)} camera(s) ({len(poses)} total frames) to {root_path}"
    )
    return result


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

    rig_prim = stage.DefinePrim(root_path, "Xform")
    rig_xform = UsdGeom.Xformable(rig_prim)

    usd_time_code_per_second = stage.GetTimeCodesPerSecond()
    usd_timestamp_scale = usd_time_code_per_second * 1e-06

    rig_transform_op = rig_xform.AddTransformOp()

    usd_start_time_code = float("inf")
    usd_end_time_code = 0.0

    for i in range(num_frames):
        usd_pose = poses[i] @ _CAMERA_COORD_FLIP
        usd_matrix = column_vector_4x4_to_usd_matrix(usd_pose)

        if timestamps_us is not None:
            usd_time_code = usd_timestamp_scale * (timestamps_us[i] - timestamp_offset_us)
        else:
            usd_time_code = float(i)

        usd_start_time_code = min(usd_start_time_code, usd_time_code)
        usd_end_time_code = max(usd_end_time_code, usd_time_code)
        rig_transform_op.Set(usd_matrix, usd_time_code)

    if usd_start_time_code <= usd_end_time_code:
        stage.SetMetadata("startTimeCode", usd_start_time_code)
        stage.SetMetadata("endTimeCode", usd_end_time_code)

    if timestamps_us is not None:
        stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", timestamp_offset_us)

    camera_path = f"{root_path}/{camera_name}"
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    camera = UsdGeom.Camera(camera_prim)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(DEFAULT_NEAR_CLIP, DEFAULT_FAR_CLIP))

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

    xformable = UsdGeom.Xformable(camera_prim)
    transform_op = xformable.AddTransformOp()
    transform_op.Set(Gf.Matrix4d(1.0))

    imageable = UsdGeom.Imageable(camera_prim)
    imageable.CreateVisibilityAttr().Set("inherited" if visible else "invisible")

    logger.info(f"Exported camera rig with {num_frames} frames to {root_path}")
    return root_path
