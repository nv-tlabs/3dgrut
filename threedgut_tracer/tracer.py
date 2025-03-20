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

import logging, math
from dataclasses import dataclass

import torch
import torch.utils.cpp_extension

import numpy as np

from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_3dgut_plugin = None


def load_3dgut_plugin(conf):
    global _3dgut_plugin
    if _3dgut_plugin is None:
        try:
            from . import lib3dgut_cc as tdgut  # type: ignore
        except ImportError:
            from .setup_3dgut import setup_3dgut

            setup_3dgut(conf)
            import lib3dgut_cc as tdgut  # type: ignore
        _3dgut_plugin = tdgut


# ----------------------------------------------------------------------------
#


@dataclass
class SensorPose3D:
    T_world_sensors: list  # represents two tquat [t,q] poses
    timestamps_us: list


class SensorPose3DModel:
    def __init__(self, R, T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(SensorPose3DModel, self).__init__()
        self.R = R
        self.T = T
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = (
            torch.tensor(SensorPose3DModel.__getWorld2View2(R, T, trans, scale)).transpose(0, 1).cpu()
        )
        # self.camera_center = self.world_view_transform.inverse()[3, :3].cpu()

    @staticmethod
    def __getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    @staticmethod
    def __so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
        """
        Converts a singe / batch of SO3 rotation matrices (3x3) to unit quaternion representation.

        Args:
            R: single / batch of SO3 rotation matrices [bs, 3, 3] or [3,3]
            unbatch: if the single example should be unbatched (first dimension removed) or not

        Returns:
            single / batch of unit quaternions (XYZW convention)  [bs, 4] or [4]
        """

        # Convert numpy array to torch tensor
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)

        R = R.reshape((-1, 3, 3))  # batch dimensions unconditionally
        num_rotations, D1, D2 = R.shape
        assert (D1, D2) == (3, 3), "so3_matrix_to_quat: Input has to be a Bx3x3 tensor."

        decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
        quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)

        decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
        choices = decision_matrix.argmax(dim=1)

        ind = torch.nonzero(choices != 3, as_tuple=True)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
        quat[ind, j] = R[ind, j, i] + R[ind, i, j]
        quat[ind, k] = R[ind, k, i] + R[ind, i, k]
        quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

        ind = torch.nonzero(choices == 3, as_tuple=True)[0]
        quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
        quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
        quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]

        quat = quat / torch.norm(quat, dim=1)[:, None]

        if unbatch:  # unbatch dimensions conditionally
            quat = quat.squeeze()

        return quat  # (N,4) or (4,)

    def get_sensor_pose(self):
        # construct rolling-shutter parameters for perfect pinhole (global shutter / static pose / no timestamps available)
        T_world_sensor_t = self.world_view_transform[
            3, :3
        ]  # translation is last *row* in current "transposed" conventions
        T_world_sensor_R = self.world_view_transform[:3, :3].transpose(
            0, 1
        )  # rotation matrix ("transposed" in current conventions)
        T_world_sensor_quat = SensorPose3DModel.__so3_matrix_to_quat(T_world_sensor_R)
        T_world_sensor_tquat = torch.hstack([T_world_sensor_t.cpu(), T_world_sensor_quat.cpu()])
        return SensorPose3D(
            T_world_sensors=[T_world_sensor_tquat, T_world_sensor_tquat],  # same pose for both timestamps
            timestamps_us=[0, 1],  # arbitrary timestamps
        )


# ----------------------------------------------------------------------------
#


class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            frame_id,
            n_active_features,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            sensor_params,
            sensor_poses,
        ):
            particle_density = torch.concat(
                [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1
            ).contiguous()
            particle_radiance = mog_sph.contiguous()

            ray_time = (
                torch.ones(
                    (ray_ori.shape[0], ray_ori.shape[1], ray_ori.shape[2], 1), device=ray_ori.device, dtype=torch.long
                )
                * sensor_poses.timestamps_us[0]
            )

            ray_radiance_density, ray_hit_distance = tracer_wrapper.trace(
                frame_id,
                n_active_features,
                particle_density,
                particle_radiance,
                ray_ori.contiguous(),
                ray_dir.contiguous(),
                ray_time.contiguous(),
                sensor_params,
                sensor_poses.timestamps_us[0],
                sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0],
                sensor_poses.T_world_sensors[1],
            )

            ctx.save_for_backward(
                ray_ori,
                ray_dir,
                ray_time,
                ray_radiance_density,
                ray_hit_distance,
                particle_density,
                particle_radiance,
            )

            ctx.frame_id = frame_id
            ctx.n_active_features = n_active_features
            ctx.sensor_params = sensor_params
            ctx.sensor_poses = sensor_poses
            ctx.tracer_wrapper = tracer_wrapper

            return (
                ray_radiance_density,
                ray_hit_distance,
            )

        @staticmethod
        def backward(
            ctx,
            ray_radiance_density_grd,
            ray_hit_distance_grd,
        ):
            (
                ray_ori,
                ray_dir,
                ray_time,
                ray_radiance_density,
                ray_hit_distance,
                particle_density,
                particle_radiance,
            ) = ctx.saved_variables

            frame_id = ctx.frame_id
            n_active_features = ctx.n_active_features
            sensor_params = ctx.sensor_params
            sensor_poses = ctx.sensor_poses

            particle_density_grd, particle_radiance_grd = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                n_active_features,
                particle_density,
                particle_radiance,
                ray_ori,
                ray_dir,
                ray_time,
                sensor_params,
                sensor_poses.timestamps_us[0],
                sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0],
                sensor_poses.T_world_sensors[1],
                ray_radiance_density,
                ray_radiance_density_grd,
                ray_hit_distance,
                ray_hit_distance_grd,
            )

            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            mog_sph_grd = particle_radiance_grd

            return (
                None,  # tracer_wrapper
                None,  # frame_id
                None,  # n_active_features
                None,  # ray_ori
                None,  # ray_dir
                mog_pos_grd.contiguous(),
                mog_rot_grd.contiguous(),
                mog_scl_grd.contiguous(),
                mog_dns_grd.contiguous(),
                mog_sph_grd.contiguous(),
                None,  # sensor_params
                None,  # sensor_poses
            )

    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf

        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        load_3dgut_plugin(conf)

        self.tracer_wrapper = _3dgut_plugin.SplatRaster(OmegaConf.to_container(conf))

    @property
    def timings(self):
        return self.tracer_wrapper.collect_times()

    def build_acc(self, gaussians, rebuild=True):
        pass  # no-op for 3DGUT

    def render(self, gaussians, gpu_batch: Batch, train=False, frame_id=0):
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir

        sensor, poses = Tracer.__create_camera_parameters(gpu_batch)

        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            (
                pred_rgba,
                pred_dist,
            ) = Tracer._Autograd.apply(
                self.tracer_wrapper,
                frame_id,
                gaussians.n_active_features,
                rays_o.contiguous(),
                rays_d.contiguous(),
                gaussians.positions.contiguous(),
                gaussians.get_rotation().contiguous(),
                gaussians.get_scale().contiguous(),
                gaussians.get_density().contiguous(),
                gaussians.get_features().contiguous(),
                sensor,
                poses,
            )

            pred_rgb = pred_rgba[..., :3].unsqueeze(0)
            pred_opacity = pred_rgba[..., 3:].unsqueeze(0)
            pred_dist = pred_dist.unsqueeze(0)

            pred_rgb, pred_opacity = gaussians.background(
                gpu_batch.T_to_world.contiguous(), rays_d, pred_rgb, pred_opacity, train
            )

            timings = self.tracer_wrapper.collect_times()

        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(torch.ones_like(pred_rgb), dim=3),
            "hits_count": torch.zeros_like(pred_rgb[..., :1]),
            "frame_time_ms": timings["forward_render"] if "forward_render" in timings else 0.0,
        }

    @staticmethod
    def __fov2focal(fov_radians: float, pixels: int):
        return pixels / (2 * math.tan(fov_radians / 2))

    @staticmethod
    def __focal2fov(focal: float, pixels: int):
        return 2 * math.atan(pixels / (2 * focal))

    @staticmethod
    def __create_camera_parameters(gpu_batch):
        from threedgrut.datasets.camera_models import ShutterType

        SHUTTER_TYPE_MAP = {
            ShutterType.ROLLING_TOP_TO_BOTTOM: _3dgut_plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT: _3dgut_plugin.ShutterType.ROLLING_LEFT_TO_RIGHT,
            ShutterType.ROLLING_BOTTOM_TO_TOP: _3dgut_plugin.ShutterType.ROLLING_BOTTOM_TO_TOP,
            ShutterType.ROLLING_RIGHT_TO_LEFT: _3dgut_plugin.ShutterType.ROLLING_RIGHT_TO_LEFT,
            ShutterType.GLOBAL: _3dgut_plugin.ShutterType.GLOBAL,
        }

        # Process the camera extrinsics
        pose = gpu_batch.T_to_world.squeeze()
        assert pose.ndim == 2
        C2W = np.concatenate((pose[:3, :4].cpu().detach().numpy(), np.zeros((1, 4))))
        C2W[3, 3] = 1.0

        # Get the world-to-camera transform and set R, T
        W2C = np.linalg.inv(C2W)
        R = np.transpose(W2C[:3, :3])
        T = W2C[:3, 3]
        pose_model = SensorPose3DModel(R=R, T=T)

        # Process the camera intrinsics
        if (K := gpu_batch.intrinsics) is not None:
            focalx, focaly, cx, cy = K[0], K[1], K[2], K[3]
            orig_w = int(2 * cx)
            orig_h = int(2 * cy)
            FovX = Tracer.__focal2fov(focalx, orig_w)
            FovY = Tracer.__focal2fov(focaly, orig_h)
            # Compute the camera model parameters
            camera_model_parameters = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=np.array([orig_w, orig_h], dtype=np.uint64),
                shutter_type=_3dgut_plugin.ShutterType.GLOBAL,
                principal_point=np.array([orig_w, orig_h], dtype=np.float32) / 2,
                focal_length=np.array(
                    [orig_w / (2.0 * math.tan(FovX * 0.5)), orig_h / (2.0 * math.tan(FovY * 0.5))], dtype=np.float32
                ),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            return camera_model_parameters, pose_model.get_sensor_pose()

        elif (K := gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters) is not None:
            camera_model_parameters = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                tangential_coeffs=K["tangential_coeffs"],
                thin_prism_coeffs=K["thin_prism_coeffs"],
            )
            return camera_model_parameters, pose_model.get_sensor_pose()

        elif (K := gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters) is not None:
            camera_model_parameters = _3dgut_plugin.fromOpenCVFisheyeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                max_angle=K["max_angle"],
            )
            return camera_model_parameters, pose_model.get_sensor_pose()

        raise ValueError(
            f"Camera intrinsics unavailable or unsupported, input keys are [{', '.join(gpu_batch.keys())}]"
        )
