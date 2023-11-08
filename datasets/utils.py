from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Sequence

import torch
import numpy as np
from plyfile import PlyData

from datasets.ncore_utils import Batch as NCoreBatch, RayFlags as NCoreRayFlags

DEFAULT_DEVICE = torch.device('cuda')

def move_to_gpu(batch : dict | NCoreBatch) -> dict[str, torch.Tensor]:
    gpu_batch: dict[str, torch.Tensor]  # expecting 'rays_ori', 'rays_dir', 'rgb_gt', optional 'sky_mask'

    if isinstance(batch, dict):
        gpu_batch = {key: tensor.to(device=DEFAULT_DEVICE) for key,tensor in batch.items()}
    else:
        # Reshape NCore batch to target sizes (B x H x W x F) / (B x 1 x N x F)
        if (h := batch.h) is not None and (w := batch.w) is not None:
            assert len(h) == 1 and len(w) == 1
            out_shape3 = (1, h[0], w[0], 3)
            out_shape1 = (1, h[0], w[0], 1)
        else:
            out_shape3 = (1, 1, len(batch.rays_cam), 3)
            out_shape1 = (1, 1, len(batch.rays_cam), 1)

        gpu_batch = {}
        gpu_batch["rays_ori"] = batch.rays_cam[:, :3].reshape(out_shape3).to(device=DEFAULT_DEVICE)
        gpu_batch["rays_dir"] = batch.rays_cam[:, 3:6].reshape(out_shape3).to(device=DEFAULT_DEVICE)
        gpu_batch["rgb_gt" ] = batch.labels.rgb.reshape(out_shape3).to(device=DEFAULT_DEVICE)
        gpu_batch["sky_mask"] = batch.rays_cam_meta.get_mask_flags_all(NCoreRayFlags.SKY_SEMANTIC).reshape(out_shape1).to(device=DEFAULT_DEVICE)

    return gpu_batch

# GSplat utils
def load_gsplat_mog(file_path):
    data = PlyData.read(file_path)
    num_gsplat = len(data['vertex'])
    gsplat_pos = np.transpose(np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), dtype=np.float32))
    gsplat_dns = data['vertex']['opacity'].astype(np.float32).reshape(num_gsplat, 1)
    gsplat_rot = np.transpose(np.stack((data['vertex']['rot_0'], data['vertex']['rot_1'], data['vertex']['rot_2'], data['vertex']['rot_3']), dtype=np.float32))
    gsplat_scl = np.transpose(np.stack((data['vertex']['scale_0'], data['vertex']['scale_1'], data['vertex']['scale_2']), dtype=np.float32))
    gsplat_sph = np.transpose(np.stack((
        data['vertex']['f_dc_0'], data['vertex']['f_dc_1'], data['vertex']['f_dc_2'],
        data['vertex']['f_rest_0'],data['vertex']['f_rest_1'],data['vertex']['f_rest_2'],data['vertex']['f_rest_3'],data['vertex']['f_rest_4'],
        data['vertex']['f_rest_5'],data['vertex']['f_rest_6'],data['vertex']['f_rest_7'],data['vertex']['f_rest_8'],data['vertex']['f_rest_9'],
        data['vertex']['f_rest_10'],data['vertex']['f_rest_11'],data['vertex']['f_rest_12'],data['vertex']['f_rest_13'],data['vertex']['f_rest_14'],
        data['vertex']['f_rest_15'],data['vertex']['f_rest_16'],data['vertex']['f_rest_17'],data['vertex']['f_rest_18'],data['vertex']['f_rest_19'],
        data['vertex']['f_rest_20'],data['vertex']['f_rest_21'],data['vertex']['f_rest_22'],data['vertex']['f_rest_23'],data['vertex']['f_rest_24'],
        data['vertex']['f_rest_25'],data['vertex']['f_rest_26'],data['vertex']['f_rest_27'],data['vertex']['f_rest_28'],data['vertex']['f_rest_29'],
        data['vertex']['f_rest_30'],data['vertex']['f_rest_31'],data['vertex']['f_rest_32'],data['vertex']['f_rest_33'],data['vertex']['f_rest_34'],
        data['vertex']['f_rest_35'],data['vertex']['f_rest_36'],data['vertex']['f_rest_37'],data['vertex']['f_rest_38'],data['vertex']['f_rest_39'],
        data['vertex']['f_rest_40'],data['vertex']['f_rest_41'],data['vertex']['f_rest_42'],data['vertex']['f_rest_43'],data['vertex']['f_rest_44']), dtype=np.float32))

    return (
        torch.from_numpy(gsplat_pos).to(DEFAULT_DEVICE),
        torch.from_numpy(gsplat_rot).to(DEFAULT_DEVICE),
        torch.from_numpy(gsplat_scl).to(DEFAULT_DEVICE),
        torch.from_numpy(gsplat_dns).to(DEFAULT_DEVICE),
        torch.from_numpy(gsplat_sph).to(DEFAULT_DEVICE)
    )

def fov2focal(fov_radians: float, pixels: int):
    return pixels / (2 * math.tan(fov_radians / 2))

def focal2fov(focal: float, pixels: int):
    return 2*math.atan(pixels/(2*focal))

def pinhole_camera_rays(x, y, f_x, f_y, w, h):
    """ return 
        ray_origin (sz_y, sz_x, 3) 
        normalized ray_direction (sz_y, sz_x, 3)
    """
    
    xs = ((x + 0.5) - 0.5 * w) / f_x
    ys = ((y + 0.5) - 0.5 * h) / f_y

    ray_lookat = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    ray_origin = np.zeros_like(ray_lookat)
    
    return ray_origin, ray_lookat/np.linalg.norm(ray_lookat, axis=-1, keepdims=True)


def camera_to_world_rays(ray_o, ray_d, poses):
    """ 
    input:
        ray_o_cam [n, 3] - ray origins in the camera coordinate system
        ray_d_cam [n, 3] - ray origins in the camera coordinate system
        poses [n, 4,4] - camera to world transformation matrices

    return 
        ray_o [n, 3] - ray origins in the world coordinate system
        ray_d [n, 3] - ray directions in the world coordinate system
    """
    ray_o = np.einsum('ijk,ik->ij', poses[:,:3,:3], ray_o) + poses[:,:3,3]

    ray_d = np.einsum('ijk,ik->ij', poses[:,:3,:3], ray_d)

    return ray_o, ray_d

@dataclass(slots=True, kw_only=True)
class PointCloud:
    """Represents a 3d point cloud consisting of corresponding start and end points"""

    xyz_start: torch.Tensor  # [N,3]
    xyz_end: torch.Tensor  # [N,3]
    device: str
    dtype = torch.float32

    def __post_init__(self) -> None:
        assert len(self.xyz_start) == len(self.xyz_end)
        assert self.xyz_start.shape[1] == self.xyz_end.shape[1] == 3

        self.xyz_start.to(self.device, dtype=self.dtype)
        self.xyz_end.to(self.device, dtype=self.dtype)

    @staticmethod
    def from_sequence(point_clouds: Sequence[PointCloud], device: str) -> PointCloud:
        point_clouds_list = list(point_clouds)

        return PointCloud(xyz_start=torch.cat([pc.xyz_start for pc in point_clouds_list]),
                          xyz_end=torch.cat([pc.xyz_end for pc in point_clouds_list]),
                          device=device)
