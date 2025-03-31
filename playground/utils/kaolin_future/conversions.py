import torch
import numpy as np
import polyscope as ps
from kaolin.render.camera import Camera


def polyscope_to_kaolin_camera(ps_camera, width, height, near=1e-2, far=1e2, device='cpu'):
    view_matrix = ps_camera.get_view_mat()
    fov_y = ps_camera.get_fov_vertical_deg() * np.pi / 180.0    # to radians

    return Camera.from_args(
        view_matrix=view_matrix,
        fov=fov_y,
        width=width, height=height,
        near=near, far=far,
        dtype=torch.float64,
        device=device
    )


def polyscope_from_kaolin_camera(camera: Camera):
    view_matrix = camera.view_matrix()
    ps_cam_param = ps.CameraParameters(
        ps.CameraIntrinsics(fov_vertical_deg=camera.fov_y.detach().cpu().numpy(), aspect=camera.width / camera.height),
        ps.CameraExtrinsics(mat=view_matrix[0].detach().cpu().numpy())
    )
    return ps_cam_param
