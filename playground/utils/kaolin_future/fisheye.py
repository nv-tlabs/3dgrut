import torch
from typing import Optional
from kaolin.render.camera import Camera, generate_centered_pixel_coords


# -- Ray gen --
def _to_ndc_coords(pixel_x, pixel_y, camera):
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0
    return pixel_x, pixel_y


def generate_fisheye_rays(camera: Camera, coords_grid: Optional[torch.Tensor] = None, eps: float = 1e-9):
    r"""Default ray generation function for perfect wide-angle fisheye cameras.

    Fisheye cameras map wide angles to screen space by introducing distortion towards the edge of the image,
    e.g. straight lines get mapped to curves in the projected image.

    This raygen function uses equidistant mapping, which preserves angular distances
    e.g. if two locations are THETA angles apart in world coordinates, they will be equally spaced in the
    projected image.
    The exact mapping is characterized by the field of view.

    Note: This function does not concern non-ideal distortion parameters, such as
    radial, tangential distortion parameters.

    Args:
        camera (kaolin.render.camera.Camera): A single camera object (batch size 1).
        coords_grid (torch.FloatTensor, optional):
            Pixel grid of ray-intersecting coordinates of shape :math:`(\text{H, W, 2})`.
            Coordinates integer parts represent the pixel :math:`(\text{i, j})` coords, and the fraction part of
            :math:`[\text{0,1}]` represents the location within the pixel itself.
            For example, a coordinate of :math:`(\text{0.5, 0.5})` represents the center of the top-left pixel.
        eps (float):
            Numerical sensitivity parameter, used to prevent division by zero.

    Returns:
        (torch.FloatTensor, torch.FloatTensor, torch.BoolTensor):
            First and second entries are the generated rays for the camera, as ray origins and ray direction tensors of
            :math:`(\text{HxW, 3})`.

            The third entry is a boolean mask that masks in which rays fall within the field of view of the camera
            (rays outside the fov region shouldn't be rendered), of shape
            :math:`(\text{HxW, 1})`.
    """
    assert len(camera) == 1, "generate_fisheye_rays() supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."

        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."

    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    u, v = _to_ndc_coords(pixel_x, pixel_y, camera)
    r = torch.sqrt(u * u + v * v)
    out_of_fov_mask = (r > 1.0)[:, :, None]

    phi_cos = torch.where(torch.abs(r) > eps, u / r, 0.0)
    phi_cos = torch.clamp(phi_cos, -1.0, 1.0)
    phi = torch.arccos(phi_cos)
    phi = torch.where(v < 0, -phi, phi)
    theta = r * camera.fov(in_degrees=False) * 0.5

    rays_dir = torch.stack(
        [torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)], dim=2
    )
    mock_dir = torch.zeros_like(rays_dir)
    mock_dir[:, :, 0] = -1.0
    mock_dir[:, :, 1] = -.05
    rays_dir = torch.where(out_of_fov_mask, mock_dir, rays_dir).unsqueeze(0)

    # Generate ray origins in world coordinates
    cam_center = camera.cam_pos()
    rays_ori = (torch.tensor(cam_center, device=camera.device, dtype=torch.float32)
                .reshape(1, 1, 1, 3)
                .expand(1, camera.height, camera.width, 3))

    rays_ori = rays_ori.reshape(-1, 3)
    rays_dir = rays_dir.reshape(-1, 3)
    out_of_fov_mask = out_of_fov_mask.reshape(-1, 1)

    return rays_ori, rays_dir, out_of_fov_mask
