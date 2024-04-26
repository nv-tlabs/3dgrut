from kornia import create_meshgrid
import cv2
from einops import rearrange
import imageio
import torch
import numpy as np

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', ray_jitter=None, return_uv=False, flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        ray_jitter: Optional RayJitter component, for whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if ray_jitter is None: # pass by the center
        directions = torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
    else:
        jitter = ray_jitter(u.shape)
        directions = torch.stack([((u + jitter[:,:,0]) - cx) / fx, ((v + jitter[:,:,1]) - cy) / fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    
    return torch.nn.functional.normalize(directions, dim=-1)

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    if c2w.ndim==2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ \
                 rearrange(c2w[..., :3], 'n a b -> n b a')
        rays_d = rearrange(rays_d, 'n 1 c -> n c')
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d)

    return rays_o, rays_d


def read_image(img_path, img_wh, return_alpha=False, bg_color=None):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        if return_alpha:
            alpha= img[:,:,-1]

        if bg_color is None:
            img = img[..., :3]
        elif bg_color == "black":
            img = img[..., :3] * img[..., -1:]
        elif bg_color == "white":
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            assert False, f"{bg_color} is not a supported background color."

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    if return_alpha:
        alpha = rearrange(alpha, 'h w -> (h w)')
        return img, alpha
    else:
        return img


def create_camera_visualization(cam_list):
    '''
    Given a list-of-dicts of camera & image info, register them in polyscope
    to create a visualization
    '''

    import polyscope as ps

    for i_cam, cam in enumerate(cam_list):
            
        ps_cam_param = ps.CameraParameters(
                    ps.CameraIntrinsics(
                        fov_vertical_deg=np.degrees(cam['fov_h']), 
                        fov_horizontal_deg=np.degrees(cam['fov_w'])
                    ),
                    ps.CameraExtrinsics(mat=cam['ext_mat'])
                )

        cam_color = (1., 1., 1.)
        if cam['split'] == 'train':
            cam_color = (1., .7, .7)
        elif cam['split'] == 'val':
            cam_color = (.7, .1, .7)

        
        ps_cam = ps.register_camera_view(f"{cam['split']}_view_{i_cam:03d}", ps_cam_param, widget_color=cam_color)

        ps_cam.add_color_image_quantity("target image", cam['rgb_img'][:,:,:3], enabled=True)
