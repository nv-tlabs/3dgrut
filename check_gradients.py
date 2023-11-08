import os
import sys
import logging
import torch 
import argparse
import logging
import torch.utils.data

from torchmetrics import PeakSignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps

from tqdm import tqdm
from omegaconf import OmegaConf

from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset
from model import MixtureOfGaussians
from datasets.utils import move_to_gpu, pinhole_camera_rays
from utils import to_np
from libs import optixtracer
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) 


DEFAULT_DEVICE = torch.device('cuda')

def main(conf):
    
    device = 'cuda'

    # Determinism, ish
    rand_gen = torch.Generator(device=device)
    rand_gen.manual_seed(42)

    # Set up the optix context and tracer
    torch.zeros(1, device=device) # Create a dummy tensor to force cuda context init
    optix_ctx = optixtracer.OptiXContext()

    # Use a GUI to check the data
    if conf.with_gui:
        # TODO
        pass

    def render_image(gauss_pos, gauss_rot, gauss_den, gauss_scale, gauss_features, ray_srcs, ray_dirs, return_all=False):

        optixtracer.build_mog_bvh(optix_ctx, gauss_pos, gauss_rot, gauss_scale, True)

        pred_rgb, pred_opacity, pred_ohit = optixtracer.trace_mog(optix_ctx, 
                ray_srcs, ray_dirs,
                gauss_pos, gauss_rot, gauss_den, gauss_scale, gauss_features)

        if return_all:
            return pred_rgb, pred_opacity, pred_ohit
        else:
            return pred_rgb


    n_guassians = 20
    max_sh_degree = 3

    if n_guassians == 3:
        ## Sample data
        gauss_pos = torch.tensor([
                [0.10, 0.10, 0.10],
                [0.11, 0.11, 0.11],
                [0.12, 0.12, 0.12],
            ], dtype=torch.float32, device=device)
        
        gauss_rot = torch.tensor([
                [1.0,  0.5, -0.5,  0.6],
                [1.0,  0.3, -0.5,  0.6],
                [1.0, -0.4, -0.5,  0.1],
            ], dtype=torch.float32, device=device)
        
        gauss_den = torch.tensor([
                [0.4, ],
                [0.5, ],
                [0.8, ],
            ], dtype=torch.float32, device=device)
        
        gauss_scale = torch.tensor([
                [1.2,  ],
                [1.8,  ],
                [1.25, ],
            ], dtype=torch.float32, device=device)
    else:
        gauss_pos = torch.rand((n_guassians,3), dtype=torch.float32, device=device)
        gauss_rot = torch.nn.functional.normalize(torch.rand((n_guassians,4), dtype=torch.float32, device=device), dim=-1)
        gauss_den = torch.rand((n_guassians,1), dtype=torch.float32, device=device)
        gauss_scale = torch.rand((n_guassians,3), dtype=torch.float32, device=device)


    gauss_features = 0.05 * torch.randn(n_guassians, 3, (max_sh_degree+1) ** 2, generator=rand_gen, dtype=torch.float32, device=device).reshape(3, -1)



    ## Generate rays
    image_w, image_h = 50, 30 
    i,j = 15, 26                # specify a single interesting ray
    f_x, f_y = 20., 20.
    # image_w, image_h = 5, 3           # other useful image sizes
    # f_x, f_y = 2., 2.
    # image_w, image_h = 1920, 1080
    # f_x, f_y = 1000., 1000.
    cam_center = np.array([0.05, 0.03, -8.]) # so the default camera direction along +z sees stuff near the origin
    u = np.tile(np.arange(image_w), image_h)
    v = np.arange(image_h).repeat(image_w)
    out_shape = (1,image_h,image_w,3)
    ray_srcs, ray_dirs = pinhole_camera_rays(u, v, f_x, f_y, image_w, image_h)
    ray_srcs = ray_srcs + cam_center[None,:]

    ray_srcs = torch.tensor(ray_srcs, dtype=torch.float32, device=device).reshape(out_shape)
    ray_dirs = torch.tensor(ray_dirs, dtype=torch.float32, device=device).reshape(out_shape)

    # Uncomment these to visualize the points
    # ps.init()
    # ps.register_point_cloud("centers", to_np(gauss_pos).reshape(-1,3))
    # ps.register_point_cloud("ray_src", to_np(ray_srcs).reshape(-1,3))
    # ps.register_point_cloud("ray_dir", to_np(ray_srcs + ray_dirs).reshape(-1,3))
    # ps.show()

    # Uncomment for a test-render to make sure you are rendering something reaonable
    # img_rgb = render_image(gauss_pos, gauss_rot, gauss_den, gauss_scale, gauss_features, ray_srcs, ray_dirs)
    # plt.imshow(to_np(img_rgb[0,...]))
    # plt.show()
    
    gradcheck_args = {
        'eps' : 1e-4,
        'atol' : 1e-3,
        'rtol' : 1e-3,
        'nondet_tol' : 1e-5,
    }


    # take just a single ray of interest
    ray_srcs = ray_srcs[:,i,j,:]
    ray_dirs = ray_dirs[:,i,j,:]
    ray_dirs = ray_dirs[:,None,None,:]
    ray_srcs = ray_srcs[:,None,None,:]

    test_val, opacity, hits = render_image(gauss_pos, gauss_rot, gauss_den, gauss_scale, gauss_features, ray_srcs, ray_dirs, return_all=True)
    print(f"Test ray contrib = {test_val}")
    print(f"Test number of hits = {hits.item()}")

    if(torch.max(torch.abs(test_val)) == 0.):
        raise ValueError("no contribution along test ray!")

    # Toggle these to test various grads
    gauss_pos.requires_grad = True
    gauss_rot.requires_grad = True
    gauss_den.requires_grad = True
    gauss_scale.requires_grad = True
    gauss_features.requires_grad = True

    inputs = (gauss_pos, gauss_rot, gauss_den, gauss_scale, gauss_features, ray_srcs, ray_dirs)
    torch.autograd.gradcheck(render_image, inputs, **gradcheck_args)


    print("\n\n ===== THE GRADIENT SQUID IS PROUD OF YOU 🦑🦑🦑🦑🦑 YOUR GRADIENT CHECK HAS PASSED =============== \n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args, remainder = parser.parse_known_args()

    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(remainder)
    conf = OmegaConf.merge(base_conf, cli_conf)

    main(conf)
