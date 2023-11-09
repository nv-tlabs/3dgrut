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

from tqdm import tqdm
from omegaconf import OmegaConf
    
import polyscope as ps
import polyscope.imgui as psim

from datasets.utils import PointCloud
from datasets.utils import move_to_gpu, pinhole_camera_rays
from loss_utils import ssim
from utils import to_np
from libs import optixtracer
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) 


DEFAULT_DEVICE = torch.device('cuda')

def main(conf):

    # Determinism, ish
    rand_gen = torch.Generator(device=DEFAULT_DEVICE)
    rand_gen.manual_seed(42)

    # Set up the optix context and tracer
    torch.zeros(1, device=DEFAULT_DEVICE) # Create a dummy tensor to force cuda context init
    optix_ctx = optixtracer.OptiXContext(
        params = optixtracer.OptixMogTracingParams(
            hit_mode = conf.render.hit_mode,
            max_hit_per_slab = conf.render.max_hit_per_slab,
            max_num_slabs = conf.render.max_num_slabs,
            topk_hits = conf.render.topk_hits,
            patch_size = conf.render.patch_size,
            sph_degree = conf.render.sph_degree,
            gaussian_sigma_threshold = conf.render.gaussian_sigma_threshold,
            min_transmittance = conf.render.min_transmittance,
        )
    )

    ## Manage a collection of Gaussians
    n_gaussians = 0
    max_sh_degree = 0
    sh_dim = (max_sh_degree+1) ** 2
    gauss_pos = torch.zeros((0,3), dtype=torch.float32, device=DEFAULT_DEVICE)
    gauss_rot = torch.zeros((0,4), dtype=torch.float32, device=DEFAULT_DEVICE)
    gauss_den = torch.zeros((0,1), dtype=torch.float32, device=DEFAULT_DEVICE)
    gauss_scale = torch.zeros((0,3), dtype=torch.float32, device=DEFAULT_DEVICE)
    gauss_features = torch.zeros((0,sh_dim,3), dtype=torch.float32, device=DEFAULT_DEVICE)


    def add_gaussian():
        nonlocal n_gaussians, gauss_pos, gauss_rot, gauss_scale, gauss_den, gauss_features
        n_gaussians += 1

        gauss_pos = torch.cat((gauss_pos, torch.tensor([[0.0, 0.0, 0.0],], dtype=torch.float32, device=DEFAULT_DEVICE)), dim=0)
        gauss_rot = torch.cat((gauss_rot, torch.tensor([[1.0, 0.0, 0.0, 0.0],], dtype=torch.float32, device=DEFAULT_DEVICE)), dim=0)
        gauss_den = torch.cat((gauss_den, torch.tensor([[0.5,]], dtype=torch.float32, device=DEFAULT_DEVICE)), dim=0)
        gauss_scale = torch.cat((gauss_scale, torch.tensor([[1., 1., 1.],], dtype=torch.float32, device=DEFAULT_DEVICE)), dim=0)

        new_feat = 0.05 * torch.randn((1,sh_dim,3), dtype=torch.float32, device=DEFAULT_DEVICE)
        # new_feat[0,0,:] = 0.0
        # if n_gaussians <= 3:
        #     new_feat[0,0,n_gaussians-1] = 0.5

        gauss_features = torch.cat((gauss_features, new_feat), dim=0)
        gauss_rot = torch.nn.functional.normalize(gauss_rot, dim=-1)

        optixtracer.build_mog_bvh(optix_ctx, gauss_pos, gauss_rot, gauss_scale, True)

    add_gaussian() # single initial Gaussian

    def remove_gaussian():
        nonlocal n_gaussians, gauss_pos, gauss_rot, gauss_scale, gauss_den, gauss_features

        if n_gaussians == 0: 
            return

        n_gaussians -= 1
        
        gauss_pos = gauss_pos[:-1,:]
        gauss_rot = gauss_rot[:-1,:]
        gauss_den = gauss_den[:-1]
        gauss_scale = gauss_scale[:-1,:]
        gauss_features = gauss_features[:-1,...]
        
        optixtracer.build_mog_bvh(optix_ctx, gauss_pos, gauss_rot, gauss_scale, True)

    def build_gaussian_ui():
        nonlocal n_gaussians, gauss_pos, gauss_rot, gauss_scale, gauss_den, gauss_features
        any_changed = False

        psim.Separator()

        psim.TextUnformatted(f"Gaussians: {n_gaussians}")
        psim.SameLine()
        if psim.Button("Add"):
            add_gaussian()
        psim.SameLine()
        if psim.Button("Remove"):
            remove_gaussian()
        
        gauss_pos = gauss_pos.cpu()
        gauss_rot = gauss_rot.cpu()
        gauss_den = gauss_den.cpu()
        gauss_scale = gauss_scale.cpu()
        gauss_features = gauss_features.cpu()

        for iG in range(n_gaussians):

            gstr = f"##{iG:04d}"
            psim.PushId(gstr)

            psim.SetNextItemOpen(iG == 0, psim.ImGuiCond_FirstUseEver)
            if(psim.TreeNode(f"Gaussian {iG}")):

                changed, new_vals = psim.SliderFloat3("position", to_np(gauss_pos[iG,:]), -1., 1.)
                if changed:
                    any_changed = True
                    for i in range(3): 
                        gauss_pos[iG,i] = new_vals[i]
                
                changed, new_vals = psim.SliderFloat3("rotation", to_np(gauss_rot[iG,1:]), -1., 1.)
                if changed:
                    any_changed = True
                    for i in range(3): 
                        gauss_rot[iG,i+1] = new_vals[i]
                
                changed, new_vals = psim.SliderFloat("density", to_np(gauss_den[iG,:]), 0., 1.)
                if changed:
                    any_changed = True
                    gauss_den[iG,0] = new_vals

                changed, new_vals = psim.SliderFloat3("scale", to_np(gauss_scale[iG,:]), 0., 1.)
                if changed:
                    any_changed = True
                    for i in range(3): 
                        gauss_scale[iG,i] = new_vals[i]
            
                if(psim.TreeNode("Spherical Harmonics Coefs")):

                    for j in range(sh_dim):
                        changed, new_vals = psim.SliderFloat3(f"coeff {j:02}", to_np(gauss_features[iG,j,:]), -1., 1.)
                        if changed:
                            any_changed = True
                            for i in range(3): 
                                gauss_features[iG,j,i] = new_vals[i]
                
                    psim.TreePop()
                
                psim.TreePop()

            psim.PopID()

        gauss_pos = gauss_pos.cuda()
        gauss_rot = gauss_rot.cuda()
        gauss_den = gauss_den.cuda()
        gauss_scale = gauss_scale.cuda()
        gauss_features = gauss_features.cuda()

        if any_changed:
            gauss_rot = torch.nn.functional.normalize(gauss_rot, dim=-1)
            optixtracer.build_mog_bvh(optix_ctx, gauss_pos, gauss_rot, gauss_scale, True)

    ## Set up Polyscope

    ps.set_use_prefs_file(False)
    ps.set_up_dir("y_up")
    ps.set_navigation_style("turntable")
    ps.set_enable_vsync(False)
    ps.set_max_fps(-1)
    ps.set_background_color((0., 0., 0.))
    ps.set_ground_plane_mode("none")
    ps.set_window_resizable(True)
    ps.set_window_size(1920, 1080)
    ps.set_give_focus_on_show(True)
    
    ps.set_automatically_compute_scene_extents(False)
    s = 8.
    ps.set_bounding_box((-s, -s, -s), (s, s, s))
    ps.set_length_scale(s)

    # viz stateful parameters & options
    viz_render_styles = ['color', 'density', 'hit count']
    viz_render_style_ind = 0
    viz_curr_render_size = None
    viz_curr_render_style_ind = None
    viz_render_color_buffer = None
    viz_render_scalar_buffer = None
    viz_render_name = 'render'
    
    ps.init()

    ps_point_cloud = ps.register_point_cloud("centers", to_np(gauss_pos), radius=1e-2)
    ps_point_cloud_buffer = ps_point_cloud.get_buffer("points")

        
    def update_cloud_viz():
        nonlocal ps_point_cloud, ps_point_cloud_buffer

        # re-initialize the viz
        if ps_point_cloud is None or ps_point_cloud.n_points() != gauss_pos.shape[0]:
            ps_point_cloud = ps.register_point_cloud("centers", to_np(gauss_pos))
            ps_point_cloud_buffer = ps_point_cloud.get_buffer("points")

        # direct on-GPU update, must not have changed size
        ps_point_cloud_buffer.update_data_from_device(gauss_pos.detach())

    def render_from_current_ps_view():

        window_w, window_h = ps.get_window_size()
        view_params = ps.get_view_camera_parameters()
        cam_center = view_params.get_position()
        corner_rays = view_params.generate_camera_ray_corners()
        c_ul, c_ur, c_ll, c_lr = [torch.tensor(a, device=DEFAULT_DEVICE, dtype=torch.float32) for a in corner_rays]

        # generate view camera ray origins and directions
        rays_ori = torch.tensor(cam_center, device=DEFAULT_DEVICE, dtype=torch.float32).reshape(1,1,1,3).expand(1,window_h,window_w,3)
        interp_x, interp_y= torch.meshgrid(
                        torch.linspace(0., 1., window_w, device=DEFAULT_DEVICE, dtype=torch.float32),
                        torch.linspace(0., 1., window_h, device=DEFAULT_DEVICE, dtype=torch.float32),
                        indexing='xy')
        interp_x = interp_x.unsqueeze(-1)
        interp_y = interp_y.unsqueeze(-1)
        rays_dir = c_ul + interp_x * (c_ur - c_ul) + interp_y * (c_ll - c_ul)
        rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
        rays_dir = rays_dir.unsqueeze(0)
            
        gauss_features_flat = gauss_features.reshape(-1,3).contiguous()

        with torch.no_grad():

            pred_rgb, pred_opacity, pred_ohit = optixtracer.trace_mog(optix_ctx, 
                    rays_ori, rays_dir,
                    gauss_pos, gauss_rot, gauss_scale, gauss_den, gauss_features_flat)

        return pred_rgb, pred_opacity, pred_ohit

    def update_render_view_viz():
        nonlocal viz_curr_render_size, viz_curr_render_style_ind, viz_render_color_buffer, viz_render_scalar_buffer

        window_w, window_h = ps.get_window_size()

        # re-initialize if needed
        style = viz_render_styles[viz_render_style_ind]
        if viz_curr_render_style_ind != viz_render_style_ind or viz_curr_render_size != (window_w, window_h):
       
            viz_curr_render_style_ind = viz_render_style_ind
            viz_curr_render_size = (window_w, window_h)

            if style == "color":

                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)

                ps.add_color_alpha_image_quantity(
                    viz_render_name,
                    dummy_image,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                )

                viz_render_color_buffer = ps.get_quantity_buffer(viz_render_name, "colors")
                viz_render_scalar_buffer = None
            
            elif style == "density":
            
                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                ps.add_scalar_image_quantity(
                    viz_render_name,
                    dummy_vals,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="blues",
                    vminmax=(0, 1),
                )

                viz_render_color_buffer = None
                viz_render_scalar_buffer = ps.get_quantity_buffer(viz_render_name, "values")
            
            elif style == "hit count":
            
                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 3.0  # hack so the default polyscope scale gets set more nicely

                ps.add_scalar_image_quantity(
                    viz_render_name,
                    dummy_vals,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="blues",
                    vminmax=(0, 3),
                )

                viz_render_color_buffer = None
                viz_render_scalar_buffer = ps.get_quantity_buffer(viz_render_name, "values")


        # do the actual rendering
        sple_orad, sple_odns, sple_ohit = render_from_current_ps_view()

        # print(f"rad max: {sple_orad.max()} dens max: {sple_odns.max()}")

        # update the data
        if style == "color":
            # append 1s for alpha
            sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:,:,:,0:1])), dim=-1)
            # sple_orad = torch.cat((sple_orad, sple_orad[...,0:1]), dim=-1)
            viz_render_color_buffer.update_data_from_device(sple_orad.detach())

        elif style == "density":
            viz_render_scalar_buffer.update_data_from_device(sple_odns.detach())
        
        elif style == "hit count":
            viz_render_scalar_buffer.update_data_from_device(sple_ohit.detach())


    def ps_ui_callback():
        nonlocal viz_render_style_ind

        # Create a little ImGUI UI
        _, viz_render_style_ind = psim.Combo("Render Display", viz_render_style_ind, viz_render_styles)

        update_cloud_viz()
        update_render_view_viz()

        build_gaussian_ui()


    ps.set_user_callback(ps_ui_callback)

    ps.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args, remainder = parser.parse_known_args()

    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(remainder)
    conf = OmegaConf.merge(base_conf, cli_conf)

    main(conf)
