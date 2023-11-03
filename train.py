import os
import sys

from regex import D
import torch 
import argparse
import matplotlib.pyplot as plt
import torch.utils.data

from torchmetrics import PeakSignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm
from datasets.colmap_dataset import ColmapDataset
from datasets.utils import load_gsplat_mog, move_to_gpu
from loss_utils import ssim

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) 
from libs import optixtracer


DEFAULT_DEVICE = torch.device('cuda')

def OptiXContext():
    print("Cuda path", torch.utils.cpp_extension.CUDA_HOME)
    torch.zeros(1, device='cuda') # Create a dummy tensor to force cuda context init
    return optixtracer.OptiXContext()

def show_image(imgs, figAxes=None):
    """Utility to show a list of image tensors"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    newFigure = not figAxes
    if newFigure:
        _, axes = plt.subplots(ncols=len(imgs), squeeze=False)
        figAxes = []
    for i, img in enumerate(imgs):
        if newFigure:
            figAxes.append(axes[0, i].imshow(img))
            axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        else:
            figAxes[i].set_data(img)
    return figAxes    

# Returns the result of running `fn()` and the time it took for `fn()` to run in milliseconds.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def load_splats_and_build_bvh(ply_path, optix_ctx, maxNumGSplat=15000000):

    # Load the pretrained point cloud from the Gaussian splatting
    sple_pos, sple_rot, sple_scl, sple_dns, sple_sph = load_gsplat_mog(ply_path)

    if maxNumGSplat<sple_pos.shape[0]:
        splt_rd_id = torch.randperm(sple_pos.shape[0])[0:maxNumGSplat]
        sple_pos = sple_pos[splt_rd_id]
        sple_rot = sple_rot[splt_rd_id]
        sple_scl = sple_scl[splt_rd_id]
        sple_dns = sple_dns[splt_rd_id]
        sple_sph = sple_sph[splt_rd_id]

    _, sple_build_ms = timed(lambda: optixtracer.build_mog_bvh(optix_ctx, sple_pos, torch.nn.functional.normalize(sple_rot, dim=1), torch.exp(sple_scl), 3, True))

    print(f"==========> build_mog_bvh = {sple_build_ms} ms.")

    return sple_pos, sple_rot, sple_scl, sple_dns, sple_sph

def main(args):
    # Run the training process
    gsplat_plyfile = os.path.join(args.path, "point_cloud.ply")
    n_iterations = 10e4
    val_period = 1
    use_ssim = False

    train_dataset = ColmapDataset(args.path, split='train', sample_full_image=True)
    val_dataset = ColmapDataset(args.path, split='val', sample_full_image=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False)

    # Build the BVH 
    optix_ctx = OptiXContext()
    sple_pos, sple_rot, sple_scl, sple_dns, sple_sph = load_splats_and_build_bvh(gsplat_plyfile, optix_ctx, maxNumGSplat=15000000)

    # Set all the variables that we want to optimize
    sple_dns.requires_grad = True
    sple_sph.requires_grad = True
    optimizable_parameters = [sple_dns, sple_sph]
    if args.optimize_position:
        sple_pos.requires_grad = True
        optimizable_parameters.append(sple_pos)

    if args.optimize_rotation:
        sple_rot.requires_grad = True
        optimizable_parameters.append(sple_rot)

    if args.optimize_scale:
        sple_scl.requires_grad = True
        optimizable_parameters.append(sple_scl)

    if args.with_gui:

        import polyscope as ps
        import polyscope.imgui as psim

        ps.set_use_prefs_file(False)
        ps.set_up_dir("y_up")
        ps.set_navigation_style("free")
        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_background_color((0., 0., 0.))
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(1920, 1080)
        ps.set_give_focus_on_show(True)
        
        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box((0, 0, 0), (1, 1, 1))

        # viz stateful parameters & options
        viz_do_train = True
        viz_render_styles = ['color', 'density']
        viz_render_style_ind = 0
        viz_curr_render_size = None
        viz_curr_render_style_ind = None
        viz_render_color_buffer = None
        viz_render_scalar_buffer = None
        viz_render_name = 'render'
        
        ps.init()

        ps_point_cloud = ps.register_point_cloud("centers", sple_pos.detach().cpu().numpy(), 
                                radius=1e-3, point_render_mode='quad')
        ps_point_cloud_buffer = ps_point_cloud.get_buffer("points")

        
        def update_cloud_viz():
            # direct on-GPU update, must not have changed size
            ps_point_cloud_buffer.update_data_from_device(sple_pos.detach())

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

            # Render a frame
            sple_orad, sple_odns, sple_ohit = optixtracer.trace_mog(optix_ctx, rays_ori, rays_dir, sple_pos, 
            torch.nn.functional.normalize(sple_rot, dim=1), torch.exp(sple_scl), torch.sigmoid(sple_dns), sple_sph)

            return sple_orad, sple_odns, sple_ohit

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


            # do the actual rendering
            sple_orad, sple_odns, sple_ohit = render_from_current_ps_view()

            # update the data
            if style == "color":
                # append 1s for alpha
                sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:,:,:,0:1])), dim=-1)
                viz_render_color_buffer.update_data_from_device(sple_orad.detach())

            elif style == "density":
                # append 1s for alpha
                viz_render_scalar_buffer.update_data_from_device(sple_odns.detach())

     
        # test evaluations
        update_cloud_viz()
        render_from_current_ps_view()
        update_render_view_viz()

        def ps_ui_callback():
            nonlocal viz_do_train, viz_render_style_ind

            # Create a little ImGUI UI
            _, viz_do_train = psim.Checkbox("Train", viz_do_train)

            _, viz_render_style_ind = psim.Combo("Render Display", viz_render_style_ind, viz_render_styles)

            update_render_view_viz()

        ps.set_user_callback(ps_ui_callback)
       


    # Initialize the optimizer and pass it the parameters
    optimizer = torch.optim.Adam(optimizable_parameters, lr=0.001)

    n_epochs = int(n_iterations/train_dataset.__len__())
    iteration = 0

    # Criterions that we log during training
    criterions = {"psnr":  PeakSignalNoiseRatio(data_range=1).to("cuda")}

    # BVH update frequncy only important if optimizing one of the parameters that changes the position of the Gaussians
    bvh_update_frequency = args.bvh_update_frequency if any([sple_rot.requires_grad, sple_scl.requires_grad, sple_pos.requires_grad]) else -1
    print(bvh_update_frequency)
    writer = SummaryWriter()
    for i in range(n_epochs):
        if i % val_period == 0:
            val_iteration = 0
            with tqdm(val_dataloader) as pbar:
                pbar.set_description("Validation:" )
                for batch in pbar:
                    with torch.no_grad():
                        rays_ori, rays_dir, rgb_gt = move_to_gpu(batch)
                        rays_ori = rays_ori.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                        rays_dir = rays_dir.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                        rgb_gt = rgb_gt.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)

                        # Render a frame
                        sple_orad, sple_odns, sple_ohit = optixtracer.trace_mog(optix_ctx, rays_ori, rays_dir, sple_pos, 
                        torch.nn.functional.normalize(sple_rot, dim=1), torch.exp(sple_scl), torch.sigmoid(sple_dns), sple_sph)

                        # Compute the loss
                        loss = torch.abs(sple_orad - rgb_gt).mean()
                        psnr = criterions["psnr"](sple_orad, rgb_gt).item()
                        pbar.set_postfix({'iteration': iteration, 'psnr': psnr, 'loss': loss.item()})

                        writer.add_scalar("psnr/val", psnr, i)
                        writer.add_scalar("loss_l1/val", loss.item(), i)

                        if val_iteration == 0:
                            writer.add_image('image/val', sple_orad[-1].clip(0,1.0), iteration, dataformats='HWC')
                            writer.add_image('image/gt', rgb_gt[-1].clip(0,1.0), iteration, dataformats='HWC')
                        val_iteration += 1


        with tqdm(train_dataloader) as pbar:
            for batch in pbar:
                # Move data to GPU
                rays_ori, rays_dir, rgb_gt = move_to_gpu(batch)
                rays_ori = rays_ori.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                rays_dir = rays_dir.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                rgb_gt = rgb_gt.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)

                # Render a frame
                sple_orad, sple_odns, sple_ohit = optixtracer.trace_mog(optix_ctx, rays_ori, rays_dir, sple_pos, 
                        torch.nn.functional.normalize(sple_rot, dim=1), torch.exp(sple_scl), torch.sigmoid(sple_dns), sple_sph)
                
                # Compute the loss
                loss = torch.abs(sple_orad - rgb_gt).mean()
                writer.add_scalar("Loss_l1/train", loss.item(), iteration)
                if use_ssim:
                    loss_ssim = ssim(torch.permute(sple_orad, (0, 3, 1, 2)), torch.permute(rgb_gt, (0, 3, 1, 2)))
                    loss += loss_ssim
                    writer.add_scalar("Loss_ssim/train", loss_ssim.item(), iteration)

                # backpropagate the gradients and update the parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                psnr = criterions["psnr"](sple_orad, rgb_gt).item()
                iteration += 1
                pbar.set_postfix({'iteration': iteration, 'psnr': psnr, 'loss': loss.item()})
                writer.add_scalar("psnr/train", psnr, iteration)

                # Update the BVH if required
                if iteration > 0 and bvh_update_frequency > 0 and iteration % bvh_update_frequency == 0:
                    print(f"==========> rebuilt the BVH <==========")
                    optixtracer.build_mog_bvh(optix_ctx, sple_pos, torch.nn.functional.normalize(sple_rot, dim=1), torch.exp(sple_scl), 3, True)

                if args.with_gui:

                    if sple_pos.requires_grad:
                        update_cloud_viz()
        
                    update_render_view_viz()

                    ps.frame_tick()
                    while not viz_do_train:
                        ps.frame_tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Update frequency of the BVH structure in iterations - will only be performed if position, scale, rotation are being optimized")
    parser.add_argument("--bvh-update-frequency", type=int, default=50, help="Update frequency of the BVH structure in iterations - will only be performed if position, scale, rotation are being optimized")
    parser.add_argument("--optimize-rotation", action='store_true', help="Set true if the rotation should be optimized as well")
    parser.add_argument("--optimize-scale", action='store_true', help="Set true if the scale should be optimized as well")
    parser.add_argument("--optimize-position", action='store_true', help="Set true if the position should be optimized as well")
    parser.add_argument("--with-gui", action='store_true', help="Enable a polyscope GUI")
    args = parser.parse_args()

    main(args)
