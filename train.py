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
from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset
from model import MixtureOfGaussians
from datasets.utils import move_to_gpu
from loss_utils import ssim
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) 


DEFAULT_DEVICE = torch.device('cuda')

logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

def main(args):
    # Run the training process
    n_iterations = 10e4
    val_period = 1
    use_ssim = False

    if args.data_type == 'nerf':
        train_dataset = NeRFDataset(args.path, split='train', sample_full_image=True)
        val_dataset = NeRFDataset(args.path, split='val', sample_full_image=True)
    elif args.data_type == 'colmap':
        train_dataset = ColmapDataset(args.path, split='train', sample_full_image=True)
        val_dataset = ColmapDataset(args.path, split='val', sample_full_image=True)
    else:
        raise ValueError(f'Unsupported dataset type: {args.data_type}. Choose between: ["colmap", "nerf"]. ')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False)

    # Initialize the model and the optix context
    model = MixtureOfGaussians(args)
    model.set_optix_context()

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

        ps_point_cloud = ps.register_point_cloud("centers", model.get_position.detach().cpu().numpy(), 
                                radius=1e-3, point_render_mode='quad')
        ps_point_cloud_buffer = ps_point_cloud.get_buffer("points")

        
        def update_cloud_viz():
            # direct on-GPU update, must not have changed size
            ps_point_cloud_buffer.update_data_from_device(model.get_position.detach())

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
            with torch.no_grad():
                outputs = model(rays_ori, rays_dir)

            return outputs['pred_rgb'], outputs['pred_opacity'], outputs['pred_ohit']

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
       

    try:
        ply_path = os.path.join(args.path, "point_cloud.ply")
        model.load_from_pretrained_point_cloud(ply_path)
    except FileNotFoundError as e:
        # Since this data set has no colmap data, we start with random points
        logging.info(f"PLY point cloud not found under path: {ply_path}")
        model.randomize_point_cloud()

    if args.resume:
        logging.info(f"Loading a pretrained checkpoint from {args.resume}!")
        checkpoint = torch.load(args.resume)
        model.init_from_checkpoint(checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['global_step']

    else:
        ply_path = None
        try:
            model.init_from_pretrained_point_cloud(os.path.join(args.path, "point_cloud.ply"))
        except FileNotFoundError as e:
            # Since this data set has no colmap data, we start with random points
            logging.info(f"PLY point cloud not found under path: {ply_path}")
            model.randomize_point_cloud()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        global_step = 0

    n_epochs = int(n_iterations/train_dataset.__len__())


    # Criterions that we log during training
    criterions = {"psnr":  PeakSignalNoiseRatio(data_range=1).to("cuda")}

    # Initialize the tensorboard writer
    if args.experiment_name and os.path.exists(f'runs/{args.experiment_name}'):
        logging.warning("The selected experiment name already exists and the checkpoints could be overwritten!")

    writer = SummaryWriter(log_dir=f'runs/{args.experiment_name}' if args.experiment_name else None)


    for epoch_idx in range(n_epochs):
        if epoch_idx % val_period == 0:
            val_iteration = 0
            with tqdm(val_dataloader) as pbar:
                pbar.set_description("Validation:" )
                val_psnr = []
                val_loss = []
                for batch in pbar:
                    with torch.no_grad():
                        rays_ori, rays_dir, rgb_gt = move_to_gpu(batch)
                        rays_ori = rays_ori.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                        rays_dir = rays_dir.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                        rgb_gt = rgb_gt.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)

                        # Compute the outputs of a single batch
                        outputs = model(rays_ori, rays_dir)

                        # Compute the loss
                        val_loss.append(torch.abs(outputs['pred_rgb'] - rgb_gt).mean().item())
                        val_psnr.append(criterions["psnr"](outputs['pred_rgb'], rgb_gt).item())

                        pbar.set_postfix({'iteration': val_iteration, 'psnr': val_psnr[-1], 'loss': val_loss[-1]})

                        if val_iteration == 0:
                            writer.add_image('image/val', outputs['pred_rgb'][-1].clip(0,1.0), global_step, dataformats='HWC')
                            writer.add_image('image/gt', rgb_gt[-1].clip(0,1.0), global_step, dataformats='HWC')
                        val_iteration += 1

                writer.add_scalar("psnr/val", np.mean(val_psnr), global_step)
                writer.add_scalar("loss_l1/val", np.mean(val_loss), global_step)

        with tqdm(train_dataloader) as pbar:
            for batch in pbar:
                # Move data to GPU
                rays_ori, rays_dir, rgb_gt = move_to_gpu(batch)
                rays_ori = rays_ori.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                rays_dir = rays_dir.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)
                rgb_gt = rgb_gt.reshape(rays_ori.shape[0], train_dataset.image_h, train_dataset.image_w, 3)

                # Compute the outputs of a single batch
                outputs = model(rays_ori, rays_dir)
                
                # Compute the loss
                loss = torch.abs(outputs['pred_rgb'] - rgb_gt).mean()
                writer.add_scalar("Loss_l1/train", loss.item(), global_step)
                if use_ssim:
                    loss_ssim = ssim(torch.permute(outputs['pred_rgb'], (0, 3, 1, 2)), torch.permute(rgb_gt, (0, 3, 1, 2)))
                    loss += loss_ssim
                    writer.add_scalar("Loss_ssim/train", loss_ssim.item(), global_step)

                # backpropagate the gradients and update the parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                psnr = criterions["psnr"](outputs['pred_rgb'], rgb_gt).item()
                global_step += 1
                pbar.set_postfix({'iteration': global_step, 'psnr': psnr, 'loss': loss.item()})
                writer.add_scalar("psnr/train", psnr, global_step)

                # Update the BVH if required
                if global_step > 0 and args.bvh_update_frequency > 0 and global_step % args.bvh_update_frequency:
                    model.build_bvh()

                # Update the BVH if required
                if global_step > 0 and global_step % args.checkpoint_frequency == 0:
                    parameters = model.get_model_parameters()
                    parameters |= {'optimizer': optimizer.state_dict(), "global_step": global_step, "epoch": epoch_idx}
                    torch.save(parameters, os.path.join(writer.get_logdir(), f"ckpt_{global_step}.pt"))

                if args.with_gui:
                    if model.get_position.requires_grad:
                        update_cloud_viz()
        
                    update_render_view_viz()

                    ps.frame_tick()
                    while not viz_do_train:
                        ps.frame_tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Update frequency of the BVH structure in iterations - will only be performed if position, scale, rotation are being optimized")
    parser.add_argument("--data-type", type=str, default='colmap', help="Type of dataset class to load and parse content in --path. One of [colmap, nerf]")
    parser.add_argument("--experiment-name", type=str, default="", help="If provided, the tensorboard logs and checkpoints will be stored in this folder name under ./runs/. If not, a name will be a defined automatically")
    parser.add_argument("--resume", type=str, default="", help="If the checkpoint path is provided, it will be used to initialize the model and continue training.")
    parser.add_argument("--bvh-update-frequency", type=int, default=50, help="Update frequency of the BVH structure in iterations - will only be performed if position, scale, rotation are being optimized")
    parser.add_argument("--density-activation", type=str, default='sigmoid', help="The name of the activation function that will be used for the density. One of [exp, sigmoid, normalize]")
    parser.add_argument("--scale-activation", type=str, default='exp', help="The name of the activation function that will be used for the scale. One of [exp, sigmoid, normalize]")
    parser.add_argument("--optimize-density", action='store_true', help="Set false if the density should be optimized as well")
    parser.add_argument("--optimize-features", action='store_true', help="Set true if the features should be optimized as well")
    parser.add_argument("--optimize-rotation", action='store_true', help="Set true if the rotation should be optimized as well")
    parser.add_argument("--optimize-scale", action='store_true', help="Set true if the scale should be optimized as well")
    parser.add_argument("--optimize-position", action='store_true', help="Set true if the position should be optimized as well")
    parser.add_argument("--checkpoint-frequency", type=int, default=1000, help="Specifies the period in which the checkpoints will be stored in terms of number of iterations")
    parser.add_argument("--with-gui", action='store_true', help="Enable a polyscope GUI")
    args = parser.parse_args()

    main(args)
