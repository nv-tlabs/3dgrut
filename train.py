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

from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset
from datasets.ngp_dataset import NGPDataset
from datasets.ncore_dataset import NCoreDataset
from datasets.ncore_utils import Batch as NCoreBatch
from datasets.utils import PointCloud
from models.model import MixtureOfGaussians
from models.background import BackgroundColor
from datasets.utils import move_to_gpu
from loss_utils import ssim
from utils import to_np
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) 


DEFAULT_DEVICE = torch.device('cuda')

logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

def main(conf):
    # Run the training process
    n_iterations = 10e4
    val_frequency = conf.val_frequency
    use_ssim = False
    scene_extent = 1.
    train_collate_fn = None
    val_collate_fn = None
    if conf.dataset.type == 'nerf':
        train_dataset = NeRFDataset(
            conf.path, 
            split='train', 
            sample_full_image=conf.dataset.train.sample_full_image, 
            batch_size=conf.dataset.train.batch_size,
            return_alphas=True
        )
        val_dataset = NeRFDataset(
            conf.path, 
            split='val', 
            sample_full_image=True,
            return_alphas=False
        )
    elif conf.dataset.type == 'colmap':
        train_dataset = ColmapDataset(
            conf.path, 
            split='train', 
            sample_full_image=conf.dataset.train.sample_full_image, 
            batch_size=conf.dataset.train.batch_size,
            downsample_factor=conf.dataset.downsample_factor
        )
        val_dataset = ColmapDataset(
            conf.path,
            split='val',
            sample_full_image=True,
            downsample_factor=conf.dataset.downsample_factor
        )
        val_dataset = ColmapDataset(conf.path, split='val', sample_full_image=True)
    elif conf.dataset.type == 'ngp':
        train_dataset = NGPDataset(
            conf.path, 
            split='train', 
            sample_full_image=conf.dataset.train.sample_full_image, 
            batch_size=conf.dataset.train.batch_size,
            use_lidar=True,
            use_aux=conf.dataset.get("use_aux_data", False)
        )
        val_dataset = NGPDataset(
            conf.path, 
            split='val', 
            sample_full_image=True, 
            val_downsample=5, 
            val_frame_subsample=5, 
            use_aux=conf.dataset.get("use_aux_data", False)
        )
        pc = train_dataset.get_point_cloud(step_frame=10)
        scene_extent = ((pc.xyz_end.max(0).values - pc.xyz_end.min(0).values)**2).sum().sqrt()
    elif conf.dataset.type == 'ncore':
        # TODO: add all of the dataset parameters to config
        duration_sec = 2.0
        train_dataset = NCoreDataset(
            conf.path, 
            split='train', 
            duration_sec=duration_sec,
        )
        val_dataset = NCoreDataset(
            conf.path,
            split='val',
            duration_sec=duration_sec,
        )
        pc = PointCloud.from_sequence(train_dataset.get_point_clouds(step_frame=10, non_dynamic_points_only=True), device="cpu")
        scene_extent = train_dataset.scene_extent_m
       
        train_collate_fn = NCoreBatch.collate_fn
        val_collate_fn = NCoreBatch.collate_fn
    else:
        raise ValueError(f'Unsupported dataset type: {conf.dataset.type}. Choose between: ["colmap", "nerf", "ngp", "ncore"]. ')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=conf.num_workers, batch_size=1, shuffle=True, collate_fn=train_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=conf.num_workers, batch_size=1, shuffle=False, collate_fn=val_collate_fn)

    # Initialize the model and the optix context
    model = MixtureOfGaussians(conf)
    model.set_optix_context()

    if conf.resume: # Load a checkpoint
        logging.info(f"Loading a pretrained checkpoint from {conf.resume}!")
        checkpoint = torch.load(conf.resume)
        model.init_from_checkpoint(checkpoint)
        global_step = checkpoint['global_step']

    else: # Initialize

        if conf.initialization.method == 'colmap':
            model.init_from_colmap(conf.path)

        elif conf.initialization.method == 'point_cloud':
            ply_path = None
            try:
                ply_path = os.path.join(conf.path, "point_cloud.ply")
                model.init_from_pretrained_point_cloud(ply_path)
            except FileNotFoundError as e:
                logging.exception(e)
                raise e
         
        elif conf.initialization.method == 'random':
            model.init_from_random_point_cloud(num_gsplat=conf.initialization.num_gaussians)

        elif conf.initialization.method == 'lidar':
            assert conf.dataset.type in ['ngp', 'ncore'], 'can only initialize from lidar with the NGPDataset / NCoreDataset'
            model.init_from_lidar(point_cloud = pc) 
        else:
           raise ValueError(f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random]")

        model.build_bvh()
        model.setup_optimizer()
        global_step = 0

    if conf.with_gui:
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
        scene_bboxmin, scene_bboxmax = train_dataset.get_bbox()
        ps.set_bounding_box(to_np(scene_bboxmin), to_np(scene_bboxmax))

        # viz stateful parameters & options
        viz_do_train = False
        viz_render_styles = ['color', 'density']
        viz_render_style_ind = 0
        viz_curr_render_size = None
        viz_curr_render_style_ind = None
        viz_render_color_buffer = None
        viz_render_scalar_buffer = None
        viz_render_name = 'render'
        
        ps.init()

        ps_point_cloud = ps.register_point_cloud("centers", to_np(model.get_positions()), 
                                radius=1e-3, point_render_mode='quad')
        ps_point_cloud_buffer = ps_point_cloud.get_buffer("points")

        
        def update_cloud_viz():
            nonlocal ps_point_cloud, ps_point_cloud_buffer

            # re-initialize the viz
            if ps_point_cloud is None or ps_point_cloud.n_points() != model.get_positions().shape[0]:
                ps_point_cloud = ps.register_point_cloud("centers", to_np(model.get_positions()))
                ps_point_cloud_buffer = ps_point_cloud.get_buffer("points")

            # direct on-GPU update, must not have changed size
            ps_point_cloud_buffer.update_data_from_device(model.get_positions().detach())

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
                viz_render_scalar_buffer.update_data_from_device(sple_odns.detach())


        def ps_ui_callback():
            nonlocal viz_do_train, viz_render_style_ind

            # Create a little ImGUI UI
            _, viz_do_train = psim.Checkbox("Train", viz_do_train)

            _, viz_render_style_ind = psim.Combo("Render Display", viz_render_style_ind, viz_render_styles)

            update_render_view_viz()

        ps.set_user_callback(ps_ui_callback)


    n_epochs = int(n_iterations/train_dataset.__len__())


    # Criterions that we log during training
    criterions = {"psnr":  PeakSignalNoiseRatio(data_range=1).to("cuda")}

    # Initialize the tensorboard writer
    if conf.experiment_name and os.path.exists(f'{conf.out_dir}/{conf.experiment_name}'):
        logging.warning("The selected experiment name already exists and the checkpoints could be overwritten!")

    writer = SummaryWriter(log_dir=f'{conf.out_dir}/{conf.experiment_name}' if conf.experiment_name else None)

    assert model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
    
    for epoch_idx in range(n_epochs):
        if epoch_idx > 0 and epoch_idx % val_frequency == 0:
            val_iteration = 0
            with tqdm(val_dataloader) as pbar:
                pbar.set_description("Validation:" )
                val_psnr = []
                val_loss = []
                for batch in pbar:
                    with torch.no_grad():
                        gpu_batch = move_to_gpu(batch)
                        rays_ori, rays_dir, rgb_gt = gpu_batch["rays_ori"], gpu_batch["rays_dir"], gpu_batch["rgb_gt"]

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
                gpu_batch = move_to_gpu(batch)
                rays_ori, rays_dir, rgb_gt = gpu_batch["rays_ori"], gpu_batch["rays_dir"], gpu_batch["rgb_gt"]
                scene_updated = False

                # Compute the outputs of a single batch
                outputs = model(rays_ori, rays_dir)
                
                # Check if alphas are given and if the background is a fix color
                if isinstance(model.background, BackgroundColor):
                    assert "alpha" in gpu_batch
                    alpha = gpu_batch["alpha"]
                    rgb_gt = rgb_gt * alpha + model.background.color * (1 - alpha)

                # Compute the loss
                loss = torch.abs(outputs['pred_rgb'] - rgb_gt).mean()
                writer.add_scalar("loss_l1/train", loss.item(), global_step)
                if use_ssim:
                    loss_ssim = ssim(torch.permute(outputs['pred_rgb'], (0, 3, 1, 2)), torch.permute(rgb_gt, (0, 3, 1, 2)))
                    loss += loss_ssim
                    writer.add_scalar("loss_ssim/train", loss_ssim.item(), global_step)
                if conf.model.lambda_background > 0.0:
                    assert "sky_mask" in gpu_batch, "Sky ray mask missing for background-loss evaluation"
                    # Push all background rays to have opacity 0 and non-background rays to have opacity 1 withing the FV
                    foreground_mask = torch.ones_like(outputs["pred_opacity"])
                    foreground_mask[gpu_batch['sky_mask']] = 0.0
                    loss_background = torch.nn.functional.mse_loss(outputs["pred_opacity"], foreground_mask)
                    loss += conf.model.lambda_background * loss_background
                    writer.add_scalar("loss_background/train", loss_background.item(), global_step)

                # backpropagate the gradients and update the parameters
                loss.backward()
                model.optimizer.step()
                model.optimizer.zero_grad()

                # Make a scheduler step
                model.scheduler_step(global_step)

                psnr = criterions["psnr"](outputs['pred_rgb'], rgb_gt).item()
                global_step += 1
                pbar.set_postfix({'iteration': global_step, 'psnr': psnr, 'loss': loss.item()})
                writer.add_scalar("psnr/train", psnr, global_step)

                # Save the checkpoint
                if global_step > 0 and global_step % conf.checkpoint.frequency == 0:
                    parameters = model.get_model_parameters()
                    parameters |= {"global_step": global_step, "epoch": epoch_idx}
                    torch.save(parameters, os.path.join(writer.get_logdir(), f"ckpt_{global_step}.pt"))

                # Densify the Gaussians
                if global_step > conf.model.densify.start_iteration and global_step < conf.model.densify.end_iteration and global_step % conf.model.densify.frequency == 0:
                    model.densify_gaussians(scene_extent=scene_extent)
                    scene_updated = True

                # Prune the Gaussians
                if global_step > conf.model.prune.start_iteration and global_step < conf.model.prune.end_iteration and global_step % conf.model.prune.frequency == 0:
                    model.prune_gaussians()
                    scene_updated = True

                # Reset the Gaussian density 
                if global_step > conf.model.reset_density.start_iteration and global_step < conf.model.reset_density.end_iteration and global_step % conf.model.reset_density.frequency == 0:
                    model.reset_density()
                    scene_updated = True

                # SH: Every N its we increase the levels of SH up to a maximum degree
                # MLP: Every N we further unmask additional dimensions
                if model.progressive_training and global_step > 0 and global_step % model.feature_dim_increase_interval == 0:
                    model.increase_num_active_features()

                # Update the BVH if required
                if scene_updated or (global_step > 0 and conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0):
                    model.build_bvh()

                if conf.with_gui:
                    if scene_updated or model.get_positions().requires_grad:
                        update_cloud_viz()
        
                    update_render_view_viz()

                    ps.frame_tick()
                    while not viz_do_train:
                        ps.frame_tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args, remainder = parser.parse_known_args()

    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(remainder)
    conf = OmegaConf.merge(base_conf, cli_conf)

    main(conf)
