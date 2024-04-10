import os
import sys
import logging
import torch
import logging
import torch.utils.data
import hydra

from torchmetrics import PeakSignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import datasets
from models import antialiasing
from models.model import MixtureOfGaussians
from models.background import BackgroundColor
from datasets.utils import move_to_gpu, PointCloud, MultiEpochsDataLoader
from models.losses import ssim
from models.background import SkyMlp
from utils.gui import GUI
from utils.recorder import TrainingRecorder
from render import Renderer
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


DEFAULT_DEVICE = torch.device('cuda')

logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

def jet_map(map:torch.Tensor, max_val: float) -> torch.Tensor:
     vs = (map/max_val).clip(0,1)
     return torch.concat([
          (4.0 * (vs - 0.375)).clip(0,1) * (-4.0 * (vs - 1.125)).clip(0,1),
          (4.0 * (vs - 0.125)).clip(0,1) * (-4.0 * (vs - 0.875)).clip(0,1),
          (4.0 * vs + 0.5).clip(0,1) * (-4.0 * (vs - 0.625)).clip(0,1)],dim=2)

@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    # Run the training process
    n_iterations = conf.n_iterations
    val_frequency = conf.val_frequency
    scene_extent: float = 1.
    scene_bbox: tuple[torch.Tensor, torch.Tensor] # Tuple of vec3 (min,max)
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset

    ray_jitter = antialiasing.make(conf.dataset.train.ray_jittering.type, conf)
    train_dataset, val_dataset, train_collate_fn, val_collate_fn = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=ray_jitter)

    train_dataloader = MultiEpochsDataLoader(train_dataset, num_workers=conf.num_workers, batch_size=1, shuffle=True, collate_fn=train_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=conf.num_workers, batch_size=1, shuffle=False, collate_fn=val_collate_fn)

    scene_extent = train_dataset.get_scene_extent()
    scene_bbox = train_dataset.get_scene_bbox()

    # Initialize the model and the optix context
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    model.set_optix_context()

    if conf.resume: # Load a checkpoint
        logging.info(f"Loading a pretrained checkpoint from {conf.resume}!")
        checkpoint = torch.load(conf.resume)
        model.init_from_checkpoint(checkpoint)
        global_step = checkpoint['global_step']

    else: # Initialize
        match conf.initialization.method:
            case 'colmap':
                observer_points = torch.tensor(train_dataset.get_observer_points(), dtype=torch.float32, device=DEFAULT_DEVICE)
                model.init_from_colmap(conf.path, observer_points)
            case 'point_cloud':
                observer_points = torch.tensor(train_dataset.get_observer_points(), dtype=torch.float32, device=DEFAULT_DEVICE)
                ply_path = None
                try:
                    ply_path = os.path.join(conf.path, "point_cloud.ply")
                    model.init_from_pretrained_point_cloud(ply_path)
                except FileNotFoundError as e:
                    logging.exception(e)
                    raise e           
            case 'random':
                model.init_from_random_point_cloud(num_gsplat=conf.initialization.num_gaussians)
            case 'lidar':
                pc = PointCloud.from_sequence(list(train_dataset.get_point_clouds(step_frame=10, non_dynamic_points_only=True)), device="cpu")
                assert conf.dataset.type in ['ngp', 'ncore'], 'can only initialize from lidar with the NGPDataset / NCoreDataset'
                observer_points = torch.tensor(train_dataset.get_observer_points(), dtype=torch.float32, device=DEFAULT_DEVICE)
                model.init_from_lidar(pc, observer_points) 
            case 'auxiliary':
                model.init_from_auxiliary_data(dataset=train_dataset,
                                               scene_bbox=scene_bbox,
                                               spacing=conf.initialization.spacing,
                                               hit_count_threshold=0,
                                               sky_step_frame=conf.initialization.sky_step_frame,
                                               sky_step_pixel=conf.initialization.sky_step_pixel,
                                               pc_step_frame=conf.initialization.pc_step_frame,
                                               pc_step_points=conf.initialization.pc_step_points)
            case _:
                raise ValueError(f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random, auxiliary]")

        model.build_bvh()
        model.setup_optimizer()
        global_step = 0

    gui = None
    if conf.with_gui:
        import polyscope as ps
        gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox)

    n_epochs = int(n_iterations/train_dataset.__len__())

    # Criterions that we log during training
    criterions = {"psnr":  PeakSignalNoiseRatio(data_range=1).to("cuda")}

    # Initialize the tensorboard writer
    if conf.experiment_name and os.path.exists(f'{conf.out_dir}/{conf.experiment_name}'):
        logging.warning("The selected experiment name already exists and the checkpoints could be overwritten!")

    recorder = TrainingRecorder(enabled=conf.record_training)
    object_name = TrainingRecorder.get_run_name(conf.path)
    run_name = f'{object_name}-' + TrainingRecorder.get_timestamp()
    if conf.use_wandb:
        import wandb
        wandb.init(config=OmegaConf.to_container(conf), project=conf.wandb_project, group=conf.experiment_name, name=run_name)
        wandb.tensorboard.patch(root_logdir=f'{conf.out_dir}/{conf.experiment_name}' if conf.experiment_name else None, save=False)
    writer = SummaryWriter(log_dir=f'{conf.out_dir}/{conf.experiment_name}' if conf.experiment_name else None)
    out_dir = os.path.join(writer.get_logdir(), run_name) if conf.experiment_name else writer.get_logdir()
    os.makedirs(out_dir, exist_ok=True)

    it_start = torch.cuda.Event(enable_timing=True)
    it_end = torch.cuda.Event(enable_timing=True)

    # Store parsed config for reference
    with open(os.path.join(writer.get_logdir(), "parsed.yaml"), "w") as fp:
        OmegaConf.save(config=conf, f=fp)

    assert model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
    
    if val_frequency % len(train_dataset) != 0:
        logging.warning(f"The selected val_frequency {val_frequency} is not a multiple of the train_dataset size {len(train_dataset)}")
        val_frequency -= val_frequency % len(train_dataset) 
        logging.warning(f"setting val_frequency to {val_frequency}")

    progress_bar = tqdm(range(conf.n_iterations), desc="Training progress")
    for epoch_idx in range(n_epochs):
        if conf.model.log_rolling_buffers:
            model.reset_rolling_buffers()

        if (global_step > 0 or conf.validate_first) and global_step % val_frequency == 0:
                val_iteration = 0
                with tqdm(val_dataloader) as pbar:
                    pbar.set_description("Validation:" )
                    val_psnr = []
                    val_loss = []
                    for batch in pbar:
                        with torch.cuda.nvtx.range(f"train.validation_step_{global_step}"):
                            with torch.no_grad():
                                gpu_batch = move_to_gpu(batch)
                                rays_ori, rays_dir, rgb_gt = gpu_batch["rays_ori"], gpu_batch["rays_dir"], gpu_batch["rgb_gt"]

                                # Compute the outputs of a single batch
                                outputs = model(rays_ori, rays_dir, train=False)

                                # Compute the loss
                                val_loss.append(torch.abs(outputs['pred_rgb'] - rgb_gt).mean().item())
                                val_psnr.append(criterions["psnr"](outputs['pred_rgb'], rgb_gt).item())

                                pbar.set_postfix({'iteration': val_iteration, 'psnr': val_psnr[-1], 'loss': val_loss[-1]})

                                if val_iteration == 0:
                                    writer.add_image('val_hits/val', jet_map(outputs['hits_count'][-1],conf.writer.max_num_hits), global_step, dataformats='HWC')
                                    writer.add_image('image/val', outputs['pred_rgb'][-1].clip(0,1.0), global_step, dataformats='HWC')
                                    writer.add_image('image/gt', rgb_gt[-1].clip(0,1.0), global_step, dataformats='HWC')
                                    
                                val_iteration += 1

                    writer.add_scalar("psnr/val", np.mean(val_psnr), global_step)
                    writer.add_scalar("loss_l1/val", np.mean(val_loss), global_step)

                    writer.add_scalar("hits/min", outputs['hits_count'].min().cpu(), global_step)
                    writer.add_scalar("hits/mean", outputs['hits_count'].mean().cpu(), global_step)
                    writer.add_scalar("hits/std", outputs['hits_count'].std().cpu(), global_step)


                    print(f"Validation: {np.mean(val_psnr):.3f}")

                    recorder.record_metrics(iteration=global_step, psnr=val_psnr, loss=val_loss)

        for batch in train_dataloader:
            if ray_jitter is not None and conf.dataset.train.ray_jittering.start_iteration >= global_step:
                ray_jitter.enabled = True
            with torch.cuda.nvtx.range(f"train.train_step_{global_step}"):
                it_start.record()

                error_buffer_batch = False
                # Sample a new batch from the error buffer (the actual batch from the dataloader will be skipped)
                if model.error_based_sampling and global_step > model.error_buffer_update_frequency and global_step % model.error_sampling_frequency == 0:
                    batch = model.sample_from_error_buffer(batch["rays_ori"].shape[:-1])
                    error_buffer_batch = True

                # Move data to GPU
                gpu_batch = move_to_gpu(batch)
                rays_ori, rays_dir, rgb_gt = gpu_batch["rays_ori"], gpu_batch["rays_dir"], gpu_batch["rgb_gt"]
                scene_updated = False

                # Compute the outputs of a single batch
                error_target = torch.zeros_like(model.density)
                error_target.requires_grad = True
                outputs = model(rays_ori, rays_dir, error_target, train=True)

                # hits 
                if global_step % conf.writer.hit_stat_frequency == 0:
                    writer.add_scalar("train_hits/mean", outputs['hits_count'].mean().item(), global_step)
                    writer.add_scalar("train_hits/std", outputs['hits_count'].std().item(), global_step)
                    writer.add_scalar("train_hits/max", outputs['hits_count'].max().item(), global_step)
                    writer.add_scalar("train_hits/min", outputs['hits_count'].min().item(), global_step)

                # Check if alphas are given and if the background is a fix color
                if isinstance(model.background, BackgroundColor):
                    if "alpha" in gpu_batch:
                        with torch.cuda.nvtx.range(f"handling-bg-color"):
                            alpha = gpu_batch["alpha"]
                            rgb_gt = rgb_gt * alpha + model.background.color * (1 - alpha)

                # Compute the loss
                with torch.cuda.nvtx.range(f"loss-l1"):
                    loss_l1 = torch.abs(outputs['pred_rgb'] - rgb_gt).mean()
                if conf.enable_writer:
                    writer.add_scalar("loss_l1/train", loss_l1.item(), global_step)

                if conf.loss.use_l2:
                    with torch.cuda.nvtx.range(f"loss-l2"):
                        loss_ssim = None
                        loss_l2 = torch.nn.MSELoss()(outputs['pred_rgb'], rgb_gt) * conf.loss.lambda_l2
                        loss = loss_l2
                        if conf.enable_writer:
                            writer.add_scalar("loss_l2/train", loss_l2.item(), global_step)
                elif conf.loss.use_ssim and ~error_buffer_batch and conf.dataset.train.get("sample_full_image", False):
                    loss_ssim = ssim(torch.permute(outputs['pred_rgb'], (0, 3, 1, 2)), torch.permute(rgb_gt, (0, 3, 1, 2)))
                    loss = (1.0 - conf.loss.lambda_ssim) * loss_l1 + conf.loss.lambda_ssim * (1.0 - loss_ssim)
                    if conf.enable_writer:
                        writer.add_scalar("loss_ssim/train", (1.0 - loss_ssim).item(), global_step)
                else:
                    loss_ssim = None
                    loss = loss_l1

                if conf.loss.lambda_reg_density > 0:
                    with torch.cuda.nvtx.range(f"loss-reg-density"):
                        loss_reg_density = torch.nn.functional.relu(1.0-torch.abs(2*model.get_density(True)-1),inplace=True).mean()
                        loss += conf.loss.lambda_reg_density * loss_reg_density 
                    if conf.enable_writer:
                        writer.add_scalar("loss_reg_density/train_loss", loss_reg_density.item(), global_step)
                        if global_step % 111 == 0: 
                            writer.add_histogram("loss_reg_density/train_density",model.get_density(True).detach().cpu())

                if conf.loss.use_lidardistance and conf.loss.lambda_lidardistance > 0.0:
                    lidar_rays_ori = gpu_batch["lidar_rays_ori"]
                    lidar_rays_dir = gpu_batch["lidar_rays_dir"]
                    lidar_dist_gt = gpu_batch["lidar_dist_gt"]

                    # Compute the outputs of the lidar rays
                    outputs_lidar = model(lidar_rays_ori, lidar_rays_dir)

                    # Compute distance loss
                    loss_lidar = torch.nn.L1Loss(reduction="mean")(outputs_lidar['pred_dist'], lidar_dist_gt) 
                    loss += conf.loss.lambda_lidardistance * loss_lidar
                    
                    if conf.enable_writer:
                        writer.add_scalar("loss_lidar/train", loss_lidar.item(), global_step)

                if conf.loss.use_scalereg and conf.loss.lambda_scalereg > 0.0:
                    # Regularization to prevent needle-like degenerate geometries (excessive ratio of largest to smallest scale)
                    scale = model.get_scale()
                    min_scales = torch.min(scale, dim=-1).values
                    max_scales = torch.max(scale, dim=-1).values
                    scale_ratio = torch.log(max_scales) - torch.log(min_scales) # positive value, larger means bigger ratio
                    loss_scalereg = torch.mean(torch.square(scale_ratio))
                    loss += conf.loss.lambda_scalereg * loss_scalereg
                    if conf.enable_writer:
                        writer.add_scalar("loss_scalereg/train", loss_scalereg.item(), global_step)

                if conf.model.lambda_background > 0.0:
                    assert "sky_mask" in gpu_batch, "Sky ray mask missing for background-loss evaluation"
                    # Push all background rays to have opacity 0 and non-background rays to have opacity 1 withing the FV
                    foreground_mask = torch.ones_like(outputs["pred_opacity"])
                    foreground_mask[gpu_batch['sky_mask']] = 0.0
                    loss_background = torch.nn.functional.mse_loss(outputs["pred_opacity"], foreground_mask)
                    loss += conf.model.lambda_background * loss_background
                    if conf.enable_writer:
                        writer.add_scalar("loss_background/train", loss_background.item(), global_step)
                elif conf.model.lambda_opacity > 0.0:
                    with torch.cuda.nvtx.range(f"loss-opacity"):
                        opacity_gt = torch.ones_like(outputs["pred_opacity"])
                        loss_opacity = torch.abs(outputs["pred_opacity"] - opacity_gt).mean()
                        loss += conf.model.lambda_opacity * loss_opacity
                    if conf.enable_writer:
                        writer.add_scalar("loss_opacity/train", loss_opacity.item(), global_step)                       

            # backpropagate the gradients and update the parameters
            with torch.cuda.nvtx.range("backward"):
                if conf.model.log_rolling_buffers:
                    ray_err_abs = torch.abs(outputs['pred_rgb'] - rgb_gt)
                    # horrible hacks to abuse gradients: distributing error statistics back to the
                    # gaussians can be viewed as backprop'nig through a proxy weight
                    fake_loss = loss + torch.sum(ray_err_abs.detach() * outputs['err_backprop_proxy'])
                    fake_loss.backward()
                else:
                    loss.backward()

            # update densification buffer:
            if global_step < conf.model.densify.end_iteration:
                if conf.model.log_rolling_buffers:
                    model.update_rolling_buffers(error_target.grad, outputs['g_weights'])
                if conf.model.densify.method == 'gradient-buffer':
                    model.update_gradient_buffer(rays_ori, rays_dir)


            # clamp density
            if global_step>0 and conf.model.density_activation == "none":
                model.clamp_density()

            it_end.record()

            # Make a scheduler step
            model.scheduler_step(global_step)

            global_step += 1

            # Logging
            if conf.enable_writer:
                with torch.cuda.nvtx.range(f"criterions_psnr"):
                    psnr = criterions["psnr"](outputs['pred_rgb'], rgb_gt).item()
                progress_bar.set_postfix({'psnr': psnr, 'loss': loss.item()})
                progress_bar.update(1)
                if global_step > 0 and global_step % conf.log_frequency == 0:
                    writer.add_scalar("psnr/train", psnr, global_step)
            elif global_step % 10 == 0:
                progress_bar.set_postfix({'loss': loss.item()})
                progress_bar.update(10)

            if conf.enable_writer:
                recorder.record_train_step(model, global_step, it_start.elapsed_time(it_end),
                                        loss_l1, loss_ssim, loss, psnr)
                recorder.report_statistics(writer=writer)

            # Save the checkpoint
            with torch.cuda.nvtx.range(f"ckpt_save"):
                if global_step > 0 and global_step % conf.checkpoint.frequency == 0:
                    parameters = model.get_model_parameters()
                    parameters |= {"global_step": global_step, "epoch": epoch_idx}
                    os.makedirs(os.path.join(out_dir,  f"ours_{int(global_step)}"), exist_ok=True)
                    torch.save(parameters, os.path.join(out_dir,  f"ours_{int(global_step)}", f"ckpt_{global_step}.pt"))

            # Densify the Gaussians
            if global_step > conf.model.densify.start_iteration and  global_step < conf.model.densify.end_iteration and global_step % conf.model.densify.frequency == 0:
                model.densify_gaussians(scene_extent=scene_extent)
                scene_updated = True

            # Prune the Gaussians based on their opacity
            if global_step > conf.model.prune.start_iteration and global_step < conf.model.prune.end_iteration and global_step % conf.model.prune.frequency == 0:
                model.prune_gaussians_opacity()
                scene_updated = True

            # Prune the Gaussians based on their contribution weight
            if global_step > conf.model.prune_weight.start_iteration and global_step < conf.model.prune_weight.end_iteration and global_step % conf.model.prune_weight.frequency == 0:
                if conf.model.log_rolling_buffers:
                    model.prune_gaussians_weight()
                scene_updated = True

            # Prune the Gaussians based on their scales
            if global_step > conf.model.prune_scale.start_iteration and global_step < conf.model.prune_scale.end_iteration and global_step % conf.model.prune_scale.frequency == 0:
                model.prune_gaussians_scale(train_dataset)
                scene_updated = True

            # Prune the needle Gaussians 
            if global_step > conf.model.prune_needles.start_iteration and global_step < conf.model.prune_needles.end_iteration and global_step % conf.model.prune_needles.frequency == 0:
                model.prune_needles()
                scene_updated = True

            # Prune the sky Gaussians
            if isinstance(model.background,SkyMlp) and global_step > conf.model.prune_sky_mlp.start_iteration and global_step < conf.model.prune_sky_mlp.end_iteration and global_step % conf.model.prune_sky_mlp.frequency == 0:
                model.prune_sky_gaussians(train_dataset.get_sky_rays(step_frame=conf.model.prune_sky_mlp.step_frame,step_pixel=conf.model.prune_sky_mlp.step_pixel))
                scene_updated = True

            # Decay the density values
            if global_step > conf.model.density_decay.start_iteration and global_step < conf.model.density_decay.end_iteration and global_step % conf.model.density_decay.frequency == 0:
                model.decay_density()

            # Reset the Gaussian density 
            if global_step > conf.model.reset_density.start_iteration and global_step < conf.model.reset_density.end_iteration and global_step % conf.model.reset_density.frequency == 0:
                model.reset_density()

            # SH: Every N its we increase the levels of SH up to a maximum degree
            # MLP: Every N we further unmask additional dimensions
            if model.progressive_training and global_step > 0 and global_step % model.feature_dim_increase_interval == 0:
                model.increase_num_active_features()

            # Update the BVH if required
            if scene_updated:
                model.build_bvh(full_build=True)
            elif (global_step > 0 and conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0):
                model.build_bvh(full_build=False)

            # Updating the GUI
            if gui is not None:
                if gui.live_update:
                    if scene_updated or model.get_positions().requires_grad:
                        gui.update_cloud_viz()
                    gui.update_render_view_viz()

                ps.frame_tick()
                while not gui.viz_do_train:
                    ps.frame_tick()

            # Optimizer step
            with torch.cuda.nvtx.range("backpropagation"):
                model.optimizer.step()
                model.optimizer.zero_grad()

            # Update the error buffers with new values
            if global_step > 0 and model.error_based_sampling and global_step % model.error_buffer_update_frequency == 0:
                with torch.cuda.nvtx.range(f"update_error_buffer[step {global_step}]"):
                    ray_origins = []
                    ray_directions = []
                    rgbs = []
                    alphas = []
                    errors = []

                    model.build_bvh()
                    ds_factor = model.error_downsampling_factor
                    #TODO: implement random shift for downsampling
                    with torch.no_grad():
                        qbar = tqdm(range(len(train_dataset)))
                        qbar.set_description("Computing error maps:" )
                        for batch_idx in qbar:
                            batch = train_dataset[batch_idx]
                            gpu_batch = move_to_gpu(batch)
                            rays_ori = gpu_batch["rays_ori"][::ds_factor, ::ds_factor, :].unsqueeze(0)
                            rays_dir = gpu_batch["rays_dir"][::ds_factor, ::ds_factor, :].unsqueeze(0)
                            rgb_gt = gpu_batch["rgb_gt"][::ds_factor, ::ds_factor, :].unsqueeze(0)

                            # Append all the tensors
                            ray_origins.append(rays_ori.reshape(-1,3).cpu())
                            ray_directions.append(rays_dir.reshape(-1,3).cpu())
                            rgbs.append(rgb_gt.reshape(-1,3).cpu())

                            # Compute the outputs of a single batch
                            outputs = model(rays_ori, rays_dir)

                            # Check if alphas are given and if the background is a fix color
                            if isinstance(model.background, BackgroundColor):
                                assert "alpha" in gpu_batch
                                alpha = gpu_batch["alpha"][::ds_factor, ::ds_factor, :].unsqueeze(0)
                                rgb_gt = rgb_gt * alpha + model.background.color * (1 - alpha)
                                alphas.append(alpha.reshape(-1,1).cpu())

                            # Compute the loss
                            error_map = torch.abs(outputs['pred_rgb'] - rgb_gt).mean(dim=-1, keepdim=True)
                            errors.append(error_map.reshape(-1,1))

                            if batch_idx == 0:
                                writer.add_image('image/error_map', error_map[-1].clip(0,1.0), global_step, dataformats='HWC')

                    model.update_error_buffer(torch.vstack(ray_origins), torch.vstack(ray_directions),
                                              torch.vstack(errors), torch.vstack(rgbs),
                                              torch.vstack(alphas) if len(alphas) else None)
        
    recorder.submit_recording(
        dataset=train_dataset,
        scene_extent=scene_bbox,
        train_path=conf.path,
        model=model
    )

    # Updating the GUI
    if gui is not None:
        gui.training_done = True
        while gui.viz_final:
            ps.frame_tick()

    if conf.test_last:
        logging.info(f"running on test set...")
        if train_dataloader is not None:
            del train_dataloader
        if val_dataloader is not None:
            del val_dataloader
        if train_dataset is not None:
            del train_dataset
        if val_dataset is not None:
            del val_dataset
        parameters = model.get_model_parameters()
        parameters |= {"global_step": global_step, "epoch": n_epochs-1}
        last_ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, last_ckpt_path)
        renderer = Renderer.from_preloaded_model(
            model=model,
            out_dir=out_dir,
            path=conf.path,
            save_gt=False,
            writer=writer,
            global_step=global_step
        )
        renderer.render_all()

if __name__ == "__main__":
    main()
