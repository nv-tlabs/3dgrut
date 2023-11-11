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

from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset
from datasets.ngp_dataset import NGPDataset
from datasets.ncore_dataset import NCoreDataset
from datasets.ncore_utils import Batch as NCoreBatch
from datasets.utils import PointCloud
from model import MixtureOfGaussians
from background import BackgroundColor
from datasets.utils import move_to_gpu
from loss_utils import ssim
from gui import GUI
from recorder import TrainingRecorder
from render import Renderer
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


DEFAULT_DEVICE = torch.device('cuda')

logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    # Run the training process
    n_iterations = conf.n_iterations
    val_frequency = conf.val_frequency
    scene_extent: float = 1.
    scene_bbox: tuple[torch.Tensor, torch.Tensor] # Tuple of vec3 (min,max)
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
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
            split='test', # TODO : change back to val, but ww can directly monitor what we will get :)
            sample_full_image=True,
            return_alphas=False
        )
        scene_extent = train_dataset.cameras_extent # taken from gsplat code
        # Scene extend from bbox poses
        scene_bbox = train_dataset.get_bbox()
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
        scene_extent = train_dataset.cameras_extent # taken from gsplat code
        # Scene extend from bbox poses
        scene_bbox = train_dataset.get_bbox()
    elif conf.dataset.type == 'ngp':
        train_dataset = NGPDataset(
            conf.path, 
            split='train', 
            sample_full_image=conf.dataset.train.sample_full_image, 
            batch_size=conf.dataset.train.batch_size,
            use_lidar=True,
            use_dynamic_masks=True,
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

        # Scene extend from bbox of point-cloud
        scene_bbox = (pc.xyz_end.min(0).values, pc.xyz_end.max(0).values)
        scene_extent = torch.linalg.norm(scene_bbox[1] - scene_bbox[0]) #TODO implement as in colmap dataset and nerf dataset
    elif conf.dataset.type == 'ncore':
        # TODO: add all of the dataset parameters to config
        duration_sec = 2.0
        n_train_sample_timepoints = 5

        train_dataset = NCoreDataset(
            conf.path, 
            split='train', 
            duration_sec=duration_sec,
            n_train_sample_timepoints=n_train_sample_timepoints,
        )
        val_dataset = NCoreDataset(
            conf.path,
            split='val',
            duration_sec=duration_sec,
        )
        pc = PointCloud.from_sequence(list(train_dataset.get_point_clouds(step_frame=10, non_dynamic_points_only=True)), device="cpu")
        
        # Scene extend from bbox of point-cloud
        scene_bbox = (pc.xyz_end.min(0).values, pc.xyz_end.max(0).values)
        scene_extent = torch.linalg.norm(scene_bbox[1] - scene_bbox[0]) #TODO implement as in colmap dataset and nerf dataset

        # Dataset produces NCoreBatch requiring dedicated collate_fns
        train_collate_fn = NCoreBatch.collate_fn
        val_collate_fn = NCoreBatch.collate_fn
    else:
        raise ValueError(f'Unsupported dataset type: {conf.dataset.type}. Choose between: ["colmap", "nerf", "ngp", "ncore"]. ')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=conf.num_workers, batch_size=1, shuffle=True, collate_fn=train_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=conf.num_workers, batch_size=1, shuffle=False, collate_fn=val_collate_fn)

    # Initialize the model and the optix context
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
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
    if conf.use_wandb:
        import wandb
        object_name = TrainingRecorder.get_run_name(conf.path)
        run_name = f'{object_name}-' + TrainingRecorder.get_timestamp()
        wandb.init(config=OmegaConf.to_container(conf), project='3dgrt', group=conf.experiment_name, name=run_name)
        wandb.tensorboard.patch(root_logdir=f'{conf.out_dir}/{conf.experiment_name}' if conf.experiment_name else None, save=False)
    writer = SummaryWriter(log_dir=f'{conf.out_dir}/{conf.experiment_name}' if conf.experiment_name else None)
    it_start = torch.cuda.Event(enable_timing=True)
    it_end = torch.cuda.Event(enable_timing=True)

    # Store parsed config for reference
    with open(os.path.join(writer.get_logdir(), "parsed.yaml"), "w") as fp:
        OmegaConf.save(config=conf, f=fp)

    assert model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
    
    for epoch_idx in range(n_epochs):
        if epoch_idx > 0 and epoch_idx % val_frequency == 0:
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

                    recorder.record_metrics(iteration=global_step, psnr=val_psnr, loss=val_loss)

        with tqdm(train_dataloader) as pbar:
            for batch in pbar:
                with torch.cuda.nvtx.range(f"train.train_step_{global_step}"):
                    it_start.record()
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
                    loss_l1 = torch.abs(outputs['pred_rgb'] - rgb_gt).mean()
                    writer.add_scalar("loss_l1/train", loss_l1.item(), global_step)

                    if conf.loss.use_ssim and conf.dataset.train.get("sample_full_image", False):
                        loss_ssim = ssim(torch.permute(outputs['pred_rgb'], (0, 3, 1, 2)), torch.permute(rgb_gt, (0, 3, 1, 2)))
                        loss = (1.0 - conf.loss.lambda_ssim) * loss_l1 + conf.loss.lambda_ssim * (1.0 - loss_ssim)
                        writer.add_scalar("loss_ssim/train", (1.0 - loss_ssim).item(), global_step)
                    else:
                        loss_ssim = None
                        loss = loss_l1

                    if conf.loss.use_scalereg and conf.loss.lambda_scalereg > 0.0:
                        # Regularization to prevent needle-like degenerate geometries (excessive ratio of largest to smallest scale)
                        scale = model.get_scale()
                        min_scales = torch.min(scale, dim=-1).values
                        max_scales = torch.max(scale, dim=-1).values
                        scale_ratio = torch.log(max_scales) - torch.log(min_scales) # positive value, larger means bigger ratio
                        loss_scalereg = torch.mean(torch.square(scale_ratio))
                        loss += conf.loss.lambda_scalereg * loss_scalereg
                        writer.add_scalar("loss_scalereg/train", loss_scalereg.item(), global_step)

                    if conf.model.lambda_background > 0.0:
                        assert "sky_mask" in gpu_batch, "Sky ray mask missing for background-loss evaluation"
                        # Push all background rays to have opacity 0 and non-background rays to have opacity 1 withing the FV
                        foreground_mask = torch.ones_like(outputs["pred_opacity"])
                        foreground_mask[gpu_batch['sky_mask']] = 0.0
                        loss_background = torch.nn.functional.mse_loss(outputs["pred_opacity"], foreground_mask)
                        loss += conf.model.lambda_background * loss_background
                        writer.add_scalar("loss_background/train", loss_background.item(), global_step)

                # backpropagate the gradients and update the parameters
                with torch.cuda.nvtx.range("backward"):
                    loss.backward()

                with torch.cuda.nvtx.range("update-positional-grad"):
                    if global_step < conf.model.densify.end_iteration:
                        hit_cts = model.get_hit_counts(rays_ori, rays_dir)
                        mask = (hit_cts > 0).squeeze()
                        model.update_positional_grad(mask)

                with torch.cuda.nvtx.range("backpropagation"):
                    model.optimizer.step()
                    model.optimizer.zero_grad()

                it_end.record()

                # Make a scheduler step
                model.scheduler_step(global_step)

                psnr = criterions["psnr"](outputs['pred_rgb'], rgb_gt).item()
                global_step += 1
                pbar.set_postfix({'iteration': global_step, 'psnr': psnr, 'loss': loss.item()})
                writer.add_scalar("psnr/train", psnr, global_step)

                recorder.record_train_step(model, global_step, it_start.elapsed_time(it_end),
                                           loss_l1, loss_ssim, loss, psnr)
                recorder.report_statistics(writer=writer)

                # Save the checkpoint
                if global_step > 0 and global_step % conf.checkpoint.frequency == 0:
                    parameters = model.get_model_parameters()
                    parameters |= {"global_step": global_step, "epoch": epoch_idx}
                    torch.save(parameters, os.path.join(writer.get_logdir(), f"ckpt_{global_step}.pt"))

                # Densify the Gaussians
                if global_step > conf.model.densify.start_iteration and  global_step < conf.model.densify.end_iteration and global_step % conf.model.densify.frequency == 0:
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

                if gui is not None:
                    if gui.live_update:
                        if scene_updated or model.get_positions().requires_grad:
                            gui.update_cloud_viz()
            
                        gui.update_render_view_viz()

                    ps.frame_tick()
                    while not gui.viz_do_train:
                        ps.frame_tick()

    recorder.submit_recording(
        dataset=train_dataset,
        scene_extent=scene_bbox,
        train_path=conf.path,
        model=model
    )

    if conf.test_last:
        parameters = model.get_model_parameters()
        parameters |= {"global_step": global_step, "epoch": n_epochs-1}
        last_ckpt_path = os.path.join(writer.get_logdir(), "ckpt_last.pt")
        torch.save(parameters, last_ckpt_path)
        renderer = Renderer(checkpoint_path=last_ckpt_path,
                            out_dir=writer.get_logdir(),
                            path=conf.path,
                            save_gt=False,
                            writer=writer,
                            model = model)
        renderer.render_all()

if __name__ == "__main__":
    main()
