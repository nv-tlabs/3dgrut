# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union
import torchvision

import numpy as np
import cv2

import torch
import torch.utils.data

from addict import Dict
from omegaconf import DictConfig, OmegaConf

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import MultiEpochsDataLoader, DEFAULT_DEVICE
from threedgrut.export.ingp_exporter import INGPExporter
from threedgrut.export.ply_exporter import PLYExporter
from threedgrut.export.usdz_exporter import USDZExporter
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.gui import GUI
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import CudaTimer
from threedgrut.utils.misc import jet_map, create_summary_writer, check_step_condition
from threedgrut.optimizers import SelectiveAdam
from PIL import Image

class Trainer3DGRUT:
    """Trainer for paper: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" """

    model: MixtureOfGaussians
    """ Gaussian Model """

    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset

    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    scene_extent: float = 1.0
    """TODO: Add docstring"""

    scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
    """TODO: Add docstring"""

    strategy: BaseStrategy
    """ Strategy for optimizing the Gaussian model in terms of densification, pruning, etc. """

    gui = None
    """ If GUI is enabled, references the GUI interface """

    criterions: Dict
    """ Contains functors required to compute evaluation metrics, i.e. psnr, ssim, lpips """

    tracking: Dict
    """ Contains all components used to report progress of training """

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        """Create a new trainer from a checkpoint file"""

        conf.resume = resume
        conf.import_ingp.enabled = False
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ingp(ply_path: str, conf: DictConfig):
        """Create a new trainer from an INGP file"""

        conf.resume = ""
        conf.import_ingp.enabled = True
        conf.import_ingp.path = ply_path
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        """Create a new trainer from a PLY file"""

        conf.resume = ""
        conf.import_ingp.enabled = False
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        """Set up a new training session, or continue an existing one based on configuration"""

        # Keep track of useful fields
        self.conf = conf
        """ Global configuration of model, scene, optimization, etc"""
        self.device = device if device is not None else DEFAULT_DEVICE
        """ Device used for training and visualizations """
        self.global_step = 0
        """ Current global iteration of the trainer """
        self.n_iterations = conf.n_iterations
        """ Total number of train iterations to take (for multiple passes over the dataset) """
        self.n_epochs = 0
        """ Total number of train epochs / passes, e.g. single pass over the dataset."""
        self.val_frequency = conf.val_frequency
        """ Validation frequency, in terms on global steps """

        # Setup the trainer and components
        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_gui(conf, self.model, self.train_dataset, self.val_dataset, self.scene_bbox)
        self.use_customized_mask = True
        self.customized_mask_dir = conf.customized_mask_dir
        
    def create_customized_mask(self, image_shape, customized_mask_dir, device):
        """Create customized mask for fisheye images to exclude black border & tripod regions.
        
        Args:
            image_shape: Shape of the image tensor (batch, height, width, channels)
            customized_mask_dir (Optional)
        
        Returns:
            torch.Tensor: Binary mask with 1 for valid pixels, 0 for invalid
        """
        batch, height, width, channels = image_shape
        
        # Create custom mask

        mask_custom = torch.from_numpy(cv2.imread("mask_train_4.png", cv2.IMREAD_GRAYSCALE).astype(bool)).to(device)
        mask_custom = (mask_custom).float()
        
        # Expand to match batch and channel dimensions
        mask_custom = mask_custom.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
        mask_custom = mask_custom.repeat(batch, 1, 1, 1)      # [B, H, W, 1]
        return mask_custom


    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform
        
        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_dataloader_kwargs = configure_dataloader_for_platform({
            'num_workers': conf.num_workers,
            'batch_size': 1,
            'shuffle': True,
            'pin_memory': True,
            'persistent_workers': True if conf.num_workers > 0 else False,
        })
        
        val_dataloader_kwargs = configure_dataloader_for_platform({
            'num_workers': conf.num_workers,
            'batch_size': 1,
            'shuffle': False,
            'pin_memory': True,
            'persistent_workers': True if conf.num_workers > 0 else False,
        })
        
        train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)
        
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        """Initializes the gaussian model and the optix context"""
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        """Set pre-train / post-train iteration logic. i.e. densification and pruning"""
        assert self.model is not None
        match self.conf.strategy.method:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy
                self.strategy = GSStrategy(conf, self.model)
                logger.info("🔆 Using GS strategy")
            case "MCMCStrategy":
                from threedgrut.strategy.mcmc import MCMCStrategy
                self.strategy = MCMCStrategy(conf, self.model)
                logger.info("🔆 Using MCMC strategy")
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.strategy.method}")

    def setup_training(self, conf: DictConfig, model: MixtureOfGaussians, train_dataset: BoundedMultiViewDataset):
        """
        Performs required steps to setup the optimization:
        1. Initialize the gaussian model fields: load previous weights from checkpoint, or initialize from scratch.
        2. Build BVH acceleration structure for gaussian model, if not loaded with checkpoint
        3. Set up the optimizer to optimize the gaussian model params
        4. Initialize the densification buffers in the densificaiton strategy
        """

        # Initialize
        if conf.resume:  # Load a checkpoint
            logger.info(f"🤸 Loading a pretrained checkpoint from {conf.resume}!")
            checkpoint = torch.load(conf.resume)
            model.init_from_checkpoint(checkpoint)
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]
        elif conf.import_ingp.enabled:
            ingp_path = (
                conf.import_ingp.path
                if conf.import_ingp.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.inpg"
            )
            logger.info(f"Loading a pretrained ingp model from {ingp_path}!")
            model.init_from_ingp(ingp_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ingp.init_global_step
        elif conf.import_ply.enabled:
            ply_path = (
                conf.import_ply.path
                if conf.import_ply.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.ply"
            )
            logger.info(f"Loading a ply model from {ply_path}!")
            model.init_from_ply(ply_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ply.init_global_step
        else:
            logger.info(f"🤸 Initiating new 3dgrut training..")
            match conf.initialization.method:
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
                case "colmap":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(), dtype=torch.float32, device=self.device
                    )
                    model.init_from_colmap(conf.path, observer_points)
                case "point_cloud":
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(conf.initialization.path)
                    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
                case _:
                    raise ValueError(
                        f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random, checkpoint]"
                    )

            self.strategy.init_densification_buffer()

            model.build_acc()
            model.setup_optimizer()
            global_step = 0

        self.global_step = global_step
        self.n_epochs = int((conf.n_iterations + len(train_dataset) - 1) / len(train_dataset))

    def init_gui(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
        val_dataset: BoundedMultiViewDataset,
        scene_bbox,
    ):
        gui = None
        if conf.with_gui:
            gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox)
        self.gui = gui

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        # Initialize the tensorboard writer
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, conf.out_dir, conf.experiment_name, conf.use_wandb
        )
        logger.info(f"📊 Training logs & will be saved to: {out_dir}")

        # Store parsed config for reference
        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # Pack all components used to track progress of training
        self.tracking = Dict(writer=writer, run_name=run_name, object_name=object_name, output_dir=out_dir)

    @torch.cuda.nvtx.range("get_metrics")
    def get_metrics(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        profilers: dict[str, CudaTimer],
        split: str = "training",
        iteration: Optional[int] = None,
    ) -> dict[str, Union[int, float]]:
        """Computes dictionary of single batch metrics based on current batch output.
        Args:
            gpu_batch: GT data of current batch
            output: model prediction for current batch
            losses: dictionary of loss terms computed for current batch
            split: name of split metrics are computed for - 'training' or 'validation'
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        step = self.global_step

        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]

        if self.use_customized_mask:
            customized_mask = self.create_customized_mask(rgb_gt.shape, self.customized_mask_dir, self.device)
            if customized_mask.shape[-1] == 1 and rgb_gt.shape[-1] == 3:
                customized_mask = customized_mask.repeat(1, 1, 1, 3)
            
            # Apply mask
            rgb_gt = rgb_gt * customized_mask
            rgb_pred = rgb_pred * customized_mask

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        # Move losses to cpu once
        metrics["losses"] = {k: v.detach().item() for k, v in losses.items()}

        is_compute_train_hit_metrics = (split == "training") and (step % self.conf.writer.hit_stat_frequency == 0)
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()

        if is_compute_validation_metrics:
            with torch.cuda.nvtx.range(f"criterions_psnr"):
                metrics["psnr"] = psnr(rgb_pred, rgb_gt).item()

            rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
            pred_rgb_full = rgb_pred.permute(0, 3, 1, 2)
            pred_rgb_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)

            with torch.cuda.nvtx.range(f"criterions_ssim"):
                metrics["ssim"] = ssim(pred_rgb_full, rgb_gt_full).item()
            with torch.cuda.nvtx.range(f"criterions_lpips"):
                metrics["lpips"] = lpips(pred_rgb_full_clipped, rgb_gt_full).item()

            if iteration in self.conf.writer.log_image_views:
                metrics["img_hit_counts"] = jet_map(outputs["hits_count"][-1], self.conf.writer.max_num_hits)
                metrics["img_gt"] = rgb_gt[-1].clip(0, 1.0) #gpu_batch.rgb_gt[-1].clip(0, 1.0)
                metrics["img_pred"] = rgb_pred[-1].clip(0, 1.0) #outputs["pred_rgb"][-1].clip(0, 1.0)
                metrics["img_pred_dist"] = jet_map(outputs["pred_dist"][-1], 100)
                metrics["img_pred_opacity"] = jet_map(outputs["pred_opacity"][-1], 1)

        if profilers:
            timings = {}
            for key, timer in profilers.items():
                if timer.enabled:
                    timings[key] = timer.timing()
            if timings:
                metrics["timings"] = timings

        return metrics

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of losses for current batch.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
        Returns:
            losses: dictionary of loss terms computed for current batch.
        """
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]
        mask = gpu_batch.mask
        dilated_mask = gpu_batch.dilated_mask
        
        # Mask out the invalid pixels if the mask is provided
        if mask is not None:
            def rgb_to_xyz_srgb_d65(x):
                M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]],
                                device=x.device, dtype=x.dtype)
                return torch.tensordot(x, M.t(), dims=1)

            def xyz_to_lab(XYZ):
                Xn, Yn, Zn = 0.95047, 1.0, 1.08883
                x = XYZ[...,0]/Xn; y = XYZ[...,1]/Yn; z = XYZ[...,2]/Zn
                eps, k = 216/24389, 24389/27
                def f(t): 
                    return torch.where(t > eps, t.pow(1/3), (k*t + 16)/116)
                fx, fy, fz = f(x), f(y), f(z)
                L = 116*fy - 16
                return L

            pred_L = xyz_to_lab(rgb_to_xyz_srgb_d65(rgb_pred.clamp(0,1)))
            gt_L   = xyz_to_lab(rgb_to_xyz_srgb_d65(rgb_gt.clamp(0,1)))
            luma = pred_L[...,None] < gt_L[...,None] # ablation: + 1.0
  
            # dilate 
            dilate = (1 - dilated_mask).bool()
            
            # combine
            combined = (luma | dilate)
            
            # distractor 
            distractor = (1 - mask).bool()

            ready    = distractor & combined # ablation: distractor & luma
            rgb_gt   = rgb_gt   * ready
            rgb_pred = rgb_pred * ready

            # Save with meaningful paths and step numbering
            if self.global_step % 5000 == 0:
                out_dir = self.tracking.output_dir
                masked_dir = os.path.join(out_dir, "training_images", "masked")
                os.makedirs(masked_dir, exist_ok=True)
                gt_masked_path = os.path.join(masked_dir, f"gt_masked_step_{self.global_step:06d}.png")
                pred_masked_path = os.path.join(masked_dir, f"pred_masked_step_{self.global_step:06d}.png")
                torchvision.utils.save_image(
                    rgb_gt[0].permute(2, 0, 1), gt_masked_path
                )
                torchvision.utils.save_image(
                    rgb_pred[0].permute(2, 0, 1), pred_masked_path
                ) 
                
                out_dir = os.path.join(self.tracking.output_dir, "trainig_masks")
                os.makedirs(out_dir, exist_ok=True)
                
                dilate_np     = (dilate    .detach().to(torch.uint8) * 255).cpu().numpy().squeeze()
                luma_np       = (luma      .detach().to(torch.uint8) * 255).cpu().numpy().squeeze()
                combined_np   = (combined  .detach().to(torch.uint8) * 255).cpu().numpy().squeeze()
                distractor_np = (distractor.detach().to(torch.uint8) * 255).cpu().numpy().squeeze()
                final_np      = (ready     .detach().to(torch.uint8) * 255).cpu().numpy().squeeze()
                
                Image.fromarray(dilate_np,     mode="L").save(os.path.join(out_dir, f"dilate_{self.global_step}.png"))
                Image.fromarray(luma_np,       mode="L").save(os.path.join(out_dir, f"luma_{self.global_step}.png"))
                Image.fromarray(combined_np,   mode="L").save(os.path.join(out_dir, f"combined_{self.global_step}.png"))
                Image.fromarray(distractor_np, mode="L").save(os.path.join(out_dir, f"distractor_{self.global_step}.png"))
                Image.fromarray(final_np,      mode="L").save(os.path.join(out_dir, f"final_{self.global_step}.png"))


        if self.use_customized_mask:
            customized_mask = self.create_customized_mask(rgb_gt.shape, self.customized_mask_dir, self.device)
            rgb_gt = rgb_gt * customized_mask
            rgb_pred = rgb_pred * customized_mask
            
        # L1 loss
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                loss_l2 = torch.nn.functional.mse_loss(outputs["pred_rgb"], rgb_gt)
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_rgb_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_rgb_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity:
            with torch.cuda.nvtx.range(f"loss-opacity"):
                loss_opacity = torch.abs(self.model.get_density()).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale:
            with torch.cuda.nvtx.range(f"loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale

        # Total loss
        loss = lambda_l1*loss_l1 + lambda_ssim * loss_ssim + lambda_opacity * loss_opacity + lambda_scale * loss_scale
        return dict(total_loss=loss, l1_loss=lambda_l1 * loss_l1, l2_loss=lambda_l2 * loss_l2, ssim_loss=lambda_ssim * loss_ssim, opacity_loss=lambda_opacity * loss_opacity, scale_loss=lambda_scale * loss_scale)

    @torch.cuda.nvtx.range("log_validation_iter")
    def log_validation_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single validation iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        logger.log_progress(
            task_name="Validation",
            advance=1,
            iteration=f"{str(iteration)}",
            psnr=batch_metrics["psnr"],
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_validation_pass")
    def log_validation_pass(self, metrics: dict[str, Any]) -> None:
        """Log information after a single validation pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if "img_pred" in metrics:
            writer.add_images("image/pred/val", torch.stack(metrics["img_pred"]), global_step, dataformats="NHWC")
        if "img_gt" in metrics:
            writer.add_images("image/gt", torch.stack(metrics["img_gt"]), global_step, dataformats="NHWC")
        if "img_hit_counts" in metrics:
            writer.add_images(
                "image/hit_counts/val", torch.stack(metrics["img_hit_counts"]), global_step, dataformats="NHWC"
            )
        if "img_pred_dist" in metrics:
            writer.add_images("image/dist/val", torch.stack(metrics["img_pred_dist"]), global_step, dataformats="NHWC")
        if "img_pred_opacity" in metrics:
            writer.add_images(
                "image/opacity/val", torch.stack(metrics["img_pred_opacity"]), global_step, dataformats="NHWC"
            )

        mean_timings = {}
        if "timings" in metrics:
            for time_key in metrics["timings"]:
                mean_timings[time_key] = np.mean(metrics["timings"][time_key])
                writer.add_scalar("time/" + time_key + "/val", mean_timings[time_key], global_step)

        writer.add_scalar("num_particles/val", self.model.num_gaussians, self.global_step)

        mean_psnr = np.mean(metrics["psnr"])
        writer.add_scalar("psnr/val", mean_psnr, global_step)
        writer.add_scalar("ssim/val", np.mean(metrics["ssim"]), global_step)
        writer.add_scalar("lpips/val", np.mean(metrics["lpips"]), global_step)
        writer.add_scalar("hits/min/val", np.mean(metrics["hits_min"]), global_step)
        writer.add_scalar("hits/max/val", np.mean(metrics["hits_max"]), global_step)
        writer.add_scalar("hits/mean/val", np.mean(metrics["hits_mean"]), global_step)

        loss = np.mean(metrics["losses"]["total_loss"])
        writer.add_scalar("loss/total/val", loss, global_step)
        if self.conf.loss.use_l1:
            l1_loss = np.mean(metrics["losses"]["l1_loss"])
            writer.add_scalar("loss/l1/val", l1_loss, global_step)
        if self.conf.loss.use_l2:
            l2_loss = np.mean(metrics["losses"]["l2_loss"])
            writer.add_scalar("loss/l2/val", l2_loss, global_step)
        if self.conf.loss.use_ssim:
            ssim_loss = np.mean(metrics["losses"]["ssim_loss"])
            writer.add_scalar("loss/ssim/val", ssim_loss, global_step)

        table = {k: np.mean(v) for k, v in metrics.items() if k in ("psnr", "ssim", "lpips")}
        for time_key in mean_timings:
            table[time_key] = f"{'{:.2f}'.format(mean_timings[time_key])}" + " ms/it"
        logger.log_table(f"📊 Validation Metrics - Step {global_step}", record=table)

    @torch.cuda.nvtx.range(f"log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single training iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            loss = np.mean(batch_metrics["losses"]["total_loss"])
            writer.add_scalar("loss/total/train", loss, global_step)
            if self.conf.loss.use_l1:
                l1_loss = np.mean(batch_metrics["losses"]["l1_loss"])
                writer.add_scalar("loss/l1/train", l1_loss, global_step)
            if self.conf.loss.use_l2:
                l2_loss = np.mean(batch_metrics["losses"]["l2_loss"])
                writer.add_scalar("loss/l2/train", l2_loss, global_step)
            if self.conf.loss.use_ssim:
                ssim_loss = np.mean(batch_metrics["losses"]["ssim_loss"])
                writer.add_scalar("loss/ssim/train", ssim_loss, global_step)
            if self.conf.loss.use_opacity:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar("loss/opacity/train", opacity_loss, global_step)
            if self.conf.loss.use_scale:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("loss/scale/train", scale_loss, global_step)
            if "psnr" in batch_metrics:
                writer.add_scalar("psnr/train", batch_metrics["psnr"], self.global_step)
            if "ssim" in batch_metrics:
                writer.add_scalar("ssim/train", batch_metrics["ssim"], self.global_step)
            if "lpips" in batch_metrics:
                writer.add_scalar("lpips/train", batch_metrics["lpips"], self.global_step)
            if "hits_mean" in batch_metrics:
                writer.add_scalar("hits/mean/train", batch_metrics["hits_mean"], self.global_step)
            if "hits_std" in batch_metrics:
                writer.add_scalar("hits/std/train", batch_metrics["hits_std"], self.global_step)
            if "hits_min" in batch_metrics:
                writer.add_scalar("hits/min/train", batch_metrics["hits_min"], self.global_step)
            if "hits_max" in batch_metrics:
                writer.add_scalar("hits/max/train", batch_metrics["hits_max"], self.global_step)

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        "time/" + time_key + "/train", batch_metrics["timings"][time_key], self.global_step
                    )

            writer.add_scalar("num_particles/train", self.model.num_gaussians, self.global_step)
            writer.add_scalar("train/num_GS", self.model.num_gaussians, self.global_step)

            # # NOTE: hack to easily compare with 3DGS
            # writer.add_scalar("train_loss_patches/total_loss", loss, global_step)
            # writer.add_scalar("gaussians/count", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{str(self.global_step)}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range(f"log_training_pass")
    def log_training_pass(self, metrics):
        """Log information after a single training pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        pass

    @torch.cuda.nvtx.range(f"on_training_end")
    def on_training_end(self):
        """Callback that prompts at the end of training."""
        conf = self.conf
        out_dir = self.tracking.output_dir

        # Export the mixture-of-3d-gaussians in mogt file
        logger.log_rule("Exporting Models")
        if conf.export_ingp.enabled:
            ingp_path = conf.export_ingp.path if conf.export_ingp.path else os.path.join(out_dir, "export_last.ingp")
            exporter = INGPExporter()
            exporter.export(self.model, Path(ingp_path), dataset=self.train_dataset, conf=conf, force_half=conf.export_ingp.force_half)
        if conf.export_ply.enabled:
            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            exporter = PLYExporter()
            exporter.export(self.model, Path(ply_path), dataset=self.train_dataset, conf=conf)
        if conf.export_usdz.enabled:
            usdz_path = conf.export_usdz.path if conf.export_usdz.path else os.path.join(out_dir, "export_last.usdz")
            exporter = USDZExporter()
            exporter.export(self.model, Path(usdz_path), dataset=self.train_dataset, conf=conf)

        # Evaluate on test set
        if conf.test_last:
            logger.log_rule("Evaluation on Test Set")

            self.teardown_dataloaders()
            self.save_checkpoint(last_checkpoint=True)

            # Renderer test split
            renderer = Renderer.from_preloaded_model(
                model=self.model,
                out_dir=out_dir,
                path=conf.path,
                save_gt=False,
                writer=self.tracking.writer,
                global_step=self.global_step,
                compute_extra_metrics=conf.compute_extra_metrics,
            )
            renderer.render_all()

    @torch.cuda.nvtx.range(f"save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.
        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step
        """
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def render_gui(self, scene_updated):
        """Render & refresh a single frame for the gui"""
        gui = self.gui
        if gui is not None:
            import polyscope as ps

            if gui.live_update:
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_cloud_viz()
                gui.update_render_view_viz()

            ps.frame_tick()
            while not gui.viz_do_train:
                ps.frame_tick()

            if ps.window_requests_close():
                logger.warning("Terminating training from GUI window is not supported. Please terminate it from the terminal.")

    def save_training_images(self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor], global_step: int):
        """Save GT and rendered images during training at specified intervals."""
        import os
        from PIL import Image
        import torch
        import numpy as np
        
        # Create directories for saving images
        out_dir = self.tracking.output_dir
        gt_dir = os.path.join(out_dir, "training_images", "gt")
        render_dir = os.path.join(out_dir, "training_images", "renders")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)
        
        # Get GT and predicted images
        rgb_gt = gpu_batch.rgb_gt[0].detach().cpu()  # Take first image from batch
        rgb_pred = outputs["pred_rgb"][0].detach().cpu()  # Take first image from batch
        
        # Clip values to [0, 1] range
        rgb_gt = torch.clamp(rgb_gt, 0, 1)
        rgb_pred = torch.clamp(rgb_pred, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        gt_np = (rgb_gt.numpy() * 255).astype(np.uint8)
        pred_np = (rgb_pred.numpy() * 255).astype(np.uint8)
        
        # Save images
        gt_path = os.path.join(gt_dir, f"gt_step_{global_step:06d}.png")
        render_path = os.path.join(render_dir, f"render_step_{global_step:06d}.png")
        
        Image.fromarray(gt_np).save(gt_path)
        Image.fromarray(pred_np).save(render_path)
        
        # Helper function to safely convert tensor to image array
        def tensor_to_image_array(tensor, normalize_range=None):
            """Convert tensor to numpy array suitable for PIL Image."""
            # Move to CPU and detach
            tensor = tensor.detach().cpu()
            
            # Handle different tensor shapes
            if tensor.dim() == 4:  # (1, H, W, C) or (1, C, H, W)
                tensor = tensor.squeeze(0)  # Remove batch dimension
            elif tensor.dim() == 3 and tensor.shape[0] == 1:  # (1, H, W)
                tensor = tensor.squeeze(0)  # Remove first dimension if it's 1
            
            # Normalize if specified
            if normalize_range is not None:
                min_val, max_val = normalize_range
                tensor = torch.clamp(tensor / max_val, 0, 1)
            else:
                tensor = torch.clamp(tensor, 0, 1)
            
            # Convert to numpy
            np_array = tensor.numpy()
            
            # Handle grayscale to RGB conversion
            if np_array.ndim == 2:  # Grayscale (H, W)
                np_array = np.stack([np_array, np_array, np_array], axis=-1)
            elif np_array.ndim == 3 and np_array.shape[-1] == 1:  # (H, W, 1)
                np_array = np.repeat(np_array, 3, axis=-1)
            
            # Scale to [0, 255] and convert to uint8
            return (np_array * 255).astype(np.uint8)
        
        # Optional: Save additional outputs if available
        # if "pred_dist" in outputs:
        #     try:
        #         dist_dir = os.path.join(out_dir, "training_images", "distance")
        #         os.makedirs(dist_dir, exist_ok=True)
                
        #         pred_dist = outputs["pred_dist"][0]
        #         dist_np = tensor_to_image_array(pred_dist, normalize_range=(0, 100))
                
        #         dist_path = os.path.join(dist_dir, f"distance_step_{global_step:06d}.png")
        #         Image.fromarray(dist_np).save(dist_path)
        #     except Exception as e:
        #         logger.warning(f"Failed to save distance image: {e}")
        
        # if "pred_opacity" in outputs:
        #     try:
        #         opacity_dir = os.path.join(out_dir, "training_images", "opacity")
        #         os.makedirs(opacity_dir, exist_ok=True)
                
        #         pred_opacity = outputs["pred_opacity"][0]
        #         opacity_np = tensor_to_image_array(pred_opacity)
                
        #         opacity_path = os.path.join(opacity_dir, f"opacity_step_{global_step:06d}.png")
        #         Image.fromarray(opacity_np).save(opacity_path)
        #     except Exception as e:
        #         logger.warning(f"Failed to save opacity image: {e}")
        
        # Save hit counts visualization if available
        # if "hits_count" in outputs:
        #     try:
        #         hits_dir = os.path.join(out_dir, "training_images", "hits")
        #         os.makedirs(hits_dir, exist_ok=True)
                
        #         hits_count = outputs["hits_count"][0]
        #         hits_np = tensor_to_image_array(hits_count, normalize_range=(0, 50))  # Adjust max hits as needed
                
        #         hits_path = os.path.join(hits_dir, f"hits_step_{global_step:06d}.png")
        #         Image.fromarray(hits_np).save(hits_path)
        #     except Exception as e:
        #         logger.warning(f"Failed to save hits image: {e}")

        # Compute and save error map
        try:
            error_dir = os.path.join(out_dir, "training_images", "error_maps")
            os.makedirs(error_dir, exist_ok=True)
            
            # Ensure both images have the same shape
            gt_for_error = rgb_gt.numpy()
            pred_for_error = rgb_pred.numpy()
            
            # Compute error map (L1 distance)
            error_map = np.abs(gt_for_error - pred_for_error)
                    
            error_rgb = (error_map * 255).astype(np.uint8)
                    
            error_path = os.path.join(error_dir, f"error_step_{global_step:06d}.png")
            Image.fromarray(error_rgb).save(error_path)
            
        except Exception as e:
            logger.warning(f"Failed to save error map: {e}")
        
        logger.info(f"💾 Saved training images at step {global_step} to {out_dir}/training_images/")

    # Add these methods to your Trainer3DGRUT class:

    def save_gradient_analysis(self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor], global_step: int):
        """Save gradient norm analysis to understand why large Gaussians aren't being split."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        try:
            out_dir = self.tracking.output_dir
            grad_dir = os.path.join(out_dir, "gradient_analysis")
            os.makedirs(grad_dir, exist_ok=True)
            
            # Get the strategy to access gradient information
            strategy = self.strategy
            
            # Check if we're using GS strategy which has gradient accumulation
            if not hasattr(strategy, 'densify_grad_norm_accum') or strategy.densify_grad_norm_accum is None:
                logger.warning("Gradient norm not available for current strategy")
                return
            
            # Log sizes for debugging
            logger.info(f"Gradient accumulator size: {strategy.densify_grad_norm_accum.shape}")
            logger.info(f"Model num gaussians: {self.model.num_gaussians}")
            
            # Calculate accumulated gradient norms (same as in densify_gaussians method)
            with torch.no_grad():
                grad_norms = strategy.densify_grad_norm_accum / strategy.densify_grad_norm_denom
                grad_norms[grad_norms.isnan()] = 0.0
                grad_norms = grad_norms.squeeze()
            
            # Clone to CPU for visualization
            grad_norms_cpu = grad_norms.clone().detach().cpu()
            
            # Get the split threshold
            split_threshold = strategy.split_grad_threshold
            clone_threshold = strategy.clone_grad_threshold

            if 'mog_visibility' in outputs:
                visibility = outputs['mog_visibility'].detach().cpu()
                
                # IMPORTANT: Squeeze visibility IMMEDIATELY to ensure it's 1D
                while visibility.dim() > 1 and 1 in visibility.shape:
                    visibility = visibility.squeeze()
                
                # If still not 1D, flatten it
                if visibility.dim() > 1:
                    visibility = visibility.flatten()
                
                logger.info(f"Visibility shape after processing: {visibility.shape}, grad_norms shape: {grad_norms_cpu.shape}")
                
                # Handle size mismatch
                if len(visibility) != len(grad_norms_cpu):
                    logger.warning(f"Visibility size {len(visibility)} doesn't match grad_norms size {len(grad_norms_cpu)}")
                    # Use the minimum size
                    min_size = min(len(visibility), len(grad_norms_cpu))
                    visibility = visibility[:min_size]
                    grad_norms_cpu = grad_norms_cpu[:min_size]
                
                visibility = visibility.bool()
            else:
                visibility = torch.ones(len(grad_norms_cpu), dtype=torch.bool)
            
            # Create comprehensive visualizations
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Gradient Norm Analysis - Step {global_step}', fontsize=16)
            
            # 1. Histogram of gradient norms
            ax = axes[0, 0]
            visible_grads = grad_norms_cpu[visibility]
            if len(visible_grads) > 0:
                # Filter out zeros for better visualization
                non_zero_grads = visible_grads[visible_grads > 0]
                if len(non_zero_grads) > 0:
                    ax.hist(non_zero_grads.numpy(), bins=100, alpha=0.7, color='blue', edgecolor='black')
                    ax.axvline(split_threshold, color='red', linestyle='--', linewidth=2, label=f'Split Threshold: {split_threshold}')
                    ax.axvline(clone_threshold, color='orange', linestyle='--', linewidth=2, label=f'Clone Threshold: {clone_threshold}')
                    ax.set_xlabel('Gradient Norm')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Gradient Norms (Visible Gaussians, non-zero)')
                    ax.legend()
                    ax.set_yscale('log')
                    ax.set_xscale('log')
            
            # 2. Gradient norms vs Gaussian scale
            ax = axes[0, 1]
            try:
                scales = self.model.get_scale().detach().cpu()
                # Ensure scales match our gradient norms size
                scales = scales[:len(grad_norms_cpu)]
                max_scale = scales.max(dim=1)[0]  # Max scale across dimensions
                
                non_zero_mask = (grad_norms_cpu > 0) & visibility
                
                if non_zero_mask.any():
                    scatter = ax.scatter(max_scale[non_zero_mask], grad_norms_cpu[non_zero_mask], 
                                    alpha=0.5, c=grad_norms_cpu[non_zero_mask], cmap='viridis', s=1)
                    ax.axhline(split_threshold, color='red', linestyle='--', linewidth=2, label=f'Split Threshold: {split_threshold}')
                    ax.axhline(clone_threshold, color='orange', linestyle='--', linewidth=2, label=f'Clone Threshold: {clone_threshold}')
                    
                    # Add scene extent threshold line for scale
                    scene_extent = self.scene_extent
                    size_threshold = strategy.relative_size_threshold * scene_extent
                    ax.axvline(size_threshold, color='green', linestyle='--', linewidth=2, label=f'Size Threshold: {size_threshold:.4f}')
                    
                    ax.set_xlabel('Max Gaussian Scale')
                    ax.set_ylabel('Gradient Norm')
                    ax.set_title('Gradient Norm vs Gaussian Scale')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.legend()
                    plt.colorbar(scatter, ax=ax, label='Gradient Norm')
            except Exception as e:
                logger.warning(f"Failed to create scale plot: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            
            # 3. Gradient norms vs opacity
            ax = axes[0, 2]
            try:
                opacities = self.model.get_density().detach().cpu().squeeze()
                # Ensure opacities match our gradient norms size
                opacities = opacities[:len(grad_norms_cpu)]
                
                non_zero_mask = (grad_norms_cpu > 0) & visibility
                
                if non_zero_mask.any():
                    scatter = ax.scatter(opacities[non_zero_mask], grad_norms_cpu[non_zero_mask], 
                                    alpha=0.5, c=grad_norms_cpu[non_zero_mask], cmap='viridis', s=1)
                    ax.axhline(split_threshold, color='red', linestyle='--', linewidth=2, label=f'Split Threshold: {split_threshold}')
                    ax.axhline(clone_threshold, color='orange', linestyle='--', linewidth=2, label=f'Clone Threshold: {clone_threshold}')
                    ax.axvline(strategy.prune_density_threshold, color='purple', linestyle='--', linewidth=2, label=f'Prune Threshold: {strategy.prune_density_threshold}')
                    ax.set_xlabel('Gaussian Opacity')
                    ax.set_ylabel('Gradient Norm')
                    ax.set_title('Gradient Norm vs Opacity')
                    ax.set_yscale('log')
                    ax.legend()
                    plt.colorbar(scatter, ax=ax, label='Gradient Norm')
            except Exception as e:
                logger.warning(f"Failed to create opacity plot: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            
            # 4. Why large Gaussians aren't splitting - detailed analysis
            ax = axes[1, 0]
            try:
                scene_extent = self.scene_extent
                size_threshold = strategy.relative_size_threshold * scene_extent
                
                # Identify different categories of Gaussians
                large_mask = max_scale > size_threshold
                high_grad_mask = grad_norms_cpu >= split_threshold
                
                # Categories
                categories = []
                labels = []
                colors = []
                
                if (large_mask & high_grad_mask & visibility).any():
                    mask = large_mask & high_grad_mask & visibility
                    categories.append((max_scale[mask], grad_norms_cpu[mask]))
                    labels.append('Large + High Grad (WILL SPLIT)')
                    colors.append('red')
                
                if (large_mask & ~high_grad_mask & visibility).any():
                    mask = large_mask & ~high_grad_mask & visibility
                    categories.append((max_scale[mask], grad_norms_cpu[mask]))
                    labels.append('Large + Low Grad (NO SPLIT)')
                    colors.append('blue')
                
                if (~large_mask & (grad_norms_cpu >= clone_threshold) & visibility).any():
                    mask = ~large_mask & (grad_norms_cpu >= clone_threshold) & visibility
                    categories.append((max_scale[mask], grad_norms_cpu[mask]))
                    labels.append('Small + High Grad (WILL CLONE)')
                    colors.append('green')
                
                for i, (scales_cat, grads_cat) in enumerate(categories):
                    ax.scatter(scales_cat, grads_cat, c=colors[i], label=labels[i], alpha=0.6, s=20)
                
                ax.axhline(split_threshold, color='red', linestyle='--', linewidth=2, label=f'Split Threshold')
                ax.axhline(clone_threshold, color='orange', linestyle='--', linewidth=2, label=f'Clone Threshold')
                ax.axvline(size_threshold, color='green', linestyle='--', linewidth=2, label=f'Size Threshold')
                
                ax.set_xlabel('Gaussian Scale')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Densification Decision Map')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Failed to create decision map: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            
            # 5. Accumulation statistics
            ax = axes[1, 1]
            try:
                denom_values = strategy.densify_grad_norm_denom.squeeze().cpu()
                denom_values = denom_values[:len(grad_norms_cpu)]  # Match size
                
                if len(denom_values) > 0:
                    ax.hist(denom_values.numpy(), bins=50, alpha=0.7, color='purple', edgecolor='black')
                    ax.set_xlabel('Accumulation Count')
                    ax.set_ylabel('Number of Gaussians')
                    ax.set_title('Gradient Accumulation Counts\n(How many views contributed to each Gaussian)')
                    ax.set_yscale('log')
                    
                    # Add statistics
                    mean_accum = denom_values.mean().item()
                    max_accum = denom_values.max().item()
                    ax.text(0.7, 0.9, f'Mean: {mean_accum:.1f}\nMax: {max_accum}', 
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                logger.warning(f"Failed to create accumulation plot: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            
            # 6. Detailed statistics
            ax = axes[1, 2]
            ax.axis('off')
            
            stats_text = f"Gradient Norm Analysis (Step {global_step})\n"
            stats_text += "=" * 40 + "\n\n"
            
            try:
                if len(visible_grads) > 0:
                    stats_text += f"Total Gaussians: {self.model.num_gaussians:,}\n"
                    stats_text += f"Gradient buffer size: {len(grad_norms_cpu):,}\n"
                    stats_text += f"Visible Gaussians: {visibility.sum().item():,}\n"
                    stats_text += f"With Non-Zero Gradient: {(grad_norms_cpu > 0).sum().item():,}\n\n"
                    
                    stats_text += f"Thresholds:\n"
                    stats_text += f"  Split: {split_threshold}\n"
                    stats_text += f"  Clone: {clone_threshold}\n"
                    stats_text += f"  Size: {size_threshold:.4f}\n"
                    stats_text += f"  Prune Opacity: {strategy.prune_density_threshold}\n\n"
                    
                    non_zero_grads = visible_grads[visible_grads > 0]
                    if len(non_zero_grads) > 0:
                        stats_text += "Gradient Norm Statistics (non-zero):\n"
                        stats_text += f"  Mean: {non_zero_grads.mean():.6f}\n"
                        stats_text += f"  Std: {non_zero_grads.std():.6f}\n"
                        stats_text += f"  Min: {non_zero_grads.min():.6f}\n"
                        stats_text += f"  Max: {non_zero_grads.max():.6f}\n"
                        stats_text += f"  Median: {non_zero_grads.median():.6f}\n\n"
                    
                    # Densification action predictions
                    if 'max_scale' in locals():
                        will_split = (large_mask & high_grad_mask).sum().item()
                        will_clone = (~large_mask & (grad_norms_cpu >= clone_threshold)).sum().item()
                        
                        stats_text += f"Predicted Actions:\n"
                        stats_text += f"  Will Split: {will_split:,}\n"
                        stats_text += f"  Will Clone: {will_clone:,}\n\n"
                        
                        # Large Gaussian analysis
                        n_large = large_mask.sum().item()
                        if n_large > 0:
                            large_grads = grad_norms_cpu[large_mask]
                            large_high_grad = (large_grads >= split_threshold).sum().item()
                            
                            stats_text += f"Large Gaussians (scale>{size_threshold:.4f}):\n"
                            stats_text += f"  Count: {n_large:,}\n"
                            stats_text += f"  With high gradient: {large_high_grad:,} ({100*large_high_grad/n_large:.1f}%)\n"
                            if (large_grads > 0).any():
                                stats_text += f"  Mean gradient: {large_grads[large_grads > 0].mean():.6f}\n"
            except Exception as e:
                stats_text += f"\nError computing statistics: {str(e)}\n"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            save_path = os.path.join(grad_dir, f'gradient_analysis_step_{global_step:06d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"💾 Saved gradient norm analysis to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save gradient analysis: {e}")
            import traceback
            traceback.print_exc()


    @torch.cuda.nvtx.range(f"run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        """Runs a single train epoch over the dataset."""
        global_step = self.global_step
        model = self.model

        metrics = []
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "build_as": CudaTimer(enabled=self.conf.enable_frame_timings),
        }

        for iter, batch in enumerate(self.train_dataloader):

            # Check if we have reached the maximum number of iterations
            if self.global_step >= conf.n_iterations:
                return

            # Access the GPU-cache batch data
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)

            # Perform validation if required
            is_time_to_validate = (global_step > 0 or conf.validate_first) and (global_step % self.val_frequency == 0)
            if is_time_to_validate:
                self.run_validation_pass(conf)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
                profilers["inference"].start()
                outputs = model(gpu_batch, train=True, frame_id=global_step)
                profilers["inference"].end()

            # Compute the losses of a single batch
            with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
                batch_losses = self.get_losses(gpu_batch, outputs)

            # Backward strategy step
            with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
                self.strategy.pre_backward(step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset, batch=gpu_batch, writer=self.tracking.writer)

            # Back-propagate the gradients and update the parameters
            with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
                profilers["backward"].start()
                batch_losses["total_loss"].backward()
                profilers["backward"].end()

            # Post backward strategy step
            with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
                scene_updated = self.strategy.post_backward(
                    step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset, batch=gpu_batch, writer=self.tracking.writer
                )

            # Optimizer step
            with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
                if isinstance(model.optimizer, SelectiveAdam):
                    assert outputs['mog_visibility'].shape == model.density.shape, f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {model.density.shape}"
                    model.optimizer.step(outputs['mog_visibility'])
                else:
                    model.optimizer.step()
                model.optimizer.zero_grad()

            # Scheduler step
            with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
                model.scheduler_step(global_step)

            # Post backward strategy step
            with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
                scene_updated = self.strategy.post_optimizer_step(
                    step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset, batch=gpu_batch, writer=self.tracking.writer
                )

            # Update the SH if required
            if self.model.progressive_training and check_step_condition(global_step, 0, 1e6, self.model.feature_dim_increase_interval):
                self.model.increase_num_active_features()

            # Update the BVH if required
            if scene_updated or (
                conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0
            ):
                with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                    profilers["build_as"].start()
                    model.build_acc(rebuild=True)
                    profilers["build_as"].end()

            # Increment the global step
            self.global_step += 1
            global_step = self.global_step

            # Compute metrics
            batch_metrics = self.get_metrics(
                gpu_batch, outputs, batch_losses, profilers, split="training", iteration=iter
            )

            if "forward_render" in model.renderer.timings:
                batch_metrics["timings"]["forward_render_cuda"] = model.renderer.timings["forward_render"]
            if "backward_render" in model.renderer.timings:
                batch_metrics["timings"]["backward_render_cuda"] = model.renderer.timings["backward_render"]
            metrics.append(batch_metrics)

            # !!! Below global step has been incremented !!!
            with torch.cuda.nvtx.range(f"train_{global_step-1}_log_iter"):
                self.log_training_iter(gpu_batch, outputs, batch_metrics, iter)

            with torch.cuda.nvtx.range(f"train_{global_step-1}_save_images"):
                if global_step % 5000 == 0:
                    self.save_training_images(gpu_batch, outputs, global_step)
                    #self.save_gradient_analysis(gpu_batch, outputs, global_step)

            with torch.cuda.nvtx.range(f"train_{global_step-1}_save_ckpt"):
                if global_step in conf.checkpoint.iterations:
                    self.save_checkpoint()

            with torch.cuda.nvtx.range(f"train_{global_step-1}_update_gui"):
                self.render_gui(scene_updated)  # Updating the GUI

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range(f"run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        """Runs a single validation epoch over the dataset.
        Returns:
             dictionary of metrics computed and aggregated over validation set.
        """

        profilers = {
            "inference": CudaTimer(),
        }
        metrics = []
        logger.info(f"Step {self.global_step} -- Running validation..")
        logger.start_progress(task_name="Validation", total_steps=len(self.val_dataloader), color="medium_purple3")

        for val_iteration, batch_idx in enumerate(self.val_dataloader):

            # Access the GPU-cache batch data
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(f"train.validation_step_{self.global_step}"):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                profilers["inference"].end()
                batch_losses = self.get_losses(gpu_batch, outputs)
                batch_metrics = self.get_metrics(
                    gpu_batch, outputs, batch_losses, profilers, split="validation", iteration=val_iteration
                )

                self.log_validation_iter(gpu_batch, outputs, batch_metrics, iteration=val_iteration)
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        """
        Converts list of dicts -> dict of lists.
        Supports flattening of up to 2 levels of dict hierarchies
        """
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = defaultdict(list) if k not in flat_dict else flat_dict[k]
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        """Initiate training logic for n_epochs.
        Training and validation are controlled by the config.
        """
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")

        # Training loop
        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")

        for epoch_idx in range(self.n_epochs):
            self.run_train_pass(conf)

        logger.end_progress(task_name="Training")

        # Report training statistics
        stats = logger.finished_tasks["Training"]
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{stats['elapsed']:.2f} s",
            iteration_speed=f"{self.global_step / stats['elapsed']:.2f} it/s",
        )
        logger.log_table(f"🎊 Training Statistics", record=table)

        # Perform testing
        self.on_training_end()
        logger.info(f"🥳 Training Complete.")

        # Updating the GUI
        if self.gui is not None:
            self.gui.training_done = True
            logger.info(f"🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
