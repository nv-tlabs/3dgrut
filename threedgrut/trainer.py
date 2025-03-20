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

import numpy as np

import torch
import torch.utils.data

from addict import Dict
from omegaconf import DictConfig, OmegaConf

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset, Batch
from threedgrut.datasets.utils import MultiEpochsDataLoader, DEFAULT_DEVICE
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.gui import GUI
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import CudaTimer
from threedgrut.utils.misc import jet_map, create_summary_writer


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

    def init_dataloaders(self, conf: DictConfig):
        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_dataloader = MultiEpochsDataLoader(
            train_dataset,
            num_workers=conf.num_workers,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=conf.num_workers,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
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
        match self.conf.model.strategy:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy

                self.strategy = GSStrategy(conf, self.model)
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.model.strategy}")

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
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
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

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        # Move losses to cpu once
        metrics["losses"] = {k: v.item() for k, v in losses.items()}

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
                metrics["img_gt"] = gpu_batch.rgb_gt[-1].clip(0, 1.0)
                metrics["img_pred"] = outputs["pred_rgb"][-1].clip(0, 1.0)
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

        # Total loss
        loss = lambda_l1 * loss_l1 + lambda_l2 * loss_l2 + lambda_ssim * loss_ssim
        return dict(total_loss=loss, l1_loss=loss_l1, l2_loss=loss_l2, ssim_loss=loss_ssim)

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
        writer.add_scalar("train_loss_patches/total_loss", loss, global_step)  # hack to compare with 3DGS
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
            self.model.export_ingp(ingp_path, conf.export_ingp.force_half)
        if conf.export_ply.enabled:
            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            self.model.export_ply(ply_path)

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
                self.strategy.pre_backward(step=global_step)

            # Back-propagate the gradients and update the parameters
            with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
                profilers["backward"].start()
                batch_losses["total_loss"].backward()
                profilers["backward"].end()

            # Update densification buffer:
            if global_step < conf.model.densify.end_iteration:
                with torch.cuda.nvtx.range(f"train_{global_step}_grad_buffer"):
                    self.strategy.update_gradient_buffer(sensor_position=gpu_batch.T_to_world[0, :3, 3])

            # Clamp density
            if global_step > 0 and conf.model.density_activation == "none":
                with torch.cuda.nvtx.range(f"train_{global_step}_clamp_density"):
                    model.clamp_density()

            # Make a scheduler step
            with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
                model.scheduler_step(global_step)

            # Post backward strategy step
            with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
                scene_updated = self.strategy.post_backward(
                    step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset
                )

            # Optimizer step
            with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
                model.optimizer.step()
                model.optimizer.zero_grad()

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
