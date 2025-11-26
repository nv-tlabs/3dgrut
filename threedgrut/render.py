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
from pathlib import Path
import cv2

import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer


class Renderer:
    def __init__(
        self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None, compute_extra_metrics=True
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.use_customized_mask = True
        self.customized_mask_dir = conf.customized_mask_dir


        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)
        
        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform({
            'num_workers': 8,
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': None,
        })
        
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # # overrides
        # if conf["render"]["method"] == "3dgrt":
        #     conf["render"]["particle_kernel_density_clamping"] = True
        #     conf["render"]["min_transmittance"] = 0.03
        # conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint)
        model.build_acc()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
        )

    @classmethod
    def from_preloaded_model(
        cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None, compute_extra_metrics=False
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
        )
    
    def create_customized_mask(self, image_shape, customized_mask_dir, device=None):
        """Create customized mask for fisheye images to exclude black border & tripod regions.
        
        Args:
            image_shape: Shape of the image tensor (batch, height, width, channels)
            customized_mask_dir (Optional)
        
        Returns:
            torch.Tensor: Binary mask with 1 for valid pixels, 0 for invalid
        """
        batch, height, width, channels = image_shape
        
        try:
            mask_custom = torch.from_numpy(cv2.imread("mask_test.png", cv2.IMREAD_GRAYSCALE).astype(bool)).to(device)
        except:
            mask_custom = torch.from_numpy(cv2.imread("mask_test_4.png", cv2.IMREAD_GRAYSCALE).astype(bool)).to(device)
        mask_custom = (mask_custom).float()
        
        # Expand to match batch and channel dimensions
        mask_custom = mask_custom.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
        mask_custom = mask_custom.repeat(batch, 1, 1, 1)      # [B, H, W, 1]
        return mask_custom

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        
        output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
        os.makedirs(output_path_gt, exist_ok=True)
        
        metrics_dir = os.path.join(self.out_dir, f"ours_{int(self.global_step)}")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_fpath = os.path.join(metrics_dir, "metrics.txt")
        mf = open(metrics_fpath, "w", encoding="utf-8")

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)
            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            if self.use_customized_mask:
                customized_mask = self.create_customized_mask(gpu_batch.rgb_gt.shape, self.customized_mask_dir, gpu_batch.rgb_gt.device)
                if customized_mask.shape[-1] == 1 and gpu_batch.rgb_gt.shape[-1] == 3:
                    customized_mask = customized_mask.repeat(1, 1, 1, 3)
        
                pred_rgb_metric = pred_rgb_full * customized_mask
                rgb_gt_metric = rgb_gt_full * customized_mask
            mask = gpu_batch.mask
            if mask is not None:
                rgb_gt_metric = rgb_gt_metric * (1-mask)
                pred_rgb_metric = pred_rgb_metric * (1-mask)
                
            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            torchvision.utils.save_image(
                rgb_gt_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
            )

            # Compute the loss
            #psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr_single_img = criterions["psnr"](pred_rgb_metric, rgb_gt_metric).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_metric.permute(0, 3, 1, 2),
                    rgb_gt_metric.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_metric.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_metric.permute(0, 3, 1, 2),
                ).item()
            )
            frame_name = f"{iteration:05d}.png"
            mf.write(
                    f"Frame {iteration}, PSNR: {psnr_single_img:.4f}, "
                    f"SSIM: {ssim[-1]:.4f}, LPIPS: {lpips[-1]:.4f}\n"
                    )
            mf.flush()
            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        mf.write("\n# Summary\n")
        mf.write(f"mean_psnr\t{mean_psnr:.6f}\n")
        mf.write(f"std_psnr\t{std_psnr:.6f}\n")
        mf.write(f"mean_ssim\t{mean_ssim:.6f}\n")
        mf.write(f"mean_lpips\t{mean_lpips:.6f}\n")
        if self.conf.render.enable_kernel_timings:
            mf.write(f"mean_inference_time_ms\t{mean_inference_time:.2f}\n")
        mf.close()

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
