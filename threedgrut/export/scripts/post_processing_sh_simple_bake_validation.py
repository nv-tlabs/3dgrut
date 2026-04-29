# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validate one-shot baking of a fixed PPISP transform into Gaussian SH DC coefficients.

The reference is the checkpoint render followed by PPISP at one camera/frame with
vignetting disabled. The baked render is the cloned model whose DC SH terms have
been mutated to encode the camera's mean PPISP transform; it is rendered without
any post-processing and clamped to [0, 1].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torchvision
from ppisp import PPISP
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.export.usd.post_processing_sh_bake import FixedPPISP
from threedgrut.export.usd.post_processing_sh_simple_bake import simple_bake
from threedgrut.render import Renderer
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.logger import logger
from threedgrut.utils.render import apply_post_processing


def _render_reference(reference_model, fixed_ppisp, gpu_batch) -> torch.Tensor:
    with torch.no_grad():
        outputs = reference_model(gpu_batch)
        outputs = apply_post_processing(fixed_ppisp, outputs, gpu_batch, training=True)
        return outputs["pred_rgb"].detach()


def _validate_arguments(args, ppisp: PPISP) -> None:
    assert isinstance(ppisp, PPISP), f"Expected PPISP module, got {type(ppisp).__name__}."
    num_frames = int(ppisp.exposure_params.shape[0])
    num_cameras = int(ppisp.crf_params.shape[0])
    assert 0 <= args.frame_id < num_frames, (
        f"frame_id must be in [0, {num_frames - 1}], got {args.frame_id}."
    )
    assert 0 <= args.camera_id < num_cameras, (
        f"camera_id must be in [0, {num_cameras - 1}], got {args.camera_id}."
    )


@torch.no_grad()
def _evaluate(
    reference_model,
    baked_model,
    fixed_ppisp,
    dataset,
    dataloader,
    output_root: Path,
    compute_extra_metrics: bool,
) -> dict:
    criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}
    if compute_extra_metrics:
        criterions |= {
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
        }

    reference_path = output_root / "reference"
    baked_path = output_root / "baked"
    reference_path.mkdir(parents=True, exist_ok=True)
    baked_path.mkdir(parents=True, exist_ok=True)

    psnr_values = []
    ssim_values = []
    lpips_values = []
    cc_psnr_values = []
    cc_ssim_values = []
    cc_lpips_values = []
    inference_time_values = []

    logger.start_progress(task_name="Evaluating simple SH bake", total_steps=len(dataloader), color="orange1")
    for iteration, batch in enumerate(dataloader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)

        reference_rgb = _render_reference(reference_model, fixed_ppisp, gpu_batch)
        baked_outputs = baked_model(gpu_batch)
        baked_rgb = torch.clamp(baked_outputs["pred_rgb"], 0, 1)

        torchvision.utils.save_image(
            reference_rgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
            reference_path / f"{iteration:05d}.png",
        )
        torchvision.utils.save_image(
            baked_rgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
            baked_path / f"{iteration:05d}.png",
        )

        psnr_values.append(criterions["psnr"](baked_rgb, reference_rgb).item())
        if compute_extra_metrics:
            ssim_values.append(
                criterions["ssim"](
                    baked_rgb.permute(0, 3, 1, 2),
                    reference_rgb.permute(0, 3, 1, 2),
                ).item()
            )
            lpips_values.append(
                criterions["lpips"](
                    baked_rgb.clip(0, 1).permute(0, 3, 1, 2),
                    reference_rgb.clip(0, 1).permute(0, 3, 1, 2),
                ).item()
            )

            baked_rgb_cc = color_correct_affine(baked_rgb, reference_rgb)
            cc_psnr_values.append(criterions["psnr"](baked_rgb_cc, reference_rgb).item())
            cc_ssim_values.append(
                criterions["ssim"](
                    baked_rgb_cc.permute(0, 3, 1, 2),
                    reference_rgb.permute(0, 3, 1, 2),
                ).item()
            )
            cc_lpips_values.append(
                criterions["lpips"](
                    baked_rgb_cc.clip(0, 1).permute(0, 3, 1, 2),
                    reference_rgb.clip(0, 1).permute(0, 3, 1, 2),
                ).item()
            )

        if "frame_time_ms" in baked_outputs:
            inference_time_values.append(baked_outputs["frame_time_ms"])

        logger.log_progress(
            task_name="Evaluating simple SH bake",
            advance=1,
            iteration=str(iteration),
            psnr=psnr_values[-1],
        )
    logger.end_progress(task_name="Evaluating simple SH bake")

    metrics = {
        "mean_psnr": float(np.mean(psnr_values)),
        "std_psnr": float(np.std(psnr_values)),
    }
    if compute_extra_metrics:
        metrics |= {
            "mean_ssim": float(np.mean(ssim_values)),
            "mean_lpips": float(np.mean(lpips_values)),
            "mean_cc_psnr": float(np.mean(cc_psnr_values)),
            "mean_cc_ssim": float(np.mean(cc_ssim_values)),
            "mean_cc_lpips": float(np.mean(cc_lpips_values)),
        }
    if inference_time_values:
        metrics["mean_inference_time"] = f"{np.mean(inference_time_values):.2f} ms/frame"

    with open(output_root / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    logger.log_table("Simple SH Bake Validation", record=metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the pretrained checkpoint.")
    parser.add_argument("--path", type=str, default="", help="Path to test data; if not provided taken from ckpt.")
    parser.add_argument("--out-dir", dest="out_dir", required=True, type=str, help="Output path.")
    parser.add_argument("--camera-id", dest="camera_id", default=0, type=int, help="PPISP camera id to bake.")
    parser.add_argument(
        "--frame-id",
        dest="frame_id",
        default=0,
        type=int,
        help="PPISP frame id used to render the reference (the bake itself uses the camera's mean).",
    )
    parser.add_argument(
        "--higher-order",
        dest="higher_order",
        action="store_true",
        help="Also bake the per-Gaussian Jacobian into higher-order SH coefficients (linearization at DC).",
    )
    parser.add_argument(
        "--compute-extra-metrics",
        dest="compute_extra_metrics",
        action="store_false",
        help="If set, extra image metrics will not be computed [True by default].",
    )
    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=args.compute_extra_metrics,
    )
    if renderer.post_processing is None:
        raise ValueError("Checkpoint does not contain PPISP post-processing.")

    _validate_arguments(args, renderer.post_processing)

    fixed_ppisp = FixedPPISP(
        renderer.post_processing,
        args.camera_id,
        args.frame_id,
        "cuda",
        include_vignetting=False,
    ).eval()

    reference_model = renderer.model.eval()
    baked_model = renderer.model.clone().eval()

    suffix = "_ho" if args.higher_order else ""
    output_root = Path(renderer.out_dir) / (
        f"post_processing_sh_simple_bake_ci{args.camera_id}_fi{args.frame_id}{suffix}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.make_train(name=renderer.conf.dataset.type, config=renderer.conf, ray_jitter=None)
    logger.info(
        f"Simple-baking SH for camera_id={args.camera_id} "
        f"(mean exposure/color over its training frames; CRF; no vignetting; higher_order={args.higher_order})"
    )
    exposure_mean, color_mean = simple_bake(
        model=baked_model,
        ppisp=renderer.post_processing,
        train_dataset=train_dataset,
        camera_id=args.camera_id,
        higher_order=args.higher_order,
    )
    baked_model.build_acc()

    logger.info(
        f"Bake done. exposure_mean={exposure_mean:.6f}; color_mean={[float(v) for v in color_mean.tolist()]}"
    )

    _evaluate(
        reference_model=reference_model,
        baked_model=baked_model,
        fixed_ppisp=fixed_ppisp,
        dataset=renderer.dataset,
        dataloader=renderer.dataloader,
        output_root=output_root,
        compute_extra_metrics=args.compute_extra_metrics,
    )


if __name__ == "__main__":
    main()
