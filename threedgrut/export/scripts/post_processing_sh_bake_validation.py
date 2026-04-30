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

"""Validate baking one fixed PPISP transform into Gaussian SH coefficients.

The reference is the checkpoint render followed by PPISP from one camera/frame,
including that camera's chromatic vignetting. The fitted method optimizes only a
cloned model's SH coefficients, with a temporary achromatic vignette applied in
the fitting loss to isolate chromatic vignette effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.render import Renderer
from threedgrut.datasets.utils import configure_dataloader_for_platform
from threedgrut.export.usd.post_processing_sh_bake import (
    MODE_PPISP_BAKE_VIGNETTING_NONE,
    FixedPPISP,
    apply_achromatic_vignetting,
    normalize_ppisp_bake_vignetting_mode,
)
from threedgrut.export.usd.post_processing_sh_simple_bake import simple_bake
from threedgrut.utils.logger import logger
from threedgrut.utils.post_processing_linear_to_srgb import linear_to_srgb
from threedgrut.utils.render import apply_post_processing

BAKE_FLAVOR_FIT = "fit"
BAKE_FLAVOR_SIMPLE = "simple"
BAKE_FLAVOR_SIMPLE_HIGHER_ORDER = "simple-higher-order"
BAKE_FLAVOR_ALL = "all"


def _setShFitParameters(model) -> Iterable[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    fitParameters = []
    for fieldName in ("features_albedo", "features_specular"):
        parameter = getattr(model, fieldName)
        parameter.requires_grad_(True)
        fitParameters.append(parameter)
    return fitParameters


def _renderReference(referenceModel, fixedPpisp, gpuBatch) -> torch.Tensor:
    with torch.no_grad():
        outputs = referenceModel(gpuBatch)
        outputs = apply_post_processing(fixedPpisp, outputs, gpuBatch, training=True)
        return outputs["pred_rgb"].detach()


def _applyAchromaticVignetting(rgb: torch.Tensor, fixedPpisp, gpuBatch, vignettingMode: str) -> torch.Tensor:
    if vignettingMode == MODE_PPISP_BAKE_VIGNETTING_NONE:
        return rgb
    _, height, width, _ = rgb.shape
    return apply_achromatic_vignetting(
        rgb=rgb,
        ppisp=fixedPpisp.ppisp,
        camera_id=fixedPpisp.camera_id,
        pixel_coords=gpuBatch.pixel_coords,
        resolution=(width, height),
    )


def _createTrainDataloader(conf):
    trainDataset = datasets.make_train(name=conf.dataset.type, config=conf, ray_jitter=None)
    dataloaderKwargs = configure_dataloader_for_platform(
        {
            "num_workers": conf.num_workers,
            "batch_size": 1,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True if conf.num_workers > 0 else False,
        }
    )
    trainDataloader = torch.utils.data.DataLoader(trainDataset, **dataloaderKwargs)
    return trainDataset, trainDataloader


def _fitBakedSh(
    referenceModel,
    bakedModel,
    fixedPpisp,
    dataset,
    dataloader,
    fitEpochs: int,
    learningRate: float,
    vignettingMode: str,
) -> None:
    if fitEpochs < 1:
        raise ValueError(f"fitEpochs must be >= 1, got {fitEpochs}.")

    fitParameters = list(_setShFitParameters(bakedModel))
    optimizer = torch.optim.Adam(fitParameters, lr=learningRate)

    totalSteps = fitEpochs * len(dataloader)
    logger.start_progress(task_name="Fitting baked SH", total_steps=totalSteps, color="cyan")
    globalStep = 0
    for fitEpoch in range(fitEpochs):
        for batch in dataloader:
            globalStep += 1
            gpuBatch = dataset.get_gpu_batch_with_intrinsics(batch)
            referenceRgb = _renderReference(referenceModel, fixedPpisp, gpuBatch)

            optimizer.zero_grad(set_to_none=True)
            bakedOutputs = bakedModel(gpuBatch)
            fittedRgb = torch.clamp(
                linear_to_srgb(
                    _applyAchromaticVignetting(bakedOutputs["pred_rgb"], fixedPpisp, gpuBatch, vignettingMode)
                ),
                0,
                1,
            )
            loss = torch.nn.functional.mse_loss(fittedRgb, referenceRgb)

            loss.backward()
            optimizer.step()

            logger.log_progress(
                task_name="Fitting baked SH",
                advance=1,
                iteration=f"{fitEpoch + 1}/{fitEpochs}:{globalStep}",
                loss=float(loss.detach().item()),
            )
    logger.end_progress(task_name="Fitting baked SH")


@torch.no_grad()
def _evaluateBakedSh(
    referenceModel,
    bakedModel,
    simpleBakedModels: Dict[str, nn.Module],
    fixedPpisp,
    fullFixedPpisp,
    dataset,
    dataloader,
    outputRoot: Path,
    computeExtraMetrics: bool,
    vignettingMode: str,
) -> dict:
    criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}
    if computeExtraMetrics:
        criterions |= {
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
        }

    fullReferencePath = outputRoot / "full_ppisp_reference"
    referencePath = outputRoot / "reference"
    unfittedPath = outputRoot / "unfitted"
    fullReferencePath.mkdir(parents=True, exist_ok=True)
    referencePath.mkdir(parents=True, exist_ok=True)
    unfittedPath.mkdir(parents=True, exist_ok=True)
    bakedPath = outputRoot / "baked" if bakedModel is not None else None
    assistedPath = outputRoot / "baked_assisted" if bakedModel is not None else None
    if bakedPath is not None:
        bakedPath.mkdir(parents=True, exist_ok=True)
    if assistedPath is not None:
        assistedPath.mkdir(parents=True, exist_ok=True)
    simplePaths = {name: outputRoot / f"{name}_baked" for name in simpleBakedModels}
    for simplePath in simplePaths.values():
        simplePath.mkdir(parents=True, exist_ok=True)

    unfittedPsnrValues = []
    psnrValues = []
    ssimValues = []
    lpipsValues = []
    assistedPsnrValues = []
    assistedSsimValues = []
    assistedLpipsValues = []
    inferenceTimeValues = []
    simpleMetricValues = {
        name: {
            "psnr": [],
            "ssim": [],
            "lpips": [],
        }
        for name in simpleBakedModels
    }

    logger.start_progress(task_name="Evaluating baked SH", total_steps=len(dataloader), color="orange1")
    for iteration, batch in enumerate(dataloader):
        gpuBatch = dataset.get_gpu_batch_with_intrinsics(batch)

        fullReferenceRgb = _renderReference(referenceModel, fullFixedPpisp, gpuBatch)
        referenceRgb = _renderReference(referenceModel, fixedPpisp, gpuBatch)
        unfittedOutputs = referenceModel(gpuBatch)
        unfittedRgb = unfittedOutputs["pred_rgb"]

        torchvision.utils.save_image(
            fullReferenceRgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
            fullReferencePath / f"{iteration:05d}.png",
        )
        torchvision.utils.save_image(
            referenceRgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
            referencePath / f"{iteration:05d}.png",
        )
        torchvision.utils.save_image(
            unfittedRgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
            unfittedPath / f"{iteration:05d}.png",
        )

        unfittedPsnrValues.append(criterions["psnr"](unfittedRgb, referenceRgb).item())

        if bakedModel is not None:
            bakedOutputs = bakedModel(gpuBatch)
            bakedRgb = torch.clamp(linear_to_srgb(bakedOutputs["pred_rgb"]), 0, 1)
            assistedRgb = torch.clamp(
                linear_to_srgb(
                    _applyAchromaticVignetting(bakedOutputs["pred_rgb"], fixedPpisp, gpuBatch, vignettingMode)
                ),
                0,
                1,
            )
            torchvision.utils.save_image(
                bakedRgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
                bakedPath / f"{iteration:05d}.png",
            )
            torchvision.utils.save_image(
                assistedRgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
                assistedPath / f"{iteration:05d}.png",
            )

            psnrValues.append(criterions["psnr"](bakedRgb, referenceRgb).item())
            assistedPsnrValues.append(criterions["psnr"](assistedRgb, referenceRgb).item())
            if computeExtraMetrics:
                ssimValues.append(
                    criterions["ssim"](bakedRgb.permute(0, 3, 1, 2), referenceRgb.permute(0, 3, 1, 2)).item()
                )
                lpipsValues.append(
                    criterions["lpips"](
                        bakedRgb.clip(0, 1).permute(0, 3, 1, 2), referenceRgb.clip(0, 1).permute(0, 3, 1, 2)
                    ).item()
                )
                assistedSsimValues.append(
                    criterions["ssim"](assistedRgb.permute(0, 3, 1, 2), referenceRgb.permute(0, 3, 1, 2)).item()
                )
                assistedLpipsValues.append(
                    criterions["lpips"](
                        assistedRgb.clip(0, 1).permute(0, 3, 1, 2), referenceRgb.clip(0, 1).permute(0, 3, 1, 2)
                    ).item()
                )

            if "frame_time_ms" in bakedOutputs:
                inferenceTimeValues.append(bakedOutputs["frame_time_ms"])

        for simpleName, simpleModel in simpleBakedModels.items():
            simpleOutputs = simpleModel(gpuBatch)
            simpleRgb = torch.clamp(simpleOutputs["pred_rgb"], 0, 1)
            torchvision.utils.save_image(
                simpleRgb.squeeze(0).permute(2, 0, 1).clip(0, 1),
                simplePaths[simpleName] / f"{iteration:05d}.png",
            )
            simpleValues = simpleMetricValues[simpleName]
            simpleValues["psnr"].append(criterions["psnr"](simpleRgb, referenceRgb).item())
            if computeExtraMetrics:
                simpleValues["ssim"].append(
                    criterions["ssim"](simpleRgb.permute(0, 3, 1, 2), referenceRgb.permute(0, 3, 1, 2)).item()
                )
                simpleValues["lpips"].append(
                    criterions["lpips"](
                        simpleRgb.clip(0, 1).permute(0, 3, 1, 2),
                        referenceRgb.clip(0, 1).permute(0, 3, 1, 2),
                    ).item()
                )

        progressPsnr = psnrValues[-1] if psnrValues else unfittedPsnrValues[-1]
        logger.log_progress(task_name="Evaluating baked SH", advance=1, iteration=str(iteration), psnr=progressPsnr)
    logger.end_progress(task_name="Evaluating baked SH")

    metrics = {
        "vignetting_mode": vignettingMode,
        "unfitted_mean_psnr": float(np.mean(unfittedPsnrValues)),
        "unfitted_std_psnr": float(np.std(unfittedPsnrValues)),
    }
    if psnrValues:
        metrics |= {
            "mean_psnr": float(np.mean(psnrValues)),
            "std_psnr": float(np.std(psnrValues)),
            "assisted_mean_psnr": float(np.mean(assistedPsnrValues)),
            "assisted_std_psnr": float(np.std(assistedPsnrValues)),
        }
    if computeExtraMetrics:
        if ssimValues:
            metrics |= {
                "mean_ssim": float(np.mean(ssimValues)),
                "mean_lpips": float(np.mean(lpipsValues)),
                "assisted_mean_ssim": float(np.mean(assistedSsimValues)),
                "assisted_mean_lpips": float(np.mean(assistedLpipsValues)),
            }
    for simpleName, simpleValues in simpleMetricValues.items():
        metrics[f"{simpleName}_mean_psnr"] = float(np.mean(simpleValues["psnr"]))
        metrics[f"{simpleName}_std_psnr"] = float(np.std(simpleValues["psnr"]))
        if computeExtraMetrics:
            metrics |= {
                f"{simpleName}_mean_ssim": float(np.mean(simpleValues["ssim"])),
                f"{simpleName}_mean_lpips": float(np.mean(simpleValues["lpips"])),
            }
    if inferenceTimeValues:
        metrics["mean_inference_time"] = f"{np.mean(inferenceTimeValues):.2f} ms/frame"

    with open(outputRoot / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    psnrMetrics = {key: value for key, value in metrics.items() if "psnr" in key}
    logger.log_table("Post-Processing SH Bake Validation PSNR", record=psnrMetrics)
    return metrics


def _validateArguments(args, ppisp: nn.Module) -> None:
    if not hasattr(ppisp, "vignetting_params"):
        raise ValueError("Checkpoint post-processing is not PPISP-like: missing vignetting_params.")
    if not hasattr(ppisp, "exposure_params") or not hasattr(ppisp, "crf_params"):
        raise ValueError("Checkpoint post-processing is not PPISP-like: missing exposure_params or crf_params.")

    numFrames = int(ppisp.exposure_params.shape[0])
    numCameras = int(ppisp.crf_params.shape[0])
    if args.frameId < 0 or args.frameId >= numFrames:
        raise ValueError(f"frameId must be in [0, {numFrames - 1}], got {args.frameId}.")
    if args.cameraId < 0 or args.cameraId >= numCameras:
        raise ValueError(f"cameraId must be in [0, {numCameras - 1}], got {args.cameraId}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the pretrained checkpoint.")
    parser.add_argument("--path", type=str, default="", help="Path to test data, if not provided taken from ckpt.")
    parser.add_argument("--out-dir", dest="outDir", required=True, type=str, help="Output path.")
    parser.add_argument("--camera-id", dest="cameraId", default=0, type=int, help="PPISP camera id to bake.")
    parser.add_argument("--frame-id", dest="frameId", default=0, type=int, help="PPISP frame id to bake.")
    parser.add_argument(
        "--fit-epochs",
        dest="fitEpochs",
        default=1,
        type=int,
        help="Number of sequential passes over the train/reference set.",
    )
    parser.add_argument("--learning-rate", dest="learningRate", default=1.0e-3, type=float, help="SH fitting LR.")
    parser.add_argument(
        "--bake-flavor",
        dest="bakeFlavor",
        choices=[
            BAKE_FLAVOR_FIT,
            BAKE_FLAVOR_SIMPLE,
            BAKE_FLAVOR_SIMPLE_HIGHER_ORDER,
            BAKE_FLAVOR_ALL,
        ],
        default=BAKE_FLAVOR_FIT,
        help=(
            "Bake flavor to evaluate. 'fit' optimizes SH; 'simple' one-shot bakes DC SH; "
            "'simple-higher-order' also linearizes higher-order SH; 'all' compares every flavor."
        ),
    )
    parser.add_argument(
        "--vignetting-mode",
        dest="vignettingMode",
        choices=["none", "achromatic-fit"],
        default="achromatic-fit",
        help=(
            "Vignetting handling for the bake. 'none' disables PPISP vignetting; "
            "'achromatic-fit' uses chromatic PPISP reference and an achromatic fit-only vignette."
        ),
    )
    parser.add_argument(
        "--compute-extra-metrics",
        dest="computeExtraMetrics",
        action="store_false",
        help="If set, extra image metrics will not be computed [True by default].",
    )
    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.outDir,
        save_gt=False,
        computes_extra_metrics=args.computeExtraMetrics,
    )
    if renderer.post_processing is None:
        raise ValueError("Checkpoint does not contain PPISP post-processing.")

    _validateArguments(args, renderer.post_processing)
    vignettingMode = normalize_ppisp_bake_vignetting_mode(args.vignettingMode)
    fixedPpisp = FixedPPISP(
        renderer.post_processing,
        args.cameraId,
        args.frameId,
        "cuda",
        include_vignetting=vignettingMode != MODE_PPISP_BAKE_VIGNETTING_NONE,
    ).eval()
    fullFixedPpisp = FixedPPISP(
        renderer.post_processing,
        args.cameraId,
        args.frameId,
        "cuda",
        include_vignetting=True,
    ).eval()

    referenceModel = renderer.model.eval()

    outputRoot = Path(renderer.out_dir) / f"post_processing_sh_bake_ci{args.cameraId}_fi{args.frameId}"
    outputRoot.mkdir(parents=True, exist_ok=True)

    trainDataset, trainDataloader = _createTrainDataloader(renderer.conf)

    runFit = args.bakeFlavor in (BAKE_FLAVOR_FIT, BAKE_FLAVOR_ALL)
    simpleFlavorHigherOrderFlags = []
    if args.bakeFlavor in (BAKE_FLAVOR_SIMPLE, BAKE_FLAVOR_ALL):
        simpleFlavorHigherOrderFlags.append(("simple", False))
    if args.bakeFlavor in (BAKE_FLAVOR_SIMPLE_HIGHER_ORDER, BAKE_FLAVOR_ALL):
        simpleFlavorHigherOrderFlags.append(("simple_higher_order", True))

    bakedModel = None
    if runFit:
        bakedModel = renderer.model.clone().eval()
        bakedModel.build_acc()
        logger.info(f"Fitting SH coefficients to fixed PPISP camera={args.cameraId} frame={args.frameId}")
        _fitBakedSh(
            referenceModel=referenceModel,
            bakedModel=bakedModel,
            fixedPpisp=fixedPpisp,
            dataset=trainDataset,
            dataloader=trainDataloader,
            fitEpochs=args.fitEpochs,
            learningRate=args.learningRate,
            vignettingMode=vignettingMode,
        )

    simpleBakedModels = {}
    for simpleName, higherOrder in simpleFlavorHigherOrderFlags:
        simpleModel = renderer.model.clone().eval()
        logger.info(
            f"Simple-baking SH for camera_id={args.cameraId} "
            f"frame_id={args.frameId} (fixed exposure/color; higher_order={higherOrder})"
        )
        exposure, color = simple_bake(
            model=simpleModel,
            ppisp=renderer.post_processing,
            camera_id=args.cameraId,
            frame_id=args.frameId,
            higher_order=higherOrder,
        )
        simpleModel.build_acc()
        simpleBakedModels[simpleName] = simpleModel
        logger.info(
            f"{simpleName} bake done. exposure={exposure:.6f}; " f"color={[float(value) for value in color.tolist()]}"
        )

    _evaluateBakedSh(
        referenceModel=referenceModel,
        bakedModel=bakedModel,
        simpleBakedModels=simpleBakedModels,
        fixedPpisp=fixedPpisp,
        fullFixedPpisp=fullFixedPpisp,
        dataset=renderer.dataset,
        dataloader=renderer.dataloader,
        outputRoot=outputRoot,
        computeExtraMetrics=args.computeExtraMetrics,
        vignettingMode=vignettingMode,
    )


if __name__ == "__main__":
    main()
