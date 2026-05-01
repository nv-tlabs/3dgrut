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

"""Fit fixed post-processing transforms into Gaussian SH coefficients for export."""

from __future__ import annotations

import copy
import logging
from typing import Iterable

import torch
import torch.nn as nn

from threedgrut.datasets.utils import configure_dataloader_for_platform
from threedgrut.utils.render import apply_post_processing

logger = logging.getLogger(__name__)


class PostProcessingBakeAdapter:
    """Adapter interface for baking one fixed post-processing transform."""

    name = "post-processing"

    def validate(self, post_processing: nn.Module) -> None:
        del post_processing

    def create_fixed_post_processing(self, post_processing: nn.Module, device: str) -> nn.Module:
        return copy.deepcopy(post_processing).to(device).eval()

    def apply_fit_transform(self, rgb: torch.Tensor, fixed_post_processing: nn.Module, gpu_batch) -> torch.Tensor:
        del fixed_post_processing, gpu_batch
        return rgb

    def initialize_fit(self, baked_model, post_processing: nn.Module) -> None:
        """Optionally warm-start the SH fit with a closed-form initialization.

        Default is a no-op: the cloned ``baked_model`` keeps its checkpoint
        SH coefficients as the starting point. Subclasses (e.g. PPISP) can
        override to apply a one-shot bake before Adam takes over.
        """
        del baked_model, post_processing

    def log_context(self) -> str:
        return ""


def _set_sh_fit_parameters(model) -> Iterable[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    fit_parameters = []
    for field_name in ("features_albedo", "features_specular"):
        parameter = getattr(model, field_name)
        parameter.requires_grad_(True)
        fit_parameters.append(parameter)
    return fit_parameters


def _create_train_dataloader(conf, train_dataset):
    num_workers = int(getattr(conf, "num_workers", 8))
    dataloader_kwargs = configure_dataloader_for_platform(
        {
            "num_workers": num_workers,
            "batch_size": 1,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True if num_workers > 0 else False,
        }
    )
    return torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)


def _render_reference(reference_model, fixed_post_processing, gpu_batch) -> torch.Tensor:
    with torch.no_grad():
        outputs = reference_model(gpu_batch)
        outputs = apply_post_processing(fixed_post_processing, outputs, gpu_batch, training=True)
        return outputs["pred_rgb"].detach()


def bake_post_processing_into_sh(
    model,
    post_processing: nn.Module,
    train_dataset,
    conf,
    *,
    adapter: PostProcessingBakeAdapter,
    epochs: int = 1,
    learning_rate: float = 1.0e-3,
    device: str = "cuda",
    view_sampling_mode: str = "training",
    interpolated_views_seed: int | None = None,
    trajectory_weight_position: float = 1.0,
    trajectory_weight_rotation: float = 0.5,
):
    """Return a cloned model whose SH coefficients approximate fixed post-processing output.

    ``view_sampling_mode`` controls what the optimizer sees each step:

    * ``"training"`` (default) -- iterate the training dataloader as usual.
    * ``"random-pair-slerp"`` -- pick two random training views and slerp
      between them at a random ``s ∈ [0, 1]``.
    * ``"trajectory"`` -- order the training views along an approximate
      Hamiltonian path (NN + 2-opt on a position+direction metric),
      arc-length-parameterise the path on ``[0, 1]``, sample random
      ``t ∈ [0, 1]``, slerp inside the bracketing segment.

    The interpolated-view modes synthesise a ``Batch`` per step from the
    template of the first training batch, replacing ``T_to_world`` with
    the interpolated pose. ``steps_per_epoch`` matches
    ``len(train_dataloader)`` so total step count is unchanged.
    """
    from threedgrut.export.usd.post_processing_view_interpolation import (
        InterpolatedViewSampler,
        VIEW_SAMPLING_TRAINING,
        normalize_view_sampling_mode,
    )

    if not hasattr(model, "clone"):
        raise TypeError("Post-processing SH bake export requires a cloneable MixtureOfGaussians model.")
    if train_dataset is None:
        raise ValueError("Post-processing SH bake export requires a train dataset. Pass --dataset if it is missing.")
    if post_processing is None:
        raise ValueError("Post-processing SH bake export requires a post_processing module.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    view_sampling_mode = normalize_view_sampling_mode(view_sampling_mode)

    adapter.validate(post_processing)
    reference_model = model.to(device).eval()
    reference_model.build_acc()
    baked_model = model.clone().to(device).eval()
    baked_model.build_acc()
    fixed_post_processing = adapter.create_fixed_post_processing(post_processing, device)

    # Warm-start the cloned SH state with the adapter's closed-form bake
    # (PPISP: simple_bake on the chosen camera/frame, with sRGB→linear so
    # the resulting SH lives in linear scene-referred space). Adam takes
    # over from there. Reduces the iterations needed and avoids fitting
    # from a checkpoint state that's far from the optimum.
    adapter.initialize_fit(baked_model, post_processing)

    fit_parameters = list(_set_sh_fit_parameters(baked_model))
    optimizer = torch.optim.Adam(fit_parameters, lr=learning_rate)
    train_dataloader = _create_train_dataloader(conf, train_dataset)
    steps_per_epoch = len(train_dataloader)

    sampler: InterpolatedViewSampler | None = None
    if view_sampling_mode != VIEW_SAMPLING_TRAINING:
        # Cache one real training batch to seed the synthetic sampler with
        # valid intrinsics / rays / pixel coords; only T_to_world rotates
        # per step.
        first_batch = next(iter(train_dataloader))
        template = train_dataset.get_gpu_batch_with_intrinsics(first_batch)
        sampler = InterpolatedViewSampler(
            train_dataset,
            template_gpu_batch=template,
            mode=view_sampling_mode,
            steps_per_epoch=steps_per_epoch,
            seed=interpolated_views_seed,
            weight_position=trajectory_weight_position,
            weight_rotation=trajectory_weight_rotation,
        )

    logger.info(
        "Fitting %s SH bake: mode=%s epochs=%s steps_per_epoch=%s%s",
        adapter.name,
        view_sampling_mode,
        epochs,
        steps_per_epoch,
        adapter.log_context(),
    )

    def _gpu_batches():
        if sampler is None:
            for batch in train_dataloader:
                yield train_dataset.get_gpu_batch_with_intrinsics(batch)
        else:
            for gpu_batch in sampler:
                yield gpu_batch

    with torch.enable_grad():
        global_step = 0
        total_steps = epochs * steps_per_epoch
        for epoch in range(epochs):
            for gpu_batch in _gpu_batches():
                global_step += 1
                reference_rgb = _render_reference(reference_model, fixed_post_processing, gpu_batch)

                optimizer.zero_grad(set_to_none=True)
                baked_outputs = baked_model(gpu_batch)
                fitted_rgb = adapter.apply_fit_transform(
                    baked_outputs["pred_rgb"],
                    fixed_post_processing,
                    gpu_batch,
                )
                loss = torch.nn.functional.mse_loss(fitted_rgb, reference_rgb)

                loss.backward()
                optimizer.step()

                if global_step == 1 or global_step % 50 == 0 or global_step == total_steps:
                    logger.info(
                        "%s SH bake epoch %s/%s step %s/%s loss=%.6g",
                        adapter.name,
                        epoch + 1,
                        epochs,
                        global_step,
                        total_steps,
                        float(loss.detach()),
                    )

    for parameter in baked_model.parameters():
        parameter.requires_grad_(False)
    baked_model.eval()
    logger.info("%s SH bake complete", adapter.name)
    return baked_model


MODE_PPISP_BAKE_VIGNETTING_NONE = "none"
MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT = "achromatic-fit"
PPISP_BAKE_VIGNETTING_MODES = {
    MODE_PPISP_BAKE_VIGNETTING_NONE,
    MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
}


class FixedPPISP(nn.Module):
    """Wrap PPISP as one fixed camera/frame color transform."""

    def __init__(
        self,
        ppisp: nn.Module,
        camera_id: int,
        frame_id: int,
        device: str,
        include_vignetting: bool = True,
    ) -> None:
        super().__init__()
        self.camera_id = int(camera_id)
        self.frame_id = int(frame_id)
        self.ppisp = copy.deepcopy(ppisp).to(device).eval()

        if hasattr(self.ppisp, "config") and hasattr(self.ppisp.config, "use_controller"):
            self.ppisp.config.use_controller = False
        if not include_vignetting and hasattr(self.ppisp, "vignetting_params"):
            with torch.no_grad():
                self.ppisp.vignetting_params.zero_()

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution: tuple[int, int],
        camera_idx=None,
        frame_idx=None,
        exposure_prior=None,
    ) -> torch.Tensor:
        del camera_idx, frame_idx, exposure_prior
        return self.ppisp(
            rgb,
            pixel_coords,
            resolution=resolution,
            camera_idx=self.camera_id,
            frame_idx=self.frame_id,
            exposure_prior=None,
        )


def normalize_ppisp_bake_vignetting_mode(mode: str | None) -> str:
    normalized = MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT if mode is None else str(mode).strip().lower()
    if normalized not in PPISP_BAKE_VIGNETTING_MODES:
        raise ValueError(
            f"Unsupported PPISP bake vignetting mode '{mode}'. "
            f"Expected one of: {sorted(PPISP_BAKE_VIGNETTING_MODES)}"
        )
    return normalized


def estimate_achromatic_vignetting(
    ppisp: nn.Module,
    camera_id: int,
    pixel_coords: torch.Tensor,
    resolution: tuple[int, int],
) -> torch.Tensor:
    """Estimate luminance falloff from PPISP's chromatic camera vignette."""
    if not hasattr(ppisp, "vignetting_params"):
        raise ValueError("PPISP-like module is missing vignetting_params.")

    width, height = resolution
    del height
    vig_params = ppisp.vignetting_params[int(camera_id)].to(device=pixel_coords.device, dtype=pixel_coords.dtype)

    u = (pixel_coords[..., 0] - float(width) * 0.5) / float(width)
    v = (pixel_coords[..., 1] - float(resolution[1]) * 0.5) / float(width)
    uv = torch.stack([u, v], dim=-1)

    channel_falloff = []
    for channel in range(3):
        center = vig_params[channel, 0:2]
        delta = uv - center
        r2 = torch.sum(delta * delta, dim=-1)
        falloff = (
            1.0 + vig_params[channel, 2] * r2 + vig_params[channel, 3] * r2 * r2 + vig_params[channel, 4] * r2 * r2 * r2
        )
        channel_falloff.append(torch.clamp(falloff, 0.0, 1.0))

    rgb_falloff = torch.stack(channel_falloff, dim=-1)
    luminance_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=pixel_coords.device, dtype=pixel_coords.dtype)
    return torch.sum(rgb_falloff * luminance_weights, dim=-1, keepdim=True)


def apply_achromatic_vignetting(
    rgb: torch.Tensor,
    ppisp: nn.Module,
    camera_id: int,
    pixel_coords: torch.Tensor,
    resolution: tuple[int, int],
) -> torch.Tensor:
    return rgb * estimate_achromatic_vignetting(ppisp, camera_id, pixel_coords, resolution)


class PPISPPostProcessingBakeAdapter(PostProcessingBakeAdapter):
    name = "PPISP post-processing"

    def __init__(
        self,
        camera_id: int = 0,
        frame_id: int = 0,
        vignetting_mode: str = MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
    ) -> None:
        self.camera_id = int(camera_id)
        self.frame_id = int(frame_id)
        self.vignetting_mode = normalize_ppisp_bake_vignetting_mode(vignetting_mode)

    def validate(self, post_processing: nn.Module) -> None:
        if not hasattr(post_processing, "exposure_params") or not hasattr(post_processing, "crf_params"):
            raise ValueError("PPISP SH bake export requires a PPISP-like post_processing module.")

        num_frames = int(post_processing.exposure_params.shape[0])
        num_cameras = int(post_processing.crf_params.shape[0])
        if self.frame_id < 0 or self.frame_id >= num_frames:
            raise ValueError(f"frame_id must be in [0, {num_frames - 1}], got {self.frame_id}.")
        if self.camera_id < 0 or self.camera_id >= num_cameras:
            raise ValueError(f"camera_id must be in [0, {num_cameras - 1}], got {self.camera_id}.")

    def create_fixed_post_processing(self, post_processing: nn.Module, device: str) -> nn.Module:
        return FixedPPISP(
            post_processing,
            self.camera_id,
            self.frame_id,
            device,
            include_vignetting=self.vignetting_mode == MODE_PPISP_BAKE_VIGNETTING_ACHROMATIC_FIT,
        ).eval()

    def apply_fit_transform(self, rgb: torch.Tensor, fixed_post_processing: nn.Module, gpu_batch) -> torch.Tensor:
        # PPISP reference is display-referred; baked side must encode to sRGB to match.
        from threedgrut.utils.post_processing_linear_to_srgb import linear_to_srgb

        if self.vignetting_mode == MODE_PPISP_BAKE_VIGNETTING_NONE:
            return torch.clamp(linear_to_srgb(rgb), 0.0, 1.0)
        _, height, width, _ = rgb.shape
        vignetted = apply_achromatic_vignetting(
            rgb=rgb,
            ppisp=fixed_post_processing.ppisp,
            camera_id=fixed_post_processing.camera_id,
            pixel_coords=gpu_batch.pixel_coords,
            resolution=(width, height),
        )
        return torch.clamp(linear_to_srgb(vignetted), 0.0, 1.0)

    def initialize_fit(self, baked_model, post_processing: nn.Module) -> None:
        """Warm-start with the higher-order simple-bake on the chosen
        (camera, frame), in linear scene-referred space."""
        # Late import: avoid pulling ppisp into modules that don't need it.
        from threedgrut.export.usd.post_processing_sh_simple_bake import simple_bake

        logger.info(
            "PPISP SH bake init: applying simple_bake (camera=%d, frame=%d, "
            "higher_order=True, apply_srgb_to_linear=True) before fitting.",
            self.camera_id, self.frame_id,
        )
        simple_bake(
            baked_model,
            post_processing,
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            higher_order=True,
            apply_srgb_to_linear=True,
        )

    def log_context(self) -> str:
        return f" camera={self.camera_id} frame={self.frame_id} vignetting={self.vignetting_mode}"
