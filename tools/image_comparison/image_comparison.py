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

import argparse
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

COMPARISON_MODES = ["slider", "checkerboard", "diff"]
SLIDER_DIRECTIONS = ["vertical", "horizontal"]
DIFF_METRICS = ["l1", "l2", "psnr", "ssim", "lpips", "flip"]
DISPLAY_MODES = ["fit_largest_dimension", "fit"]


@dataclass
class MetricResults:
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    flip: Optional[float]
    lpips_error: Optional[str] = None
    flip_error: Optional[str] = None


def import_viser():
    try:
        import viser
    except ImportError:
        print('viser not installed, please install the gui extra or run "pip install viser"')
        sys.exit(1)

    return viser


@dataclass(frozen=True)
class ImagePair:
    name: str
    image_a_path: Path
    image_b_path: Path


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def collect_images_by_name(folder: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    duplicate_names: List[str] = []

    for path in sorted(folder.rglob("*")):
        if not is_image_path(path):
            continue

        image_name = path.name
        if image_name in images:
            duplicate_names.append(image_name)
            continue

        images[image_name] = path

    if duplicate_names:
        duplicate_list = ", ".join(sorted(set(duplicate_names)))
        raise ValueError(f"Duplicate image names found in {folder}: {duplicate_list}")

    return images


def build_specific_image_pair(image_a_path: Path, image_b_path: Path) -> List[ImagePair]:
    if not is_image_path(image_a_path):
        raise ValueError(f"Invalid image path: {image_a_path}")
    if not is_image_path(image_b_path):
        raise ValueError(f"Invalid image path: {image_b_path}")

    return [
        ImagePair(
            name=f"{image_a_path.name} <-> {image_b_path.name}",
            image_a_path=image_a_path,
            image_b_path=image_b_path,
        )
    ]


def build_folder_image_pairs(folder_a_path: Path, folder_b_path: Path) -> List[ImagePair]:
    if not folder_a_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder_a_path}")
    if not folder_b_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder_b_path}")

    folder_a_images = collect_images_by_name(folder_a_path)
    folder_b_images = collect_images_by_name(folder_b_path)
    matched_names = sorted(set(folder_a_images).intersection(folder_b_images))

    if not matched_names:
        raise ValueError(f"No matching image names found between {folder_a_path} and {folder_b_path}")

    return [
        ImagePair(
            name=image_name,
            image_a_path=folder_a_images[image_name],
            image_b_path=folder_b_images[image_name],
        )
        for image_name in matched_names
    ]


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    resized = Image.fromarray(float_image_to_uint8(image)).resize((width, height), resample_filter)
    return np.asarray(resized, dtype=np.float32) / 255.0


def load_aligned_pair(image_pair: ImagePair) -> Tuple[np.ndarray, np.ndarray, str]:
    image_a = load_image_rgb(image_pair.image_a_path)
    image_b = load_image_rgb(image_pair.image_b_path)

    if image_a.shape == image_b.shape:
        status = f"{image_pair.name}: {image_a.shape[1]}x{image_a.shape[0]}"
        return image_a, image_b, status

    image_b = resize_image(image_b, (image_a.shape[1], image_a.shape[0]))
    status = (
        f"{image_pair.name}: A {image_a.shape[1]}x{image_a.shape[0]}, "
        f"B resized to match from {image_pair.image_b_path.name}"
    )
    return image_a, image_b, status


def render_slider_comparison(
    image_a: np.ndarray,
    image_b: np.ndarray,
    slider_position: float,
    slider_direction: str,
) -> np.ndarray:
    output = image_b.copy()
    height, width = image_a.shape[:2]
    slider_position = float(np.clip(slider_position, 0.0, 1.0))

    if slider_direction == "horizontal":
        split_row = int(round(height * slider_position))
        output[:split_row, :] = image_a[:split_row, :]
        if 0 < split_row < height:
            output[max(0, split_row - 1) : min(height, split_row + 1), :] = 1.0
    else:
        split_col = int(round(width * slider_position))
        output[:, :split_col] = image_a[:, :split_col]
        if 0 < split_col < width:
            output[:, max(0, split_col - 1) : min(width, split_col + 1)] = 1.0

    return output


def render_checkerboard_comparison(image_a: np.ndarray, image_b: np.ndarray, checker_size: int) -> np.ndarray:
    checker_size = max(1, checker_size)
    height, width = image_a.shape[:2]
    y_indices, x_indices = np.indices((height, width))
    checker_mask = ((x_indices // checker_size) + (y_indices // checker_size)) % 2 == 0
    return np.where(checker_mask[..., None], image_a, image_b)


def render_diff_comparison(image_a: np.ndarray, image_b: np.ndarray, diff_scale: float) -> np.ndarray:
    return render_diff_metric(image_a=image_a, image_b=image_b, diff_metric="l1", diff_scale=diff_scale)


def render_diff_metric(image_a: np.ndarray, image_b: np.ndarray, diff_metric: str, diff_scale: float) -> np.ndarray:
    error_map = compute_error_map(image_a=image_a, image_b=image_b, diff_metric=diff_metric)
    scaled_error = np.clip(error_map * diff_scale, 0.0, 1.0)
    return apply_jet_colormap(scaled_error)


def compute_error_map(image_a: np.ndarray, image_b: np.ndarray, diff_metric: str) -> np.ndarray:
    if diff_metric == "l2":
        return np.sqrt(np.mean(np.square(image_a - image_b), axis=-1))
    if diff_metric == "psnr":
        return compute_psnr_error_map(image_a=image_a, image_b=image_b)
    if diff_metric == "ssim":
        return 1.0 - compute_ssim_map(image_a=image_a, image_b=image_b)
    if diff_metric == "lpips":
        lpips_value, _ = compute_lpips_metric(image_a=image_a, image_b=image_b)
        return scalar_metric_to_map(lpips_value, image_a.shape[:2], higher_is_worse=True)
    if diff_metric == "flip":
        flip_map, flip_value, _ = compute_flip_metric(image_a=image_a, image_b=image_b)
        if flip_map is not None:
            return normalize_error_map(flip_map)
        return scalar_metric_to_map(flip_value, image_a.shape[:2], higher_is_worse=True)

    return np.mean(np.abs(image_a - image_b), axis=-1)


def compute_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    mse = float(np.mean(np.square(image_a - image_b)))
    if mse <= 1.0e-12:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def compute_psnr_error_map(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    mse = np.mean(np.square(image_a - image_b), axis=-1)
    psnr_map = -10.0 * np.log10(np.maximum(mse, 1.0e-12))
    return 1.0 - np.clip(psnr_map / 60.0, 0.0, 1.0)


def compute_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    return float(np.mean(compute_ssim_map(image_a=image_a, image_b=image_b)))


def compute_ssim_map(image_a: np.ndarray, image_b: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    image_a = np.clip(image_a, 0.0, 1.0)
    image_b = np.clip(image_b, 0.0, 1.0)

    mu_a = box_filter(image_a, kernel_size=kernel_size)
    mu_b = box_filter(image_b, kernel_size=kernel_size)
    mu_a_squared = np.square(mu_a)
    mu_b_squared = np.square(mu_b)
    mu_ab = mu_a * mu_b

    sigma_a_squared = box_filter(np.square(image_a), kernel_size=kernel_size) - mu_a_squared
    sigma_b_squared = box_filter(np.square(image_b), kernel_size=kernel_size) - mu_b_squared
    sigma_ab = box_filter(image_a * image_b, kernel_size=kernel_size) - mu_ab

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_squared + mu_b_squared + c1) * (sigma_a_squared + sigma_b_squared + c2)
    ssim = numerator / np.maximum(denominator, 1.0e-12)
    return np.clip(np.mean(ssim, axis=-1), 0.0, 1.0)


def box_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    summed = (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )
    return summed / float(kernel_size * kernel_size)


def scalar_metric_to_map(value: Optional[float], shape: Tuple[int, int], higher_is_worse: bool) -> np.ndarray:
    if value is None or not np.isfinite(value):
        return np.zeros(shape, dtype=np.float32)

    if higher_is_worse:
        normalized_value = float(np.clip(value, 0.0, 1.0))
    else:
        normalized_value = 1.0 - float(np.clip(value, 0.0, 1.0))

    return np.full(shape, normalized_value, dtype=np.float32)


def normalize_error_map(error_map: np.ndarray) -> np.ndarray:
    if error_map.ndim == 3:
        error_map = np.mean(error_map[..., :3], axis=-1)
    return np.clip(error_map.astype(np.float32), 0.0, 1.0)


def image_to_torch_tensor(image: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()


def compute_lpips_metric(image_a: np.ndarray, image_b: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    try:
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        return None, f"LPIPS unavailable: {exc}"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)
        with torch.no_grad():
            value = metric(image_to_torch_tensor(image_a).to(device), image_to_torch_tensor(image_b).to(device))
        return float(value.detach().cpu().item()), None
    except Exception as exc:
        return None, f"LPIPS failed: {exc}"


def compute_flip_metric(
    image_a: np.ndarray, image_b: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    try:
        import flip_evaluator
    except ImportError as exc:
        try:
            import nbflip as flip_evaluator
        except ImportError:
            return (
                None,
                None,
                f"FLIP unavailable: install NVIDIA FLIP with `pip install flip-evaluator`, not `flip`: {exc}",
            )

    try:
        flip_map, mean_flip, _ = flip_evaluator.evaluate(
            np.ascontiguousarray(image_a.astype(np.float32)),
            np.ascontiguousarray(image_b.astype(np.float32)),
            "ldr",
            True,
            False,
            True,
            {},
        )
        return normalize_error_map(flip_map), float(mean_flip), None
    except Exception as exc:
        return None, None, f"FLIP failed: {exc}"


def compute_metric_results(
    image_a: np.ndarray,
    image_b: np.ndarray,
    include_perceptual_metrics: bool = False,
) -> MetricResults:
    if include_perceptual_metrics:
        lpips_value, lpips_error = compute_lpips_metric(image_a=image_a, image_b=image_b)
        _, flip_value, flip_error = compute_flip_metric(image_a=image_a, image_b=image_b)
    else:
        lpips_value = None
        flip_value = None
        lpips_error = "LPIPS not computed"
        flip_error = "FLIP not computed"

    return MetricResults(
        psnr=compute_psnr(image_a=image_a, image_b=image_b),
        ssim=compute_ssim(image_a=image_a, image_b=image_b),
        lpips=lpips_value,
        flip=flip_value,
        lpips_error=lpips_error,
        flip_error=flip_error,
    )


def format_metric_value(value: Optional[float], precision: int = 5) -> str:
    if value is None:
        return "unavailable"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def apply_jet_colormap(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def float_image_to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def resize_uint8_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    resized = Image.fromarray(image).resize((max(1, width), max(1, height)), resample_filter)
    return np.asarray(resized, dtype=np.uint8)


def get_image_canvas_rect(
    image_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    display_mode: str,
) -> Tuple[float, float, float, float]:
    image_width, image_height = image_size
    canvas_width, canvas_height = canvas_size
    canvas_width = max(1, canvas_width)
    canvas_height = max(1, canvas_height)

    if display_mode == "fit":
        return 0.0, 0.0, float(canvas_width), float(canvas_height)

    scale = min(canvas_width / image_width, canvas_height / image_height)
    display_width = image_width * scale
    display_height = image_height * scale
    image_x0 = 0.5 * (canvas_width - display_width)
    image_y0 = 0.5 * (canvas_height - display_height)
    return image_x0, image_y0, display_width, display_height


def fit_image_to_canvas(
    image: np.ndarray,
    canvas_size: Tuple[int, int],
    display_mode: str,
) -> np.ndarray:
    canvas_width, canvas_height = canvas_size
    canvas_width = max(1, canvas_width)
    canvas_height = max(1, canvas_height)
    image_height, image_width = image.shape[:2]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    if display_mode == "fit":
        return resize_uint8_image(image, (canvas_width, canvas_height))

    _, _, display_width, display_height = get_image_canvas_rect(
        image_size=(image_width, image_height),
        canvas_size=(canvas_width, canvas_height),
        display_mode=display_mode,
    )
    resized_width = max(1, int(round(display_width)))
    resized_height = max(1, int(round(display_height)))
    resized = resize_uint8_image(image, (resized_width, resized_height))

    source_x0 = max(0, (resized_width - canvas_width) // 2)
    source_y0 = max(0, (resized_height - canvas_height) // 2)
    target_x0 = max(0, (canvas_width - resized_width) // 2)
    target_y0 = max(0, (canvas_height - resized_height) // 2)
    copy_width = min(resized_width - source_x0, canvas_width - target_x0)
    copy_height = min(resized_height - source_y0, canvas_height - target_y0)

    canvas[target_y0 : target_y0 + copy_height, target_x0 : target_x0 + copy_width] = resized[
        source_y0 : source_y0 + copy_height,
        source_x0 : source_x0 + copy_width,
    ]
    return canvas


class ImageComparisonViewer:
    def __init__(self, image_pairs: List[ImagePair], host: str, port: int, target_fps: float) -> None:
        self.image_pairs = image_pairs
        self.host = host
        self.port = port
        self.target_fps = target_fps
        self.viser = import_viser()
        self.server = self.viser.ViserServer(host=self.host, port=self.port)
        self.need_update = True
        self.image_cache: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
        self.metric_cache: Dict[str, MetricResults] = {}
        self.error_map_cache: Dict[Tuple[str, str], np.ndarray] = {}

        self.image_pair_dropdown = None
        self.display_mode_dropdown = None
        self.mode_dropdown = None
        self.slider_direction_dropdown = None
        self.slider_position_slider = None
        self.checker_size_slider = None
        self.diff_metric_dropdown = None
        self.diff_scale_slider = None
        self.psnr_text = None
        self.ssim_text = None
        self.lpips_text = None
        self.flip_text = None
        self.compute_perceptual_metrics_button = None
        self.status_text = None

        self.init_ui()

        @self.server.on_client_connect
        def _(client) -> None:
            self.need_update = True

    def init_ui(self) -> None:
        with self.server.gui.add_folder("Image Comparison"):
            image_pair_names = [image_pair.name for image_pair in self.image_pairs]
            self.image_pair_dropdown = self.server.gui.add_dropdown(
                "Image Pair",
                options=image_pair_names,
                initial_value=image_pair_names[0],
            )
            previous_image_button = self.server.gui.add_button("Previous Image")
            next_image_button = self.server.gui.add_button("Next Image")
            self.display_mode_dropdown = self.server.gui.add_dropdown(
                "Display Mode",
                options=DISPLAY_MODES,
                initial_value=DISPLAY_MODES[0],
            )
            self.mode_dropdown = self.server.gui.add_dropdown(
                "Mode",
                options=COMPARISON_MODES,
                initial_value=COMPARISON_MODES[0],
            )
            self.slider_direction_dropdown = self.server.gui.add_dropdown(
                "Slider Direction",
                options=SLIDER_DIRECTIONS,
                initial_value=SLIDER_DIRECTIONS[0],
            )
            self.slider_position_slider = self.server.gui.add_slider(
                "Slider Position",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.5,
            )
            self.checker_size_slider = self.server.gui.add_slider(
                "Checker Size",
                min=4,
                max=256,
                step=1,
                initial_value=32,
            )
            self.diff_metric_dropdown = self.server.gui.add_dropdown(
                "Diff Metric",
                options=DIFF_METRICS,
                initial_value=DIFF_METRICS[0],
            )
            self.diff_scale_slider = self.server.gui.add_slider(
                "Diff Scale",
                min=0.1,
                max=20.0,
                step=0.1,
                initial_value=4.0,
            )
            reload_button = self.server.gui.add_button("Reload Images")
            self.status_text = self.server.gui.add_text("Status", initial_value="Loading", disabled=True)

        with self.server.gui.add_folder("Metrics"):
            self.psnr_text = self.server.gui.add_text("PSNR", initial_value="unavailable", disabled=True)
            self.ssim_text = self.server.gui.add_text("SSIM", initial_value="unavailable", disabled=True)
            self.lpips_text = self.server.gui.add_text("LPIPS", initial_value="unavailable", disabled=True)
            self.flip_text = self.server.gui.add_text("FLIP", initial_value="unavailable", disabled=True)
            self.compute_perceptual_metrics_button = self.server.gui.add_button("Compute LPIPS / FLIP")

        controls = [
            self.image_pair_dropdown,
            self.display_mode_dropdown,
            self.mode_dropdown,
            self.slider_direction_dropdown,
            self.slider_position_slider,
            self.checker_size_slider,
            self.diff_metric_dropdown,
            self.diff_scale_slider,
        ]

        for control in controls:

            @control.on_update
            def _(_) -> None:
                self.need_update = True

        @reload_button.on_click
        def _(_) -> None:
            self.image_cache.clear()
            self.metric_cache.clear()
            self.error_map_cache.clear()
            self.need_update = True

        @previous_image_button.on_click
        def _(_) -> None:
            self.select_relative_image_pair(offset=-1)

        @next_image_button.on_click
        def _(_) -> None:
            self.select_relative_image_pair(offset=1)

        @self.compute_perceptual_metrics_button.on_click
        def _(_) -> None:
            self.compute_selected_perceptual_metrics()
            self.need_update = True

    def select_relative_image_pair(self, offset: int) -> None:
        selected_name = self.image_pair_dropdown.value
        image_pair_names = [image_pair.name for image_pair in self.image_pairs]
        try:
            selected_index = image_pair_names.index(selected_name)
        except ValueError:
            selected_index = 0

        next_index = (selected_index + offset) % len(self.image_pairs)
        self.image_pair_dropdown.value = image_pair_names[next_index]
        self.need_update = True

    def get_selected_pair(self) -> ImagePair:
        selected_name = self.image_pair_dropdown.value
        for image_pair in self.image_pairs:
            if image_pair.name == selected_name:
                return image_pair

        return self.image_pairs[0]

    def get_aligned_pair(self, image_pair: ImagePair) -> Tuple[np.ndarray, np.ndarray, str]:
        if image_pair.name not in self.image_cache:
            self.image_cache[image_pair.name] = load_aligned_pair(image_pair)
        return self.image_cache[image_pair.name]

    def get_metric_results(self, image_pair: ImagePair, image_a: np.ndarray, image_b: np.ndarray) -> MetricResults:
        if image_pair.name not in self.metric_cache:
            self.metric_cache[image_pair.name] = compute_metric_results(image_a=image_a, image_b=image_b)
        return self.metric_cache[image_pair.name]

    def compute_selected_perceptual_metrics(self) -> None:
        image_pair = self.get_selected_pair()
        image_a, image_b, _ = self.get_aligned_pair(image_pair)
        self.metric_cache[image_pair.name] = compute_metric_results(
            image_a=image_a,
            image_b=image_b,
            include_perceptual_metrics=True,
        )
        self.update_metric_widgets(metric_results=self.metric_cache[image_pair.name])

    def update_metric_widgets(self, metric_results: MetricResults) -> None:
        self.psnr_text.value = format_metric_value(metric_results.psnr, precision=4)
        self.ssim_text.value = format_metric_value(metric_results.ssim, precision=5)
        self.lpips_text.value = format_metric_value(metric_results.lpips, precision=5)
        self.flip_text.value = format_metric_value(metric_results.flip, precision=5)

    def render_current_diff(
        self,
        image_pair: ImagePair,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> np.ndarray:
        diff_metric = self.diff_metric_dropdown.value
        cache_key = (image_pair.name, diff_metric)
        if cache_key not in self.error_map_cache:
            self.error_map_cache[cache_key] = compute_error_map(
                image_a=image_a,
                image_b=image_b,
                diff_metric=diff_metric,
            )

        scaled_error = np.clip(self.error_map_cache[cache_key] * float(self.diff_scale_slider.value), 0.0, 1.0)
        return apply_jet_colormap(scaled_error)

    def render_current_comparison(self) -> np.ndarray:
        image_pair = self.get_selected_pair()
        image_a, image_b, status = self.get_aligned_pair(image_pair)
        metric_results = self.get_metric_results(image_pair=image_pair, image_a=image_a, image_b=image_b)
        self.update_metric_widgets(metric_results=metric_results)
        mode = self.mode_dropdown.value

        if mode == "checkerboard":
            output = render_checkerboard_comparison(
                image_a=image_a,
                image_b=image_b,
                checker_size=int(self.checker_size_slider.value),
            )
        elif mode == "diff":
            output = self.render_current_diff(image_pair=image_pair, image_a=image_a, image_b=image_b)
        else:
            output = render_slider_comparison(
                image_a=image_a,
                image_b=image_b,
                slider_position=float(self.slider_position_slider.value),
                slider_direction=self.slider_direction_dropdown.value,
            )

        if mode == "slider":
            self.status_text.value = f"{status} | mode: {mode}"
        elif mode == "diff":
            diff_metric = self.diff_metric_dropdown.value
            warning = self.get_metric_warning(metric_results=metric_results, diff_metric=diff_metric)
            self.status_text.value = f"{status} | mode: {mode} | diff: {diff_metric}{warning}"
        else:
            self.status_text.value = f"{status} | mode: {mode}"
        return float_image_to_uint8(output)

    def get_metric_warning(self, metric_results: MetricResults, diff_metric: str) -> str:
        if diff_metric == "lpips" and metric_results.lpips_error is not None:
            return f" | {metric_results.lpips_error}"
        if diff_metric == "flip" and metric_results.flip_error is not None:
            return f" | {metric_results.flip_error}"
        return ""

    def display_output(self, output: np.ndarray) -> None:
        display_mode = self.display_mode_dropdown.value
        for client in self.server.get_clients().values():
            canvas_width = int(client.camera.image_width or output.shape[1])
            canvas_height = int(client.camera.image_height or output.shape[0])
            display_image = fit_image_to_canvas(
                image=output,
                canvas_size=(canvas_width, canvas_height),
                display_mode=display_mode,
            )
            client.scene.set_background_image(display_image, format="jpeg")

    def update(self) -> None:
        if not self.need_update:
            return

        output = self.render_current_comparison()
        self.display_output(output)

        self.need_update = False

    def run(self) -> None:
        print_server_urls(host=self.host, port=self.port)
        while True:
            self.update()
            time.sleep(max(0.001, 1.0 / self.target_fps))


def get_candidate_host_addresses() -> List[str]:
    addresses = ["127.0.0.1"]
    try:
        hostname = socket.gethostname()
        for address_info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            address = address_info[4][0]
            if address not in addresses and not address.startswith("127."):
                addresses.append(address)
    except OSError:
        pass
    return addresses


def print_server_urls(host: str, port: int) -> None:
    if host in ("0.0.0.0", "::"):
        print("Viser is listening on all interfaces. Try these URLs:")
        for address in get_candidate_host_addresses():
            print(f"  http://{address}:{port}")
    else:
        print(f"Viser URL: http://{host}:{port}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Viser based image comparison viewer.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--images",
        nargs=2,
        metavar=("IMAGE_A", "IMAGE_B"),
        help="Compare two specific images.",
    )
    input_group.add_argument(
        "--folders",
        nargs=2,
        metavar=("FOLDER_A", "FOLDER_B"),
        help="Compare matching image names from two folders.",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host/interface.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--target_fps", type=float, default=20.0, help="Maximum UI refresh rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.images is not None:
        image_pairs = build_specific_image_pair(Path(args.images[0]), Path(args.images[1]))
    else:
        image_pairs = build_folder_image_pairs(Path(args.folders[0]), Path(args.folders[1]))

    viewer = ImageComparisonViewer(
        image_pairs=image_pairs,
        host=args.host,
        port=args.port,
        target_fps=args.target_fps,
    )
    viewer.run()


if __name__ == "__main__":
    main()
