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

"""
Visibility filtering for Gaussian splatting models.

Computes average visibility of each Gaussian over the training dataset by:
1. Rendering all training views
2. Accumulating mog_visibility from each render
3. Computing per-Gaussian average visibility

Usage:
    from threedgrut.export import compute_average_visibility
    
    visibility = compute_average_visibility(model, train_dataset, conf)
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_average_visibility(
    model,
    train_dataset,
    conf=None,
    num_workers: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute average visibility for each Gaussian over the training dataset.
    
    Renders all training views and accumulates the mog_visibility buffer
    returned by the tracer, then computes the average per Gaussian.
    
    Args:
        model: The Gaussian splatting model (MixtureOfGaussians)
        train_dataset: Training dataset with camera poses
        conf: Configuration object (optional, uses model.conf if None)
        num_workers: Number of DataLoader workers (default 0 for single-threaded)
        show_progress: Show progress bar
        
    Returns:
        np.ndarray: Per-Gaussian average visibility [N] where N is num_gaussians
    """
    if conf is None:
        conf = getattr(model, 'conf', None)
    
    num_gaussians = model.num_gaussians
    device = next(model.parameters()).device
    
    # Accumulate visibility counts
    visibility_sum = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
    num_views = 0
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    
    logger.info(f"Computing visibility over {len(dataloader)} training views...")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Build BVH if needed
    model.build_acc(rebuild=True)
    
    iterator = tqdm(dataloader, desc="Computing visibility", disable=not show_progress)
    
    for batch in iterator:
        # Get GPU batch with intrinsics
        gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch)
        
        # Render
        outputs = model(gpu_batch, train=False)
        
        # Accumulate visibility
        # mog_visibility is [N, 1] boolean or float tensor
        mog_visibility = outputs["mog_visibility"]
        
        # Convert to float and squeeze if needed
        if mog_visibility.dtype == torch.bool:
            mog_visibility = mog_visibility.float()
        
        mog_visibility = mog_visibility.squeeze(-1)  # [N]
        
        visibility_sum += mog_visibility
        num_views += 1
    
    # Compute average
    if num_views > 0:
        average_visibility = visibility_sum / num_views
    else:
        logger.warning("No views processed for visibility computation, returning zero visibility")
        average_visibility = visibility_sum
    
    # Convert to numpy
    result = average_visibility.cpu().numpy()
    
    # Log statistics
    logger.info(f"Visibility statistics over {num_views} views:")
    logger.info(f"  Min: {result.min():.4f}")
    logger.info(f"  Max: {result.max():.4f}")
    logger.info(f"  Mean: {result.mean():.4f}")
    logger.info(f"  Median: {np.median(result):.4f}")
    
    # Count Gaussians at different visibility thresholds
    for threshold in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
        count_below = np.sum(result < threshold)
        pct_below = 100.0 * count_below / num_gaussians
        logger.info(f"  Gaussians with visibility < {threshold}: {count_below} ({pct_below:.2f}%)")
    
    return result


def compute_visibility_and_filter(
    model,
    train_dataset,
    conf=None,
    visibility_threshold: float = 0.0,
    num_workers: int = 0,
    show_progress: bool = True,
) -> tuple:
    """
    Compute visibility and return filtering mask.
    
    Convenience function that computes visibility and returns both the
    visibility array and a boolean mask for filtering.
    
    Args:
        model: The Gaussian splatting model
        train_dataset: Training dataset
        conf: Configuration object
        visibility_threshold: Minimum visibility threshold (default 0.0 = no filtering)
        num_workers: DataLoader workers
        show_progress: Show progress bar
        
    Returns:
        Tuple of (visibility array [N], filter mask [N] where True = keep)
    """
    visibility = compute_average_visibility(
        model=model,
        train_dataset=train_dataset,
        conf=conf,
        num_workers=num_workers,
        show_progress=show_progress,
    )
    
    if visibility_threshold > 0.0:
        mask = visibility >= visibility_threshold
        num_filtered = np.sum(~mask)
        logger.info(
            f"Visibility filtering: {num_filtered} Gaussians below threshold {visibility_threshold} "
            f"({100.0 * num_filtered / len(visibility):.2f}%)"
        )
    else:
        mask = np.ones(len(visibility), dtype=bool)
    
    return visibility, mask
