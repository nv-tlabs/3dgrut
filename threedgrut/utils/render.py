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


## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def apply_post_processing(
    post_processing,
    outputs: dict,
    gpu_batch,
    training: bool = False,
) -> dict:
    """Apply post-processing to rendered output.

    Args:
        post_processing: Post-processing module
        outputs: Model outputs including pred_rgb
        gpu_batch: Batch containing camera_idx, frame_idx, pixel_coords, exposure
        training: If True, use actual frame_idx; if False, use -1 for novel view mode

    Returns:
        Updated outputs dict with post-processed pred_rgb
    """
    assert outputs["pred_rgb"].shape[0] == 1, "Post-processing requires batch_size=1"

    pred_rgb = outputs["pred_rgb"]
    camera_idx = gpu_batch.camera_idx
    frame_idx = gpu_batch.frame_idx if training else -1
    H, W = pred_rgb.shape[1], pred_rgb.shape[2]

    # Flatten: [1, H, W, 3] -> [H*W, 3]
    # Ensure contiguous memory for CUDA kernels
    pred_rgb_flat = pred_rgb.contiguous().view(-1, 3)
    pixel_coords_flat = gpu_batch.pixel_coords.contiguous().view(-1, 2)

    # Apply post-processing
    pred_rgb_pp = post_processing(
        pred_rgb_flat,
        pixel_coords_flat,
        resolution=(W, H),
        camera_idx=camera_idx,
        frame_idx=frame_idx,
        exposure_prior=gpu_batch.exposure,
    )

    # Reshape back: [H*W, 3] -> [1, H, W, 3]
    outputs["pred_rgb"] = pred_rgb_pp.view(pred_rgb.shape)
    return outputs
