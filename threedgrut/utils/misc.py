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
from typing import Callable, Optional
from datetime import datetime

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from omegaconf import OmegaConf, DictConfig

OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("eq", lambda a, b: a == b)


def to_torch(data: npt.NDArray, device: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Converts a numpy array to a torch tensor on target device with optional type-casting"""
    return torch.from_numpy(data).to(device=device, dtype=dtype)


def to_np(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().cpu().numpy()


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


ACTIVATION_DICT: dict[str, Callable[..., torch.Tensor]] = {
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
    "normalize": torch.nn.functional.normalize,
    "none": lambda x: x,
}

INVERSE_ACTIVATION_DICT: dict[str, Callable[..., torch.Tensor]] = {
    "sigmoid": inverse_sigmoid,
    "exp": torch.log,
    "none": lambda x: x,
}


def get_activation_function(activation_function: str, inverse=False) -> Callable:
    if not inverse:
        return ACTIVATION_DICT[activation_function]
    else:
        return INVERSE_ACTIVATION_DICT[activation_function]


def quaternion_to_so3(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), dtype=r.dtype, device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def exponential_scheduler(lr_init, lr_final, max_steps=1000000, type=""):
    def helper(step):
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


def skip_scheduler(type=""):
    def helper(step):
        return None

    return helper


SCHEDULER_DICT: dict[str, Callable] = {"exp": exponential_scheduler, "skip": skip_scheduler}


def get_scheduler(scheduler: str) -> Callable:
    return SCHEDULER_DICT[scheduler]


def sh_degree_to_specular_dim(degree):
    """Number of dimensions used by SH of deg [1..degree], inclusive"""
    return 3 * ((degree + 1) ** 2 - 1)


def sh_degree_to_num_features(degree):
    """Number of dimensions used by SH of deg [0..degree], inclusive"""
    return sh_degree_to_specular_dim(degree) + 3


def jet_map(map: torch.Tensor, max_val: float) -> torch.Tensor:
    """A colormap for 1D maps in [0.0, 1.0]"""
    vs = (map / max_val).clip(0, 1)
    return torch.concat(
        [
            (4.0 * (vs - 0.375)).clip(0, 1) * (-4.0 * (vs - 1.125)).clip(0, 1),
            (4.0 * (vs - 0.125)).clip(0, 1) * (-4.0 * (vs - 0.875)).clip(0, 1),
            (4.0 * vs + 0.5).clip(0, 1) * (-4.0 * (vs - 0.625)).clip(0, 1),
        ],
        dim=2,
    )


def create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb):
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    run_name = f"{object_name}-" + timestamp

    assert out_dir is not None, "Output directory must be specified"
    out_dir = os.path.join(out_dir, experiment_name) if experiment_name else out_dir
    out_dir = os.path.join(out_dir, run_name)

    if use_wandb:
        import wandb

        wandb.login()
        wandb.init(
            config=OmegaConf.to_container(DictConfig(conf)),
            project=conf.wandb_project,
            group=experiment_name,
            name=run_name,
        )
        wandb.tensorboard.patch(root_logdir=out_dir, save=False)

    writer = SummaryWriter(log_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return writer, out_dir, run_name
