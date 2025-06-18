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

import subprocess
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options
from pathlib import Path

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])


def main(object_path, config_name: str = 'apps/colmap_3dgrt.yaml') -> None:
    from threedgrut.trainer import Trainer3DGRUT
    from threedgrut.model.model import MixtureOfGaussians

    def load_default_config():
        from hydra.compose import compose
        from hydra.initialize import initialize
        with initialize(version_base=None, config_path='./configs'):
            conf = compose(config_name=config_name)
        return conf

    if object_path.endswith('.pt'):
        checkpoint = torch.load(object_path)
        conf = checkpoint["config"]
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        object_name = conf.experiment_name
    elif object_path.endswith('.ingp'):
        conf = load_default_config()
        model = MixtureOfGaussians(conf)
        model.init_from_ingp(object_path, init_model=False)
        object_name = Path(object_path).stem
    elif object_path.endswith('.ply'):
        conf = load_default_config()
        model = MixtureOfGaussians(conf)
        model.init_from_ply(object_path, init_model=False)
        object_name = Path(object_path).stem
    else:
        raise ValueError(f"Unknown object type: {object_path}")

    model.export_ply(f"{object_name}_exported.ply")
    print(f"Exported model to: {object_name}_exported.ply")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
