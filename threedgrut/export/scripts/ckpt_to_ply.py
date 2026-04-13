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

"""
Utility script to convert 3DGRT model files (checkpoints, INGP, or PLY) into a standard PLY format.
This is useful for exporting trained models to a format that can be viewed or processed by other tools.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from threedgrut.utils.logger import logger

# Register the 'int_list' resolver for OmegaConf to parse lists of integers in yaml configs
OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])


def main(object_path: str, config_name: str = "apps/colmap_3dgrt.yaml", output_filename: str = None) -> None:
    """
    Main function to load a 3DGRT model and export it to a PLY file.

    Args:
        object_path (str): Path to the input model file. Supported formats are:
            - .pt: A PyTorch checkpoint containing the model weights and configuration.
            - .ingp: An Instant-NGP format exported model.
            - .ply: An already existing PLY model (can be used to re-export or clean up).
        config_name (str, optional): The hydra config name to use when loading .ingp or .ply files,
            since they do not contain their own configuration. Defaults to "apps/colmap_3dgrt.yaml".
        output_filename (str, optional): The desired output filename. If None, it will automatically
            append '_exported.ply' to the object name. Defaults to None.

    Raises:
        ValueError: If the `object_path` does not end with one of the supported extensions.
    """
    from threedgrut.model.model import MixtureOfGaussians

    def load_default_config():
        """Helper to load the default hydra configuration."""
        from hydra.compose import compose
        from hydra.initialize import initialize

        # Initialize hydra and compose the specified configuration
        with initialize(version_base=None, config_path="./configs"):
            conf = compose(config_name=config_name)
        return conf

    # Load from PyTorch checkpoint (.pt)
    if object_path.endswith(".pt"):
        checkpoint = torch.load(object_path, weights_only=False)
        conf = checkpoint["config"]
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        object_name = conf.experiment_name

    # Load from Instant-NGP format (.ingp)
    elif object_path.endswith(".ingp"):
        conf = load_default_config()
        model = MixtureOfGaussians(conf)
        model.init_from_ingp(object_path, init_model=False)
        object_name = Path(object_path).stem

    # Load from PLY format (.ply)
    elif object_path.endswith(".ply"):
        conf = load_default_config()
        model = MixtureOfGaussians(conf)
        model.init_from_ply(object_path, init_model=False)
        object_name = Path(object_path).stem
    else:
        raise ValueError(f"Unknown object type: {object_path}. Supported types are .pt, .ingp, .ply")

    # Export the loaded model to a PLY file
    if output_filename is None:
        output_filename = f"{object_name}_exported.ply"
    model.export_ply(output_filename)
    print(f"Exported model to: {output_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert 3DGRT model files to PLY format.")
    parser.add_argument("object_path", type=str, help="Path to the input model file (.pt, .ingp, or .ply)")
    parser.add_argument(
        "--config_name",
        type=str,
        default="apps/colmap_3dgrt.yaml",
        help="Hydra config name to use when loading .ingp or .ply files (default: apps/colmap_3dgrt.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output filename. If not provided, it will be derived from the input filename.",
    )

    args = parser.parse_args()

    main(args.object_path, config_name=args.config_name, output_filename=args.output)
