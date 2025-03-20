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

from __future__ import annotations
from playground.playground import Playground
import os
import argparse


def run_demo(gs_object, mesh_assets_folder, default_gs_config, buffer_mode):
    """
    How to run:
    > python playground.py --gs_object <ckpt_path>
                          [--mesh_assets <mesh_folder_path>]
                          [--default_gs_config <config_name>]
                          [--buffer_mode <host2device | device2device>]
    """
    playground = Playground(gs_object, mesh_assets_folder, default_gs_config, buffer_mode)
    playground.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gs_object',
        type=str,
        required=True,
        help="Path of pretrained 3dgrt checkpoint, as .pt / .ingp / .ply file."
    )
    parser.add_argument(
        '--mesh_assets',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'playground', 'assets'),
        help="Path to folder containing mesh assets of .obj or .glb format."
    )
    parser.add_argument(
        '--default_gs_config',
        type=str,
        default='apps/colmap_3dgrt.yaml',
        help="Name of default config to use for .ingp, .ply files, or .pt files not trained with 3dgrt."
    )
    parser.add_argument(
        '--buffer_mode',
        type=str,
        choices=["host2device", "device2device"],
        default="device2device",
        help="Buffering mode for passing rendered data from CUDA to OpenGL screen buffer."
             "Using device2device is recommended."
    )
    args = parser.parse_args()

    run_demo(
        gs_object=args.gs_object,
        mesh_assets_folder=args.mesh_assets,
        default_gs_config=args.default_gs_config,
        buffer_mode=args.buffer_mode
    )
