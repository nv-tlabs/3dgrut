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
from threedgrut.render import Renderer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument("--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt")
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--save-gt", action="store_false", help="If set, the GT images will not be saved [True by default]")
    parser.add_argument("--compute-extra-metrics", action="store_false", help="If set, extra image metrics will not be computed [True by default]")
    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
                        checkpoint_path=args.checkpoint,
                        path=args.path,
                        out_dir=args.out_dir,
                        save_gt=args.save_gt,
                        computes_extra_metrics=args.compute_extra_metrics)

    renderer.render_all()