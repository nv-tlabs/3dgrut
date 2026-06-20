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

"""Shared CLI arguments for the canonical object-frame option (export + transcode)."""

import argparse


def add_frame_arguments(
    parser: argparse.ArgumentParser, *, include_cameras: bool = False, default: str = "none"
) -> None:
    """Add the unified frame options to a CLI parser.

    ``--frame`` selects the scene-normalizing transform source: 'none' (identity),
    'pca' (robust geometry-based canonical frame), and (when ``include_cameras``) 'cameras'
    (from dataset poses). Authored on /World for USD/NuRec; baked into PLY.
    """
    choices = ["none", "pca"] + (["cameras"] if include_cameras else [])
    parser.add_argument(
        "--frame",
        dest="frame_mode",
        choices=choices,
        default=default,
        help=(
            "Re-frame the object: 'none' keeps the source frame; 'pca' centers the centroid and "
            "aligns axes to the principal axes"
            + (" ; 'cameras' uses dataset camera poses" if include_cameras else "")
            + ". Authored on /World for USD/NuRec, baked into PLY."
        ),
    )
    parser.add_argument(
        "--up-axis",
        dest="up_axis",
        choices=["y", "z"],
        default="y",
        help="World up axis for the canonical frame and USD upAxis metadata. Default: y.",
    )
    parser.add_argument(
        "--frame-origin",
        dest="frame_origin",
        choices=["centroid", "plane"],
        default="centroid",
        help=(
            "Origin for --frame pca: 'centroid' (weighted center of mass) or 'plane' (in-plane "
            "centroid with up=0 at a robust low percentile, i.e. resting on the base)."
        ),
    )
