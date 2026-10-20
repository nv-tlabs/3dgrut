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

"""ParticleField schema hint tokens supported by usd-core 26.5+."""

DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT = "cameraDistance"

PARTICLE_FIELD_SORTING_MODE_HINTS = (
    "zDepth",
    "cameraDistance",
    "rayHitDistance",
)


def normalize_particle_field_sorting_mode_hint(value: str) -> str:
    """Normalize and validate a ParticleField sortingModeHint token."""
    normalized = str(value).strip()
    if normalized not in PARTICLE_FIELD_SORTING_MODE_HINTS:
        raise ValueError(
            f"Unsupported ParticleField sortingModeHint '{value}'. "
            f"Expected one of: {list(PARTICLE_FIELD_SORTING_MODE_HINTS)}"
        )
    return normalized


def resolve_particle_field_sorting_mode_hint(user_hint, conf=None) -> str:
    """Pick the ParticleField sortingModeHint for an export.

    Precedence: an explicit ``user_hint`` always wins. Otherwise it is derived from the
    training render settings in ``conf``: 3DGRT sorts by ray hit distance; 3DGUT
    uses ray hit distance when its k-buffer is enabled, otherwise it follows
    ``render.splat.global_z_order`` (True: zDepth, False: cameraDistance).
    Without a known render method (e.g. a PLY source) the default is used.
    """
    if user_hint is not None:
        return normalize_particle_field_sorting_mode_hint(user_hint)
    render = getattr(conf, "render", None)
    method = getattr(render, "method", None)
    if method == "3dgrt":
        return "rayHitDistance"
    if method == "3dgut":
        splat = getattr(render, "splat", None)
        k_buffer_size = getattr(splat, "k_buffer_size", 0)
        if k_buffer_size is not None and k_buffer_size > 0:
            return "rayHitDistance"
        global_z_order = getattr(splat, "global_z_order", None)
        if global_z_order is not None:
            return "zDepth" if global_z_order else "cameraDistance"
    return DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT
