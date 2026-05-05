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
