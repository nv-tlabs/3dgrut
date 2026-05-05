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

"""OpenUSD validation helpers for exported ParticleField / LightField stages."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Stage-wide checks used by export tests; ParticleField-specific validators may be added as USD exposes them.
_LIGHTFIELD_VALIDATOR_NAMES = (
    "usdValidation:StageMetadataChecker",
    "usdValidation:CompositionErrorTest",
)


def validate_exported_usd_stage(path: Path) -> None:
    """
    Run OpenUSD validation on a written .usd / .usda / .usdc / .usdz.

    Intended for outputs from :class:`~threedgrut.export.usd.exporter.USDExporter`
    (ParticleField3DGaussianSplat / LightField). NuRec exports are not validated here.

    If ``UsdValidation`` is missing, validators fail to load, or the registry API is
    unavailable, this function logs at DEBUG and returns without error.

    Args:
        path: Path to the package root file on disk.

    Raises:
        ValueError: Stage cannot be opened, or validators reported errors.
    """
    path = Path(path)
    try:
        from pxr import Usd, UsdValidation
    except ImportError:
        logger.debug("pxr not available; skipping USD validation for %s", path)
        return

    try:
        registry = UsdValidation.ValidationRegistry()
        validators = registry.GetOrLoadValidatorsByName(list(_LIGHTFIELD_VALIDATOR_NAMES))
    except Exception as exc:
        logger.debug("UsdValidation unavailable (%s); skipping USD validation for %s", exc, path)
        return

    if not validators:
        logger.debug("No USD validators loaded; skipping validation for %s", path)
        return

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"USD validation could not open stage: {path}")

    logger.info("Running OpenUSD stage validation on %s", path)
    ctx = UsdValidation.ValidationContext(validators)
    result = ctx.Validate(stage)
    errors = list(result) if result else []
    if errors:
        msg = "\n".join(e.GetMessage() for e in errors)
        raise ValueError(f"USD validation failed for {path}:\n{msg}")
    logger.info("OpenUSD stage validation passed for %s", path)
