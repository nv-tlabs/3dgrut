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

"""
Background USD writer for exporting environment lighting.

Exports 3DGRUT background models to USD DomeLight with environment maps.
- Solid colors (non-black) export as 1x1 textures
- Black backgrounds are omitted
"""

import io
import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdLux

from threedgrut.model.background import BackgroundColor, SkipBackground

logger = logging.getLogger(__name__)

# Default DomeLight intensity
DEFAULT_DOME_INTENSITY = 1.0


def _tensor_to_tuple(color: torch.Tensor) -> Tuple[float, float, float]:
    """Convert a torch tensor color to a tuple of floats."""
    if color.is_cuda:
        color = color.cpu()
    arr = color.numpy()
    return tuple(float(c) for c in arr[:3])


def get_background_color(
    background,
    conf=None,
) -> Optional[Tuple[float, float, float]]:
    """
    Extract the background color from a 3DGRUT background model.

    Args:
        background: Background model (BackgroundColor, SkipBackground, etc.)
        conf: Optional configuration

    Returns:
        Tuple of (r, g, b) in [0, 1] range, or None if no solid color
    """
    if background is None:
        return None

    # Handle SkipBackground - no color to export
    if isinstance(background, SkipBackground):
        logger.debug("SkipBackground detected, no color to export")
        return None

    # Handle BackgroundColor
    if isinstance(background, BackgroundColor):
        color_type = background.background_color_type
        if color_type == "black":
            return (0.0, 0.0, 0.0)
        elif color_type == "white":
            return (1.0, 1.0, 1.0)
        elif color_type == "random":
            # Random backgrounds default to black when not training
            return (0.0, 0.0, 0.0)
        else:
            # Unexpected color type - log and try to use stored color
            logger.warning(f"Unexpected background_color_type: {color_type}")
            return _tensor_to_tuple(background.color)

    # Unknown background type - log warning and return None
    logger.warning(f"Unknown background type: {type(background).__name__}, cannot extract color")
    return None


def is_black_background(color: Optional[Tuple[float, float, float]], threshold: float = 1e-6) -> bool:
    """Check if the background color is effectively black."""
    if color is None:
        return True
    return all(c < threshold for c in color)


def create_1x1_envmap_bytes(color: Tuple[float, float, float], format: str = "PNG") -> bytes:
    """
    Create a 1x1 environment map texture as bytes.

    Args:
        color: RGB color tuple in [0, 1] range
        format: Image format (PNG or HDR)

    Returns:
        Image bytes
    """
    # Create 1x1 RGB image
    r = int(np.clip(color[0] * 255, 0, 255))
    g = int(np.clip(color[1] * 255, 0, 255))
    b = int(np.clip(color[2] * 255, 0, 255))

    img = Image.new("RGB", (1, 1), (r, g, b))

    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


def create_envmap_hdr_bytes(color: Tuple[float, float, float]) -> bytes:
    """
    Create a 1x1 HDR environment map.

    Simple Radiance HDR format (RGBE encoding).

    Args:
        color: RGB color tuple in [0, 1] range (or higher for HDR)

    Returns:
        HDR image bytes
    """
    # Simple 1x1 HDR in Radiance format
    r, g, b = color
    max_val = max(r, g, b)

    if max_val < 1e-32:
        # Black - use all zeros with exponent 0
        rgbe = bytes([0, 0, 0, 0])
    else:
        # RGBE encoding
        exp_val = int(math.ceil(math.log2(max_val)))
        scale = 255.0 / (2.0**exp_val)
        re = int(np.clip(r * scale, 0, 255))
        ge = int(np.clip(g * scale, 0, 255))
        be = int(np.clip(b * scale, 0, 255))
        e = exp_val + 128
        rgbe = bytes([re, ge, be, e])

    # Radiance HDR header
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n"
    return header + rgbe


def export_background_to_usd(
    stage: Usd.Stage,
    background=None,
    conf=None,
    root_path: str = "/World/Environment",
    envmap_filename: str = "envmap.png",
) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Export 3DGRUT background to USD as a DomeLight with environment map.

    Per the spec:
    - Solid non-black colors export as 1x1 textures
    - Black backgrounds are omitted (return None)

    Args:
        stage: USD stage to export to
        background: 3DGRUT background model
        conf: Optional configuration
        root_path: USD path for environment lights
        envmap_filename: Filename for the environment map texture

    Returns:
        Tuple of (dome_light_path, envmap_bytes) or (None, None) if black/skipped
    """
    # Get background color
    color = get_background_color(background, conf)

    # Skip black backgrounds
    if is_black_background(color):
        logger.info("Black background detected, skipping environment export")
        return None, None

    logger.info(f"Exporting background color {color} to DomeLight")

    # Create environment root
    env_xform = UsdGeom.Xform.Define(stage, root_path)

    # Create DomeLight
    dome_path = f"{root_path}/DomeLight"
    dome_light = UsdLux.DomeLight.Define(stage, dome_path)

    # Set intensity
    dome_light.GetIntensityAttr().Set(DEFAULT_DOME_INTENSITY)

    # Create 1x1 environment map texture
    envmap_bytes = create_1x1_envmap_bytes(color, format="PNG")

    # Set texture file path (relative path in USDZ)
    texture_path = f"./{envmap_filename}"
    dome_light.GetTextureFileAttr().Set(Sdf.AssetPath(texture_path))

    # Set texture format hint
    dome_light.GetTextureFormatAttr().Set("latlong")

    logger.info(f"Created DomeLight at {dome_path} with 1x1 envmap")
    return dome_path, envmap_bytes


def export_background_color_to_usd(
    stage: Usd.Stage,
    color: Tuple[float, float, float],
    root_path: str = "/World/Environment",
    envmap_filename: str = "envmap.png",
) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Export a specific RGB color as a DomeLight environment.

    Convenience function when you have a color tuple directly.

    Args:
        stage: USD stage to export to
        color: RGB color tuple in [0, 1] range
        root_path: USD path for environment lights
        envmap_filename: Filename for the environment map texture

    Returns:
        Tuple of (dome_light_path, envmap_bytes) or (None, None) if black
    """
    # Skip black
    if is_black_background(color):
        logger.info("Black color provided, skipping environment export")
        return None, None

    logger.info(f"Exporting color {color} to DomeLight")

    # Create environment root
    UsdGeom.Xform.Define(stage, root_path)

    # Create DomeLight
    dome_path = f"{root_path}/DomeLight"
    dome_light = UsdLux.DomeLight.Define(stage, dome_path)

    # Set intensity
    dome_light.GetIntensityAttr().Set(DEFAULT_DOME_INTENSITY)

    # Create 1x1 environment map texture
    envmap_bytes = create_1x1_envmap_bytes(color, format="PNG")

    # Set texture file path
    texture_path = f"./{envmap_filename}"
    dome_light.GetTextureFileAttr().Set(Sdf.AssetPath(texture_path))

    # Set texture format
    dome_light.GetTextureFormatAttr().Set("latlong")

    return dome_path, envmap_bytes
