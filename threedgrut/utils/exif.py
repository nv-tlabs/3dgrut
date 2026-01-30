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

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import piexif  # type: ignore


def _extract_shutter_time(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    TAG_EXPOSURE_TIME = 33434  # ExposureTime (seconds)
    TAG_SHUTTER_SPEED_VALUE = 37377  # ShutterSpeedValue (APEX Tv)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    if TAG_EXPOSURE_TIME in exif_ifd:
        num, den = exif_ifd[TAG_EXPOSURE_TIME]
        seconds = num / den
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    if TAG_SHUTTER_SPEED_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_SHUTTER_SPEED_VALUE]
        tv = num / den
        seconds = math.pow(2.0, -tv)
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    return None


def _extract_aperture_fnumber(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    TAG_FNUMBER = 33437  # FNumber (f-number)
    TAG_APERTURE_VALUE = 37378  # ApertureValue (APEX Av)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    if TAG_FNUMBER in exif_ifd:
        num, den = exif_ifd[TAG_FNUMBER]
        fnum = num / den
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    if TAG_APERTURE_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_APERTURE_VALUE]
        av = num / den
        fnum = math.pow(2.0, av / 2.0)
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    return None


def _extract_iso(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    # PhotographicSensitivity / ISOSpeedRatings
    TAG_PHOTOGRAPHIC_SENSITIVITY = 34855
    TAG_STANDARD_OUTPUT_SENSITIVITY = 34857  # StandardOutputSensitivity (SOS)
    TAG_RECOMMENDED_EXPOSURE_INDEX = 34858  # RecommendedExposureIndex (REI)
    TAG_ISO_SPEED = 34859  # ISOSpeed
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    candidates: List[int] = [
        TAG_PHOTOGRAPHIC_SENSITIVITY,
        TAG_RECOMMENDED_EXPOSURE_INDEX,
        TAG_STANDARD_OUTPUT_SENSITIVITY,
        TAG_ISO_SPEED,
    ]

    for tag in candidates:
        if tag in exif_ifd:
            value = float(exif_ifd[tag])
            if value > 0.0 and math.isfinite(value):
                return value

    return None


def compute_exposure_from_exif(path: Path) -> Optional[float]:
    """Return exposure in EV stops (log2 of relative exposure) or None if unavailable.

    Relative exposure is computed as (seconds / f^2 * ISO) then converted via log2.
    Returns None if the file format doesn't support EXIF (e.g., PNG).
    """
    try:
        exif = piexif.load(str(path))
    except piexif.InvalidImageDataError:
        # File format doesn't support EXIF (e.g., PNG)
        return None
    shutter_s = _extract_shutter_time(exif)
    aperture_f = _extract_aperture_fnumber(exif)
    iso_value = _extract_iso(exif)

    # If none of the components are available, we cannot compute exposure
    if shutter_s is None and aperture_f is None and iso_value is None:
        return None

    # Use available components; treat missing ones as 1 for exposure calculation
    seconds = shutter_s if shutter_s is not None else 1.0
    f_number = aperture_f if aperture_f is not None else 1.0
    iso = iso_value if iso_value is not None else 1.0

    rel_exposure = (seconds / (f_number * f_number)) * iso
    if rel_exposure <= 0.0 or not math.isfinite(rel_exposure):
        return None
    return math.log2(rel_exposure)


def load_exif_exposures(image_paths: List[Path]) -> List[Optional[float]]:
    """Load EXIF exposure data for a list of images and return mean-normalized values.

    Extracts exposure time, aperture, and ISO from EXIF metadata for each image,
    computes log2 relative exposure, subtracts the mean across all valid frames,
    and returns the normalized values.

    Args:
        image_paths: Paths to all images (both train and val, before splitting)

    Returns:
        List of mean-normalized log2 exposure values. Returns None for images
        where EXIF data is unavailable or invalid. The mean is computed only
        from frames with valid EXIF data.
    """
    from threedgrut.utils.logger import logger

    raw_exposures: List[Optional[float]] = []
    for path in image_paths:
        try:
            exp = compute_exposure_from_exif(path)
        except Exception:
            exp = None
        raw_exposures.append(exp)

    # Compute mean from valid values only
    valid_exposures = [e for e in raw_exposures if e is not None]
    valid_count = len(valid_exposures)
    total_count = len(image_paths)

    if valid_count > 0:
        mean_exposure = sum(valid_exposures) / valid_count
        # Normalize: subtract mean from valid values
        normalized: List[Optional[float]] = [(e - mean_exposure) if e is not None else None for e in raw_exposures]
    else:
        mean_exposure = 0.0
        normalized = raw_exposures  # All None

    # Log summary
    if valid_count == 0:
        logger.info(f"📷 EXIF: No exposure data found in {total_count} images")
    elif valid_count == total_count:
        logger.info(f"📷 EXIF: Loaded exposure for all {total_count} images " f"(mean: {mean_exposure:.2f} EV)")
    else:
        logger.info(
            f"📷 EXIF: Loaded exposure for {valid_count}/{total_count} images "
            f"(mean: {mean_exposure:.2f} EV, {total_count - valid_count} missing)"
        )

    return normalized
