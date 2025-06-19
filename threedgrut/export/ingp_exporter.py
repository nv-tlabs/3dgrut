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

import gzip
from pathlib import Path
from typing import Any

import msgpack
import torch

from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.utils.logger import logger


class INGPExporter(ModelExporter):
    """Exporter for Instant-NGP (INGP) format files.

    Implements export functionality for Gaussian models
    in the INGP file format, which uses msgpack with gzip compression.
    """

    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, force_half: bool = False, **kwargs) -> None:
        """Export the model to an INGP file.

        Args:
            model: The model to export (must implement ExportableModel)
            output_path: Path where the INGP file will be saved
            dataset: Optional dataset (not used for INGP export)
            conf: Optional configuration (not used for INGP export)
            force_half: Whether to force use of float16 precision
            **kwargs: Additional parameters (not used for INGP export)
        """
        positions = model.get_positions()
        export_dtype = torch.float16 if force_half else positions.dtype
        logger.info(f"exporting ingp file to {output_path}...")
        mogt_config: dict[str, Any] = {}
        mogt_config["nre_data"] = {"version": "0.0.1", "model": "mogt"}
        mogt_config["precision"] = "half" if export_dtype == torch.float16 else "single"
        mogt_config["mog_num"] = positions.shape[0]
        mogt_config["mog_sph_degree"] = model.get_max_n_features()
        mogt_config["mog_positions"] = (
            positions.flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_scales"] = (
            model.get_scale(preactivation=True).flatten().to(
                dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_rotations"] = (
            model.get_rotation(preactivation=True).flatten().to(
                dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        mogt_config["mog_densities"] = (
            model.get_density(preactivation=True).flatten().to(
                dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        features = torch.cat((model.get_features_albedo(),
                             model.get_features_specular()), dim=1)
        mogt_config["mog_features"] = (
            features.flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        )
        with gzip.open(output_path, "wb") as f:
            packed = msgpack.packb(mogt_config)
            f.write(packed)
