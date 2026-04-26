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
USD prim writers for Gaussian splatting data.

Provides schema-agnostic interface for writing Gaussian data to USD:
- GaussianLightFieldWriter: ParticleField3DGaussianSplat schema
- export_cameras_to_usd: one Camera prim per physical camera, animated xforms
- create_render_products: /Render scope with per-camera RenderProducts
- add_ppisp_to_all_render_products: PPISP SPG shader on RenderProducts
- add_ov_post_processing: Omniverse RTX post-processing PPISP workaround
"""

from threedgrut.export.usd.writers.background import export_background_to_usd
from threedgrut.export.usd.writers.base import GaussianUSDWriter, create_gaussian_writer
from threedgrut.export.usd.writers.camera import export_cameras_to_usd
from threedgrut.export.usd.writers.lightfield import GaussianLightFieldWriter
from threedgrut.export.usd.writers.ov_post_processing import add_ov_post_processing
from threedgrut.export.usd.writers.ppisp_writer import add_ppisp_to_all_render_products
from threedgrut.export.usd.writers.render_product import create_render_products

__all__ = [
    "GaussianUSDWriter",
    "GaussianLightFieldWriter",
    "create_gaussian_writer",
    "export_cameras_to_usd",
    "export_background_to_usd",
    "create_render_products",
    "add_ppisp_to_all_render_products",
    "add_ov_post_processing",
]
