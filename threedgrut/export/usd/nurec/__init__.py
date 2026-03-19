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
NuRec/Omniverse-specific USD export format.

This format is maintained for backward compatibility with Omniverse Kit
and Isaac Sim. For new integrations, prefer the standard USDExporter
with ParticleField3DGaussianSplat schema.
"""

from threedgrut.export.usd.nurec.exporter import NuRecExporter
from threedgrut.export.usd.nurec.serializer import serialize_nurec_usd
from threedgrut.export.usd.nurec.templates import NamedSerialized, fill_3dgut_template

__all__ = [
    "NuRecExporter",
    "NamedSerialized",
    "fill_3dgut_template",
    "serialize_nurec_usd",
]
