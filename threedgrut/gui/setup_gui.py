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

import os
from pathlib import Path

from threedgrut.utils import jit


# ----------------------------------------------------------------------------
#
def setup_gui():
    ROOT = os.path.dirname(__file__)

    # Make sure we can find the necessary compiler and libary binaries.
    include_paths = []
    include_paths.append(os.path.join(ROOT, "include"))

    # Compiler options.
    cflags = ["-DUSE_CUGL_INTEROP=1"]
    cuda_cflags = []
    source_files = ["bindings.cpp"]

    # Compile and load.
    source_paths = [os.path.abspath(os.path.join(ROOT, fn)) for fn in source_files]
    jit.load(
        name="lib3dgrut_gui_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
    )
