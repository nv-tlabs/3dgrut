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
import torch
from threedgrut.utils import jit

def setup_mcmc():
    """Setup MCMC CUDA extensions.
    
    Args:
        conf: Configuration object containing MCMC-specific settings
    """
    
    # Get build directory
    build_dir = torch.utils.cpp_extension._get_build_directory("lib_mcmc_cc", verbose=True)

    # Setup include paths
    include_paths = []
    prefix = os.path.dirname(__file__)
    include_paths.append(os.path.join(prefix, "include"))
    include_paths.append(build_dir)

    cuda_cflags = [
        "-use_fast_math", "-O3",
    ]

    # Source files
    source_files = [
        "gaussian_mcmc.cu",
        "bindings.cpp",
    ]

    # Compile and load
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    jit.load(
        name="lib_mcmc_cc",
        sources=source_paths,
        extra_cflags=[],
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
    )