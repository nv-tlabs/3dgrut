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

import os, math
import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME


def load(
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    with_cuda=True,
    verbose=True,
    *args,
    **kwargs,
):

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == "nt":

        def find_cl_path():
            import glob

            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64"
                        % edition
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError(
                    "Could not locate a supported Microsoft Visual C++ installation"
                )
            os.environ["PATH"] += ";" + cl_path

    elif os.name == "posix":
        pass

    # Compiler flags.
    cflags = [
        "-DNVDR_TORCH",
    ]
    # Add Windows-specific flags
    if os.name == "nt":
        cflags.append("/DNOMINMAX")
    
    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags = [
        "-DNVDR_TORCH",
        "-std=c++17",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fno-strict-aliasing",
    ]
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    # Linker options.
    if os.name == "posix":
        ldflags = [
            # NOTE: ad-hoc fix for CUDA 12.8.1
            f"-L{os.path.join(CUDA_HOME, 'lib', 'stubs')}",
            f"-L{os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'lib')}",
            f"-L{os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'lib', 'stubs')}",
            "-lcuda",
            "-lnvrtc",
        ]
    elif os.name == "nt":
        ldflags = [
            "cuda.lib",
            "advapi32.lib",
            "nvrtc.lib",
        ]
    if extra_ldflags is not None:
        ldflags += extra_ldflags

    # Include paths.
    include_paths = [
        # NOTE: ad-hoc fix for CUDA 12.8.1
        os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include"),
    ]
    if extra_include_paths is not None:
        include_paths += extra_include_paths

    # Load
    return torch.utils.cpp_extension.load(
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        with_cuda=with_cuda,
        verbose=verbose,
        *args,
        **kwargs,
    )
