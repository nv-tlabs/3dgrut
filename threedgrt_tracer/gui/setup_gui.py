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
import torch.utils.cpp_extension
from pathlib import Path

# ----------------------------------------------------------------------------
#
def setup_gui():
    THREEDGRT_ROOT = os.path.join(str(Path(os.path.dirname(__file__)).parent.parent), 'threedgrt_tracer')

    # Make sure we can find the necessary compiler and libary binaries.
    include_paths = []
    include_paths.append(os.path.join(THREEDGRT_ROOT, "include"))
    include_paths.append(os.path.join(THREEDGRT_ROOT, "gui", "include"))

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == "nt":
        def find_cl_path():
            import glob

            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path

    elif os.name == "posix":
        pass

    # Compiler options.
    cc_flags = ["-DNVDR_TORCH"]

    nvcc_flags = [
        "-DNVDR_TORCH",
        "-std=c++17",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fno-strict-aliasing",
    ]

    # Linker options.
    if os.name == "posix":
        ldflags = ["-lcuda", "-lnvrtc"]
    elif os.name == "nt":
        ldflags = ["cuda.lib", "advapi32.lib", "nvrtc.lib"]

    source_files = [
        "gui/bindings.cpp",
    ]
    cc_flags.append("-DUSE_CUGL_INTEROP=1")

    # Compile and load.
    source_paths = [os.path.abspath(os.path.join(THREEDGRT_ROOT, fn)) for fn in source_files]
    torch.utils.cpp_extension.load(
        name="lib3dgrt_gui_cc",
        sources=source_paths,
        extra_cflags=cc_flags,
        extra_cuda_cflags=nvcc_flags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        with_cuda=True,
        verbose=True,
    )
