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
#
#
# Selective Adam implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/optimizers/selective_adam.py),
# which is based on the original implementation https://github.com/humansensinglab/taming-3dgs that uderlines the work
#
# Taming 3DGS: High-Quality Radiance Fields with Limited Resources by
# Saswat Subhajyoti Mallick*, Rahul Goel*, Bernhard Kerbl, Francisco Vicente Carrasco, Markus Steinberger and Fernando De La Torre
#
# If you use this code in your research, please cite the above works.


import os
import torch


def setup_lib_optimizers_cc():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    cpp_standard = 17

    nvcc_flags = [
        f"-std=c++{cpp_standard}",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        # The following definitions must be undefined
        # since TCNN requires half-precision operation.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]

    if os.name == "posix":
        cflags = [f"-std=c++{cpp_standard}"]
        nvcc_flags += [
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
        ]
    elif os.name == "nt":
        cflags = [f"/std:c++{cpp_standard}", "/DNOMINMAX"]

    include_paths = [root_dir]

    # Linker options.
    if os.name == "posix":
        ldflags = ["-lcuda", "-lnvrtc"]
    elif os.name == "nt":
        ldflags = ["cuda.lib", "advapi32.lib", "nvrtc.lib"]

    build_dir = torch.utils.cpp_extension._get_build_directory(
        "lib_optimizers_cc", verbose=True
    )

    return torch.utils.cpp_extension.load(
        name="lib_optimizers_cc",
        sources=[
            os.path.join(root_dir, "optimizers.cu"),
            os.path.join(root_dir, "optimizers.cpp"),
        ],
        extra_cflags=cflags,
        extra_cuda_cflags=nvcc_flags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
        with_cuda=True,
        verbose=True,
    )
