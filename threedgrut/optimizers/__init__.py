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
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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
        cflags = [f"/std:c++{cpp_standard}"]

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


_CC = None

if _CC is None:
    _CC = setup_lib_optimizers_cc()


class SelectiveAdam(torch.optim.Adam):
    """
    A custom optimizer that extends the standard Adam optimizer by
    incorporating selective updates.

    This class is useful for situations where only a subset of parameters
    should be updated at each step, such as in sparse models or in cases where
    parameter visibility is controlled by an external mask.

    Additionally, the operations are fused into a single kernel. This optimizer
    leverages the `adam` function from a CUDA backend for
    optimized sparse updates.

    This is one of the two optimizers mentioned in the Taming3DGS paper. This implementation
    also references gsplat's SelectiveAdam optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).

    Examples:

        >>> N = 100
        >>> param = torch.randn(N, requires_grad=True)
        >>> optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        >>> visibility_mask = torch.cat([torch.ones(50), torch.zeros(50)])  # Visible first half, hidden second half

        >>> # Forward pass
        >>> loss = torch.sum(param ** 2)

        >>> # Backward pass
        >>> loss.backward()

        >>> # Optimization step with selective updates
        >>> optimizer.step(visibility=visibility_mask)

    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params=params, lr=lr, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, visibility):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "More than one tensor in group is not supported"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            import pdb; pdb.set_trace()
            _CC.selective_adam_update(
                param.contiguous(),
                param.grad.contiguous(),
                exp_avg.contiguous(),
                exp_avg_sq.contiguous(),
                visibility.squeeze(),
                lr,
                beta1,
                beta2,
                eps,
            )