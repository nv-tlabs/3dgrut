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

import glob
import pathlib

from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

sources = glob.glob("*.cpp") + glob.glob("*.cu")

setup(
    name="gaussian_mcmc",
    version="0.1",
    author="zgojcic",
    author_email="zgojcic@nvidia.com",
    description="Helper functions for the MCMC Gaussians",
    long_description="Helper functions for the MCMC Gaussians",
    ext_modules=[
        CUDAExtension(
            name="gaussian_mcmc",
            sources=sources,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
