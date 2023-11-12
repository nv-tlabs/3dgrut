# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import glob

from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

sources = glob.glob("*.cpp") + glob.glob("*.cu")

setup(
    name="packed_ops",
    version="0.1",
    author="zgojcic",
    author_email="zgojcic@nvidia.com",
    description="cuda packed tensors operations",
    long_description="cuda packed tensors operations",
    ext_modules=[
        CUDAExtension(
            name="packed_ops",
            sources=sources,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
