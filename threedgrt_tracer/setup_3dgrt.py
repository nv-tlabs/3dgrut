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


# ----------------------------------------------------------------------------
#
def setup_3dgrt(conf):
    include_paths = []

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == "nt":
        include_paths.append(os.path.dirname(__file__) + r"\include")
        include_paths.append(os.path.dirname(__file__) + r"\dependencies\optix-dev\include")

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
        include_paths.append(os.path.dirname(__file__) + r"/include")
        include_paths.append(os.path.dirname(__file__) + r"/dependencies/optix-dev/include")

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

    # List of sources.
    source_files = [
        "src/optixTracer.cpp",
        "src/particlePrimitives.cu",
        "bindings.cpp",
    ]

    # Compile slang kernels
    # TODO: do not overwrite files, use config hash to register the needed version
    import importlib, subprocess

    slang_mod = importlib.import_module("slangtorch")
    slang_build_env = os.environ
    slang_build_env["PATH"] += ";" if os.name == "nt" else ":"
    slang_build_env["PATH"] += os.path.join(os.path.dirname(slang_mod.__file__), "bin")
    slang_build_inc_dir = os.path.join(os.path.dirname(__file__), "include")
    slang_build_file_dir = os.path.join(os.path.dirname(__file__), "include/3dgrt/kernels/slang/")
    subprocess.check_call(
        [
            "slangc",
            "-target", "cuda",
            "-I", slang_build_inc_dir,
            "-line-directive-mode", "none",
            "-matrix-layout-row-major", # NB : this is required for cuda target
            "-O2",
            f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
            f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
            f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
            f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
            f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
            f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={conf.render.enable_normals}",
            f"-DGAUSSIAN_PARTICLE_SURFEL={conf.render.primitive_type=='trisurfel'}",
            f"{os.path.join(slang_build_file_dir,'models/gaussianParticles.slang')}",
            f"{os.path.join(slang_build_file_dir,'models/shRadiativeParticles.slang')}",
            "-o",
            f"{os.path.join(slang_build_file_dir,'gaussianParticles.cuh')}",
        ],
        env=slang_build_env,
    )

    if os.name == "posix":
        use_cugl_interop = False
        try:
            import polyscope
            use_cugl_interop = True   # polyscope installed, possibly without cupy / cuda-python available
            import cupy
            import cuda
            use_cugl_interop = False  # polyscope has access to preset bindings from cupy / cuda-python
        except ImportError:
            use_cugl_interop = False  # polyscope not installed or polyscope installed + cupy / cuda-python available
        if use_cugl_interop:
            include_paths.append(os.path.dirname(__file__) + r"/dependencies/cugl/include")
            cc_flags.append("-DUSE_CUGL_INTEROP=1")

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(
        name="lib3dgrt_cc",
        sources=source_paths,
        extra_cflags=cc_flags,
        extra_cuda_cflags=nvcc_flags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        with_cuda=True,
        verbose=True,
    )
