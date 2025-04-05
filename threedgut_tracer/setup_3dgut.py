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


# ----------------------------------------------------------------------------
#
def setup_3dgut(conf):
    include_paths = []

    build_dir = torch.utils.cpp_extension._get_build_directory("lib3dgut_cc", verbose=True)

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

    prefix = os.path.dirname(__file__)
    include_paths.append(os.path.join(prefix, "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies", "fmt", "include"))
    include_paths.append(build_dir)

    def to_cpp_bool(value):
        return "true" if value else "false"

    # Compiler options

    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha*ut_alpha*(ut_d+ut_kappa))

    defines = [
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        # Specific to the 3DGUT renderer
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={conf.render.splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={conf.render.splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={to_cpp_bool(conf.render.splat.global_z_order)}",
        # -- Unscented Transform --
        f"-DGAUSSIAN_UT_ALPHA={ut_alpha}",
        f"-DGAUSSIAN_UT_BETA={ut_beta}",
        f"-DGAUSSIAN_UT_KAPPA={ut_kappa}",
        f"-DGAUSSIAN_UT_DELTA={ut_delta}",
        f"-DGAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR={conf.render.splat.ut_in_image_margin_factor}",
        f"-DGAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID={to_cpp_bool(conf.render.splat.ut_require_all_sigma_points_valid)}",
        # -- Culling --
        f"-DGAUSSIAN_RECT_BOUNDING={to_cpp_bool(conf.render.splat.rect_bounding)}",
        f"-DGAUSSIAN_TIGHT_OPACITY_BOUNDING={to_cpp_bool(conf.render.splat.tight_opacity_bounding)}",
        f"-DGAUSSIAN_TILE_BASED_CULLING={to_cpp_bool(conf.render.splat.tile_based_culling)}",
    ]

    cc_flags = [
        "-DNVDR_TORCH",
        "-DTCNN_MIN_GPU_ARCH=70",
        *defines,
    ]

    nvcc_flags = [
        "-DNVDR_TORCH",
        "-DTCNN_MIN_GPU_ARCH=70",
        "-std=c++17",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-Wno-float-conversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-use_fast_math",
        *defines,
    ]

    # Linker options.
    if os.name == "posix":
        ldflags = ["-lcuda", "-lnvrtc"]
    elif os.name == "nt":
        ldflags = ["cuda.lib", "advapi32.lib", "nvrtc.lib"]

    # List of sources.
    source_files = [
        "src/splatRaster.cu",
        "src/gutRenderer.cu",
        "src/cudaBuffer.cpp",
        "bindings.cpp",
    ]

    # Compile slang kernels
    # TODO: do not overwrite files, use config hash to register the needed version
    import importlib, subprocess

    slang_mod = importlib.import_module("slangtorch")
    slang_dir = os.path.dirname(slang_mod.__file__)

    slang_build_env = os.environ
    slang_build_env["PATH"] += ";" if os.name == "nt" else ":"
    slang_build_env["PATH"] += os.path.join(slang_dir, "bin")
    slang_build_inc_dir = os.path.join(os.path.dirname(__file__), "include", "3dgut")

    subprocess.check_call(
        [
            "slangc", "-target", "cuda",
            "-I", os.path.join(os.path.dirname(__file__), "include"),
            "-I", os.path.join(os.path.dirname(__file__), "..", "threedgrt_tracer", "include"),
            "-line-directive-mode", "none",
            "-matrix-layout-row-major", # NB : this is required for cuda target
            "-Wno-41018",
            "-O2",
            *defines,
            f"{os.path.join(slang_build_inc_dir,'threedgut.slang')}",
            "-o", f"{os.path.join(build_dir,'threedgutSlang.cuh')}",
        ],
        env=slang_build_env,
    )

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]

    torch.utils.cpp_extension.load(
        name="lib3dgut_cc",
        sources=source_paths,
        extra_cflags=cc_flags,
        extra_cuda_cflags=nvcc_flags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
        with_cuda=True,
        verbose=True,
    )
