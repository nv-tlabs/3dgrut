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

import math
import os

import torch

from threedgrut.utils import jit
from threedgrut.model.features import Features


# ----------------------------------------------------------------------------
#
def setup_3dgut(conf):
    build_dir = torch.utils.cpp_extension._get_build_directory("lib3dgut_cc", verbose=True)

    include_paths = []
    prefix = os.path.dirname(__file__)
    include_paths.append(os.path.join(prefix, "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies", "fmt", "include"))

    def to_cpp_bool(value):
        return "true" if value else "false"

    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha * ut_alpha * (ut_d + ut_kappa))

    feat = Features(conf)
    transform_defines = [
        f"-DPARTICLE_FEATURE_DIM={feat.particle_feature_dim}",
        f"-DRAY_FEATURE_DIM={feat.ray_feature_dim}",
        f"-DFEATURE_TRANSFORM_TYPE={feat.transform_type}",
    ]
    nht_defines = [
        f"-DFEATURE_INTERPOLATION_TYPE={feat.interpolation_type}",
        f"-DFEATURE_INTERPOLATION_SUPPORT={feat.interpolation_support}",
        f"-DFEATURE_ACTIVATION_TYPE={feat.activation_type}",
        f"-DFEATURE_ACTIVATION_NUM_FREQUENCIES={feat.activation_num_frequencies}",
        f"-DINTERP_POINT_FEATURE_DIM={feat.interp_point_feature_dim}",
    ]
    half_defines = [
        f"-DPARTICLE_FEATURE_HALF={1 if conf.render.particle_feature_half else 0}",
        f"-DFEATURE_OUTPUT_HALF={1 if conf.render.feature_output_half else 0}",
    ]

    defines = [
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={to_cpp_bool(conf.render.enable_normals)}",
        f"-DGAUSSIAN_PARTICLE_SURFEL={to_cpp_bool(conf.render.primitive_type=='trisurfel')}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        f"-DGAUSSIAN_ENABLE_HIT_COUNT={to_cpp_bool(conf.render.enable_hitcounts)}",
        # Feature-based radiance dimensions
        *transform_defines,
        *nht_defines,
        # Feature buffer memory layout
        *half_defines,
        # Specific to the 3DGUT renderer
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={conf.render.splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={conf.render.splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={to_cpp_bool(conf.render.splat.global_z_order)}",
        f"-DFINE_GRAINED_LOAD_BALANCING={to_cpp_bool(getattr(conf.render.splat, 'fine_grained_load_balancing', False))}",
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

    cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        *defines,
    ]

    cuda_cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-use_fast_math",
        "-O3",
        *defines,
    ]
    # Diagnostic: dump ptxas register / smem / spill stats per kernel.
    # Enable with `export NHT_PTXAS_VERBOSE=1` before launching training.
    if os.environ.get("NHT_PTXAS_VERBOSE", "0") == "1":
        cuda_cflags += [
            "-Xptxas=-v",
            "--resource-usage",
        ]
    # When PARTICLE_FEATURE_HALF=1 the Slang-generated header uses __half types;
    # the Slang prelude only pulls in <cuda_fp16.h> and defines __half when
    # SLANG_CUDA_ENABLE_HALF is set.
    if conf.render.particle_feature_half or conf.render.feature_output_half:
        cuda_cflags.append("-DSLANG_CUDA_ENABLE_HALF=1")

    # List of sources.
    source_files = [
        "src/splatRaster.cpp",
        "src/gutRenderer.cu",
        "src/cudaBuffer.cpp",
        "bindings.cpp",
    ]

    # Compile slang kernels
    slang_build_inc_dir = os.path.join(os.path.dirname(__file__), "include", "3dgut")
    slang_output_file = os.path.join(os.path.dirname(__file__), "include", "threedgutSlang.cuh")

    jit.compile_slang_kernel(
        kernel_files=[f"{os.path.join(slang_build_inc_dir, 'threedgut.slang')}"],
        output_file=slang_output_file,
        include_paths=[
            os.path.join(os.path.dirname(__file__), "include"),
            os.path.join(os.path.dirname(__file__), "..", "threedgrt_tracer", "include"),
        ],
        defines=defines,
    )

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    return jit.load(
        name="lib3dgut_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
    )
