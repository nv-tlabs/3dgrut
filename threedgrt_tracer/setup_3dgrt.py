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

from threedgrut.utils import jit
from threedgrut.model.features import Features


# ----------------------------------------------------------------------------
#
def setup_3dgrt(conf):
    def to_cpp_bool(value):
        return "true" if value else "false"

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

    include_paths = []
    include_paths.append(os.path.join(os.path.dirname(__file__), "include"))
    include_paths.append(os.path.join(os.path.dirname(__file__), "dependencies", "optix-dev", "include"))

    # Compiler options. Same -D for feature dims so pipelineParameters.h and JIT OptiX pipeline (generateDefines) stay in sync.
    cflags = [
        *transform_defines,
    ]
    cuda_flags = [
        # Feature-based radiance dimensions (must match Slang compilation)
        *transform_defines,
        *nht_defines,
        *half_defines,
        # Other particle parameters
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
    ]
    # When PARTICLE_FEATURE_HALF=1 the Slang-generated header uses __half types;
    # the Slang prelude only pulls in <cuda_fp16.h> and defines __half when
    # SLANG_CUDA_ENABLE_HALF is set.
    if conf.render.particle_feature_half or conf.render.feature_output_half:
        cuda_flags.append("-DSLANG_CUDA_ENABLE_HALF=1")

    # List of sources.
    source_files = [
        "src/optixTracer.cpp",
        "src/particlePrimitives.cu",
        "bindings.cpp",
    ]

    # Compile slang kernels
    slang_build_dir = os.path.join(os.path.dirname(__file__), "include", "3dgrt", "kernels", "slang")
    jit.compile_slang_kernel(
        kernel_files=[
            f"{os.path.join(slang_build_dir,'models/gaussianParticles.slang')}",
            f"{os.path.join(slang_build_dir,'models/radiativeParticles.slang')}",
        ],
        output_file=f"{os.path.join(slang_build_dir, 'gaussianParticles.cuh')}",
        defines=[
            f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
            f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
            f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
            f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
            f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
            f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={to_cpp_bool(conf.render.enable_normals)}",
            f"-DGAUSSIAN_PARTICLE_SURFEL={to_cpp_bool(conf.render.primitive_type=='trisurfel')}",
            # Feature-based radiance dimensions
            *transform_defines,
            *nht_defines,
            *half_defines,
        ],
        include_paths=[
            os.path.join(os.path.dirname(__file__), "include"),
        ],
    )

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    return jit.load(
        name="lib3dgrt_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=include_paths,
    )
