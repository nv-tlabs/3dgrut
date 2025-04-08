// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
// clang-format off
#include <3dgrt/mathUtils.h>
#include <3dgrt/pipelineDefinitions.h>
// #include <3dgrt/particlePrimitives.h>
// clang-format on


namespace {
inline __host__ uint32_t div_round_up(uint32_t val, uint32_t divisor) {
    return (val + divisor - 1) / divisor;
}

} // namespace

__global__ void computeMeshFaceBufferKernel(
    const uint32_t fNum,
    const float3* verts,
    const int3* faces,
    float3* __restrict__ fPrimVrt,
    int3* __restrict__ fPrimTri) {

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < fNum) {
        const int3 f = faces[idx];
        const float3 v0 = verts[f.x];
        const float3 v1 = verts[f.y];
        const float3 v2 = verts[f.z];
        fPrimVrt[f.x] = v0;
        fPrimVrt[f.y] = v1;
        fPrimVrt[f.z] = v2;
        fPrimTri[idx] = f;
    }
}

void computeMeshFaceBuffer(uint32_t fNum,
                           const float3* verts,
                           const int3* faces,
                           float3* fPrimVrt,
                           int3* fPrimTri,
                           cudaStream_t stream) {
    const uint32_t threads = 1024;
    const uint32_t blocks  = div_round_up(static_cast<uint32_t>(fNum), threads);

    computeMeshFaceBufferKernel<<<blocks, threads, 0, stream>>>(fNum,
                                                                verts,
                                                                faces,
                                                                fPrimVrt,
                                                                fPrimTri);
}
