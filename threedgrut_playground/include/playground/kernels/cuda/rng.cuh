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

#pragma once

#include <playground/kernels/cuda/mathUtils.cuh>

#ifdef __CUDACC__

template <unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int& prev)
{
    prev = (prev * 8121 + 28411) % 134456;
    return prev;
}

// Generate random float in [0, 1) - using Linear Congruential Generator (LCG)
static __host__ __device__ __inline__ float rnd(unsigned int& prev)
{
    return ((float)lcg(prev) / (float)0x01000000);
}

// Generate random float3 in [0, 1) - using Linear Congruential Generator (LCG)
static __host__ __device__ __inline__ float3 rnd3(unsigned int& prev)
{
    return make_float3(rnd(prev),rnd(prev),rnd(prev));
}

// Generate random float3 in [0, 1) - using Permuted Congruential Generator (PCG)
static __device__ __inline__ float3 rnd_pcg3d(uint3 v)
{
  v.x = v.x * 1664525u + 1013904223u;
  v.y = v.y * 1664525u + 1013904223u;
  v.z = v.z * 1664525u + 1013904223u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  v.x ^= v.x >> 16u; v.y ^= v.y >> 16u; v.z ^= v.z >> 16u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  return make_float3(v.x, v.y, v.z) * (1.0/float(0xffffffffu));
}

#endif