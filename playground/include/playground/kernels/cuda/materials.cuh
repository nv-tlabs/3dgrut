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
#ifdef __PLAYGROUND__MODE__

#include <optix.h>
#include <playground/pipelineParameters.h>
#include <playground/kernels/cuda/mathUtils.cuh>
#include <playground/kernels/cuda/trace.cuh>
#include <playground/kernels/cuda/rng.cuh>

extern "C"
{
    #ifndef __PLAYGROUND__PARAMS__
    __constant__ PlaygroundPipelineParameters params;
    #define __PLAYGROUND__PARAMS__ 1
    #endif
}

static __device__ __inline__ const PBRMaterial& get_material(const unsigned int matId)
{
    return params.materials[matId];
}

static __device__ __inline__ float3 get_diffuse_color(const float3 ray_d, float3 normal)
{
    const unsigned int triId = optixGetPrimitiveIndex();
    const unsigned int v0_idx = params.triangles[triId][0];
    const unsigned int v1_idx = params.triangles[triId][1];
    const unsigned int v2_idx = params.triangles[triId][2];

    const unsigned int materialId = params.matID[triId][0];
    const auto material = get_material(materialId);

    const float2 uv0 = make_float2(params.matUV[v0_idx][0], params.matUV[v0_idx][1]);
    const float2 uv1 = make_float2(params.matUV[v1_idx][0], params.matUV[v1_idx][1]);
    const float2 uv2 = make_float2(params.matUV[v2_idx][0], params.matUV[v2_idx][1]);
    const float2 barycentric = optixGetTriangleBarycentrics();
    float2 texCoords = (1 - barycentric.x - barycentric.y) * uv0 + barycentric.x * uv1 + barycentric.y * uv2;

    float3 diffuse;
    bool disableTextures = params.playgroundOpts & PGRNDRenderDisablePBRTextures;

    float3 diffuseFactor = make_float3(material.diffuseFactor.x, material.diffuseFactor.y, material.diffuseFactor.z);
    if (!material.useDiffuseTexture || disableTextures)
    {
        diffuse = diffuseFactor;
    }
    else
    {
        cudaTextureObject_t diffuseTex = material.diffuseTexture;
        float4 diffuse_fp4 = tex2D<float4>(diffuseTex, texCoords.x, texCoords.y);
        diffuse = make_float3(diffuse_fp4.x, diffuse_fp4.y, diffuse_fp4.z);
        diffuse *= diffuseFactor;
    }

    float shade = fabsf(dot(ray_d, normal));
    return diffuse * shade;
}


#endif