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

#include <optix.h>

void computeGaussianEnclosingIcosaHedron(uint32_t gNum,
                                         const float3* gPos,
                                         const float4* gRot,
                                         const float3* gScl,
                                         const float* gDns,
                                         float kernelMinResponse,
                                         uint32_t opts,
                                         float degree,
                                         float3* gPrimVrt,
                                         int3* gPrimTri,
                                         OptixAabb* gPrimAABB,
                                         cudaStream_t stream);

void computeGaussianEnclosingOctaHedron(uint32_t gNum,
                                        const float3* gPos,
                                        const float4* gRot,
                                        const float3* gScl,
                                        const float* gDns,
                                        float kernelMinResponse,
                                        uint32_t opts,
                                        float degree,
                                        float3* gPrimVrt,
                                        int3* gPrimTri,
                                        OptixAabb* gPrimAABB,
                                        cudaStream_t stream);

void computeGaussianEnclosingTriHexa(uint32_t gNum,
                                     const float3* gPos,
                                     const float4* gRot,
                                     const float3* gScl,
                                     const float* gDns,
                                     float kernelMinResponse,
                                     uint32_t opts,
                                     float degree,
                                     float3* gPrimVrt,
                                     int3* gPrimTri,
                                     OptixAabb* gPrimAABB,
                                     cudaStream_t stream);

void computeGaussianEnclosingTriSurfel(uint32_t gNum,
                                       const float3* gPos,
                                       const float4* gRot,
                                       const float3* gScl,
                                       const float* gDns,
                                       float kernelMinResponse,
                                       uint32_t opts,
                                       float degree,
                                       float3* gPrimVrt,
                                       int3* gPrimTri,
                                       OptixAabb* gPrimAABB,
                                       float4* gNormalDensity,
                                       cudaStream_t stream);

void computeGaussianEnclosingTetraHedron(uint32_t gNum,
                                         const float3* gPos,
                                         const float4* gRot,
                                         const float3* gScl,
                                         const float* gDns,
                                         float kernelMinResponse,
                                         uint32_t opts,
                                         float degree,
                                         float3* gPrimVrt,
                                         int3* gPrimTri,
                                         OptixAabb* gPrimAABB,
                                         cudaStream_t stream);

void computeGaussianEnclosingDiamond(uint32_t gNum,
                                     const float3* gPos,
                                     const float4* gRot,
                                     const float3* gScl,
                                     const float* gDns,
                                     float kernelMinResponse,
                                     uint32_t opts,
                                     float degree,
                                     float3* gPrimVrt,
                                     int3* gPrimTri,
                                     OptixAabb* gPrimAABB,
                                     cudaStream_t stream);

void computeGaussianEnclosingSphere(uint32_t gNum,
                                    const float3* gPos,
                                    const float4* gRot,
                                    const float3* gScl,
                                    const float* gDns,
                                    float kernelMinResponse,
                                    uint32_t opts,
                                    float degree,
                                    float3* gPrimCenter,
                                    float* gPrimRadius,
                                    OptixAabb* gPrimAABB,
                                    cudaStream_t stream);

void computeGaussianEnclosingAABB(uint32_t gNum,
                                  const float3* gPos,
                                  const float4* gRot,
                                  const float3* gScl,
                                  const float* gDns,
                                  float kernelMinResponse,
                                  uint32_t opts,
                                  float degree,
                                  OptixAabb* gPrimAABB,
                                  OptixAabb* gAABB,
                                  cudaStream_t stream);

void computeGaussianEnclosingInstances(uint32_t gNum,
                                       const float3* gPos,
                                       const float4* gRot,
                                       const float3* gScl,
                                       const float* gDns,
                                       float kernelMinResponse,
                                       uint32_t opts,
                                       float degree,
                                       OptixTraversableHandle ias,
                                       OptixInstance* gPrimInstances,
                                       OptixAabb* gAABB,
                                       cudaStream_t stream);
