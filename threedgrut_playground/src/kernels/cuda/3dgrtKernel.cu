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

#include <3dgrt/pipelineParameters.h>
extern "C" {
    __constant__ PipelineParameters params;
}
#include <3dgrt/kernels/cuda/3dgrtTracer.cuh>


extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    float3 rayOrigin    = params.rayWorldOrigin(idx);
    float3 rayDirection = params.rayWorldDirection(idx);
    RayData rayData;
    rayData.initialize();

    traceVolumetricGS(rayData, rayOrigin, rayDirection, 1e-9, 1e9);

    params.rayRadiance[idx.z][idx.y][idx.x][0]    = rayData.radiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = rayData.radiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = rayData.radiance.z;
    params.rayDensity[idx.z][idx.y][idx.x][0]     = rayData.density;
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = rayData.hitDistance;
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = rayData.rayLastHitDistance;
#ifdef ENABLE_NORMALS
    params.rayNormal[idx.z][idx.y][idx.x][0] = rayData.normal.x;
    params.rayNormal[idx.z][idx.y][idx.x][1] = rayData.normal.y;
    params.rayNormal[idx.z][idx.y][idx.x][2] = rayData.normal.z;
#endif
#ifdef ENABLE_HIT_COUNTS
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = rayData.hitCount;
#endif
}

extern "C" __global__ void __intersection__is() {
    intersectVolumetricGS();
}

extern "C" __global__ void __anyhit__ah() {
    anyhitSortVolumetricGS();
}
