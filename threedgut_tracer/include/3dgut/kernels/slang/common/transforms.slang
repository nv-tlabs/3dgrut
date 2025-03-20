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

module transforms;

namespace transforms
{

[BackwardDifferentiable]
float3x3 rotationMatrixTranspose(float4 quaternion)
{
    // quaternion represented as (r,x,y,z)
    const float xx = quaternion.y * quaternion.y;
    const float yy = quaternion.z * quaternion.z;
    const float zz = quaternion.w * quaternion.w;
    const float xy = quaternion.y * quaternion.z;
    const float xz = quaternion.y * quaternion.w;
    const float yz = quaternion.z * quaternion.w;
    const float rx = quaternion.x * quaternion.y;
    const float ry = quaternion.x * quaternion.z;
    const float rz = quaternion.x * quaternion.w;

    // Compute rotation matrix from quaternion
    return float3x3(float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry)),
                    float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx)),
                    float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy))));
}

}
