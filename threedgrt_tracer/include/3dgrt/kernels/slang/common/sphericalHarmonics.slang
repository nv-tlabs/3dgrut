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

module sphericalHarmonics;

namespace sphericalHarmonics
{

[BackwardDifferentiable]
vector<float, Dim> decode<let Dim : int, let MaxNumCoefficients : int>(
    no_diff in int degree,
    in vector<float, Dim> coefficients[MaxNumCoefficients], 
    in float3 direction)
{
    static const float C0 = 0.28209479177387814;
    static const float C1 = 0.4886025119029199;
    static const float C2[] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                                   -1.0925484305920792, 0.5462742152960396 };
    static const float C3[] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                                   -0.4570457994644658, 1.445305721320277, -0.5900435899266435 };

    vector<float, Dim> features = C0 * coefficients[0];
    if (degree > 0)
    {
        const float x = direction.x;
        const float y = direction.y;
        const float z = direction.z;
        features = features - C1 * y * coefficients[1] + C1 * z * coefficients[2] -
              C1 * x * coefficients[3];

        if (degree > 1)
        {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            features = features + C2[0] * xy * coefficients[4] + C2[1] * yz * coefficients[5] +
                  C2[2] * (2.0 * zz - xx - yy) * coefficients[6] +
                  C2[3] * xz * coefficients[7] + C2[4] * (xx - yy) * coefficients[8];

            if (degree > 2)
            {
                features = features + C3[0] * y * (3.0 * xx - yy) * coefficients[9] +
                      C3[1] * xy * z * coefficients[10] +
                      C3[2] * y * (4.0 * zz - xx - yy) * coefficients[11] +
                      C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[12] +
                      C3[4] * x * (4.0 * zz - xx - yy) * coefficients[13] +
                      C3[5] * z * (xx - yy) * coefficients[14] +
                      C3[6] * x * (xx - 3.0 * yy) * coefficients[15];
            }
        }
    }
    return max(features + 0.5, vector<float, Dim>(0));
}

}