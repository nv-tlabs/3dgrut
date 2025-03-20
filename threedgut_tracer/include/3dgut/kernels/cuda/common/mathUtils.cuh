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

#ifdef __CUDACC__

using float33 = float3[3]; // row major matrix

template <class T>
static __device__ __inline__ T clamp(T x, T _min, T _max) {
    return min(_max, max(_min, x));
}
// static __device__ inline float3 make_float3(float a) {
//     return make_float3(a, a, a);
//}
// static __device__ inline float3 make_float3(float4 a) {
//     return make_float3(a.x, a.y, a.z);
// }
// static __device__ inline float4 make_float4(float a) {
//     return make_float4(a, a, a, a);
// }
static __device__ inline float3 maxf3(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
static __device__ inline float3 minf3(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
static __device__ inline float2& operator/=(float2& a, const float2& b) {
    a.x /= b.x;
    a.y /= b.y;
    return a;
}
static __device__ inline float2& operator*=(float2& a, const float2& b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}
static __device__ inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}
static __device__ inline float2& operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}
static __device__ inline float2& operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
    return a;
}
static __device__ inline float2& operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
    return a;
}
static __device__ inline float2& operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
    return a;
}
static __device__ inline float2& operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
    return a;
}
// static __device__ inline float2 operator/(const float2& a, const float2& b) {
//     return make_float2(a.x / b.x, a.y / b.y);
// }
// static __device__ inline float2 operator*(const float2& a, const float2& b) {
//     return make_float2(a.x * b.x, a.y * b.y);
// }
// static __device__ inline float2 operator+(const float2& a, const float2& b) {
//     return make_float2(a.x + b.x, a.y + b.y);
// }
// static __device__ inline float2 operator-(const float2& a, const float2& b) {
//     return make_float2(a.x - b.x, a.y - b.y);
// }
static __device__ inline float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}
static __device__ inline float2 operator*(const float2& a, float b) {
    return make_float2(a.x * b, a.y * b);
}
static __device__ inline float2 operator+(const float2& a, float b) {
    return make_float2(a.x + b, a.y + b);
}
static __device__ inline float2 operator-(const float2& a, float b) {
    return make_float2(a.x - b, a.y - b);
}
static __device__ inline float2 operator/(float a, const float2& b) {
    return make_float2(a / b.x, a / b.y);
}
static __device__ inline float2 operator*(float a, const float2& b) {
    return make_float2(a * b.x, a * b.y);
}
static __device__ inline float2 operator+(float a, const float2& b) {
    return make_float2(a + b.x, a + b.y);
}
static __device__ inline float2 operator-(float a, const float2& b) {
    return make_float2(a - b.x, a - b.y);
}
// static __device__ inline float2 operator-(const float2& a) {
//     return make_float2(-a.x, -a.y);
// }
static __device__ inline float3& operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}
static __device__ inline float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}
static __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
static __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
static __device__ inline float3& operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}
static __device__ inline float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}
static __device__ inline float3& operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}
static __device__ inline float3& operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}
// static __device__ inline float3 operator/(const float3& a, const float3& b) {
//     return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
// }
// static __device__ inline float3 operator*(const float3& a, const float3& b) {
//     return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
// }
// static __device__ inline float3 operator+(const float3& a, const float3& b) {
//     return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
// }
// static __device__ inline float3 operator-(const float3& a, const float3& b) {
//     return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
// }
static __device__ inline float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
static __device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
static __device__ inline float3 operator+(const float3& a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
static __device__ inline float3 operator-(const float3& a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}
static __device__ inline float3 operator/(float a, const float3& b) {
    return make_float3(a / b.x, a / b.y, a / b.z);
}
static __device__ inline float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}
static __device__ inline float3 operator+(float a, const float3& b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}
static __device__ inline float3 operator-(float a, const float3& b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}
// static __device__ inline float3 operator-(const float3& a) {
//     return make_float3(-a.x, -a.y, -a.z);
// }
static __device__ inline float4& operator/=(float4& a, const float4& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    return a;
}
static __device__ inline float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}
static __device__ inline float4& operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}
static __device__ inline float4& operator-=(float4& a, const float4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}
static __device__ inline float4& operator/=(float4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
    return a;
}
static __device__ inline float4& operator*=(float4& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}
static __device__ inline float4& operator+=(float4& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
}
static __device__ inline float4& operator-=(float4& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    return a;
}
// static __device__ inline float4 operator/(const float4& a, const float4& b) {
//     return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
// }
// static __device__ inline float4 operator*(const float4& a, const float4& b) {
//     return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
// }
// static __device__ inline float4 operator+(const float4& a, const float4& b) {
//     return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
// }
// static __device__ inline float4 operator-(const float4& a, const float4& b) {
//     return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
// }
// static __device__ inline float4 operator/(const float4& a, float b) {
//     return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
// }
// static __device__ inline float4 operator*(const float4& a, float b) {
//     return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
// }
// static __device__ inline float4 operator+(const float4& a, float b) {
//     return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
// }
// static __device__ inline float4 operator-(const float4& a, float b) {
//     return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
// }
// static __device__ inline float4 operator/(float a, const float4& b) {
//     return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
// }
// static __device__ inline float4 operator*(float a, const float4& b) {
//     return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
// }
// static __device__ inline float4 operator+(float a, const float4& b) {
//     return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
// }
// static __device__ inline float4 operator-(float a, const float4& b) {
//     return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
// }
// static __device__ inline float4 operator-(const float4& a) {
//     return make_float4(-a.x, -a.y, -a.z, -a.w);
// }
// static __device__ inline int3 operator+(const int3& a, const int3& b) {
//     return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
// }
// static __device__ inline int3 operator*(const int3& a, const int3& b) {
//     return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
// }

static __device__ inline float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

static __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ inline void bwd_dot(float3 a, float3 b, float3& d_a, float3& d_b, float d_out) {
    d_a.x += d_out * b.x;
    d_a.y += d_out * b.y;
    d_a.z += d_out * b.z;
    d_b.x += d_out * a.x;
    d_b.y += d_out * a.y;
    d_b.z += d_out * a.z;
}

static __device__ inline float luminance(const float3 rgb) {
    return dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
}

static __device__ inline float sum(const float3 v) {
    return v.x + v.y + v.z;
}

static __device__ inline float3 cross(float3 a, float3 b) {
    float3 out;
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - a.x * b.z;
    out.z = a.x * b.y - a.y * b.x;
    return out;
}

static __device__ inline void bwd_cross(float3 a, float3 b, float3& d_a, float3& d_b, float3 d_out) {
    d_a.x += d_out.z * b.y - d_out.y * b.z;
    d_a.y += d_out.x * b.z - d_out.z * b.x;
    d_a.z += d_out.y * b.x - d_out.x * b.y;

    d_b.x += d_out.y * a.z - d_out.z * a.y;
    d_b.y += d_out.z * a.x - d_out.x * a.z;
    d_b.z += d_out.x * a.y - d_out.y * a.x;
}

__device__ static inline void bwdCross3(
    const float3& v0, const float3& v1, const float3& v2, const float3& grad, float3& g_v0, float3& g_v1, float3& g_v2) {
    float3 a      = v2 - v0;
    float3 b      = v1 - v0;
    float3 dn_dax = make_float3(0, -b.z, b.y);
    float3 dn_day = make_float3(b.z, 0, -b.x);
    float3 dn_daz = make_float3(-b.y, b.x, 0);

    float3 dn_dbx = make_float3(0, a.z, -a.y);
    float3 dn_dby = make_float3(-a.z, 0, a.x);
    float3 dn_dbz = make_float3(a.y, -a.x, 0);

    g_v0 += make_float3(dot(-dn_dax - dn_dbx, grad), dot(-dn_day - dn_dby, grad), dot(-dn_daz - dn_dbz, grad));
    g_v1 += make_float3(dot(dn_dbx, grad), dot(dn_dby, grad), dot(dn_dbz, grad));
    g_v2 += make_float3(dot(dn_dax, grad), dot(dn_day, grad), dot(dn_daz, grad));
}

static __device__ inline float3 reflect(float3 x, float3 n) {
    return n * 2.0f * dot(n, x) - x;
}

static __device__ inline void bwd_reflect(float3 x, float3 n, float3& d_x, float3& d_n, float3 d_out) {
    d_x.x += d_out.x * (2 * n.x * n.x - 1) + d_out.y * (2 * n.x * n.y) + d_out.z * (2 * n.x * n.z);
    d_x.y += d_out.x * (2 * n.x * n.y) + d_out.y * (2 * n.y * n.y - 1) + d_out.z * (2 * n.y * n.z);
    d_x.z += d_out.x * (2 * n.x * n.z) + d_out.y * (2 * n.y * n.z) + d_out.z * (2 * n.z * n.z - 1);

    d_n.x +=
        d_out.x * (2 * (2 * n.x * x.x + n.y * x.y + n.z * x.z)) + d_out.y * (2 * n.y * x.x) + d_out.z * (2 * n.z * x.x);
    d_n.y +=
        d_out.x * (2 * n.x * x.y) + d_out.y * (2 * (n.x * x.x + 2 * n.y * x.y + n.z * x.z)) + d_out.z * (2 * n.z * x.y);
    d_n.z +=
        d_out.x * (2 * n.x * x.z) + d_out.y * (2 * n.y * x.z) + d_out.z * (2 * (n.x * x.x + n.y * x.y + 2 * n.z * x.z));
}

static __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

static __device__ inline float3 safe_normalize(float3 v) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}

static __device__ inline void bwd_safe_normalize(const float3 v, float3& d_v, float3 d_out) {

    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (l > 0.0f) {
        float fac = 1.0 / powf(v.x * v.x + v.y * v.y + v.z * v.z, 1.5f);
        d_v.x += (d_out.x * (v.y * v.y + v.z * v.z) - d_out.y * (v.x * v.y) - d_out.z * (v.x * v.z)) * fac;
        d_v.y += (d_out.y * (v.x * v.x + v.z * v.z) - d_out.x * (v.y * v.x) - d_out.z * (v.y * v.z)) * fac;
        d_v.z += (d_out.z * (v.x * v.x + v.y * v.y) - d_out.x * (v.z * v.x) - d_out.y * (v.z * v.y)) * fac;
    }
}

// static __device__ inline float3 safe_normalize_bw(const float3& v, const float3& d_out)
// {
//     const float l = v.x * v.x + v.y * v.y + v.z * v.z;
//     if (l > 0.0f)
//     {
//         const float il = rsqrtf(l);
//         const float il3 = il * il * il;
//         return make_float3((d_out.x * (v.y * v.y + v.z * v.z) - d_out.y * (v.x * v.y) - d_out.z * (v.x * v.z)),
//                            (d_out.y * (v.x * v.x + v.z * v.z) - d_out.x * (v.y * v.x) - d_out.z * (v.y * v.z)),
//                            (d_out.z * (v.x * v.x + v.y * v.y) - d_out.x * (v.z * v.x) - d_out.y * (v.z * v.y))) * il3;
//     }
//     return make_float3(0);
// }

static __device__ inline float3 safe_normalize_bw(const float3& v, const float3& d_out) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    if (l > 0.0f) {
        const float il  = rsqrtf(l);
        const float il3 = (il * il * il);
        return il * d_out - il3 * make_float3(d_out.x * (v.x * v.x) + d_out.y * (v.y * v.x) + d_out.z * (v.z * v.x),
                                              d_out.x * (v.x * v.y) + d_out.y * (v.y * v.y) + d_out.z * (v.z * v.y),
                                              d_out.x * (v.x * v.z) + d_out.y * (v.y * v.z) + d_out.z * (v.z * v.z));
    }
    return make_float3(0);
}

static __device__ __inline__ float sqr(const float x) {
    return x * x;
}

static __device__ __inline__ float radians(const float degrees) {
    return 3.14159265359f * degrees / 180.0f;
}

// Code from
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
static __device__ inline void branchlessONB(const float3& n, float3& b1, float3& b2) {
    float sign    = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1            = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2            = make_float3(b, sign + n.y * n.y * a, -n.y);
}

static __device__ inline float3 operator*(const float33& m, const float3& p) {
    return make_float3(
        dot(make_float3(m[0].x, m[1].x, m[2].x), p),
        dot(make_float3(m[0].y, m[1].y, m[2].y), p),
        dot(make_float3(m[0].z, m[1].z, m[2].z), p));
}

static __device__ inline float3 operator*(const float3& p, const float33& m) {
    return make_float3(dot(m[0], p), dot(m[1], p), dot(m[2], p));
}

static __device__ inline float3 matmul_bw_vec(const float33& m, const float3& gdt) {
    return make_float3(
        gdt.x * m[0].x + gdt.y * m[1].x + gdt.z * m[2].x,
        gdt.x * m[0].y + gdt.y * m[1].y + gdt.z * m[2].y,
        gdt.x * m[0].z + gdt.y * m[1].z + gdt.z * m[2].z);
}

static __device__ inline float4 matmul_bw_quat(const float3& p, const float3& g, const float4& q) {
    float33 dmat;
    dmat[0] = g.x * p;
    dmat[1] = g.y * p;
    dmat[2] = g.z * p;

    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    float dr = 0;
    float dx = 0;
    float dy = 0;
    float dz = 0;

    // m[0] = make_float3((1.f - 2.f * (y * y + z * z)), 2.f * (x * y + r * z), 2.f * (x * z - r * y));

    // m[0].x = (1.f - 2.f * (y * y + z * z))
    dy += -4 * y * dmat[0].x;
    dz += -4 * z * dmat[0].x;
    // m[0].y = 2.f * (x * y + r * z)
    dr += 2 * z * dmat[0].y;
    dx += 2 * y * dmat[0].y;
    dy += 2 * x * dmat[0].y;
    dz += 2 * r * dmat[0].y;
    // m[0].z = 2.f * (x * z - r * y)
    dr += -2 * y * dmat[0].z;
    dx += 2 * z * dmat[0].z;
    dy += -2 * r * dmat[0].z;
    dz += 2 * x * dmat[0].z;

    // m[1] = make_float3(2.f * (x * y - r * z), (1.f - 2.f * (x * x + z * z)), 2.f * (y * z + r * x));

    // m[1].x = 2.f * (x * y - r * z)
    dr += -2 * z * dmat[1].x;
    dx += 2 * y * dmat[1].x;
    dy += 2 * x * dmat[1].x;
    dz += -2 * r * dmat[1].x;
    // m[1].y = (1.f - 2.f * (x * x + z * z))
    dx += -4 * x * dmat[1].y;
    dz += -4 * z * dmat[1].y;
    // m[1].z = 2.f * (y * z + r * x))
    dr += 2 * x * dmat[1].z;
    dx += 2 * r * dmat[1].z;
    dy += 2 * z * dmat[1].z;
    dz += 2 * y * dmat[1].z;

    // m[2] = make_float3(2.f * (x * z + r * y), 2.f * (y * z - r * x), (1.f - 2.f * (x * x + y * y)));

    // m[2].x = 2.f * (x * z + r * y)
    dr += 2 * y * dmat[2].x;
    dx += 2 * z * dmat[2].x;
    dy += 2 * r * dmat[2].x;
    dz += 2 * x * dmat[2].x;
    // m[2].y = 2.f * (y * z - r * x)
    dr += -2 * x * dmat[2].y;
    dx += -2 * r * dmat[2].y;
    dy += 2 * z * dmat[2].y;
    dz += 2 * y * dmat[2].y;
    // m[2].z = (1.f - 2.f * (x * x + y * y))
    dx += -4 * x * dmat[2].z;
    dy += -4 * y * dmat[2].z;

    return make_float4(dr, dx, dy, dz);
}

static __device__ inline void invRotationMatrix(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (y * y + z * z)), 2.f * (x * y - r * z), 2.f * (x * z + r * y));
    ret[1] = make_float3(2.f * (x * y + r * z), (1.f - 2.f * (x * x + z * z)), 2.f * (y * z - r * x));
    ret[2] = make_float3(2.f * (x * z - r * y), 2.f * (y * z + r * x), (1.f - 2.f * (x * x + y * y)));
}

static __device__ inline void rotationMatrix(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (y * y + z * z)), 2.f * (x * y + r * z), 2.f * (x * z - r * y));
    ret[1] = make_float3(2.f * (x * y - r * z), (1.f - 2.f * (x * x + z * z)), 2.f * (y * z + r * x));
    ret[2] = make_float3(2.f * (x * z + r * y), 2.f * (y * z - r * x), (1.f - 2.f * (x * x + y * y)));
}

// static __device__ inline void rotationMatrix(const float4& quat, float33& mat)
// {
//     const float q0 = quat.x;
//     const float q1 = quat.y;
//     const float q2 = quat.z;
//     const float q3 = quat.w;

//     const float q00 = q0*q0;
//     const float q01 = q0*q1;
//     const float q02 = q0*q2;
//     const float q03 = q0*q3;

//     const float q12 = q1*q2;
//     const float q13 = q1*q3;

//     const float q23 = q2*q3;

//     // column-major matrix
//     mat[0][0] = 2 * (q00 + q1*q1) - 1;
//     mat[0][1] = 2 * (q12 + q03);
//     mat[0][2] = 2 * (q13 - q02);

//     mat[1][0] = 2 * (q12 - q03);
//     mat[1][1] = 2 * (q00 + q2*q2) - 1;
//     mat[1][2] = 2 * (q23 + q01);

//     mat[2][0] = 2 * (q13 + q02);
//     mat[2][1] = 2 * (q23 - q01);
//     mat[2][2] = 2 * (q00 + q3*q3) - 1;
// }

// ===============================================================
// Implementation of the atomicMinfloat using ordered int
__forceinline__ __device__ void atomicMinFloat(float* addr, float value) {
    if (value >= 0)
        __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
    else
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
}

// ===============================================================
// Implementation of the atomicMaxFloat using ordered int
__forceinline__ __device__ void atomicMaxFloat(float* addr, float value) {
    if (value >= 0)
        __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
    else
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

#endif