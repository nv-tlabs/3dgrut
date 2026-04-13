/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Common data types and PLY I/O for 3D Gaussian Splatting visualization.
//
// This header defines the in-memory representation of a Gaussian splat scene
// (GaussianData) and the routines needed to load one from a .ply file exported
// by a 3DGS training pipeline, decode the stored SH / log-scale / logit-opacity
// parameters into rendering-ready values, and compute scene-level bounding
// information (SceneBounds).  It also provides a helper to build per-Gaussian
// affine transforms (buildTransform) used when instancing the splats via ANARI.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "happly.h"

// Lightweight linear-algebra aliases backed by std::array so they can be
// passed directly to ANARI parameter-setting functions that expect contiguous
// float data.
using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using mat4 = std::array<float, 16>; // column-major 4x4

// 0th-order spherical harmonic coefficient: 1 / (2 * sqrt(pi)).
// Used to convert the DC SH band stored in the PLY into an RGB color:
//   color_channel = clamp(SH_C0 * f_dc + 0.5, 0, 1)
static constexpr float SH_C0 = 0.28209479177387814f;

// Standard logistic sigmoid, used to decode the raw logit-space opacity
// stored in 3DGS PLY files into the [0, 1] range.
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Per-Gaussian attributes loaded from a .ply file after opacity filtering and
// parameter decoding.  All parallel vectors share the same indexing -- element i
// across positions/colors/scales/quats describes one Gaussian splat.  bboxMin
// and bboxMax form an axis-aligned bounding box that accounts for each
// Gaussian's position *plus* its largest scale axis, so they represent the
// visual extent of the scene rather than just the point centres.
struct GaussianData {
  std::vector<vec3> positions; // world-space centres
  std::vector<vec3> colors;    // linear RGB in [0, 1], decoded from SH DC band
  std::vector<vec3> scales;    // per-axis radii in world units (exp of stored log-scale)
  std::vector<vec4> quats;     // orientation as unit quaternion (w, x, y, z)
  vec3 bboxMin;                // AABB lower corner (position - max scale)
  vec3 bboxMax;                // AABB upper corner (position + max scale)
};

// Compact summary of the scene's spatial extent, derived from GaussianData's
// bounding box.  Used as a quick reference for camera placement and distance
// clamping without re-iterating the full point cloud.
struct SceneBounds {
  vec3 center;         // midpoint of the AABB
  float diagonal{0.f}; // length of the AABB diagonal
};

// Load a 3D Gaussian Splatting .ply file and return decoded, rendering-ready
// Gaussian attributes.
//
// The PLY format written by common 3DGS training pipelines stores parameters in
// their optimisation-space encoding:
//   - opacity  : logit space  -> decoded via sigmoid()
//   - scale    : log space    -> decoded via exp()
//   - color    : 0th-order SH -> decoded via SH_C0 * f_dc + 0.5
//   - rotation : un-normalised quaternion -> normalised to unit length
//
// Gaussians whose decoded opacity falls below |opacityThreshold| are discarded
// (they are nearly invisible and would only waste memory and GPU time).  The
// bounding box is expanded per surviving Gaussian by its largest scale axis so
// that it reflects the scene's visual footprint, not just point centres.
inline GaussianData loadPLY(const std::string &path, float opacityThreshold) {
  happly::PLYData ply(path);

  auto x = ply.getElement("vertex").getProperty<float>("x");
  auto y = ply.getElement("vertex").getProperty<float>("y");
  auto z = ply.getElement("vertex").getProperty<float>("z");

  auto f_dc_0 = ply.getElement("vertex").getProperty<float>("f_dc_0");
  auto f_dc_1 = ply.getElement("vertex").getProperty<float>("f_dc_1");
  auto f_dc_2 = ply.getElement("vertex").getProperty<float>("f_dc_2");

  auto opacity_raw = ply.getElement("vertex").getProperty<float>("opacity");

  auto scale_0 = ply.getElement("vertex").getProperty<float>("scale_0");
  auto scale_1 = ply.getElement("vertex").getProperty<float>("scale_1");
  auto scale_2 = ply.getElement("vertex").getProperty<float>("scale_2");

  auto rot_0 = ply.getElement("vertex").getProperty<float>("rot_0");
  auto rot_1 = ply.getElement("vertex").getProperty<float>("rot_1");
  auto rot_2 = ply.getElement("vertex").getProperty<float>("rot_2");
  auto rot_3 = ply.getElement("vertex").getProperty<float>("rot_3");

  size_t total = x.size();
  printf("PLY loaded: %zu Gaussians\n", total);

  GaussianData data;
  data.positions.reserve(total);
  data.colors.reserve(total);
  data.scales.reserve(total);
  data.quats.reserve(total);
  data.bboxMin = {1e30f, 1e30f, 1e30f};
  data.bboxMax = {-1e30f, -1e30f, -1e30f};

  for (size_t i = 0; i < total; i++) {
    float alpha = sigmoid(opacity_raw[i]);
    if (alpha < opacityThreshold)
      continue;

    float r = std::clamp(SH_C0 * f_dc_0[i] + 0.5f, 0.0f, 1.0f);
    float g = std::clamp(SH_C0 * f_dc_1[i] + 0.5f, 0.0f, 1.0f);
    float b = std::clamp(SH_C0 * f_dc_2[i] + 0.5f, 0.0f, 1.0f);

    float s0 = std::exp(scale_0[i]);
    float s1 = std::exp(scale_1[i]);
    float s2 = std::exp(scale_2[i]);

    float qw = rot_0[i], qx = rot_1[i], qy = rot_2[i], qz = rot_3[i];
    float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn > 0.f) {
      qw /= qn;
      qx /= qn;
      qy /= qn;
      qz /= qn;
    } else {
      qw = 1.f;
      qx = qy = qz = 0.f;
    }

    data.positions.push_back({x[i], y[i], z[i]});
    data.colors.push_back({r, g, b});
    data.scales.push_back({s0, s1, s2});
    data.quats.push_back({qw, qx, qy, qz});

    float maxS = std::max({s0, s1, s2});
    for (int ax = 0; ax < 3; ax++) {
      float p = data.positions.back()[ax];
      data.bboxMin[ax] = std::min(data.bboxMin[ax], p - maxS);
      data.bboxMax[ax] = std::max(data.bboxMax[ax], p + maxS);
    }
  }

  printf("After opacity filter (threshold=%.3f): %zu / %zu Gaussians kept\n", opacityThreshold, data.positions.size(), total);

  return data;
}

// Build a column-major 4x4 affine transform (TRS) for a single Gaussian.
//
// The resulting matrix encodes:  M = T * R * S
//   T  -- translation to world-space position |pos|
//   R  -- rotation from unit quaternion |q| (w, x, y, z)
//   S  -- anisotropic scale |s| multiplied by the global |sf| (scale factor)
//
// The rotation matrix is derived directly from the quaternion via the standard
// formula (no intermediate Euler angles), and scale is baked into the rotation
// columns so that only one 4x4 matrix is needed per Gaussian instance.
//
// Storage layout: m[col*4 + row]  (OpenGL / ANARI column-major convention).
inline mat4 buildTransform(const vec3 &pos, const vec4 &q, const vec3 &s, float sf) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  float s0 = s[0] * sf, s1 = s[1] * sf, s2 = s[2] * sf;

  float r00 = 1.f - 2.f * (y * y + z * z);
  float r10 = 2.f * (x * y + w * z);
  float r20 = 2.f * (x * z - w * y);
  float r01 = 2.f * (x * y - w * z);
  float r11 = 1.f - 2.f * (x * x + z * z);
  float r21 = 2.f * (y * z + w * x);
  float r02 = 2.f * (x * z + w * y);
  float r12 = 2.f * (y * z - w * x);
  float r22 = 1.f - 2.f * (x * x + y * y);

  return {{
      r00 * s0,
      r10 * s0,
      r20 * s0,
      0.f,
      r01 * s1,
      r11 * s1,
      r21 * s1,
      0.f,
      r02 * s2,
      r12 * s2,
      r22 * s2,
      0.f,
      pos[0],
      pos[1],
      pos[2],
      1.f,
  }};
}

// Derive the AABB midpoint and diagonal from a loaded GaussianData's bounding
// box.  Returns a zero-diagonal SceneBounds if no Gaussians are present.
inline SceneBounds computeSceneBounds(const GaussianData &data) {
  SceneBounds bounds;
  bounds.center = {0.f, 0.f, 0.f};
  if (data.positions.empty())
    return bounds;

  for (int ax = 0; ax < 3; ax++)
    bounds.center[ax] = (data.bboxMin[ax] + data.bboxMax[ax]) * 0.5f;

  float dx = data.bboxMax[0] - data.bboxMin[0];
  float dy = data.bboxMax[1] - data.bboxMin[1];
  float dz = data.bboxMax[2] - data.bboxMin[2];
  bounds.diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);

  return bounds;
}
