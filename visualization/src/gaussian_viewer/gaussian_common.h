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

#pragma once

#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "happly.h"

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using mat4 = std::array<float, 16>;

static constexpr float SH_C0 = 0.28209479177387814f;

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct GaussianData {
  std::vector<vec3> positions;
  std::vector<vec3> colors;
  std::vector<vec3> scales;
  std::vector<vec4> quats;
  vec3 bboxMin;
  vec3 bboxMax;
};

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

  printf("After opacity filter (threshold=%.3f): %zu / %zu Gaussians kept\n",
         opacityThreshold, data.positions.size(), total);

  return data;
}

// Column-major mat4: m[col*4 + row]
inline mat4 buildTransform(const vec3 &pos, const vec4 &q, const vec3 &s,
                           float sf) {
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

inline anari::World buildScene(anari::Device device, const GaussianData &data,
                               float scaleFactor) {
  uint32_t N = static_cast<uint32_t>(data.positions.size());

  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  vec3 origin = {0.f, 0.f, 0.f};
  anari::setParameterArray1D(device, geometry, "vertex.position", &origin, 1);
  anari::setParameter(device, geometry, "radius", 1.0f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", "color");
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto group = anari::newObject<anari::Group>(device);
  anari::setParameterArray1D(device, group, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, group);

  auto xfmArray = anari::newArray1D(device, ANARI_FLOAT32_MAT4, N);
  auto colArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, N);
  {
    auto *xfms = anari::map<mat4>(device, xfmArray);
    auto *cols = anari::map<vec3>(device, colArray);
    for (uint32_t i = 0; i < N; i++) {
      xfms[i] = buildTransform(data.positions[i], data.quats[i], data.scales[i],
                               scaleFactor);
      cols[i] = data.colors[i];
    }
    anari::unmap(device, xfmArray);
    anari::unmap(device, colArray);
  }

  auto instance = anari::newObject<anari::Instance>(device, "transform");
  anari::setAndReleaseParameter(device, instance, "group", group);
  anari::setAndReleaseParameter(device, instance, "transform", xfmArray);
  anari::setAndReleaseParameter(device, instance, "color", colArray);
  anari::commitParameters(device, instance);

  auto keyLight = anari::newObject<anari::Light>(device, "directional");
  vec3 keyDir = {-1.f, -1.f, -1.f};
  anari::setParameter(device, keyLight, "direction", keyDir);
  anari::setParameter(device, keyLight, "irradiance", 3.0f);
  anari::commitParameters(device, keyLight);

  auto fillLight = anari::newObject<anari::Light>(device, "directional");
  vec3 fillDir = {1.f, -0.5f, -0.5f};
  anari::setParameter(device, fillLight, "direction", fillDir);
  anari::setParameter(device, fillLight, "irradiance", 1.5f);
  anari::commitParameters(device, fillLight);

  auto backLight = anari::newObject<anari::Light>(device, "directional");
  vec3 backDir = {0.f, 0.5f, 1.f};
  anari::setParameter(device, backLight, "direction", backDir);
  anari::setParameter(device, backLight, "irradiance", 0.8f);
  anari::commitParameters(device, backLight);

  ANARILight lights[] = {keyLight, fillLight, backLight};

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "instance", &instance, 1);
  anari::release(device, instance);
  anari::setParameterArray1D(device, world, "light", lights, 3);
  anari::release(device, keyLight);
  anari::release(device, fillLight);
  anari::release(device, backLight);
  anari::commitParameters(device, world);

  return world;
}

inline void anariStatusFunc(const void *, ANARIDevice, ANARIObject source,
                            ANARIDataType, ANARIStatusSeverity severity,
                            ANARIStatusCode, const char *message) {
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  } else if (severity == ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[WARN ][%p] %s\n", source, message);
  }
}
