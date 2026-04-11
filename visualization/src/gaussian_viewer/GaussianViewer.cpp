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

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/ext/visrtx/makeVisRTXDevice.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "gaussian_common.h"

static void printUsage(const char *argv0) {
  printf("Usage: %s <path.ply> [options]\n", argv0);
  printf("  --scale-factor F        Multiply Gaussian scales (default: 1.0)\n");
  printf("  --opacity-threshold T   Min opacity to keep (default: 0.05)\n");
  printf("  --output FILE           Output PNG path (default: "
         "gaussian_viewer.png)\n");
  printf("  --spp N                 Samples per pixel (default: 128)\n");
  printf("  --resolution WxH        Image resolution (default: 3840x2160)\n");
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string plyPath = argv[1];
  float scaleFactor = 1.0f;
  float opacityThreshold = 0.05f;
  std::string outputPath = "gaussian_viewer.png";
  int spp = 128;
  uvec2 imageSize = {3840, 2160};

  for (int i = 2; i < argc; i++) {
    if (std::strcmp(argv[i], "--scale-factor") == 0 && i + 1 < argc)
      scaleFactor = std::strtof(argv[++i], nullptr);
    else if (std::strcmp(argv[i], "--opacity-threshold") == 0 && i + 1 < argc)
      opacityThreshold = std::strtof(argv[++i], nullptr);
    else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
      outputPath = argv[++i];
    else if (std::strcmp(argv[i], "--spp") == 0 && i + 1 < argc)
      spp = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
      unsigned w = 0, h = 0;
      if (std::sscanf(argv[++i], "%ux%u", &w, &h) == 2 && w > 0 && h > 0)
        imageSize = {w, h};
      else {
        fprintf(stderr,
                "Invalid resolution format, use WxH (e.g. 1920x1080)\n");
        return 1;
      }
    } else {
      printUsage(argv[0]);
      return 1;
    }
  }

  auto data = loadPLY(plyPath, opacityThreshold);
  if (data.positions.empty()) {
    fprintf(stderr, "No Gaussians survived filtering.\n");
    return 1;
  }

  vec3 center;
  float diagonal = 0.0f;
  {
    for (int ax = 0; ax < 3; ax++)
      center[ax] = (data.bboxMin[ax] + data.bboxMax[ax]) * 0.5f;

    float dx = data.bboxMax[0] - data.bboxMin[0];
    float dy = data.bboxMax[1] - data.bboxMin[1];
    float dz = data.bboxMax[2] - data.bboxMin[2];
    diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  printf("Scene center: (%.3f, %.3f, %.3f)  diagonal: %.3f\n", center[0],
         center[1], center[2], diagonal);
  printf("Scale factor: %.3f\n", scaleFactor);

  auto device = makeVisRTXDevice(anariStatusFunc);

  auto world = buildScene(device, data, scaleFactor);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  float pullback = 0.3f * diagonal;
  vec3 eye = {center[0], center[1], center[2] - pullback};
  vec3 dir = {0.f, 0.f, 1.f};
  vec3 up = {0.f, -1.f, 0.f};

  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);
  anari::setParameter(device, camera, "aspect",
                      imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  vec4 bgColor = {0.1f, 0.1f, 0.1f, 1.f};
  anari::setParameter(device, renderer, "background", bgColor);
  anari::setParameter(device, renderer, "ambientRadiance", 1.0f);
  anari::setParameter(device, renderer, "pixelSamples", spp);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  printf("Rendering %zu Gaussians...\n", data.positions.size());
  anari::render(device, frame);
  anari::wait(device, frame);

  float duration = 0.f;
  anari::getProperty(device, frame, "duration", duration, ANARI_NO_WAIT);
  printf("Rendered in %.2f ms\n", duration * 1000);

  stbi_flip_vertically_on_write(1);
  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  stbi_write_png(outputPath.c_str(), fb.width, fb.height, 4, fb.data,
                 4 * fb.width);
  anari::unmap(device, frame, "channel.color");
  printf("Saved: %s\n", outputPath.c_str());

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);

  return 0;
}
