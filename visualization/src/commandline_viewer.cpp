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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "renderer_core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void printUsage(const char *argv0) {
  printf("Usage: %s <path.ply> [options]\n", argv0);
  printf("  --library NAME          ANARI library (default: visrtx)\n");
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
  std::string libraryName = "visrtx";
  float scaleFactor = 1.0f;
  float opacityThreshold = 0.05f;
  std::string outputPath = "gaussian_viewer.png";
  int spp = 128;
  uvec2 imageSize = {3840, 2160};

  for (int i = 2; i < argc; i++) {
    if (std::strcmp(argv[i], "--library") == 0 && i + 1 < argc)
      libraryName = argv[++i];
    else if (std::strcmp(argv[i], "--scale-factor") == 0 && i + 1 < argc)
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
        fprintf(stderr, "Invalid resolution format, use WxH (e.g. 1920x1080)\n");
        return 1;
      }
    } else {
      printUsage(argv[0]);
      return 1;
    }
  }

  InitOptions options;
  options.plyPath = plyPath;
  options.libraryName = libraryName;
  options.scaleFactor = scaleFactor;
  options.opacityThreshold = opacityThreshold;
  options.frameSize = imageSize;
  options.rendererConfig.spp = spp;

  GaussianRendererCore renderer;
  std::string errorMessage;
  if (!renderer.init(options, &errorMessage)) {
    fprintf(stderr, "Renderer init failed: %s\n", errorMessage.c_str());
    return 1;
  }

  const auto &center = renderer.sceneCenter();
  printf("Scene center: (%.3f, %.3f, %.3f)  diagonal: %.3f\n", center[0], center[1], center[2], renderer.sceneDiagonal());
  printf("Scale factor: %.3f\n", scaleFactor);
  printf("Rendering %zu Gaussians...\n", renderer.gaussianCount());

  if (!renderer.run(&errorMessage)) {
    fprintf(stderr, "Render failed: %s\n", errorMessage.c_str());
    return 1;
  }

  printf("Rendered in %.2f ms\n", renderer.lastDurationSeconds() * 1000.f);

  stbi_flip_vertically_on_write(1);
  auto fb = renderer.mapColorHost(&errorMessage);
  if (!fb.data) {
    fprintf(stderr, "Framebuffer map failed: %s\n", errorMessage.c_str());
    return 1;
  }
  stbi_write_png(outputPath.c_str(), fb.width, fb.height, 4, fb.data, 4 * fb.width);
  renderer.unmapColorHost();
  printf("Saved: %s\n", outputPath.c_str());

  return 0;
}
