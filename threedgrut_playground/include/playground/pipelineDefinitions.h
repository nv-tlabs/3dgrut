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

enum PlaygroundPrimitiveTypes
{
    PGRNDPrimitiveNone = 0,
    PGRNDPrimitiveMirror = 1,
    PGRNDPrimitiveGlass = 2,
    PGRNDPrimitiveDiffuse = 3
};

enum PlaygroundTraceState
{
    PGRNDTraceRTLastGaussiansPass = 0,   // Tracing gaussians with volumetric rendering - until scene extents are hit
    PGRNDTracePrimitivesPass = 1,        // Tracing mirrors, glasses, meshes..
    PGRNDTraceRTGaussiansPass = 2,       // Tracing gaussians with volumetric rendering
    PGRNDTraceTerminate = 3,             // Terminate current ray
};

enum PlaygroundRenderOptions
{
    PGRNDRenderNone = 0,
    PGRNDRenderSmoothNormals = 1<<0,           // Geometry: If enabled, will interpolate precomputed vertex normals
    PGRNDRenderDisableGaussianTracing = 1<<1,  // Disable gaussian tracing -- only meshes will be rendered
    PGRNDRenderDisablePBRTextures = 1<<2       // Disable PBR textures, use base material values only
};

enum GltfAlphaMode
{
    // See: https://github.com/KhronosGroup/glTF-Sample-Models/blob/main/2.0/AlphaBlendModeTest/README.md
    GLTFOpaque = 0,
    GLTFBlend = 1,
    GLTFMask = 2
};