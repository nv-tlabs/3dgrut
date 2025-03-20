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

#include <3dgrt/tensorAccessor.h>
#include <3dgrt/pipelineParams.h>
#include <3playground/cutexture.h>

#include <optix.h>

struct PBRMaterial
{
    bool useDiffuseTexture;
    float4 diffuseFactor;
    cudaTextureObject_t diffuseTexture;

    bool useEmissiveTexture;
    float3 emissiveFactor;
    cudaTextureObject_t emissiveTexture;

    bool useMetallicRoughnessTexture;
    float metallicFactor;
    float roughnessFactor;
    cudaTextureObject_t metallicRoughnessTexture;

    bool useNormalTexture;
    cudaTextureObject_t normalTexture;

    bool useOcclussionTexture;
    cudaTextureObject_t occlussionTexture;

    unsigned int alphaMode;    // see GltfAlphaMode
    float alphaCutoff;
    float transmissionFactor;
    float ior;
};

struct PlaygroundTracingParams: PipelineParameters
{
    // -- renamed --
    // rayOri ->  rayOrigin
    // rayDir ->  rayDirection
    // mogPos, mogRot, mogScl, mogDns -> particleDensity
    // mogSph-> particleRadiance
    // rayRad -> rayRadiance
    // rayDns -> rayDensity
    // rayHit -> rayHitDistance
    // rayHitsCount -> rayHitsCount

    // -- unchanged --
    // OptixTraversableHandle handle;
    // OptixAabb aabb;
    // float minTransmittance;
    // float hitMinGaussianResponse;
    // float alphaMinThreshold;
    // unsigned int sphDegree;
    // uint2 frameBounds;
    // unsigned int frameNumber;
    // int gPrimNumTri;

    // removed:
//    PackedTensorAccessor32<float, 3> rayMaxT; ///< ray max t for termination
//    PackedTensorAccessor32<float, 2> mogHitCount; ///< output (only in HC pipeline) number of ray hits per mog
//    PackedTensorAccessor32<float, 2> mogWeightSum; ///< output (only in AH pipeline) sum of all weights that a gaussian contributed during render pass
//    float slabSpacing;
//    float alphaMaxValue;

    bool useEnvmap;
    bool useEnvmapAsBackground;
    float3 backgroundColor;     // Optional, only if useEnvmap is False
    cudaTextureObject_t envmap; // Optional, only if useEnvmap is True

    // -- Playground specific launch params --
    OptixTraversableHandle triHandle;   // Handle to BVH of mesh primitives: mirrors, glasses, pbr..

    unsigned int playgroundOpts; // see PlaygroundRenderOptions
    unsigned int maxPBRBounces; // Maximum PBR ray iterations (reflections, transmissions & refractions)
    PackedTensorAccessor32<int32_t, 4> trace_state; // Scratch buffer, stores current render pass per ray
    PackedTensorAccessor32<int32_t, 2> triangles;   // Primitive index -> vertex indices

    // Per vertex attributes
    PackedTensorAccessor32<float, 2> vNormals;     // vertex normals
    PackedTensorAccessor32<bool, 2> vHasTangents;  // has precomputed vertex tangents;
    PackedTensorAccessor32<float, 2> vTangents;    // vertex tangents

    // Materials
    PackedTensorAccessor32<float, 2> matUV;              // uv coordinates per vertex
    PackedTensorAccessor32<int32_t, 2> matID;            // id of material to use, per vertex
    PBRMaterial mat0;                        // material 0
    PBRMaterial mat1;                        // material 1
    PBRMaterial mat2;                        // material 2
    PBRMaterial mat3;                        // material 3
    PBRMaterial mat4;                        // material 4
    PBRMaterial mat5;                        // material 5
    PBRMaterial mat6;                        // material 6
    PBRMaterial mat7;                        // material 7
    PBRMaterial mat8;                        // material 8
    PBRMaterial mat9;                        // material 9
    PBRMaterial mat10;                       // material 10
    PBRMaterial mat11;                       // material 11
    PBRMaterial mat12;                       // material 12
    PBRMaterial mat13;                       // material 13
    PBRMaterial mat14;                       // material 14
    PBRMaterial mat15;                       // material 15
    PBRMaterial mat16;                       // material 16
    PBRMaterial mat17;                       // material 17
    PBRMaterial mat18;                       // material 18
    PBRMaterial mat19;                       // material 19
    PBRMaterial mat20;                       // material 20
    PBRMaterial mat21;                       // material 21
    PBRMaterial mat22;                       // material 22
    PBRMaterial mat23;                       // material 23
    PBRMaterial mat24;                       // material 24
    PBRMaterial mat25;                       // material 25
    PBRMaterial mat26;                       // material 26
    PBRMaterial mat27;                       // material 27
    PBRMaterial mat28;                       // material 28
    PBRMaterial mat29;                       // material 29
    PBRMaterial mat30;                       // material 30
    PBRMaterial mat31;                       // material 31

    // Per triangle attributes
    PackedTensorAccessor32<int32_t, 2> primType;         // see PlaygroundPrimitiveTypes
    PackedTensorAccessor32<int32_t, 2> castsShadows;     // If true, casts shadows
    PackedTensorAccessor32<float, 2> refractiveIndex;    // glass refraction, higher -> thicker glass

    unsigned int numPointLights;                         //  size of point lights buffer, dim 0
    PackedTensorAccessor32<float, 2> pointLights;        //  buffer of point lights, casting shadows
};
