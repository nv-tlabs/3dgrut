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

#include <string>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <3dgrt/pipelineDefinitions.h>

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {
    // No data needed
};
typedef SbtRecord<RayGenData> RayGenSbtRecord;

struct MissData {
    // No data needed
};
typedef SbtRecord<MissData> MissSbtRecord;

struct HitGroupData {
    // No data needed
};

typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

enum CreatePipelineFlags {
    PipelineFlag_None       = 0,
    PipelineFlag_HasCH      = 1 << 0,
    PipelineFlag_HasIS      = 1 << 1,
    PipelineFlag_HasAH      = 1 << 2,
    PipelineFlag_HasMS      = 1 << 3,
    PipelineFlag_HasRG      = 1 << 4,
    PipelineFlag_SpherePrim = 1 << 5
};

class OptixTracer {

protected:
    struct State {
        OptixDeviceContext context;
        OptixTraversableHandle gasHandle;
        CUdeviceptr gasBuffer;
        size_t gasBufferSz;
        CUdeviceptr gasBufferTmp;
        size_t gasBufferTmpSz;

        OptixAabb gasAABB;
        CUdeviceptr optixAabbPtr;
        CUdeviceptr paramsDevice;
        size_t paramsDeviceSz;

        float particleKernelDegree;
        float particleKernelMinResponse;
        bool particleKernelDensityClamping;
        int particleRadianceSphDegree;

        uint32_t gNum;         ///< current number of gaussians
        uint32_t gPrimType;    ///< type of the prim [0 : octahedron 1 : tetrahedron]
        uint32_t gPrimNumVert; ///< number of vertices per gaussian primitive
        uint32_t gPrimNumTri;  ///< number of triangles per gaussian primitive
        CUdeviceptr gPrimVrt;  ///< buffer containing the vertices of the gaussian primitive
        size_t gPrimVrtSz;
        CUdeviceptr gPrimTri; ///< buffer containing the vertices index of the gaussian primitive triangles
        size_t gPrimTriSz;
        CUdeviceptr gPrimAABB; ///< buffer containing the gaussians AABB to be usedwith custom primitives
        size_t gPrimAABBSz;
        CUdeviceptr iasBuffer;
        size_t iasBufferSz;
        CUdeviceptr gPipelineParticleData; ///< buffer containing pipeline specific particle data
        size_t gPipelineParticleDataSz;

        OptixPipeline pipelineTracingFwd;
        OptixShaderBindingTable sbtTracingFwd;
        OptixModule moduleTracingFwd;

        OptixPipeline pipelineTracingBwd;
        OptixShaderBindingTable sbtTracingBwd;
        OptixModule moduleTracingBwd;
    }* _state;

     std::vector<std::string> generateDefines(
        float particleKernelDegree,
        bool particleKernelDensityClamping,
        int particleRadianceSphDegree,
        bool enableNormals,
        bool enableHitCounts
    );
    void createPipeline(const OptixDeviceContext context,
                        const std::string& path,
                        const std::string& dependencies_path,
                        const std::string& cuda_path,
                        const std::vector<std::string>& defines,
                        const std::string& kernel_name,
                        uint32_t flags,
                        OptixModule* module,
                        OptixPipeline* pipeline,
                        OptixShaderBindingTable& sbt,
                        uint32_t numPayloadValues = 32,
                        const std::vector<std::string>& extra_includes = {});
    void reallocateBuffer(CUdeviceptr* bufferPtr, size_t& size, size_t newSize, cudaStream_t cudaStream);
    void reallocatePrimGeomBuffer(cudaStream_t stream);
    void reallocateParamsDevice(size_t sz, cudaStream_t stream);

    OptixTraversableHandle createParticleInstanceAS(cudaStream_t cudaStream);

public:
    OptixTracer(
        const std::string& path,
        const std::string& cuda_path,
        const std::string& pipeline,
        const std::string& backwardPipeline,
        const std::string& primitive,
        float particleKernelDegree,
        float particleKernelMinResponse,
        bool particleKernelDensityClamping,
        int particleRadianceSphDegree,
        bool enableNormals,
        bool enableHitCounts);

    virtual ~OptixTracer();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    virtual trace(uint32_t frameNumber,
                  torch::Tensor rayToWorld,
                  torch::Tensor rayOri,
                  torch::Tensor rayDir,
                  torch::Tensor particleDensity,
                  torch::Tensor particleRadiance,
                  uint32_t renderOpts,
                  int sphDegree,
                  float minTransmittance);

    std::tuple<torch::Tensor, torch::Tensor>
    virtual traceBwd(uint32_t frameNumber,
                     torch::Tensor rayToWorld,
                     torch::Tensor rayOri,
                     torch::Tensor rayDir,
                     torch::Tensor rayRad,
                     torch::Tensor rayDns,
                     torch::Tensor rayHit,
                     torch::Tensor rayNrm,
                     torch::Tensor particleDensity,
                     torch::Tensor particleRadiance,
                     torch::Tensor rayRadGrd,
                     torch::Tensor rayDnsGrd,
                     torch::Tensor rayHitGrd,
                     torch::Tensor rayNrmGrd,
                     uint32_t renderOpts,
                     int sphDegree,
                     float minTransmittance);

    virtual void buildBVH(torch::Tensor mogPos,
                          torch::Tensor mogRot,
                          torch::Tensor mogScl,
                          torch::Tensor mogDns,
                          unsigned int rebuild,
                          bool allow_update);
};
