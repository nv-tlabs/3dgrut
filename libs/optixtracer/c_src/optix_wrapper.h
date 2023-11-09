// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

#include <optix.h>
#include <string>

struct OptiXState
{
    OptixDeviceContext context;
    OptixTraversableHandle gasHandle;
    CUdeviceptr            gasBuffer;
    OptixAabb gasAABB;

    uint32_t pipeline;
    uint32_t hitMode;
    uint32_t maxHitsPerSlab;
    uint32_t maxNumSlabs;
    bool topKHits;
    uint32_t patchSize;
    uint32_t sphDegree;
    float gaussianSigmaThreshold;
    float minTransmittance;
    
    uint32_t gPrimNumTri; ///< number of triangles per gaussian primitive
    CUdeviceptr gPrimVrt; ///< buffer containing the vertices of the gaussian primitive
    CUdeviceptr gPrimTri; ///< buffer containing the vertices index of the gaussian primitive triangles
    CUdeviceptr gPrimAABB; ///< buffer containing the gaussians AABB to be usedwith custom primitives

    // closest hit forward pipeline : the scene is iteratively traced for a closest hit until a density threshold has been reached
    OptixPipeline pipelineMoGTracingCH;
    OptixShaderBindingTable sbtMoGTracingCH;
    OptixModule moduleMoGTracingCH;

    // any hit forward pipeline : the scene is iteratively traced for any hit on ray stabs. The candidate hits are sorted and accumulated in the raygen shader.
    OptixPipeline pipelineMoGTracingAH;
    OptixShaderBindingTable sbtMoGTracingAH;
    OptixModule moduleMoGTracingAH;

    // any hit backward pipeline : the scene is iteratively traced for any hit on ray stabs. The candidate hits are sorted and accumulated in the raygen shader.
    OptixPipeline pipelineMoGTracingAHBwd;
    OptixShaderBindingTable sbtMoGTracingAHBwd;
    OptixModule moduleMoGTracingAHBwd;

    // any hit intersection shader pipeline 
    OptixPipeline pipelineMoGTracingIS;
    OptixShaderBindingTable sbtMoGTracingIS;
    OptixModule moduleMoGTracingIS;

    // any hit multi-layer alpha tracing pipeline
    OptixPipeline pipelineMoGTracingMLAT;
    OptixShaderBindingTable sbtMoGTracingMLAT;
    OptixModule moduleMoGTracingMLAT;

};

class OptiXStateWrapper
{
public:
    OptiXStateWrapper     (
        const std::string& path, 
        const std::string& cuda_path,
        uint32_t hitMode,
        uint32_t pipeline,
        uint32_t maxHitsPerSlab,
        uint32_t maxNumSlabs,
        bool topKHits,
        uint32_t patchSize,
        uint32_t sphDegree,
        float gaussianSigmaThreshold,
        float minTransmittance
    );
    ~OptiXStateWrapper    (void);

    void setSphDegree(int degree)
    {
        if (pState)
        {
            pState->sphDegree = degree;
        }
    }
    
    OptiXState*           pState;
};

