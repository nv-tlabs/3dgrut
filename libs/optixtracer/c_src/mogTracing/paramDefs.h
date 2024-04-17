// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#pragma once

enum MOGTracingMode
{
    MOGTracingModeNone = 0,
    MOGTracingGaussianHit = 1<<0, ///< use the position on the ray with highest gaussian response
    MOGTracingSampling = 1<<1, ///< sampling mode
    MOGTracingTesseracticKernel = 1<<2, 
    MOGTracingAdaptiveKernelClamping = 1<<3, 
    MOGTracingDefaultMode = MOGTracingModeNone
};

enum MOGRenderOpts
{
    MOGRenderNone = 0,
    MOGRenderUseGWeights = 1<<0, 
    MOGRenderDnsHitSampling = 1<<1, 
    MOGRenderTesseracticKernel = 1<<2, 
    MOGRenderEnableTiming = 1<<3, 
    MOGRenderDefault = MOGRenderNone
};

enum MOGPrimitiveTypes
{
    MOGTracingIcosaHedron,
    MOGTracingOctraHedron,
    MOGTracingTetraHedron,
    MOGTracingDiamond,
    MOGTracingSphere,
    MOGTracingCustom,
    MOGTracingTriHexa,
    MOGTracingTriSurfel
};

enum MOGTracingPipeline
{
    MOGTracingPipelineCH = 0,
    MOGTracingPipelineAH = 1,
    MOGTracingPipelineIS = 2,
    MOGTracingPipelineMLAT = 3,
    MOGTracingPipelineMBOIT = 4,
    MOGTracingPipelineHC = 5,
    MOGTracingPipelineInd = 6,
};