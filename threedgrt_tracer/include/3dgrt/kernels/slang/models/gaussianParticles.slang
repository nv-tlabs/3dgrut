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

#include <3dgrt/kernels/slang/common/transforms.slang>
namespace gaussianParticle {
static const int KernelDegree               = GAUSSIAN_PARTICLE_KERNEL_DEGREE;
static const float MinParticleKernelDensity = GAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY;
static const float MinParticleAlpha = GAUSSIAN_PARTICLE_MIN_ALPHA;
static const float MaxParticleAlpha = GAUSSIAN_PARTICLE_MAX_ALPHA;
static const bool Surfel                    = false; // GAUSSIAN_PARTICLE_SURFEL == 0 ? true : false;
}; // namespace gaussianParticle

namespace gaussianParticle {

struct RawParameters : IDifferentiable {
    float3 position;
    float density;
    float4 quaternion;
    float3 scale;
    float padding;
};

struct RawParametersBuffer {
    const RawParameters* _dataPtr;
    RawParameters *_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct CommonParameters {
    RawParametersBuffer parametersBuffer;
};

[BackwardDifferentiable][ForceInline] RawParameters fetchParametersFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer) {
    RawParameters rawParameters = parametersBuffer._dataPtr[particleIdx];
    if (Surfel) {
        rawParameters.scale.z = 1e-06f;
    }
    return rawParameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] void fetchParametersFromBufferBwd(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    RawParameters rawParameters) {
    if (parametersBuffer.exclusiveGradient) {
        parametersBuffer._gradPtr[particleIdx].density += rawParameters.density;
        parametersBuffer._gradPtr[particleIdx].position += rawParameters.position;
        parametersBuffer._gradPtr[particleIdx].quaternion += rawParameters.quaternion;
        if (!Surfel) {
            parametersBuffer._gradPtr[particleIdx].scale += rawParameters.scale;
        } else {
            parametersBuffer._gradPtr[particleIdx].scale.xy += rawParameters.scale.xy;
        }
    } else {
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].density, rawParameters.density);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.x, rawParameters.position.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.y, rawParameters.position.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.z, rawParameters.position.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.x, rawParameters.quaternion.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.y, rawParameters.quaternion.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.z, rawParameters.quaternion.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.w, rawParameters.quaternion.w);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.x, rawParameters.scale.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.y, rawParameters.scale.y);
        if (!Surfel) {
            InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.z, rawParameters.scale.z);
        }
    }

}

struct Parameters : IDifferentiable {
    float3 position;
    float3 scale;
    float3x3 rotationT;
    float density;
};

[BackwardDifferentiable][ForceInline] Parameters fetchParameters(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer) {
    const RawParameters rawParameters = fetchParametersFromBuffer(particleIdx, parametersBuffer);
    return {
        rawParameters.position,
        rawParameters.scale,
        transforms.rotationMatrixTranspose(rawParameters.quaternion),
        rawParameters.density};
}
[BackwardDifferentiable][ForceInline] void cannonicalRay(
    in float3 rayOrigin,
    in float3 rayDirection,
    in Parameters parameters,
    out float3 particleRayOrigin,
    out float3 particleRayDirection, ) {
    const float3 giscl  = float3(1.0f) / parameters.scale;
    const float3 gposc  = (rayOrigin - parameters.position);
    const float3 gposcr = mul(parameters.rotationT, gposc);
    particleRayOrigin   = giscl * gposcr;

    const float3 rayDirR = mul(parameters.rotationT, rayDirection);
    const float3 grdu    = giscl * rayDirR;
    particleRayDirection = normalize(grdu);
}

[BackwardDifferentiable][ForceInline] float canonicalRayMinSquaredDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    const float3 gcrod = cross(canonicalRayDirection, canonicalRayOrigin);
    return dot(gcrod, gcrod);
}

[BackwardDifferentiable][ForceInline] float canonicalRayMaxKernelResponse<let KernelDegree : int>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    const float grayDist = canonicalRayMinSquaredDistance(canonicalRayOrigin, canonicalRayDirection);

    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
    switch (KernelDegree) {
    case 8: // Zenzizenzizenzic
    {
        /*static const*/ float s = -0.000685871056241;
        const float grayDistSq   = grayDist * grayDist;
        return exp(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        /*static const*/ float s = -0.0185185185185;
        return exp(s * grayDist * grayDist * sqrt(grayDist));
    }
    case 4: // Tesseractic
    {
        /*static const*/ float s = -0.0555555555556;
        return exp(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        /*static const*/ float s = -0.166666666667;
        return exp(s * grayDist * sqrt(grayDist));
    }
    case 1: // Laplacian
    {
        /*static const*/ float s = -1.5f;
        return exp(s * sqrt(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return max(1 + s * sqrt(grayDist), 0.f);
    }
    default: // Quadratic
    {
        /*static const*/ float s = -0.5f;
        return exp(s * grayDist);
    }
    }
}

[BackwardDifferentiable][ForceInline] float canonicalRayDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale) {
    const float3 grds = scale * canonicalRayDirection * dot(canonicalRayDirection, -1 * canonicalRayOrigin);
    return sqrt(dot(grds, grds));
}

[BackwardDifferentiable][ForceInline] float3 canonicalRayNormal<let Surfel : bool>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale,
    float3x3 rotationT) {
    
    // TODO : unify the computation of normals
    float3 surfelNm = float3(0, 0, 1); // fetchSurfelNm<MOGTRACING_PRIMITIVE_TYPE>(primId % params.gPrimNumTri);
    // resolve direction ambiguities
    if (dot(surfelNm, canonicalRayDirection) > 0) {
        surfelNm *= -1.0f;
    }
    return normalize(mul(surfelNm * scale, rotationT));
}

[BackwardDifferentiable][ForceInline]
bool hit(
    float3 rayOrigin,
    float3 rayDirection,
    Parameters parameters,
    out float alpha,
    inout float depth,
    no_diff bool enableNormal,
    inout float3 normal) {

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    const float maxResponse = canonicalRayMaxKernelResponse<KernelDegree>(
        canonicalRayOrigin,
        canonicalRayDirection);

    alpha = min(MaxParticleAlpha, maxResponse * parameters.density); // needed for the stability of the backward pass
    const bool acceptHit = ((maxResponse > MinParticleKernelDensity) && (alpha > MinParticleAlpha));
    if (acceptHit)
    {
        depth = canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);
        if (enableNormal)
        {
            normal = canonicalRayNormal<Surfel>(canonicalRayOrigin, canonicalRayDirection, parameters.scale, parameters.rotationT);
        }
    }
    return acceptHit;
}

[BackwardDifferentiable][ForceInline]
float integrateHit<let backToFront : bool>(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth,
    no_diff bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal)
{
   const float weight = backToFront ? alpha : alpha * transmittance;
   if (backToFront)
   {
       integratedDepth = lerp(integratedDepth, depth, alpha);
       if (enableNormal)
       {
           integratedNormal = lerp(integratedNormal, normal, alpha);
       }
   }
   else 
   {
       integratedDepth += depth * weight;
       if (enableNormal) 
       {
            integratedNormal += normal * weight;
       }
   }

   transmittance *= (1 - alpha);
   return weight; 
}

[BackwardDifferentiable][ForceInline]
float processHitFromBuffer<let backToFront : bool>(
    no_diff float3 rayOrigin,
    no_diff float3 rayDirection,
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    inout float transmittance,
    inout float integratedDepth,
    no_diff bool enableNormal,
    inout float3 integratedNormal)
{
    float alpha = 0.0f;
    float depth;
    float3 normal;
    if (hit(rayOrigin,
            rayDirection,
            fetchParameters(particleIdx, parametersBuffer),
            alpha,
            depth,
            enableNormal,
            normal))
    {
        return integrateHit<backToFront>(alpha, 
                                         transmittance, 
                                         depth, 
                                         integratedDepth, 
                                         enableNormal, 
                                         normal, 
                                         integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable][ForceInline] float3x3 computeCovariance(
    in Parameters parameters) {
    // Σ = RSS^TR^T
    const float3x3 StRt = float3x3(parameters.scale.x * parameters.rotationT[0],
                                  parameters.scale.y * parameters.rotationT[1],
                                  parameters.scale.z * parameters.rotationT[2]);
    return mul(transpose(StRt), StRt);
}

[BackwardDifferentiable] [ForceInline]
float3 incidentDirectionFromParameters(
    Parameters parameters,
    no_diff float3 sourcePosition
)
{
    return normalize(parameters.position - sourcePosition);
}

[BackwardDifferentiable][ForceInline]
float3 incidentDirectionFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float3 sourcePosition
)
{
    return incidentDirectionFromParameters(
        fetchParameters(particleIdx, parametersBuffer),
        sourcePosition
    );
}
} // namespace gaussianParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport] 
inline gaussianParticle.Parameters particleDensityParameters(
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters) 
{
    return gaussianParticle.fetchParameters(
        particleIdx,
        commonParameters.parametersBuffer);
}

[CudaDeviceExport]
inline bool particleDensityHit(
    float3 rayOrigin,
    float3 rayDirection,
    gaussianParticle.Parameters parameters,
    out float alpha,
    out float depth,
    bool enableNormal,
    out float3 normal)
{
    return gaussianParticle.hit(rayOrigin,
                                rayDirection,
                                parameters,
                                alpha,
                                depth,
                                enableNormal,
                                normal);
}

[CudaDeviceExport] inline float particleDensityIntegrateHit(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth,
    in bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal) 
{
    return gaussianParticle.integrateHit<false>(
        alpha,
        transmittance,
        depth,
        integratedDepth,
        enableNormal,
        normal,
        integratedNormal);
}

[CudaDeviceExport]
inline float particleDensityProcessHitFwdFromBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    inout float transmittance,
    inout float integratedDepth,
    in bool enableNormal,
    inout float3 integratedNormal) 
{
    return gaussianParticle.processHitFromBuffer<false>(
        rayOrigin,
        rayDirection,
        particleIdx,
        commonParameters.parametersBuffer,
        transmittance,
        integratedDepth,
        enableNormal,
        integratedNormal);
}

[CudaDeviceExport]
void particleDensityProcessHitBwdToBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float depth,
    inout float integratedDepth,
    inout float integratedDepthGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad) 
{
    if (alpha > 0.0f)
    {
        const float weight = 1.0f / (1.0f - alpha);
        
        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);
        
        integratedDepth = (integratedDepth - depth * alpha) * weight;
        DifferentialPair<float> integratedDepthDiff = DifferentialPair<float>(integratedDepth, integratedDepthGrad);

        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        bwd_diff(gaussianParticle.processHitFromBuffer<true>)(
            rayOrigin,
            rayDirection,
            particleIdx,
            commonParameters.parametersBuffer,
            transmittanceDiff,
            integratedDepthDiff,
            enableNormal,
            integratedNormalDiff,
            alphaGrad);

        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedDepthGrad = integratedDepthDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }
    }
}

[CudaDeviceExport]
bool particleDensityHitCustom(
    float3 rayOrigin,
    float3 rayDirection,
    int32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    float minHitDistance,
    float maxHitDistance,
    float maxParticleSquaredDistance,
    out float hitDistance
)
{
    gaussianParticle.Parameters parameters = gaussianParticle.fetchParameters(particleIdx, commonParameters.parametersBuffer);

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    gaussianParticle.cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    const float maxResponse = gaussianParticle.canonicalRayMaxKernelResponse<gaussianParticle.KernelDegree>(
        canonicalRayOrigin,
        canonicalRayDirection);

    // distance to the gaussian center projection on the ray
    hitDistance = gaussianParticle.canonicalRayDistance(canonicalRayOrigin, canonicalRayDirection, parameters.scale);

    return (hitDistance > minHitDistance) &&
           (hitDistance < maxHitDistance) &&
           (gaussianParticle.canonicalRayMinSquaredDistance(
                canonicalRayOrigin, canonicalRayDirection) < maxParticleSquaredDistance);
}

[CudaDeviceExport]
bool particleDensityHitInstance(
    float3 canonicalRayOrigin,
    float3 canonicalUnormalizedRayDirection,
    float minHitDistance,
    float maxHitDistance,
    float maxParticleSquaredDistance,
    out float hitDistance
)
{
    const float numerator = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection);
    const float denominator = rcp(dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection));
    hitDistance = numerator * denominator;
    return (hitDistance > minHitDistance) &&
           (hitDistance < maxHitDistance) &&
           (gaussianParticle.canonicalRayMinSquaredDistance(
                canonicalRayOrigin,
                normalize(canonicalUnormalizedRayDirection)) < maxParticleSquaredDistance);
}

[CudaDeviceExport] float3 particleDensityIncidentDirection(
    in gaussianParticle.Parameters parameters,
    in float3 sourcePosition
)
{
    return gaussianParticle.incidentDirectionFromParameters(parameters, sourcePosition);
}

[CudaDeviceExport] void particleDensityIncidentDirectionBwdToBuffer(
    in uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float3 sourcePosition,
    in float3 incidentDirectionGrad
)
{
    bwd_diff(gaussianParticle.incidentDirectionFromBuffer)(
        particleIdx, 
        commonParameters.parametersBuffer, 
        sourcePosition, 
        incidentDirectionGrad
    );
}
