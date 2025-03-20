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

#include <3dgut/kernels/slang/common/sphericalHarmonics.slang>

namespace shRadiativeParticle
{

struct ParametersBuffer
{
    const vector<float, Dim>* _dataPtr;
    vector<float, Dim> *_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct CommonParameters
{
    ParametersBuffer parametersBuffer;
    int sphDegree;
};

struct Parameters : IDifferentiable
{
    vector<float, Dim> sphCoefficients[RadianceMaxNumSphCoefficients];
};

[BackwardDifferentiable][ForceInline]
Parameters fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                     no_diff ParametersBuffer parametersBuffer) 
{
    Parameters parameters;
    const uint32_t particleOffset = particleIdx * RadianceMaxNumSphCoefficients;
    [unroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        parameters.sphCoefficients[i] = parametersBuffer._dataPtr[particleOffset + i];
    }
    return parameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] 
void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                  no_diff ParametersBuffer parametersBuffer,
                                  Parameters parametersGrad) 
{
    const uint32_t particleOffset = particleIdx * RadianceMaxNumSphCoefficients;
    [unroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        const vector<float, Dim> coeffs = parametersGrad.sphCoefficients[i];
        if (parametersBuffer.exclusiveGradient) {
            [unroll] for (int j = 0; j < Dim; ++j) {
                parametersBuffer._gradPtr[particleOffset + i][j] += coeffs[j];
            }
        } else {
            [unroll] for (int j = 0; j < Dim; ++j) {
                InterlockedAdd(parametersBuffer._gradPtr[particleOffset + i][j], coeffs[j]);
            }
        }
    }
}

[BackwardDifferentiable] [ForceInline]
vector<float, Dim> radianceFromBuffer(no_diff uint32_t particleIdx,
                                      no_diff float3 incidentDirection,
                                      no_diff uint32_t sphDegree,
                                      no_diff ParametersBuffer parametersBuffer)
{
    return sphericalHarmonics.decode<Dim, RadianceMaxNumSphCoefficients>(
               sphDegree,
               fetchParametersFromBuffer(particleIdx, parametersBuffer).sphCoefficients,
               incidentDirection);
}

[BackwardDifferentiable][ForceInline]
void integrateRadiance<let backToFront : bool>(float weight,
                                               in vector<float, Dim> radiance,
                                               inout vector<float, Dim> integratedRadiance)
{
    if (weight > 0.0f)
    {
        if (backToFront)
        {
            integratedRadiance = lerp(integratedRadiance, radiance, weight);
        }
        else
        {
            integratedRadiance += radiance * weight;
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromParameters<let backToFront : bool>(float3 incidentDirection,
                                                             no_diff uint32_t sphDegree,
                                                             float weight,
                                                             Parameters parameters,
                                                             inout vector<float, Dim> integratedRadiance) 
{
    if (weight > 0.0f) 
    {
        integrateRadiance<backToFront>(
            weight,
            sphericalHarmonics.decode<Dim, RadianceMaxNumSphCoefficients>(
                sphDegree,
                parameters.sphCoefficients,
                incidentDirection),
            integratedRadiance
        );
    }
}

[BackwardDifferentiable][ForceInline]
void integrateRadianceFromBuffer<let backToFront : bool>(no_diff float3 incidentDirection,
                                                         no_diff uint32_t sphDegree,
                                                         float weight,
                                                         no_diff uint32_t particleIdx,
                                                         no_diff ParametersBuffer parametersBuffer,
                                                         inout vector<float, Dim> integratedRadiance)
{
    integrateRadianceFromParameters<backToFront>(incidentDirection,
                                                 sphDegree,
                                                 weight,
                                                 fetchParametersFromBuffer(particleIdx, parametersBuffer),
                                                 integratedRadiance);
}

} // namespace shRadiativeParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport]
inline vector<float, shRadiativeParticle.Dim> particleFeaturesFromBuffer(in uint32_t particleIdx,
                                         shRadiativeParticle.CommonParameters commonParameters,
                                         in float3 incidentDirection)
{
    return sphericalHarmonics.decode<shRadiativeParticle.Dim, shRadiativeParticle.RadianceMaxNumSphCoefficients>(
        commonParameters.sphDegree,
        shRadiativeParticle.fetchParametersFromBuffer(particleIdx, commonParameters.parametersBuffer).sphCoefficients,
        incidentDirection);
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwd(in float weight,
                                         in vector<float, shRadiativeParticle.Dim> features,
                                         inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    shRadiativeParticle.integrateRadiance<false>(
        weight,
        features,
        integratedFeatures
    );
}

[CudaDeviceExport] inline void particleFeaturesIntegrateFwdFromBuffer(in float3 incidentDirection,
                                                   in float weight,
                                                   in uint32_t particleIdx,
                                                   shRadiativeParticle.CommonParameters commonParameters,
                                                   inout vector<float, shRadiativeParticle.Dim> integratedFeatures)
{
    shRadiativeParticle.integrateRadianceFromBuffer<false>(
        incidentDirection,
        commonParameters.sphDegree,
        weight,
        particleIdx,
        commonParameters.parametersBuffer,
        integratedFeatures);
}

[CudaDeviceExport] void particleFeaturesIntegrateBwd(
    in float alpha,
    inout float alphaGrad,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> featuresGrad,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad)
{
    if(alpha > 0.0f)
    {
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> featuresDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(features, featuresGrad);

        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        bwd_diff(shRadiativeParticle.integrateRadiance<true>)(
            alphaDiff,
            featuresDiff,
            integratedFeaturesDiff);

        alphaGrad = alphaDiff.getDifferential();
        featuresGrad = featuresDiff.getDifferential();
        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
    }
}

[CudaDeviceExport] void particleFeaturesIntegrateBwdToBuffer(
    in float3 incidentDirection,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> features,
    inout vector<float, shRadiativeParticle.Dim> integratedFeatures,
    inout vector<float, shRadiativeParticle.Dim> integratedFeaturesGrad) 
{
    if (alpha > 0.0f)
    {
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        
        const float weight = 1.0f / (1.0f - alpha);
        integratedFeatures = (integratedFeatures - features * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff = 
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeatures, integratedFeaturesGrad);

        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)(
            incidentDirection,
            commonParameters.sphDegree,
            alphaDiff,
            particleIdx,
            commonParameters.parametersBuffer,
            integratedFeaturesDiff);

        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
        alphaGrad = alphaDiff.getDifferential();
    }
}

[CudaDeviceExport] void particleFeaturesBwdToBuffer(
    in uint32_t particleIdx,
    shRadiativeParticle.CommonParameters commonParameters,
    in vector<float, shRadiativeParticle.Dim> featuresGrad,
    in float3 incidentDirection
)
{
    bwd_diff(shRadiativeParticle.radianceFromBuffer)(
        particleIdx,
        incidentDirection,
        commonParameters.sphDegree,
        commonParameters.parametersBuffer,
        featuresGrad);
}