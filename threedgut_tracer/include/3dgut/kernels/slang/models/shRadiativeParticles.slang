// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use it except in compliance with the License.
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
static const int RadianceMaxNumSphCoefficients = PARTICLE_RADIANCE_NUM_COEFFS;
static const int Dim = 3;
};

namespace shRadiativeParticle
{

struct ParametersBuffer
{
    Ptr<vector<float, Dim>, Access.Read> _dataPtr;
    vector<float, Dim> *_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
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
    [ForceUnroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
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
    [ForceUnroll] for (int i = 0; i < RadianceMaxNumSphCoefficients; ++i) {
        const vector<float, Dim> coeffs = parametersGrad.sphCoefficients[i];
        if (parametersBuffer.exclusiveGradient) {
            [ForceUnroll] for (int j = 0; j < Dim; ++j) {
                parametersBuffer._gradPtr[particleOffset + i][j] += coeffs[j];
            }
        } else {
            [ForceUnroll] for (int j = 0; j < Dim; ++j) {
                InterlockedAdd(parametersBuffer._gradPtr[particleOffset + i][j], coeffs[j]);
            }
        }
    }
}

[BackwardDifferentiable] [ForceInline]
vector<float, Dim> radianceFromBuffer(no_diff uint32_t particleIdx,
                                      in float3 incidentDirection,
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
inline void particleFeaturesFromBuffer(
    in uint32_t particleIdx,
    in float *radianceBufferPtr,
    in int auxParam,
    in float3 incidentDirection,
    in float3 canonicalPosition,
    out float features[shRadiativeParticle.Dim])
{
    shRadiativeParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)radianceBufferPtr;
    parametersBuffer._gradPtr = nullptr;
    parametersBuffer.exclusiveGradient = false;

    vector<float, shRadiativeParticle.Dim> featuresVec = sphericalHarmonics.decode<shRadiativeParticle.Dim, shRadiativeParticle.RadianceMaxNumSphCoefficients>(
        auxParam,
        shRadiativeParticle.fetchParametersFromBuffer(particleIdx, parametersBuffer).sphCoefficients,
        incidentDirection);

    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        features[i] = featuresVec[i];
    }
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwd(in float weight,
                                         in float features[shRadiativeParticle.Dim],
                                         inout float integratedFeatures[shRadiativeParticle.Dim])
{
    vector<float, shRadiativeParticle.Dim> featuresVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        featuresVec[i] = features[i];
    }
    vector<float, shRadiativeParticle.Dim> integratedVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedVec[i] = integratedFeatures[i];
    }
    shRadiativeParticle.integrateRadiance<false>(weight, featuresVec, integratedVec);
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedFeatures[i] = integratedVec[i];
    }
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwdFromBuffer(
    in float3 incidentDirection,
    in float3 canonicalPosition,
    in float weight,
    in uint32_t particleIdx,
    in float *radianceBufferPtr,
    in int auxParam,
    inout float integratedFeatures[shRadiativeParticle.Dim])
{
    shRadiativeParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)radianceBufferPtr;
    parametersBuffer._gradPtr = nullptr;
    parametersBuffer.exclusiveGradient = false;

    vector<float, shRadiativeParticle.Dim> integratedVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedVec[i] = integratedFeatures[i];
    }
    shRadiativeParticle.integrateRadianceFromBuffer<false>(
        incidentDirection, auxParam, weight, particleIdx, parametersBuffer, integratedVec);
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
        integratedFeatures[i] = integratedVec[i];
    }
}

[CudaDeviceExport] void particleFeaturesIntegrateBwd(
    in float alpha,
    inout float alphaGrad,
    in float features[shRadiativeParticle.Dim],
    inout float featuresGrad[shRadiativeParticle.Dim],
    inout float integratedFeatures[shRadiativeParticle.Dim],
    inout float integratedFeaturesGrad[shRadiativeParticle.Dim])
{
    if(alpha > 0.0f)
    {
        vector<float, shRadiativeParticle.Dim> featuresVec, featuresGradVec;
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            featuresVec[i] = features[i];
            featuresGradVec[i] = featuresGrad[i];
        }
        vector<float, shRadiativeParticle.Dim> integratedVec, integratedGradVec;
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedVec[i] = integratedFeatures[i];
            integratedGradVec[i] = integratedFeaturesGrad[i];
        }

        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> featuresDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(featuresVec, featuresGradVec);

        const float weight = 1.0f / (1.0f - alpha);
        integratedVec = (integratedVec - featuresVec * alpha) * weight;
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedVec, integratedGradVec);

        bwd_diff(shRadiativeParticle.integrateRadiance<true>)(
            alphaDiff,
            featuresDiff,
            integratedFeaturesDiff);

        alphaGrad = alphaDiff.getDifferential();
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            featuresGrad[i] = featuresDiff.getDifferential()[i];
        }
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeaturesGrad[i] = integratedFeaturesDiff.getDifferential()[i];
        }
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeatures[i] = integratedVec[i];
        }
    }
}

[CudaDeviceExport] void particleFeaturesIntegrateBwdToBuffer(
    in float3 incidentDirection,
    in float3 canonicalPosition,
    inout float3 canonicalPositionGrad,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    in float *featuresBufferPtr,
    in float *featuresBufferGradPtr,
    in int auxParam,
    in bool exclusiveGradient,
    in float features[shRadiativeParticle.Dim],
    inout float integratedFeatures[shRadiativeParticle.Dim],
    inout float integratedFeaturesGrad[shRadiativeParticle.Dim])
{
    if (alpha > 0.0f)
    {
        shRadiativeParticle.ParametersBuffer parametersBuffer;
        parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)featuresBufferPtr;
        parametersBuffer._gradPtr = (vector<float, 3>*)featuresBufferGradPtr;
        parametersBuffer.exclusiveGradient = exclusiveGradient;

        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);

        const float weight = 1.0f / (1.0f - alpha);
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeatures[i] = (integratedFeatures[i] - features[i] * alpha) * weight;
        }

        vector<float, shRadiativeParticle.Dim> integratedFeaturesVec;
        vector<float, shRadiativeParticle.Dim> integratedFeaturesGradVec;
        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeaturesVec[i]     = integratedFeatures[i];
            integratedFeaturesGradVec[i] = integratedFeaturesGrad[i];
        }
        DifferentialPair<vector<float, shRadiativeParticle.Dim>> integratedFeaturesDiff =
            DifferentialPair<vector<float, shRadiativeParticle.Dim>>(integratedFeaturesVec, integratedFeaturesGradVec);

        bwd_diff(shRadiativeParticle.integrateRadianceFromBuffer<true>)(
            incidentDirection,
            auxParam,
            alphaDiff,
            particleIdx,
            parametersBuffer,
            integratedFeaturesDiff);

        [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) {
            integratedFeaturesGrad[i] = integratedFeaturesDiff.getDifferential()[i];
        }
        alphaGrad += alphaDiff.getDifferential();
    }
}

[CudaDeviceExport] void particleFeaturesBwdToBuffer(
    in uint32_t particleIdx,
    in float *featuresBufferPtr,
    in float *featuresBufferGradPtr,
    in int auxParam,
    in bool exclusiveGradient,
    in float featuresGrad[shRadiativeParticle.Dim],
    in float3 incidentDirection,
    inout float3 incidentDirectionGrad
)
{
    shRadiativeParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<vector<float, 3>, Access.Read>)featuresBufferPtr;
    parametersBuffer._gradPtr = (vector<float, 3>*)featuresBufferGradPtr;
    parametersBuffer.exclusiveGradient = exclusiveGradient;

    vector<float, shRadiativeParticle.Dim> featuresGradVec;
    [ForceUnroll] for (int i = 0; i < shRadiativeParticle.Dim; ++i) featuresGradVec[i] = featuresGrad[i];

    DifferentialPair<float3> incidentDirectionDiff = DifferentialPair<float3>(incidentDirection, incidentDirectionGrad);

    bwd_diff(shRadiativeParticle.radianceFromBuffer)(
        particleIdx,
        incidentDirectionDiff,
        auxParam,
        parametersBuffer,
        featuresGradVec);

    incidentDirectionGrad += incidentDirectionDiff.getDifferential();
}
