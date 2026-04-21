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

// Neural Harmonic Features: per-particle K-dim features, output N (decoder input); decoder maps N -> RGB.
// Same CudaDeviceExport API as shRadiativeParticles.slang for drop-in replacement.
// Compile-time: FEATURE_INTERPOLATION_TYPE, FEATURE_INTERPOLATION_SUPPORT, FEATURE_ACTIVATION_TYPE, FEATURE_ACTIVATION_NUM_FREQUENCIES,
//   PARTICLE_FEATURE_DIM (total K per particle = buffer stride), INTERP_POINT_FEATURE_DIM (per-interpolation-point = K/num_points).
// Support: center -> K=interpPointDim; tetrahedra -> K=4*interpPointDim. With activation: N = interpPointDim * num_frequencies (decoder input).

namespace neuralHarmonicFeaturesParticle
{
static const int ParticleFeatureDim = PARTICLE_FEATURE_DIM;     // Total per-particle (K); not per-interpolation-point
static const int RayFeatureDim = RAY_FEATURE_DIM;              // Decoder input N = INTERP_POINT_FEATURE_DIM * FEATURE_ACTIVATION_NUM_FREQUENCIES
static const int InterpPointFeatureDim = INTERP_POINT_FEATURE_DIM;  // Per-interpolation-point dimension
static const int FeatureActivationType_None = 0;
static const int FeatureActivationType_Siren = 1;
static const int FeatureActivationType_Sincos = 2;
static const int FeatureActivationType_Relu = 3;
static const int FeatureActivationType = FEATURE_ACTIVATION_TYPE;
static const int FeatureActivationNumFrequencies = FEATURE_ACTIVATION_NUM_FREQUENCIES;

// Interpolation type (compile-time from FEATURE_INTERPOLATION_TYPE)
static const int InterpolationType_Barycentric = 0;
static const int InterpolationType_Bezier = 1; // Not supported yet
static const int InterpolationType = FEATURE_INTERPOLATION_TYPE;

// Interpolation support (compile-time from FEATURE_INTERPOLATION_SUPPORT)
static const int InterpolationSupport_Center = 0;
static const int InterpolationSupport_Tetrahedra = 1;
static const int InterpolationSupport_CoTriangles = 2; // 2 coplanar triangles / Not supported yet
static const int InterpolationSupport = FEATURE_INTERPOLATION_SUPPORT;

// Canonical regular tetrahedron that contains the unit sphere; each face is tangent to the unit sphere
// (insphere radius = 1). Same geometry as particlePrimitives.cu (enclosing tetrahedra).
// Incenter at origin; base z=-1, apex z=3; edge s=sqrt(24), height h=4, inradius r=1.
static const float tetraHedraEdge         = 4.898979485566356f;   // sqrt(24)
static const float tetraHedraFaceHeight   = 4.242640687119285f;   // s*sqrt(3)/2
static const float tetraHedraHeight       = 4.0f;                 // s*sqrt(2/3)
static const float tetraHedraFaceInRadius = 1.4142135623730951f; // s*sqrt(3)/6 = sqrt(2)
static const float tetraHedraInRadius     = 1.0f;                 // unit sphere
static const float3 canonicalTetraVerts[4] = {
    float3(-0.5f * tetraHedraEdge, -tetraHedraFaceInRadius, -1.0f),
    float3(0.0f, tetraHedraFaceHeight - tetraHedraFaceInRadius, -1.0f),
    float3(0.0f, 0.0f, tetraHedraHeight - tetraHedraInRadius),
    float3(0.5f * tetraHedraEdge, -tetraHedraFaceInRadius, -1.0f)
};
// Cramer terms for canonical tetrahedron (independent of P); used by barycentricTetrahedronCanonical.
static const float3 canonicalTetraE1 = canonicalTetraVerts[1] - canonicalTetraVerts[0];
static const float3 canonicalTetraE2 = canonicalTetraVerts[2] - canonicalTetraVerts[0];
static const float3 canonicalTetraE3 = canonicalTetraVerts[3] - canonicalTetraVerts[0];
static const float3 canonicalTetraCrossE2E3 = cross(canonicalTetraE2, canonicalTetraE3);
static const float canonicalTetraDet = dot(canonicalTetraE1, canonicalTetraCrossE2E3);
static const float canonicalTetraInvDet = 1.0f / canonicalTetraDet;

};

#if PARTICLE_FEATURE_HALF
typedef half feat_elem_t;
#else
typedef float feat_elem_t;
#endif

namespace neuralHarmonicFeaturesParticle
{

struct ParametersBuffer
{
    Ptr<feat_elem_t, Access.Read> _dataPtr;  // [N_particles, K] flat; fp16 when PARTICLE_FEATURE_HALF
    float *_gradPtr;
    bool exclusiveGradient;
};

struct Parameters : IDifferentiable
{
    Array<float, InterpPointFeatureDim> features;
};

[BackwardDifferentiable][ForceInline]
Parameters fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                     no_diff int interpPointIdx,
                                     no_diff ParametersBuffer parametersBuffer)
{
    Parameters parameters;
    const uint32_t particleOffset = particleIdx * ParticleFeatureDim;
    const uint32_t interpPointOffset = particleOffset + interpPointIdx * InterpPointFeatureDim;
    [ForceUnroll] for (int i = 0; i < InterpPointFeatureDim; ++i) {
        parameters.features[i] = parametersBuffer._dataPtr[interpPointOffset + i];
    }
    return parameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline]
void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                  no_diff int interpPointIdx,
                                  no_diff ParametersBuffer parametersBuffer,
                                  Parameters parametersGrad)
{
    const uint32_t particleOffset = particleIdx * ParticleFeatureDim;
    const uint32_t interpPointOffset = particleOffset + interpPointIdx * InterpPointFeatureDim;
    [ForceUnroll] for (int i = 0; i < InterpPointFeatureDim; ++i) {
        const float grad = parametersGrad.features[i];
        if (parametersBuffer.exclusiveGradient) {
            parametersBuffer._gradPtr[interpPointOffset + i] += grad;
        } else {
            InterlockedAdd(parametersBuffer._gradPtr[interpPointOffset + i], grad);
        }
    }
}

// Barycentric coordinates for the canonical tetrahedron only. Uses precomputed static const e1,e2,e3,invDet (independent of P).
[BackwardDifferentiable][ForceInline]
float4 barycentricTetrahedronCanonical(float3 P)
{
    float3 d = P - canonicalTetraVerts[0];
    float4 weights;
    weights.y = dot(d, canonicalTetraCrossE2E3) * canonicalTetraInvDet;
    weights.z = dot(canonicalTetraE1, cross(d, canonicalTetraE3)) * canonicalTetraInvDet;
    weights.w = dot(canonicalTetraE1, cross(canonicalTetraE2, d)) * canonicalTetraInvDet;
    weights.x = 1.0f - weights.y - weights.z - weights.w;
    return weights;
}

// Encode and activate : none -> identity; siren -> sin(b*2^f); sincos -> sin(b*2^f)+cos(b*2^f); relu -> max(0,b).
[BackwardDifferentiable][ForceInline]
float encodeAndActivate(float baseVal, no_diff int f)
{
    if (FeatureActivationType == FeatureActivationType_None)
        return baseVal;
    if (FeatureActivationType == FeatureActivationType_Relu)
        return max(0.0f, baseVal);
    float freq = ldexp(1.0f, f);
    float angle = baseVal * freq;
    if (FeatureActivationType == FeatureActivationType_Siren)
        return sin(angle);
    return sin(angle) + cos(angle);
}

// Compute blended features into baseFeatures[INTERP_POINT_FEATURE_DIM], then optionally expand by activation to features[RayFeatureDim].
// canonicalPosition is differential (hit position in particle canonical space; gradient accumulated in API canonicalPositionGrad).
[BackwardDifferentiable][ForceInline]
void featuresFromParametersBuffer(ParametersBuffer parametersBuffer,
                                  no_diff uint32_t particleIdx,
                                  float3 canonicalPosition, 
                                  out Array<float, RayFeatureDim> features
)
{
    Array<float, InterpPointFeatureDim> baseFeatures = fetchParametersFromBuffer(particleIdx, 0, parametersBuffer).features;
    if (InterpolationSupport == InterpolationSupport_Tetrahedra && InterpolationType == InterpolationType_Barycentric) {
        float4 barycentricWeights = barycentricTetrahedronCanonical(canonicalPosition);
        [ForceUnroll] for (int n = 0; n < InterpPointFeatureDim; ++n) {
            baseFeatures[n] *= barycentricWeights[0];
        }
        [ForceUnroll] for (int k = 1; k < 4; ++k) {
            Parameters parameters = fetchParametersFromBuffer(particleIdx, k, parametersBuffer);
            [ForceUnroll] for (int n = 0; n < InterpPointFeatureDim; ++n) {
                baseFeatures[n] += barycentricWeights[k] * parameters.features[n];
            }
        }
    }

    if (FeatureActivationType == FeatureActivationType_None) {
        [ForceUnroll] for (int i = 0; i < InterpPointFeatureDim; ++i) {
            features[i] = baseFeatures[i];
        }
    } else {
        [ForceUnroll] for (int k = 0; k < InterpPointFeatureDim; ++k) {
            [ForceUnroll] for (int f = 0; f < FeatureActivationNumFrequencies; ++f) {
                features[k * FeatureActivationNumFrequencies + f] = encodeAndActivate(baseFeatures[k], f);
            }
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateFeatures<let backToFront : bool>(float weight,
                                               in Array<float, RayFeatureDim> features,
                                               inout float integratedFeatures[RayFeatureDim])
{
    if (weight > 0.0f) {
        [ForceUnroll] for (int i = 0; i < RayFeatureDim; ++i) {
            if (backToFront)
                integratedFeatures[i] = lerp(integratedFeatures[i], features[i], weight);
            else
                integratedFeatures[i] += features[i] * weight;
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateFeaturesFromBuffer<let backToFront : bool>(float weight,
                                                         no_diff uint32_t particleIdx,
                                                         ParametersBuffer parametersBuffer,
                                                         float3 canonicalPosition,
                                                         inout float integratedFeatures[RayFeatureDim])
{
    if (weight > 0.0f) {
        Array<float, RayFeatureDim> features;
        featuresFromParametersBuffer(parametersBuffer, particleIdx, canonicalPosition, features);
        integrateFeatures<backToFront>(weight, features, integratedFeatures);
    }
}

} // namespace neuralHarmonicFeaturesParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points - same CudaDeviceExport API as shRadiativeParticles.slang

[CudaDeviceExport]
inline void particleFeaturesFromBuffer(
    in uint32_t particleIdx,
    in feat_elem_t *featuresBufferPtr,
    in int auxParam,
    in float3 incidentDirection,
    in float3 canonicalPosition,
    out float features[neuralHarmonicFeaturesParticle.RayFeatureDim])
{
    neuralHarmonicFeaturesParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<feat_elem_t, Access.Read>)featuresBufferPtr;
    parametersBuffer._gradPtr = nullptr;
    parametersBuffer.exclusiveGradient = false;

    neuralHarmonicFeaturesParticle.featuresFromParametersBuffer(
        parametersBuffer,
        particleIdx, 
        canonicalPosition,
        features
    );
}

[CudaDeviceExport]
inline void particleFeaturesIntegrateFwdFromBuffer(
    in float3 incidentDirection,
    in float3 canonicalPosition,
    in float weight,
    in uint32_t particleIdx,
    in feat_elem_t *featuresBufferPtr,
    in int auxParam,
    inout float integratedFeatures[neuralHarmonicFeaturesParticle.RayFeatureDim])
{
    neuralHarmonicFeaturesParticle.ParametersBuffer parametersBuffer;
    parametersBuffer._dataPtr = (Ptr<feat_elem_t, Access.Read>)featuresBufferPtr;
    parametersBuffer._gradPtr = nullptr;
    parametersBuffer.exclusiveGradient = false;

    neuralHarmonicFeaturesParticle.integrateFeaturesFromBuffer<false>(
        weight, particleIdx, parametersBuffer, canonicalPosition, integratedFeatures);
}

// canonicalPosition is differential; canonicalPositionGrad is inout and accumulated by this backward.
[CudaDeviceExport] void particleFeaturesIntegrateBwdToBuffer(
    in float3 incidentDirection,
    in float3 canonicalPosition,
    inout float3 canonicalPositionGrad,
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    in feat_elem_t *featuresBufferPtr,
    in float *featuresBufferGradPtr,
    in int auxParam,
    in bool exclusiveGradient,
    in float features[neuralHarmonicFeaturesParticle.RayFeatureDim],
    inout float integratedFeatures[neuralHarmonicFeaturesParticle.RayFeatureDim],
    inout float integratedFeaturesGrad[neuralHarmonicFeaturesParticle.RayFeatureDim])
{
    if (alpha > 0.0f)
    {
        neuralHarmonicFeaturesParticle.ParametersBuffer parametersBuffer;
        parametersBuffer._dataPtr = (Ptr<feat_elem_t, Access.Read>)featuresBufferPtr;
        parametersBuffer._gradPtr = (Ptr<float, Access.ReadWrite>)featuresBufferGradPtr;
        parametersBuffer.exclusiveGradient = exclusiveGradient;

        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);

        const float weight = 1.0f / (1.0f - alpha);
        [ForceUnroll] for (int i = 0; i < neuralHarmonicFeaturesParticle.RayFeatureDim; ++i) {
            integratedFeatures[i] = (integratedFeatures[i] - features[i] * alpha) * weight;
        }

        DifferentialPair<float[neuralHarmonicFeaturesParticle.RayFeatureDim]> integratedFeaturesDiff =
            DifferentialPair<float[neuralHarmonicFeaturesParticle.RayFeatureDim]>(integratedFeatures, integratedFeaturesGrad);

        DifferentialPair<float3> canonicalPositionDiff = DifferentialPair<float3>(canonicalPosition, canonicalPositionGrad);

        bwd_diff(neuralHarmonicFeaturesParticle.integrateFeaturesFromBuffer<true>)(
            alphaDiff,
            particleIdx,
            parametersBuffer,
            canonicalPositionDiff,
            integratedFeaturesDiff);

        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
        canonicalPositionGrad += canonicalPositionDiff.getDifferential();
        alphaGrad += alphaDiff.getDifferential();
    }
}
