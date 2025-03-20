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

#include <3dgut/sensors/sensors.h>

namespace threedgut {

template <int N>
static inline __device__ float evalPolyHorner(const tcnn::vec<N>& coeffs, float x) {
    // Evaluates a N-1 degree polynomial y=f(x) using numerically stable Horner scheme.
    // With :
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    float y = coeffs[N - 1];
#pragma unroll
    for (int i = N - 2; i >= 0; --i) {
        y = x * y + coeffs[i];
    }
    return y;
}

static inline __device__ float relativeShutterTime(const TSensorModel& sensorModel,
                                                   const tcnn::ivec2& resolution,
                                                   const tcnn::vec2& position) {
    switch (sensorModel.shutterType) {
    case TSensorModel::RollingTopToBottomShutter:
        return floorf(position.y) / (resolution.y - 1.f);
    case TSensorModel::RollingLeftToRightShutter:
        return floorf(position.x) / (resolution.x - 1.f);
    case TSensorModel::RollingBottomToTopShutter:
        return (resolution.y - ceilf(position.y)) / (resolution.y - 1.f);
    case TSensorModel::RollingRightToLeftShutter:
        return (resolution.x - ceilf(position.x)) / (resolution.x - 1.f);
    default:
        return 0.5f;
    }
};

static __forceinline__ __device__ bool withinResolution(const tcnn::vec2& resolution, float tolerance, const tcnn::vec2& p) {
    const tcnn::vec2 tolMargin = resolution * tolerance;
    return (p.x > -tolMargin.x) && (p.y > -tolMargin.y) && (p.x < resolution.x + tolMargin.x) && (p.y < resolution.y + tolMargin.y);
}

static inline __device__ bool projectPoint(const OpenCVPinholeProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {

    if (position.z <= 0.f) {
        projected = tcnn::vec2::zero();
        return false;
    }

    const tcnn::vec2 uvNormalized = position.xy() / position.z;

    // computeDistortion
    const tcnn::vec2 uvSquared = uvNormalized * uvNormalized;
    const float r2             = uvSquared.x + uvSquared.y;
    const float a1             = 2.f * uvNormalized.x * uvNormalized.y;
    const float a2             = r2 + 2.f * uvSquared.x;
    const float a3             = r2 + 2.f * uvSquared.y;

    const float icD_numerator   = 1.f + r2 * (sensorParams.radialCoeffs[0] + r2 * (sensorParams.radialCoeffs[1] + r2 * sensorParams.radialCoeffs[2]));
    const float icD_denominator = 1.f + r2 * (sensorParams.radialCoeffs[3] + r2 * (sensorParams.radialCoeffs[4] + r2 * sensorParams.radialCoeffs[5]));
    const float icD             = icD_numerator / icD_denominator;

    const tcnn::vec2 delta = tcnn::vec2{
        sensorParams.tangentialCoeffs[0] * a1 + sensorParams.tangentialCoeffs[1] * a2 + r2 * (sensorParams.thinPrismCoeffs[0] + r2 * sensorParams.thinPrismCoeffs[1]),
        sensorParams.tangentialCoeffs[0] * a3 + sensorParams.tangentialCoeffs[1] * a1 + r2 * (sensorParams.thinPrismCoeffs[2] + r2 * sensorParams.thinPrismCoeffs[3])};

    // Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
    // in case radial distortion is within limits
    const tcnn::vec2 uvND = icD * uvNormalized + delta;

    constexpr float kMinRadialDist = 0.8f, kMaxRadialDist = 1.2f;
    const bool validRadial = (icD > kMinRadialDist) && (icD < kMaxRadialDist);
    if (validRadial) {
        projected = uvND * sensorParams.focalLength + sensorParams.principalPoint;
    } else {
        // If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
        // (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
        // out of the image bounds accordingly. The coordinates will be clipped to
        // viable range and direction but the exact values cannot be trusted / are still invalid
        const float roiClippingRadius = hypotf(resolution.x, resolution.y);
        projected                     = (roiClippingRadius / sqrtf(r2)) * uvNormalized + sensorParams.principalPoint;
    }

    return validRadial && withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const OpenCVFisheyeProjectionParameters& sensorParams,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    constexpr float eps   = 1e-11f;
    const float rho       = fmaxf(tcnn::length(position.xy()), eps);
    const float thetaFull = atan2f(rho, position.z);
    // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
    // In particular for omnidirectional cameras, this prevents points outside the FOV to be
    // wrongly projected to in-image-domain points because of badly constrained polynomials outside
    // the effective FOV (which is different to the image boundaries).
    //
    // These FOV-clamped projections will be marked as *invalid*
    const float theta = fminf(thetaFull, sensorParams.maxAngle);
    // Evaluate forward polynomial
    // (radial distances to the principal point in the normalized image domain (up to focal length scales))
    const float delta =
        (theta * (evalPolyHorner<6>(sensorParams.radialCoeffs, theta * theta) + 1.0f)) / rho;
    projected = sensorParams.focalLength * position.xy() * delta + sensorParams.principalPoint;

    return (theta < sensorParams.maxAngle) && withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const TSensorModel& sensorModel,
                                           const tcnn::ivec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    switch (sensorModel.modelType) {
    case TSensorModel::OpenCVPinholeModel:
        return projectPoint(sensorModel.ocvPinholeParams, resolution, position, tolerance, projected);
    case TSensorModel::OpenCVFisheyeModel:
        return projectPoint(sensorModel.ocvFisheyeParams, resolution, position, tolerance, projected);
    default:
        projected = tcnn::vec2::zero();
        return false;
    }
}

template <int NRollingShutterIterations>
static inline __device__ bool projectPointWithShutter(const tcnn::vec3& position,
                                                      const tcnn::ivec2& resolution,
                                                      const TSensorModel& sensorModel,
                                                      const TSensorState& sensorState,
                                                      float tolerance,
                                                      tcnn::vec2& projectedPosition) {

    const tcnn::vec3 tStart = sensorState.startPose.slice<0, 3>();
    const tcnn::quat qStart = tcnn::quat{sensorState.startPose[6], sensorState.startPose[3], sensorState.startPose[4], sensorState.startPose[5]};

    bool validProjection = projectPoint(sensorModel, resolution, tcnn::to_mat3(qStart) * position + tStart, tolerance, projectedPosition);
    if (sensorModel.shutterType == TSensorModel::GlobalShutter) {
        return validProjection;
    }

    const tcnn::vec3 tEnd = sensorState.endPose.slice<0, 3>();
    const tcnn::quat qEnd = tcnn::quat{sensorState.endPose[6], sensorState.endPose[3], sensorState.endPose[4], sensorState.endPose[5]};

    if (!validProjection) {
        validProjection = projectPoint(sensorModel, resolution, tcnn::to_mat3(qEnd) * position + tEnd, tolerance, projectedPosition);
        if (!validProjection) {
            return false;
        }
    }

    // Compute the new timestamp and project again
#pragma unroll
    for (int i = 0; i < NRollingShutterIterations; ++i) {
        const float alpha = relativeShutterTime(sensorModel, resolution, projectedPosition);
        validProjection   = projectPoint(
            sensorModel,
            resolution,
            tcnn::to_mat3(tcnn::slerp(qStart, qEnd, alpha)) * position + tcnn::mix(tStart, tEnd, alpha),
            tolerance,
            projectedPosition);
    }

    return validProjection;
}

} // namespace threedgut
