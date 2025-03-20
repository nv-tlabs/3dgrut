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

// clang-format off
#include <tiny-cuda-nn/common.h>
#include <3dgut/sensors/cameraModels.h>
// clang-format on

namespace threedgut {

// tcnn::vec::slice method is not compiling
template <uint32_t Offset, uint32_t OutSize, typename T, uint32_t InSize, size_t A>
inline TCNN_HOST_DEVICE tcnn::tvec<T, OutSize, A>& sliceVec(const tcnn::tvec<T, InSize, A>& vec) {
    return *(tcnn::tvec<T, OutSize, A>*)(vec.data() + Offset);
}

using TTimestamp = int64_t;

using TSensorPose = tcnn::vec<7>; // 3D position and 3D quaternion (x,y,z,w)

using TSensorModel = CameraModelParameters;

struct TSensorState {
    TTimestamp startTimestamp;
    TSensorPose startPose;
    TTimestamp endTimestamp;
    TSensorPose endPose;
};

static inline TCNN_HOST_DEVICE TSensorPose sensorPoseInverse(const TSensorPose& pose) {
    const tcnn::mat3 invRotation   = tcnn::transpose(tcnn::to_mat3(tcnn::tquat{pose[6], pose[3], pose[4], pose[5]}));
    const tcnn::quat invQuaternion = tcnn::quat{invRotation};
    TSensorPose invPose;
    invPose.slice<0, 3>() = -1.0f * invRotation * pose.slice<0, 3>();
    invPose.slice<3, 4>() = tcnn::vec4{invQuaternion.x, invQuaternion.y, invQuaternion.z, invQuaternion.w};
    return invPose;
}

static inline TCNN_HOST_DEVICE TSensorPose interpolatedSensorPose(const TSensorPose& startPose,
                                                                  const TSensorPose& endPose,
                                                                  float relativeTime) {
    using namespace tcnn;

    const quat interpolatedQuat = slerp(quat{startPose[6], startPose[3], startPose[4], startPose[5]},
                                        quat{endPose[6], endPose[3], endPose[4], endPose[5]},
                                        relativeTime);
    TSensorPose interpolated;
    interpolated.slice<0, 3>() = mix(startPose.slice<0, 3>(), endPose.slice<0, 3>(), relativeTime);
    interpolated.slice<3, 4>() = vec4{interpolatedQuat.x, interpolatedQuat.y, interpolatedQuat.z, interpolatedQuat.w};

    return interpolated;
}

static inline TCNN_HOST_DEVICE tcnn::mat4x3 sensorPoseToMat(const TSensorPose& pose) {
    using namespace tcnn;

    const mat3 rotation = to_mat3(quat{pose[6], pose[3], pose[4], pose[5]});
    return mat4x3{rotation[0], rotation[1], rotation[2], pose.slice<0, 3>()};
}

} // namespace threedgut
