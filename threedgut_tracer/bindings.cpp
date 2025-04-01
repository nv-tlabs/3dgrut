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

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgut/splatRaster.h>

#include <3dgut/sensors/cameraModels.h>

threedgut::CameraModelParameters
fromOpenCVPinholeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::CameraModelParameters::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 6> radial_coeffs,
                                       std::array<float, 2> tangential_coeffs,
                                       std::array<float, 4> thin_prism_coeffs) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::CameraModelParameters::ShutterType>(shutter_type);
    params.modelType   = threedgut::CameraModelParameters::OpenCVPinholeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec<6>), "[3dgut] typing size mismatch");
    static_assert(sizeof(tangential_coeffs) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(thin_prism_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvPinholeParams.principalPoint   = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvPinholeParams.focalLength      = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvPinholeParams.radialCoeffs     = *reinterpret_cast<const tcnn::vec<6>*>(radial_coeffs.data());
    params.ocvPinholeParams.tangentialCoeffs = *reinterpret_cast<const tcnn::vec2*>(tangential_coeffs.data());
    params.ocvPinholeParams.thinPrismCoeffs  = *reinterpret_cast<const tcnn::vec4*>(thin_prism_coeffs.data());
    return params;
}

threedgut::CameraModelParameters
fromOpenCVFisheyeCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                       threedgut::CameraModelParameters::ShutterType shutter_type,
                                       std::array<float, 2> principal_point,
                                       std::array<float, 2> focal_length,
                                       std::array<float, 4> radial_coeffs,
                                       float max_angle) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::CameraModelParameters::ShutterType>(shutter_type);
    params.modelType   = threedgut::CameraModelParameters::OpenCVFisheyeModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(focal_length) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    static_assert(sizeof(radial_coeffs) == sizeof(tcnn::vec4), "[3dgut] typing size mismatch");
    params.ocvFisheyeParams.principalPoint = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.ocvFisheyeParams.focalLength    = *reinterpret_cast<const tcnn::vec2*>(focal_length.data());
    params.ocvFisheyeParams.radialCoeffs   = *reinterpret_cast<const tcnn::vec4*>(radial_coeffs.data());
    params.ocvFisheyeParams.maxAngle       = max_angle;
    return params;
}

threedgut::CameraModelParameters
fromFThetaCameraModelParameters(std::array<uint64_t, 2> _resolution,
                                threedgut::CameraModelParameters::ShutterType shutter_type,
                                std::array<float, 2> principal_point,
                                threedgut::FThetaProjectionParameters::PolynomialType reference_poly,
                                std::array<float, threedgut::FThetaProjectionParameters::PolynomialDegree> pixeldist_to_angle_poly, // backward polynomial
                                std::array<float, threedgut::FThetaProjectionParameters::PolynomialDegree> angle_to_pixeldist_poly, // forward polynomial
                                float max_angle,
                                std::array<float, 3> linear_cde) {
    threedgut::CameraModelParameters params;
    params.shutterType = static_cast<threedgut::CameraModelParameters::ShutterType>(shutter_type);
    params.modelType   = threedgut::CameraModelParameters::FThetaModel;
    static_assert(sizeof(principal_point) == sizeof(tcnn::vec2), "[3dgut] typing size mismatch");
    params.fthetaParams.principalPoint = *reinterpret_cast<const tcnn::vec2*>(principal_point.data());
    params.fthetaParams.referencePoly  = reference_poly;
    static_assert(sizeof(pixeldist_to_angle_poly) == sizeof(tcnn::vec<threedgut::FThetaProjectionParameters::PolynomialDegree>), "[3dgut] typing size mismatch");
    static_assert(sizeof(angle_to_pixeldist_poly) == sizeof(tcnn::vec<threedgut::FThetaProjectionParameters::PolynomialDegree>), "[3dgut] typing size mismatch");
    params.fthetaParams.pixeldistToAnglePoly = *reinterpret_cast<const tcnn::vec<threedgut::FThetaProjectionParameters::PolynomialDegree>*>(pixeldist_to_angle_poly.data());
    params.fthetaParams.angleToPixeldistPoly = *reinterpret_cast<const tcnn::vec<threedgut::FThetaProjectionParameters::PolynomialDegree>*>(angle_to_pixeldist_poly.data());
    params.fthetaParams.maxAngle             = max_angle;
    static_assert(sizeof(linear_cde) == sizeof(tcnn::vec<3>), "[3dgut] typing size mismatch");
    params.fthetaParams.linear_cde = *reinterpret_cast<const tcnn::vec<3>*>(linear_cde.data());
    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    pybind11::class_<SplatRaster>(m, "SplatRaster")
        .def(pybind11::init<const nlohmann::json&>())
        .def("trace", &SplatRaster::trace)
        .def("trace_bwd", &SplatRaster::traceBwd)
        .def("collect_times", &SplatRaster::collectTimes);

    py::enum_<threedgut::CameraModelParameters::ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", threedgut::CameraModelParameters::ShutterType::RollingTopToBottomShutter)
        .value("ROLLING_LEFT_TO_RIGHT", threedgut::CameraModelParameters::ShutterType::RollingLeftToRightShutter)
        .value("ROLLING_BOTTOM_TO_TOP", threedgut::CameraModelParameters::ShutterType::RollingBottomToTopShutter)
        .value("ROLLING_RIGHT_TO_LEFT", threedgut::CameraModelParameters::ShutterType::RollingRightToLeftShutter)
        .value("GLOBAL", threedgut::CameraModelParameters::ShutterType::GlobalShutter);

    py::enum_<threedgut::FThetaProjectionParameters::PolynomialType>(m, "PolynomialType")
        .value("PIXELDIST_TO_ANGLE", threedgut::FThetaProjectionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value("ANGLE_TO_PIXELDIST", threedgut::FThetaProjectionParameters::PolynomialType::ANGLE_TO_PIXELDIST);

    py::class_<threedgut::CameraModelParameters>(m, "CameraModelParameters")
        .def(py::init<>());

    m.def("fromOpenCVPinholeCameraModelParameters", &fromOpenCVPinholeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("tangential_coeffs"),
          py::arg("thin_prism_coeffs"));

    m.def("fromOpenCVFisheyeCameraModelParameters", &fromOpenCVFisheyeCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("focal_length"),
          py::arg("radial_coeffs"),
          py::arg("max_angle"));

    m.def("fromFThetaCameraModelParameters", &fromFThetaCameraModelParameters,
          py::arg("resolution"),
          py::arg("shutter_type"),
          py::arg("principal_point"),
          py::arg("reference_poly"),
          py::arg("pixeldist_to_angle_poly"),
          py::arg("angle_to_pixeldist_poly"),
          py::arg("max_angle"),
          py::arg("linear_cde"));
}
