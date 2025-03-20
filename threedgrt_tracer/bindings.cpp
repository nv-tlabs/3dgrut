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

#include <3dgrt/optixTracer.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<OptixTracer>(m, "OptixTracer")
        .def(pybind11::init<const std::string&, const std::string&, const std::string&, const std::string&, const std::string&, float, float, bool, int, bool, bool>())
        .def("trace", &OptixTracer::trace)
        .def("trace_bwd", &OptixTracer::traceBwd)
        .def("build_bvh", &OptixTracer::buildBVH);
}
