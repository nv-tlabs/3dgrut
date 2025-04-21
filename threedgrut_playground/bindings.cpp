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
#include <playground/hybridTracer.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<HybridOptixTracer>(m, "HybridOptixTracer")
        .def(pybind11::init<const std::string&, const std::string&, const std::string&, const std::string&, const std::string&, const std::string&, float, float, bool, int, bool, bool>())
        .def("trace", &OptixTracer::trace)
        .def("build_bvh", &OptixTracer::buildBVH)
        .def("trace_hybrid", &HybridOptixTracer::traceHybrid)
        .def("build_mesh_bvh", &HybridOptixTracer::buildMeshBVH)
        .def("denoise", &HybridOptixTracer::denoise);

    py::class_<CPBRMaterial>(m, "CPBRMaterial")
        .def(py::init<>())
        .def_readwrite("material_id", &CPBRMaterial::material_id)
        .def_readwrite("diffuseMap", &CPBRMaterial::diffuseMap)
        .def_readwrite("emissiveMap", &CPBRMaterial::emissiveMap)
        .def_readwrite("metallicRoughnessMap", &CPBRMaterial::metallicRoughnessMap)
        .def_readwrite("normalMap", &CPBRMaterial::normalMap)
        .def_readwrite("diffuseFactor", &CPBRMaterial::diffuseFactor)
        .def_readwrite("emissiveFactor", &CPBRMaterial::emissiveFactor)
        .def_readwrite("metallicFactor", &CPBRMaterial::metallicFactor)
        .def_readwrite("roughnessFactor", &CPBRMaterial::roughnessFactor)
        .def_readwrite("alphaMode", &CPBRMaterial::alphaMode)
        .def_readwrite("alphaCutoff", &CPBRMaterial::alphaCutoff)
        .def_readwrite("transmissionFactor", &CPBRMaterial::transmissionFactor)
        .def_readwrite("ior", &CPBRMaterial::ior);
}
