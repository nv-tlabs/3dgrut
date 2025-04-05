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

#ifdef USE_CUGL_INTEROP
#include <gl_interop.h>
using namespace pybind11::literals;
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

#ifdef USE_CUGL_INTEROP
    py::module_ cuda = m.def_submodule("cuda");
    cuda.def("register_gl_buffer", &cugl_register_gl_buffer, "gl_buffer"_a)
        .def("register_gl_texture", &cugl_register_gl_texture, "gl_texture"_a)
        .def("unregister_cuda_resource", &cugl_unregister_cuda_resource, "cuda_resource"_a)
        .def(
            "map_graphics_resource_ptr",
            [](void *cuda_resource) {
                size_t n_bytes;
                void *ptr = cugl_map_graphics_resource_ptr(cuda_resource, &n_bytes);
                return std::make_pair((std::uintptr_t) ptr, n_bytes);
            },
            "cuda_resource"_a)
        .def(
            "map_graphics_resource_array",
            [](void *cuda_resource, uint32_t array_index, uint32_t mip_level) {
                return (std::uintptr_t) cugl_map_graphics_resource_array(
                    cuda_resource, array_index, mip_level);
            },
            "cuda_resource"_a,
            "array_index"_a = 0,
            "mip_level"_a = 0)
        .def("unmap_graphics_resource", &cugl_unmap_graphics_resource, "cuda_resource"_a)
        .def(
            "memcpy_2d",
            [](std::uintptr_t dst, size_t dst_pitch, std::uintptr_t src, size_t src_pitch,
               size_t width, size_t height) {
                cugl_memcpy_2d(reinterpret_cast<void *>(dst), dst_pitch,
                              reinterpret_cast<void *>(src), src_pitch,
                              width, height);
            },
            "dst"_a, "dst_pitch"_a, "src"_a, "src_pitch"_a, "width"_a, "height"_a)
        .def(
            "memcpy_2d_to_array",
            [](std::uintptr_t dst, size_t w_offset, size_t h_offset, std::uintptr_t src, size_t src_pitch,
               size_t width, size_t height) {
                cugl_memcpy_2d_to_array(reinterpret_cast<void *>(dst), w_offset, h_offset,
                                       reinterpret_cast<void *>(src), src_pitch,
                                       width, height);
            },
            "dst"_a, "w_offset"_a, "h_offset"_a, "src"_a, "src_pitch"_a, "width"_a, "height"_a)
        .def(
            "memcpy_2d_to_array_async",
            [](std::uintptr_t dst, size_t w_offset, size_t h_offset, std::uintptr_t src, size_t src_pitch,
               size_t width, size_t height) {
                cugl_memcpy_2d_to_array_async(reinterpret_cast<void *>(dst), w_offset, h_offset,
                                             reinterpret_cast<void *>(src), src_pitch,
                                             width, height);
            },
            "dst"_a, "w_offset"_a, "h_offset"_a, "src"_a, "src_pitch"_a, "width"_a, "height"_a)
        ;
#endif
}
