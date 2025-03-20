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

#include <cuda.h>
#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include <3dgrt/cuoptixMacros.h>

using CUDAGraphicsResource = struct cudaGraphicsResource *;

void *cugl_register_gl_buffer(GLuint gl_buffer) {
    CUDAGraphicsResource cuda_resource;
    CUDA_CHECK(
        cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_buffer, cudaGraphicsMapFlagsWriteDiscard));

    return (void *) cuda_resource;
}

void *cugl_register_gl_texture(GLuint gl_texture) {
    CUDAGraphicsResource cuda_resource;
    CUDA_CHECK(
        cudaGraphicsGLRegisterImage(&cuda_resource, gl_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

    return (void *) cuda_resource;
}

void cugl_unregister_cuda_resource(void *cuda_resource) {
    CUDA_CHECK(cudaGraphicsUnregisterResource((CUDAGraphicsResource) cuda_resource));
}

void *cugl_map_graphics_resource_ptr(void *cuda_resource, size_t *n_bytes) {
    void *ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, (CUDAGraphicsResource *) &cuda_resource, 0));
    CUDA_CHECK(
        cudaGraphicsResourceGetMappedPointer(&ptr, n_bytes, (CUDAGraphicsResource) cuda_resource));

    return ptr;
}

void *cugl_map_graphics_resource_array(void *cuda_resource,
                                      uint32_t array_index = 0,
                                      uint32_t mip_level = 0) {
    void *ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, (CUDAGraphicsResource *) &cuda_resource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
        (cudaArray_t *) &ptr, (CUDAGraphicsResource) cuda_resource, array_index, mip_level));
    return ptr;
}

void cugl_unmap_graphics_resource(void *cuda_resource) {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, (CUDAGraphicsResource *) &cuda_resource, 0));
}


void cugl_memcpy_2d(void *dst, size_t dst_pitch, void *src, size_t src_pitch,
                   size_t width, size_t height) {
    CUDA_CHECK(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height,
                            cudaMemcpyDeviceToDevice));
}

void cugl_memcpy_2d_to_array(void *dst, size_t w_offset, size_t h_offset, void *src, size_t src_pitch,
                            size_t width, size_t height) {
    CUDA_CHECK(cudaMemcpy2DToArray(
        (cudaArray_t) dst, w_offset, h_offset, src, src_pitch, width, height,
        cudaMemcpyDeviceToDevice));
}

void cugl_memcpy_2d_to_array_async(void *dst, size_t w_offset, size_t h_offset, void *src, size_t src_pitch,
                            size_t width, size_t height) {
    CUDA_CHECK(cudaMemcpy2DToArrayAsync(
        (cudaArray_t) dst, w_offset, h_offset, src, src_pitch, width, height,
        cudaMemcpyDeviceToDevice));
}
