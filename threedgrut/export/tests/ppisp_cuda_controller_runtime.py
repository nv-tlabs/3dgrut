# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime harness for exported embedded-weight PPISP controller CUDA sidecars.

This module is test-only support. It compiles a tiny PyTorch extension
that includes exported controller CUDA source and launches the exact
``controllerPoolProcess`` and ``controllerProcess`` kernels against CUDA
tensors.

The goal is to keep controller regression and performance tests tied to
the exported artifact instead of to a hand-written Python simulation of
the same math.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import textwrap

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch


def add_ninja_to_path() -> None:
    """Compatibility shim for NRE's Bazel helper; uv/venv PATH is enough here."""
    return None


@dataclass(frozen=True)
class ControllerLaunchDims:
    block: Tuple[int, int, int]
    grid: Tuple[int, int, int]


@dataclass(frozen=True)
class ControllerLaunchPlan:
    pool: ControllerLaunchDims
    mlp: ControllerLaunchDims


def default_controller_launch_dims() -> ControllerLaunchPlan:
    return ControllerLaunchPlan(
        pool=ControllerLaunchDims(block=(256, 1, 1), grid=(25, 1, 1)),
        mlp=ControllerLaunchDims(block=(128, 1, 1), grid=(1, 1, 1)),
    )


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for candidate in ("/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.8", "/usr/local/cuda-12.6"):
        if (Path(candidate) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = candidate
            return


def _extension_source(controller_cu_path: Path) -> str:
    source = r"""
        #include <ATen/cuda/CUDAContext.h>
        #include <c10/cuda/CUDAGuard.h>
        #include <cuda_runtime.h>
        #include <torch/extension.h>

        #include <sstream>
        #include <stdexcept>

        #include "__CONTROLLER_CU__"

        static void checkCuda(cudaError_t err, const char* what)
        {
            if (err != cudaSuccess)
            {
                std::ostringstream oss;
                oss << what << ": " << cudaGetErrorString(err);
                throw std::runtime_error(oss.str());
            }
        }

        torch::Tensor run_controller(
            torch::Tensor hdr,
            double priorExposure,
            int poolBlockX,
            int poolBlockY,
            int poolBlockZ,
            int poolGridX,
            int poolGridY,
            int poolGridZ,
            int mlpBlockX,
            int mlpBlockY,
            int mlpBlockZ,
            int mlpGridX,
            int mlpGridY,
            int mlpGridZ)
        {
            TORCH_CHECK(hdr.is_cuda(), "hdr must be a CUDA tensor");
            TORCH_CHECK(hdr.scalar_type() == torch::kFloat32, "hdr must be float32");
            TORCH_CHECK(hdr.dim() == 3, "hdr must have shape [H, W, 4]");
            TORCH_CHECK(hdr.size(2) == 4, "hdr must have four float channels");
            TORCH_CHECK(hdr.is_contiguous(), "hdr must be contiguous");
            TORCH_CHECK(poolBlockX > 0 && poolBlockY > 0 && poolBlockZ > 0, "pool block dimensions must be positive");
            TORCH_CHECK(poolGridX > 0 && poolGridY > 0 && poolGridZ > 0, "pool grid dimensions must be positive");
            TORCH_CHECK(mlpBlockX > 0 && mlpBlockY > 0 && mlpBlockZ > 0, "MLP block dimensions must be positive");
            TORCH_CHECK(mlpGridX > 0 && mlpGridY > 0 && mlpGridZ > 0, "MLP grid dimensions must be positive");

            const c10::cuda::CUDAGuard deviceGuard(hdr.device());
            auto features = torch::empty({1, POOL_FEATURE_LEN}, hdr.options());
            auto output = torch::empty({1, 9}, hdr.options());

            const int height = static_cast<int>(hdr.size(0));
            const int width = static_cast<int>(hdr.size(1));
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            cudaArray_t array = nullptr;
            checkCuda(cudaMallocArray(&array, &channelDesc, static_cast<size_t>(width), static_cast<size_t>(height)),
                      "cudaMallocArray");

            checkCuda(cudaMemcpy2DToArrayAsync(
                          array,
                          0,
                          0,
                          hdr.data_ptr<float>(),
                          static_cast<size_t>(hdr.stride(0) * sizeof(float)),
                          static_cast<size_t>(width * 4 * sizeof(float)),
                          static_cast<size_t>(height),
                          cudaMemcpyDeviceToDevice,
                          stream),
                      "cudaMemcpy2DToArrayAsync");

            cudaResourceDesc resourceDesc{};
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = array;

            cudaTextureDesc textureDesc{};
            textureDesc.addressMode[0] = cudaAddressModeClamp;
            textureDesc.addressMode[1] = cudaAddressModeClamp;
            textureDesc.filterMode = cudaFilterModePoint;
            textureDesc.readMode = cudaReadModeElementType;
            textureDesc.normalizedCoords = 0;

            cudaTextureObject_t texture = 0;
            checkCuda(cudaCreateTextureObject(&texture, &resourceDesc, &textureDesc, nullptr), "cudaCreateTextureObject");

            controllerPoolProcess<<<dim3(poolGridX, poolGridY, poolGridZ), dim3(poolBlockX, poolBlockY, poolBlockZ), 0, stream>>>(
                width,
                height,
                texture,
                features.data_ptr<float>());
            checkCuda(cudaGetLastError(), "controllerPoolProcess launch");

            controllerProcess<<<dim3(mlpGridX, mlpGridY, mlpGridZ), dim3(mlpBlockX, mlpBlockY, mlpBlockZ), 0, stream>>>(
                features.data_ptr<float>(),
                static_cast<float>(priorExposure),
                output.data_ptr<float>());
            checkCuda(cudaGetLastError(), "controllerProcess launch");

            // The texture object is owned by this wrapper call. Synchronize
            // before destroying it so the launched sidecar kernel has
            // finished consuming the object.
            checkCuda(cudaStreamSynchronize(stream), "controllerProcess synchronize");
            checkCuda(cudaDestroyTextureObject(texture), "cudaDestroyTextureObject");
            checkCuda(cudaFreeArray(array), "cudaFreeArray");

            return output;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
        {
            m.def("run_controller", &run_controller, "Run exported PPISP controllerProcess CUDA kernel");
        }
    """
    return textwrap.dedent(source).replace("__CONTROLLER_CU__", str(controller_cu_path.resolve()))


@lru_cache(maxsize=8)
def _load_extension(controller_source: str):
    _ensure_cuda_home()
    add_ninja_to_path()
    from torch.utils.cpp_extension import load

    digest = hashlib.sha256(controller_source.encode("utf-8")).hexdigest()[:12]
    build_dir = Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir())) / f"ppisp_controller_cuda_{digest}"
    build_dir.mkdir(parents=True, exist_ok=True)
    controller_path = build_dir / "ppisp_controller_runtime_source.cu"
    controller_path.write_text(controller_source, encoding="utf-8")
    source = _extension_source(controller_path)
    wrapper_path = build_dir / "ppisp_controller_runtime_wrapper.cu"
    wrapper_path.write_text(source, encoding="utf-8")

    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name=f"ppisp_controller_cuda_{digest}",
        sources=[str(wrapper_path)],
        build_directory=str(build_dir),
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
        with_cuda=True,
    )


def _hdr_rgb_to_float4(hdr: torch.Tensor) -> torch.Tensor:
    if hdr.ndim != 3 or hdr.shape[-1] != 3:
        raise ValueError(f"hdr must have shape [H, W, 3], got {tuple(hdr.shape)}")
    if not hdr.is_cuda:
        raise ValueError("hdr must be a CUDA tensor")
    alpha = torch.zeros((*hdr.shape[:2], 1), device=hdr.device, dtype=hdr.dtype)
    return torch.cat((hdr.to(dtype=torch.float32), alpha), dim=-1).contiguous()


class ExportedEmbeddedCudaController:
    """Callable wrapper around a generated embedded-weight controller CUDA sidecar."""

    def __init__(self, controller_source: str | bytes, *, device: torch.device) -> None:
        if isinstance(controller_source, bytes):
            source_text = controller_source.decode("utf-8")
        else:
            source_text = controller_source
        self._device = device
        self._dims = default_controller_launch_dims()
        self._extension = _load_extension(source_text)

    def __call__(self, hdr: torch.Tensor, prior_exposure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hdr4 = _hdr_rgb_to_float4(hdr)
        prior = float(prior_exposure.reshape(-1)[0].item())
        output = self._extension.run_controller(
            hdr4,
            prior,
            *self._dims.pool.block,
            *self._dims.pool.grid,
            *self._dims.mlp.block,
            *self._dims.mlp.grid,
        )
        return output[0, 0], output[0, 1:9]
