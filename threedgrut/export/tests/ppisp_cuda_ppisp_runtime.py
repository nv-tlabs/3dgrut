# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime harness for exported PPISP CUDA image sidecars.

This is test-only support. It compiles the exported static and automatic
PPISP CUDA sidecars as real CUDA translation units, then launches their
entry points through a tiny PyTorch extension wrapper. The wrapper owns
only tensor/texture/surface setup; the PPISP math comes from the exported
``.cu`` files.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import textwrap

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch


def add_ninja_to_path() -> None:
    """Compatibility shim for NRE's Bazel helper; uv/venv PATH is enough here."""
    return None

from pxr import Usd

from threedgrut.export.usd.ppisp_spg import _SPG_DIR as CONTROLLER_SPG_DIR
from threedgrut.export.usd.ppisp_spg import _SPG_DIR as STATIC_SPG_DIR


_STATIC_CU = STATIC_SPG_DIR / "ppisp_usd_spg.cu"
_AUTO_CU = CONTROLLER_SPG_DIR / "ppisp_usd_spg_auto.cu"


_PARAM_FLOATS = (
    "responsivity",
    "exposureOffset",
    "vignettingCenterR.x",
    "vignettingCenterR.y",
    "vignettingAlpha1R",
    "vignettingAlpha2R",
    "vignettingAlpha3R",
    "vignettingCenterG.x",
    "vignettingCenterG.y",
    "vignettingAlpha1G",
    "vignettingAlpha2G",
    "vignettingAlpha3G",
    "vignettingCenterB.x",
    "vignettingCenterB.y",
    "vignettingAlpha1B",
    "vignettingAlpha2B",
    "vignettingAlpha3B",
    "colorLatentBlue.x",
    "colorLatentBlue.y",
    "colorLatentRed.x",
    "colorLatentRed.y",
    "colorLatentGreen.x",
    "colorLatentGreen.y",
    "colorLatentNeutral.x",
    "colorLatentNeutral.y",
    "crfToeR",
    "crfShoulderR",
    "crfGammaR",
    "crfCenterR",
    "crfToeG",
    "crfShoulderG",
    "crfGammaG",
    "crfCenterG",
    "crfToeB",
    "crfShoulderB",
    "crfGammaB",
    "crfCenterB",
)

_AUTO_PARAM_INDICES = (
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
)


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for candidate in ("/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.8", "/usr/local/cuda-12.6"):
        if (Path(candidate) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = candidate
            return


def _extension_source(sidecar: Path, run_source: str, binding_name: str, binding_doc: str) -> str:
    source = r"""
        #include <ATen/cuda/CUDAContext.h>
        #include <c10/cuda/CUDAGuard.h>
        #include <cuda_runtime.h>
        #include <torch/extension.h>

        #include <sstream>
        #include <stdexcept>

        #include "__PPISP_SIDECAR_CU__"

        static void checkCuda(cudaError_t err, const char* what)
        {
            if (err != cudaSuccess)
            {
                std::ostringstream oss;
                oss << what << ": " << cudaGetErrorString(err);
                throw std::runtime_error(oss.str());
            }
        }

        struct TextureSurfacePair
        {
            cudaArray_t inputArray = nullptr;
            cudaArray_t outputArray = nullptr;
            cudaTextureObject_t inputTexture = 0;
            cudaSurfaceObject_t outputSurface = 0;
        };

        static TextureSurfacePair createTextureSurface(torch::Tensor hdr, int width, int height, cudaStream_t stream)
        {
            TextureSurfacePair pair;

            cudaChannelFormatDesc inputDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            checkCuda(cudaMallocArray(
                          &pair.inputArray,
                          &inputDesc,
                          static_cast<size_t>(width),
                          static_cast<size_t>(height)),
                      "cudaMallocArray input");

            checkCuda(cudaMemcpy2DToArrayAsync(
                          pair.inputArray,
                          0,
                          0,
                          hdr.data_ptr<float>(),
                          static_cast<size_t>(hdr.stride(0) * sizeof(float)),
                          static_cast<size_t>(width * 4 * sizeof(float)),
                          static_cast<size_t>(height),
                          cudaMemcpyDeviceToDevice,
                          stream),
                      "cudaMemcpy2DToArrayAsync input");

            cudaResourceDesc inputResource{};
            inputResource.resType = cudaResourceTypeArray;
            inputResource.res.array.array = pair.inputArray;

            cudaTextureDesc textureDesc{};
            textureDesc.addressMode[0] = cudaAddressModeClamp;
            textureDesc.addressMode[1] = cudaAddressModeClamp;
            textureDesc.filterMode = cudaFilterModePoint;
            textureDesc.readMode = cudaReadModeElementType;
            textureDesc.normalizedCoords = 0;

            checkCuda(cudaCreateTextureObject(&pair.inputTexture, &inputResource, &textureDesc, nullptr),
                      "cudaCreateTextureObject input");

            cudaChannelFormatDesc outputDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            checkCuda(cudaMallocArray(
                          &pair.outputArray,
                          &outputDesc,
                          static_cast<size_t>(width),
                          static_cast<size_t>(height),
                          cudaArraySurfaceLoadStore),
                      "cudaMallocArray output");

            cudaResourceDesc outputResource{};
            outputResource.resType = cudaResourceTypeArray;
            outputResource.res.array.array = pair.outputArray;
            checkCuda(cudaCreateSurfaceObject(&pair.outputSurface, &outputResource), "cudaCreateSurfaceObject output");

            return pair;
        }

        static void destroyTextureSurface(TextureSurfacePair& pair)
        {
            if (pair.outputSurface)
            {
                checkCuda(cudaDestroySurfaceObject(pair.outputSurface), "cudaDestroySurfaceObject output");
            }
            if (pair.inputTexture)
            {
                checkCuda(cudaDestroyTextureObject(pair.inputTexture), "cudaDestroyTextureObject input");
            }
            if (pair.outputArray)
            {
                checkCuda(cudaFreeArray(pair.outputArray), "cudaFreeArray output");
            }
            if (pair.inputArray)
            {
                checkCuda(cudaFreeArray(pair.inputArray), "cudaFreeArray input");
            }
        }

        static torch::Tensor copyOutput(TextureSurfacePair& pair, torch::Tensor hdr, int width, int height, cudaStream_t stream)
        {
            auto output = torch::empty(
                {height, width, 4},
                torch::TensorOptions().device(hdr.device()).dtype(torch::kUInt8));
            checkCuda(cudaMemcpy2DFromArrayAsync(
                          output.data_ptr<unsigned char>(),
                          static_cast<size_t>(output.stride(0) * sizeof(unsigned char)),
                          pair.outputArray,
                          0,
                          0,
                          static_cast<size_t>(width * 4 * sizeof(unsigned char)),
                          static_cast<size_t>(height),
                          cudaMemcpyDeviceToDevice,
                          stream),
                      "cudaMemcpy2DFromArrayAsync output");
            return output;
        }

        static void validateHdr(torch::Tensor hdr)
        {
            TORCH_CHECK(hdr.is_cuda(), "hdr must be a CUDA tensor");
            TORCH_CHECK(hdr.scalar_type() == torch::kFloat32, "hdr must be float32");
            TORCH_CHECK(hdr.dim() == 3, "hdr must have shape [H, W, 4]");
            TORCH_CHECK(hdr.size(2) == 4, "hdr must have four float channels");
            TORCH_CHECK(hdr.is_contiguous(), "hdr must be contiguous");
        }
        __RUN_SOURCE__

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
        {
            m.def("__BINDING_NAME__", &__BINDING_NAME__, "__BINDING_DOC__");
        }
    """
    return (
        textwrap.dedent(source)
        .replace("__PPISP_SIDECAR_CU__", str(sidecar.resolve()))
        .replace("__RUN_SOURCE__", textwrap.indent(textwrap.dedent(run_source).strip(), "        "))
        .replace("__BINDING_NAME__", binding_name)
        .replace("__BINDING_DOC__", binding_doc)
    )


def _static_run_source() -> str:
    return r"""
        torch::Tensor run_static_ppisp(torch::Tensor hdr, torch::Tensor params)
        {
            validateHdr(hdr);
            TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
            TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
            TORCH_CHECK(params.dim() == 1 && params.numel() == 37, "params must have shape [37]");
            params = params.contiguous();

            const c10::cuda::CUDAGuard deviceGuard(hdr.device());
            const int height = static_cast<int>(hdr.size(0));
            const int width = static_cast<int>(hdr.size(1));
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            TextureSurfacePair pair = createTextureSurface(hdr, width, height, stream);

            auto p = params.data_ptr<float>();
            ppispProcess<<<dim3((width + 15) / 16, (height + 15) / 16, 1), dim3(16, 16, 1), 0, stream>>>(
                width,
                height,
                pair.inputTexture,
                p,
                pair.outputSurface);
            checkCuda(cudaGetLastError(), "ppispProcess launch");

            torch::Tensor output = copyOutput(pair, hdr, width, height, stream);
            checkCuda(cudaStreamSynchronize(stream), "ppispProcess synchronize");
            destroyTextureSurface(pair);
            return output;
        }
    """


def _auto_run_source() -> str:
    return r"""
        torch::Tensor run_auto_ppisp(torch::Tensor hdr, torch::Tensor controllerParams, torch::Tensor params)
        {
            validateHdr(hdr);
            TORCH_CHECK(controllerParams.is_cuda(), "controllerParams must be a CUDA tensor");
            TORCH_CHECK(controllerParams.scalar_type() == torch::kFloat32, "controllerParams must be float32");
            TORCH_CHECK(controllerParams.numel() == 9, "controllerParams must hold 9 floats");
            TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
            TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
            TORCH_CHECK(params.dim() == 1 && params.numel() == 28, "params must have shape [28]");
            controllerParams = controllerParams.contiguous();
            params = params.contiguous();

            const c10::cuda::CUDAGuard deviceGuard(hdr.device());
            const int height = static_cast<int>(hdr.size(0));
            const int width = static_cast<int>(hdr.size(1));
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            TextureSurfacePair pair = createTextureSurface(hdr, width, height, stream);

            auto p = params.data_ptr<float>();
            ppispProcessAuto<<<dim3((width + 15) / 16, (height + 15) / 16, 1), dim3(16, 16, 1), 0, stream>>>(
                width,
                height,
                pair.inputTexture,
                controllerParams.data_ptr<float>(),
                p,
                pair.outputSurface);
            checkCuda(cudaGetLastError(), "ppispProcessAuto launch");

            torch::Tensor output = copyOutput(pair, hdr, width, height, stream);
            checkCuda(cudaStreamSynchronize(stream), "ppispProcessAuto synchronize");
            destroyTextureSurface(pair);
            return output;
        }
    """


@lru_cache(maxsize=2)
def _load_extension(kind: str):
    _ensure_cuda_home()
    add_ninja_to_path()
    from torch.utils.cpp_extension import load

    if kind == "static":
        sidecar = _STATIC_CU
        binding_name = "run_static_ppisp"
        source = _extension_source(
            sidecar, _static_run_source(), binding_name, "Run exported static PPISP CUDA sidecar"
        )
    elif kind == "auto":
        sidecar = _AUTO_CU
        binding_name = "run_auto_ppisp"
        source = _extension_source(sidecar, _auto_run_source(), binding_name, "Run exported auto PPISP CUDA sidecar")
    else:
        raise ValueError(f"unknown PPISP CUDA image extension kind: {kind}")

    digest = hashlib.sha256((sidecar.read_text(encoding="utf-8") + source).encode("utf-8")).hexdigest()[:12]
    build_dir = Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir())) / f"ppisp_cuda_image_{kind}_{digest}"
    build_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = build_dir / f"ppisp_cuda_image_{kind}_runtime_wrapper.cu"
    wrapper_path.write_text(source, encoding="utf-8")

    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name=f"ppisp_cuda_image_{kind}_{digest}",
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
    alpha = torch.zeros((*hdr.shape[:2], 1), device=hdr.device, dtype=torch.float32)
    return torch.cat((hdr.to(dtype=torch.float32), alpha), dim=-1).contiguous()


def _read_float(prim: Usd.Prim, name: str) -> float:
    attr = prim.GetAttribute(f"inputs:{name}")
    if not attr.IsValid():
        raise ValueError(f"missing inputs:{name}")
    return float(attr.Get())


def _read_float2_tuple(prim: Usd.Prim, name: str) -> Tuple[float, float]:
    attr = prim.GetAttribute(f"inputs:{name}")
    if not attr.IsValid():
        raise ValueError(f"missing inputs:{name}")
    value = attr.Get()
    return float(value[0]), float(value[1])


def pack_static_ppisp_params(
    prim: Usd.Prim, *, exposure: float | None = None, color: torch.Tensor | None = None
) -> torch.Tensor:
    values = []
    for name in _PARAM_FLOATS:
        if name == "exposureOffset" and exposure is not None:
            values.append(float(exposure))
        elif name.startswith("colorLatent") and color is not None:
            color_map = {
                "colorLatentBlue.x": 0,
                "colorLatentBlue.y": 1,
                "colorLatentRed.x": 2,
                "colorLatentRed.y": 3,
                "colorLatentGreen.x": 4,
                "colorLatentGreen.y": 5,
                "colorLatentNeutral.x": 6,
                "colorLatentNeutral.y": 7,
            }
            values.append(float(color.reshape(-1)[color_map[name]].item()))
        elif name.endswith(".x"):
            values.append(_read_float2_tuple(prim, name[:-2])[0])
        elif name.endswith(".y"):
            values.append(_read_float2_tuple(prim, name[:-2])[1])
        else:
            values.append(_read_float(prim, name))
    return torch.tensor(values, dtype=torch.float32)


def pack_auto_ppisp_params(prim: Usd.Prim) -> torch.Tensor:
    static_params = pack_static_ppisp_params(prim, exposure=0.0, color=torch.zeros(8))
    return static_params[list(_AUTO_PARAM_INDICES)].contiguous()


class ExportedCudaPPISP:
    """Callable wrappers around exported static and auto PPISP CUDA sidecars."""

    def __init__(self, *, device: torch.device) -> None:
        self._device = device

    def run_static(self, hdr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        hdr4 = _hdr_rgb_to_float4(hdr)
        output_u8 = _load_extension("static").run_static_ppisp(
            hdr4, params.to(device=self._device, dtype=torch.float32)
        )
        return output_u8[..., :3].to(dtype=torch.float32) / 255.0

    def run_auto(self, hdr: torch.Tensor, controller_params: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        hdr4 = _hdr_rgb_to_float4(hdr)
        output_u8 = _load_extension("auto").run_auto_ppisp(
            hdr4,
            controller_params.reshape(-1).to(device=self._device, dtype=torch.float32),
            params.to(device=self._device, dtype=torch.float32),
        )
        return output_u8[..., :3].to(dtype=torch.float32) / 255.0
