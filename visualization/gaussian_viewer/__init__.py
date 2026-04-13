"""Python bindings for GaussianRendererCore (ANARI/VisRTX).

Provides a headless 3D Gaussian Splat renderer with CUDA framebuffer output
that can be fed directly into polyscope via device-to-device copy.

Quick start::

    import gaussian_viewer as viewer

    renderer = viewer.GaussianRendererCore()
    opts = viewer.InitOptions()
    opts.ply_path = "scene.ply"
    opts.use_float32_color = True  # default; matches polyscope's internal format
    renderer.init(opts)
    renderer.run()

    with renderer.map_color_cuda() as frame:
        # frame exposes __cuda_array_interface__
        ...
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _pkg_dir = Path(__file__).resolve().parent
    _dll_dirs = (_pkg_dir, _pkg_dir / "lib", _pkg_dir / "lib64", _pkg_dir / "bin")
    # Keep handles alive for process lifetime so loaded directories remain active.
    _dll_handles = []
    for _dll_dir in _dll_dirs:
        if _dll_dir.is_dir():
            _dll_handles.append(os.add_dll_directory(str(_dll_dir)))

from ._gaussian_renderer_core import (  # type: ignore[import-untyped]
    CameraState,
    CUDAFrameContext,
    GaussianRendererCore,
    InitOptions,
    MappedCUDAFrame,
    RendererConfig,
)

__all__ = [
    "CameraState",
    "CUDAFrameContext",
    "GaussianRendererCore",
    "InitOptions",
    "MappedCUDAFrame",
    "RendererConfig",
]
