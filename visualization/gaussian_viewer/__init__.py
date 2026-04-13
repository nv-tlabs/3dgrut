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
    _dll_dirs: list[Path] = [_pkg_dir, _pkg_dir / "lib", _pkg_dir / "lib64", _pkg_dir / "bin"]
    # Python 3.8+ on Windows ignores PATH when resolving DLLs for extension
    # modules.  Bridge PATH entries into add_dll_directory so that the user
    # can control DLL search via PATH as expected (needed for standalone builds
    # where ANARI DLLs live in install/bin, outside the package directory).
    for _p in os.environ.get("PATH", "").split(os.pathsep):
        if _p:
            _dll_dirs.append(Path(_p))
    _dll_handles = []
    for _dll_dir in _dll_dirs:
        if _dll_dir.is_dir():
            try:
                _dll_handles.append(os.add_dll_directory(str(_dll_dir)))
            except OSError:
                pass

try:
    from ._gaussian_renderer_core import (  # type: ignore[import-untyped]
        CameraState,
        CUDAFrameContext,
        GaussianRendererCore,
        InitOptions,
        MappedCUDAFrame,
        RendererConfig,
    )
except ImportError as e1:
    try:
        from _gaussian_renderer_core import (  # type: ignore[import-untyped]
            CameraState,
            CUDAFrameContext,
            GaussianRendererCore,
            InitOptions,
            MappedCUDAFrame,
            RendererConfig,
        )
    except ImportError as e2:
        _pkg_dir = Path(__file__).resolve().parent
        _install = _pkg_dir.parent / "install"
        _sep = ";" if sys.platform == "win32" else ":"
        _ib, _il, _pp = _install / "bin", _install / "lib", _pkg_dir.parent
        _win = sys.platform == "win32"

        def _env_hint(var: str, val: str) -> str:
            ref = f"$env:{var}" if _win else f"${var}"
            return f'  $env:{var} = "{val}{_sep}{ref}"' if _win else f'  export {var}="{val}{_sep}{ref}"'

        _hint = "\n".join(
            [
                _env_hint("PATH", f"{_ib}{_sep}{_il}"),
                _env_hint("PYTHONPATH", f"{_pp}{_sep}{_il}"),
            ]
        )
        raise ImportError(
            "Cannot find _gaussian_renderer_core native module.\n"
            f"  Package-relative import failed: {e1}\n"
            f"  System-level import failed:     {e2}\n"
            "If you are using a standalone build, you may need to set:\n" + _hint
        ) from e2

__all__ = [
    "CameraState",
    "CUDAFrameContext",
    "GaussianRendererCore",
    "InitOptions",
    "MappedCUDAFrame",
    "RendererConfig",
]
