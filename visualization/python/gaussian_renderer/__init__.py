"""Python bindings for GaussianRendererCore (ANARI/VisRTX).

Provides a headless 3D Gaussian Splat renderer with CUDA framebuffer output
that can be fed directly into polyscope via device-to-device copy.

Quick start::

    import gaussian_renderer as gr

    renderer = gr.GaussianRendererCore()
    opts = gr.InitOptions()
    opts.ply_path = "scene.ply"
    opts.use_float32_color = True  # default; matches polyscope's internal format
    renderer.init(opts)
    renderer.run()

    with renderer.map_color_cuda() as frame:
        # frame exposes __cuda_array_interface__
        ...
"""

from _gaussian_renderer_core import (  # type: ignore[import-untyped]
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
