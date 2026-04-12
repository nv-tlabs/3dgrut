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
    "blit_to_polyscope_buffer",
]


def blit_to_polyscope_buffer(renderer: GaussianRendererCore, ps_buffer) -> None:
    """Map the renderer's CUDA framebuffer and copy it into a polyscope managed buffer.

    This is a convenience wrapper around the context-manager API that performs a
    single device-to-device transfer with automatic map/unmap.  The renderer must
    have been initialised with ``use_float32_color=True`` (the default) so that
    the pixel format matches polyscope's ``add_color_alpha_image_quantity``
    internal layout (float32 RGBA).

    Args:
        renderer: An initialised ``GaussianRendererCore`` after ``run()`` has
            been called.
        ps_buffer: A polyscope ``ManagedBuffer`` obtained via
            ``ps.get_quantity_buffer(name, "colors")``.
    """
    with renderer.map_color_cuda() as frame:
        ps_buffer.update_data_from_device(frame)
