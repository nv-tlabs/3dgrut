#!/usr/bin/env python3
"""Example: render a 3DGS .ply with ANARI/VisRTX and display in polyscope.

Usage::

    example-polyscope /path/to/scene.ply [--scale-factor 1.0]

The ANARI renderer produces a float32 RGBA CUDA framebuffer which is copied
device-to-device into polyscope's OpenGL texture every frame — no CPU
roundtrip.

Prerequisites:
  - ``pip install gaussian-viewer[polyscope]``
  - Install the CUDA-GL interop backend:
      * ``pip install cuda-python cupy``, OR
      * Use the custom bindings from ``threedgrut.gui.ps_extension``.
"""

import argparse

import numpy as np
import polyscope as ps

import gaussian_viewer as viewer


def blit_to_polyscope_buffer(renderer: viewer.GaussianRendererCore, ps_buffer) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ANARI Gaussian viewer + polyscope")
    parser.add_argument("ply_path", help="Path to a 3DGS .ply file")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=1)
    args = parser.parse_args()

    # ── Renderer setup ────────────────────────────────────────────────────

    renderer = viewer.GaussianRendererCore()

    opts = viewer.InitOptions()
    opts.ply_path = args.ply_path
    opts.scale_factor = args.scale_factor
    opts.frame_size = (args.width, args.height)
    opts.use_float32_color = True
    opts.renderer_config.spp = args.spp

    renderer.init(opts)
    print(
        f"Scene: center={renderer.scene_center}, "
        f"diagonal={renderer.scene_diagonal:.3f}, "
        f"{renderer.gaussian_count} Gaussians"
    )

    # ── Polyscope setup ──────────────────────────────────────────────────

    try:
        import cuda  # noqa: F401
        import cupy  # noqa: F401
    except ImportError:
        from threedgrut.gui.ps_extension import initialize_cugl_interop

        initialize_cugl_interop()

    ps.set_use_prefs_file(False)
    ps.set_up_dir("neg_y_up")
    ps.set_front_dir("neg_z_front")
    ps.set_navigation_style("free")
    ps.set_enable_vsync(False)
    ps.set_max_fps(-1)
    ps.set_background_color((0.0, 0.0, 0.0))
    ps.set_ground_plane_mode("none")
    ps.set_window_size(args.width, args.height)
    ps.init()

    dummy = np.ones((args.height, args.width, 4), dtype=np.float32)
    ps.add_color_alpha_image_quantity(
        "render",
        dummy,
        enabled=True,
        image_origin="upper_left",
        show_fullscreen=True,
        show_in_imgui_window=False,
    )
    color_buf = ps.get_quantity_buffer("render", "colors")

    # ── Sync polyscope camera → ANARI and render each frame ────────────

    prev_size = (args.width, args.height)

    def callback() -> None:
        nonlocal color_buf, prev_size

        # Handle window resize
        w, h = ps.get_window_size()
        if (w, h) != prev_size and w > 0 and h > 0:
            prev_size = (w, h)
            renderer.set_frame_size((w, h))
            dummy_img = np.ones((h, w, 4), dtype=np.float32)
            ps.add_color_alpha_image_quantity(
                "render",
                dummy_img,
                enabled=True,
                image_origin="upper_left",
                show_fullscreen=True,
                show_in_imgui_window=False,
            )
            color_buf = ps.get_quantity_buffer("render", "colors")

        # Extract polyscope's interactive camera.
        # Negate the up vector to convert from polyscope's OpenGL convention
        # (camera Y-up) to ANARI's Y-down convention.
        view = ps.get_view_camera_parameters()
        ps_up = view.get_up_dir()
        cam = viewer.CameraState()
        cam.eye = tuple(view.get_position())
        cam.dir = tuple(view.get_look_dir())
        cam.up = (-ps_up[0], -ps_up[1], -ps_up[2])
        cam.aspect = w / max(h, 1)
        renderer.set_camera(cam)

        renderer.run()
        blit_to_polyscope_buffer(renderer, color_buf)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
