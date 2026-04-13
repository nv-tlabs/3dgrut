#!/usr/bin/env python3
"""Interactive 3DGS .ply viewer in polyscope with runtime renderer switching.

All available backends are initialised at startup and a polyscope ImGui dropdown
lets you switch between them on the fly.

Backends:

- **anari** — ANARI/VisRTX via the ``gaussian_viewer`` C++ module (CUDA->GL blit).
- **3dgrt** / **3dgut** — native OptiX / splat tracers from the 3DGRUT repo.

Usage::

    polyscope-viewer /path/to/scene.ply [--renderer anari|3dgrt|3dgut]

Prerequisites:

  - **anari**: ``pip install gaussian-viewer[polyscope]`` + CUDA-GL interop
    (``pip install cuda-python cupy``, or ``threedgrut.gui.ps_extension``).
  - **3dgrt / 3dgut**: editable install of threedgrut + tracers, CUDA, torch,
    hydra-core, polyscope.  Configs at ``<repo>/configs/`` are loaded via Hydra.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_omegaconf_resolvers() -> None:
    """Register custom resolvers used by repo YAML (normally done in train.py / threedgrut.utils.misc)."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("int_list"):
        OmegaConf.register_new_resolver("int_list", lambda seq: [int(x) for x in seq])
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    if not OmegaConf.has_resolver("eq"):
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b)


def _init_polyscope_cugl() -> None:
    try:
        import cuda  # noqa: F401
        import cupy  # noqa: F401
    except ImportError:
        from threedgrut.gui.ps_extension import initialize_cugl_interop

        initialize_cugl_interop()


def _polyscope_window_setup(width: int, height: int) -> None:
    ps.set_use_prefs_file(False)
    ps.set_up_dir("neg_y_up")
    ps.set_front_dir("neg_z_front")
    ps.set_navigation_style("free")
    ps.set_enable_vsync(False)
    ps.set_max_fps(-1)
    ps.set_background_color((0.0, 0.0, 0.0))
    ps.set_ground_plane_mode("none")
    ps.set_window_size(width, height)


def _load_grut_config(method: Literal["3dgrt", "3dgut"], config_name: str | None):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    _ensure_omegaconf_resolvers()

    cfg_dir = _repo_root() / "configs"
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"Hydra configs not found at {cfg_dir} (is this the 3DGRUT repo root?)")

    name = config_name
    if not name:
        name = "apps/colmap_3dgrt" if method == "3dgrt" else "apps/colmap_3dgut"

    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base=None):
        conf = compose(config_name=name)

    with open_dict(conf):
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

    return conf


def _fov2focal(fov_radians: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov_radians / 2))


def _grut_batch_from_polyscope(window_w: int, window_h: int, device):
    """Build a ``Batch`` from the current polyscope view (same convention as ``trainer_ps_gui``)."""
    import torch

    from threedgrut.datasets.protocols import Batch

    view_params = ps.get_view_camera_parameters()
    fov_deg = view_params.get_fov_vertical_deg()
    focal = _fov2focal(math.radians(fov_deg), window_h)

    interp_x, interp_y = torch.meshgrid(
        torch.linspace(0.0, window_w - 1, window_w, device=device, dtype=torch.float32),
        torch.linspace(0.0, window_h - 1, window_h, device=device, dtype=torch.float32),
        indexing="xy",
    )
    u, v = interp_x, interp_y
    xs = ((u + 0.5) - 0.5 * window_w) / focal
    ys = ((v + 0.5) - 0.5 * window_h) / focal
    rays_dir = torch.nn.functional.normalize(torch.stack((xs, ys, torch.ones_like(xs)), dim=-1), dim=-1).unsqueeze(0)

    w2c = view_params.get_view_mat()
    c2w = np.linalg.inv(w2c)
    c2w[:, 1:3] *= -1.0  # [right up back] -> [right down front]

    return Batch(
        intrinsics=[focal, focal, window_w / 2, window_h / 2],
        T_to_world=torch.tensor(c2w, dtype=torch.float32, device=device).unsqueeze(0),
        rays_ori=torch.zeros((1, window_h, window_w, 3), device=device, dtype=torch.float32),
        rays_dir=rays_dir.reshape(1, window_h, window_w, 3),
    )


def blit_to_polyscope_buffer(renderer, ps_buffer) -> None:
    """Copy ANARI float32 RGBA CUDA framebuffer into a polyscope managed buffer."""
    with renderer.map_color_cuda() as frame:
        ps_buffer.update_data_from_device(frame)


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------


def _probe_backends() -> list[str]:
    """Return list of renderer names whose dependencies are importable."""
    available: list[str] = []
    try:
        import gaussian_viewer  # noqa: F401

        available.append("anari")
    except ImportError:
        pass
    try:
        import threedgrut.model.model  # noqa: F401

        available.append("3dgrt")
        available.append("3dgut")
    except ImportError:
        pass
    return available


# ---------------------------------------------------------------------------
# Unified viewer
# ---------------------------------------------------------------------------


def run_viewer(args: argparse.Namespace, available: list[str]) -> None:
    """Initialise all available backends and run polyscope with a runtime renderer selector."""

    # -- Determine initial renderer ------------------------------------------
    if args.renderer in available:
        active_name = args.renderer
    else:
        active_name = available[0]
        print(f"Warning: --renderer {args.renderer!r} is not available; falling back to {active_name!r}")

    active_idx = available.index(active_name)

    # -- ANARI init ----------------------------------------------------------
    anari_renderer = None
    anari_viewer_mod = None
    if "anari" in available:
        import gaussian_viewer as viewer

        anari_viewer_mod = viewer
        anari_renderer = viewer.GaussianRendererCore()

        opts = viewer.InitOptions()
        opts.ply_path = args.ply_path
        opts.scale_factor = args.scale_factor
        opts.frame_size = (args.width, args.height)
        opts.use_float32_color = True
        opts.renderer_config.spp = args.spp
        anari_renderer.init(opts)
        print(
            f"[anari] {anari_renderer.gaussian_count} Gaussians, "
            f"center={anari_renderer.scene_center}, "
            f"diagonal={anari_renderer.scene_diagonal:.3f}"
        )

    # -- 3DGRT / 3DGUT init (shared Gaussian data, swappable tracer) --------
    grut_model = None
    grut_tracers: dict[str, object] = {}
    cuda_device = None

    has_3dgrt = "3dgrt" in available
    has_3dgut = "3dgut" in available

    if has_3dgrt or has_3dgut:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("3dgrt / 3dgut viewing requires CUDA.")

        cuda_device = torch.device("cuda")

        # Load the primary config to construct the model.
        # We use whichever method the user requested (or 3dgrt by default).
        primary_method: Literal["3dgrt", "3dgut"] = "3dgrt" if has_3dgrt else "3dgut"
        conf_primary = _load_grut_config(primary_method, args.config_name)

        from threedgrut.model.model import MixtureOfGaussians

        grut_model = MixtureOfGaussians(conf_primary)
        grut_model.init_from_ply(args.ply_path, init_model=False)

        # The model constructor already created the primary tracer.
        grut_tracers[primary_method] = grut_model.renderer

        # Create the other tracer if both backends are available.
        other_method: Literal["3dgrt", "3dgut"] | None = None
        if has_3dgrt and has_3dgut:
            other_method = "3dgut" if primary_method == "3dgrt" else "3dgrt"
            conf_other = _load_grut_config(other_method, args.config_name)
            if other_method == "3dgrt":
                from threedgrt_tracer.tracer import Tracer as Tracer3DGRT

                grut_tracers[other_method] = Tracer3DGRT(conf_other)
            else:
                from threedgut_tracer.tracer import Tracer as Tracer3DGUT

                grut_tracers[other_method] = Tracer3DGUT(conf_other)

        # Activate whichever tracer the user requested and build BVH.
        if active_name in grut_tracers:
            grut_model.renderer = grut_tracers[active_name]
        grut_model.build_acc()

        print(f"[3dgrt/3dgut] {grut_model.num_gaussians} Gaussians from {args.ply_path}")

    # -- Polyscope window ----------------------------------------------------
    _init_polyscope_cugl()
    _polyscope_window_setup(args.width, args.height)
    ps.init()

    def _image_origin_for_renderer(renderer_name: str) -> str:
        # ANARI framebuffer is addressed from the lower-left.
        return "lower_left" if renderer_name == "anari" else "upper_left"

    color_buf = None
    current_image_origin = _image_origin_for_renderer(active_name)

    def _recreate_render_quantity(width: int, height: int) -> None:
        nonlocal color_buf, current_image_origin
        current_image_origin = _image_origin_for_renderer(active_name)
        dummy_img = np.ones((height, width, 4), dtype=np.float32)
        ps.add_color_alpha_image_quantity(
            "render",
            dummy_img,
            enabled=True,
            image_origin=current_image_origin,
            show_fullscreen=True,
            show_in_imgui_window=False,
        )
        color_buf = ps.get_quantity_buffer("render", "colors")

    _recreate_render_quantity(args.width, args.height)

    prev_size = (args.width, args.height)

    # -- Frame callback with renderer switching UI ---------------------------
    def callback() -> None:
        nonlocal active_idx, active_name, color_buf, prev_size

        w, h = ps.get_window_size()

        # ── ImGui renderer selector ───────────────────────────────────
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Renderer"):
            changed, active_idx = psim.Combo("Backend", active_idx, available)
            if changed:
                new_name = available[active_idx]
                _switch_renderer(new_name)
                active_name = new_name
                if _image_origin_for_renderer(active_name) != current_image_origin and w > 0 and h > 0:
                    _recreate_render_quantity(w, h)
            psim.TreePop()

        # ── Handle window resize ──────────────────────────────────────
        if (w, h) != prev_size and w > 0 and h > 0:
            prev_size = (w, h)
            if anari_renderer is not None:
                anari_renderer.set_frame_size((w, h))
            _recreate_render_quantity(w, h)

        # ── Dispatch to active renderer ───────────────────────────────
        if active_name == "anari":
            _render_anari(w, h)
        else:
            _render_grut(w, h)

    def _switch_renderer(new_name: str) -> None:
        """Swap tracer on the shared model when switching between 3dgrt/3dgut."""
        if grut_model is not None and new_name in grut_tracers:
            grut_model.renderer = grut_tracers[new_name]
            grut_model.build_acc()

    def _render_anari(w: int, h: int) -> None:
        view = ps.get_view_camera_parameters()
        ps_up = view.get_up_dir()
        cam = anari_viewer_mod.CameraState()
        cam.eye = tuple(view.get_position())
        cam.dir = tuple(view.get_look_dir())
        cam.up = tuple(ps_up)
        cam.aspect = w / max(h, 1)
        cam.fovy = math.radians(view.get_fov_vertical_deg())
        anari_renderer.set_camera(cam)

        anari_renderer.run()
        blit_to_polyscope_buffer(anari_renderer, color_buf)

    def _render_grut(w: int, h: int) -> None:
        import torch

        batch = _grut_batch_from_polyscope(w, h, cuda_device)
        with torch.no_grad():
            outputs = grut_model(batch, train=False)

        rgb = outputs["pred_rgb"][0]
        opa = outputs["pred_opacity"][0]
        if opa.dim() == 2:
            opa = opa.unsqueeze(-1)
        rgba = torch.cat((rgb, opa), dim=-1).contiguous()
        color_buf.update_data_from_device(rgba)

    ps.set_user_callback(callback)
    ps.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Gaussian splat viewer + polyscope")
    parser.add_argument("ply_path", help="Path to a 3DGS .ply file")
    parser.add_argument(
        "--renderer",
        choices=("anari", "3dgrt", "3dgut"),
        default="anari",
        help="Initial active renderer (default: anari). All available backends are loaded; "
        "switch at runtime via the GUI.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Hydra config for 3dgrt/3dgut (default: apps/colmap_3dgrt or apps/colmap_3dgut)",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=1, help="Samples per pixel (anari only)")
    args = parser.parse_args()

    available = _probe_backends()
    if not available:
        raise RuntimeError(
            "No rendering backends found. Install gaussian-viewer (for anari) " "or threedgrut (for 3dgrt/3dgut)."
        )
    print(f"Available backends: {available}")

    run_viewer(args, available)


if __name__ == "__main__":
    main()
