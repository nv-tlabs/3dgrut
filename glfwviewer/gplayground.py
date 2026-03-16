"""CLI entry point for the modular GLFW viewer.

This script preserves the calling convention of the legacy
``glfwviwer.py`` entry while delegating the actual runtime behavior to
``glfwviewer.viewer.InteractiveViewer``.

Note:
    The repository root is temporarily inserted into ``sys.path`` so the
    script can still be launched directly with ``python glfwviewer/gplayground.py``.
"""

import argparse
import sys

sys.path.insert(0, '.')  # 保证直接脚本运行时可import本地包

from glfwviewer.viewer import InteractiveViewer

def main():
    """Parse CLI arguments and launch the interactive viewer.

    Returns:
        None: The function creates an ``InteractiveViewer`` instance and
        enters its main rendering loop.

    Note:
        The argument list intentionally mirrors the historical standalone
        viewer script to remain compatible with existing launch commands.
    """
    parser = argparse.ArgumentParser(description="3DGRUT Interactive Viewer")
    parser.add_argument(
        "--gs_object", 
        type=str, 
        required=True, 
        help="Path of pretrained 3dgrt checkpoint (.pt / .ingp / .ply file)"
    )
    parser.add_argument(
        "--mesh_assets",
        type=str,
        default=None,
        help="Path to folder containing mesh assets (.obj or .glb format)"
    )
    parser.add_argument(
        "--default_gs_config",
        type=str,
        default="apps/colmap_3dgrt.yaml",
        help="Default config for .ingp, .ply, or .pt files"
    )
    parser.add_argument(
        "--envmap_assets",
        type=str,
        default=None,
        help="Path to folder containing .hdr environment maps"
    )
    parser.add_argument(
        "--buffer_mode",
        type=str,
        choices=["host2device", "device2device"],
        default="device2device",
        help="Buffer mode for CUDA-OpenGL interop"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Window width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Window height"
    )
    args = parser.parse_args()
    viewer = InteractiveViewer(
        gs_object=args.gs_object,
        mesh_assets_folder=args.mesh_assets,
        default_config=args.default_gs_config,
        envmap_assets_folder=args.envmap_assets,
        width=args.width,
        height=args.height,
        buffer_mode=args.buffer_mode
    )
    viewer.run()

if __name__ == "__main__":
    main()
