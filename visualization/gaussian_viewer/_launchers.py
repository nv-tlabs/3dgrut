import os
import subprocess
import sys
from pathlib import Path


def _run(name: str) -> None:
    pkg_dir = Path(__file__).resolve().parent
    exe = pkg_dir / name
    env = None
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
        runtime_dirs = [pkg_dir, pkg_dir / "lib", pkg_dir / "lib64", pkg_dir / "bin"]
        path_prefix = os.pathsep.join(str(d) for d in runtime_dirs if d.is_dir())
        if path_prefix:
            env = os.environ.copy()
            prev_path = env.get("PATH", "")
            env["PATH"] = f"{path_prefix}{os.pathsep}{prev_path}" if prev_path else path_prefix
    raise SystemExit(subprocess.call([str(exe)] + sys.argv[1:], env=env))


def gaussian_viewer() -> None:
    _run("commandline_viewer")


def interactive_viewer() -> None:
    _run("interactive_viewer")
