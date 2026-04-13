import subprocess
import sys
from pathlib import Path


def _run(name: str) -> None:
    pkg_dir = Path(__file__).resolve().parent
    exe = pkg_dir / name
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    raise SystemExit(subprocess.call([str(exe)] + sys.argv[1:]))


def gaussian_viewer() -> None:
    _run("commandline_viewer")


def interactive_viewer() -> None:
    _run("interactive_viewer")
