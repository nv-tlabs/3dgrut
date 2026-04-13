import os
import sys
import subprocess


def _run(name: str) -> None:
    bin_dir = os.path.dirname(sys.executable)
    exe = os.path.join(bin_dir, name)
    if sys.platform == "win32" and not exe.endswith(".exe"):
        exe += ".exe"
    raise SystemExit(subprocess.call([exe] + sys.argv[1:]))


def gaussian_viewer() -> None:
    _run("commandline_viewer")


def interactive_viewer() -> None:
    _run("interactive_viewer")
