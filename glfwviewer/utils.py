"""Shared math and runtime helpers for the GLFW viewer.

The functions in this module are intentionally side-effect free and are
reused by HUD construction and diagnostics reporting.
"""

import numpy as np
import torch

def quat_wxyz_to_euler_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a quaternion in ``(w, x, y, z)`` order to Euler angles.

    Args:
        quat_wxyz: Quaternion values stored in ``(w, x, y, z)`` order.

    Returns:
        np.ndarray: Euler angles ``(roll, pitch, yaw)`` in degrees.

    Note:
        The conversion matches the orientation summary shown in the viewer HUD.
    """
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float32))

def format_runtime_memory_usage(engine, device) -> str:
    """Build a concise runtime RAM/VRAM usage summary string.

    Args:
        engine: Viewer rendering engine instance.
        device: Active torch device used by the engine.

    Returns:
        str: Human-readable text describing current RAM and VRAM usage.

    Note:
        The ``engine`` argument is accepted for interface symmetry with HUD
        helpers, even though the current implementation only needs ``device``.
    """
    ram_gb = 0.0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    ram_kb = float(parts[1])
                    ram_gb = ram_kb / (1024.0 * 1024.0)
                    break
    except Exception:
        ram_gb = 0.0
    vram_text = "N/A"
    try:
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            allocated = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024.0 ** 3)
            vram_text = f"{allocated:.2f}/{reserved:.2f} GB"
    except Exception:
        vram_text = "N/A"
    return f"Memory: VRAM={vram_text} | RAM={ram_gb:.2f} GB"
