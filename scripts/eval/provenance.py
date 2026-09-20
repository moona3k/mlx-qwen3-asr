"""Runtime provenance recorded into benchmark artifacts.

Every quality or latency number is only meaningful next to the machine and
software stack that produced it. ``runtime_provenance()`` returns the fields
the nightly lane and the eval scripts embed under ``"runtime"`` so results
from different Macs and MLX releases can be compared or kept apart.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", key], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _git_short_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def runtime_provenance(repo_root: Path | None = None) -> dict[str, object]:
    """Describe the machine and stack: chip, memory, OS, Python, MLX, git commit.

    Missing pieces are ``None`` rather than errors so artifacts are still
    written on unusual hosts (Linux CI, missing git).
    """
    try:
        import mlx.core as mx

        mlx_version: str | None = str(mx.__version__)
    except Exception:  # pragma: no cover - mlx always present on supported hosts
        mlx_version = None

    memsize = _sysctl("hw.memsize")
    memory_gb: float | None = None
    if memsize and memsize.isdigit():
        memory_gb = round(int(memsize) / (1024**3), 1)

    root = repo_root or Path(__file__).resolve().parents[2]
    return {
        "host_chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "memory_gb": memory_gb,
        "os": f"{platform.system()} {platform.release()}",
        "macos_version": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "mlx_version": mlx_version,
        "git_commit": _git_short_head(root),
    }
