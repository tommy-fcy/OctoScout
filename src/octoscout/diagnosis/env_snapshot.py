"""Environment snapshot: collect Python env, packages, CUDA, OS info."""

from __future__ import annotations

import json
import platform
import subprocess
import sys

from octoscout.models import EnvSnapshot


def collect_env_snapshot() -> EnvSnapshot:
    """Collect a snapshot of the current Python environment.

    Each sub-collector is fault-tolerant — failures are silently skipped
    so that a partial snapshot is always returned.
    """
    snap = EnvSnapshot()

    snap.python_version = _get_python_version()
    snap.os_info = _get_os_info()
    snap.installed_packages = _get_installed_packages()
    snap.cuda_version = _get_cuda_version(snap.installed_packages)
    snap.cudnn_version = _get_cudnn_version()
    snap.gpu_model = _get_gpu_model()

    return snap


# ---------------------------------------------------------------------------
# Sub-collectors (each must be individually fault-tolerant)
# ---------------------------------------------------------------------------


def _get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_os_info() -> str:
    return platform.platform()


def _get_installed_packages() -> dict[str, str]:
    """Get installed packages via pip list --format=json."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            return {p["name"].lower(): p["version"] for p in packages}
    except Exception:
        pass
    return {}


def _get_cuda_version(packages: dict[str, str]) -> str | None:
    """Try to detect CUDA version from torch or nvidia-smi."""
    # Method 1: from torch
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Method 2: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"driver {result.stdout.strip()}"
    except Exception:
        pass

    return None


def _get_cudnn_version() -> str | None:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '')",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_gpu_model() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None
