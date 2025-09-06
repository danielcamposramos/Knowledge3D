from __future__ import annotations

"""
Environment guard and accelerator logging utilities.

Goals
- Prevent heavy training jobs from running on bare Debian hosts without
  containment (Conda/Docker), unless explicitly overridden.
- Provide optional logging to confirm GPU/CPU code paths during builds.

Usage
  from .utils.env_guard import enforce_containment, accel_log
  enforce_containment("HF intent training")
  accel_log("Using FAISS GPU flat index")

Override
- Set K3D_ALLOW_NATIVE=1 to bypass the Debian containment guard.
"""

import os
from pathlib import Path
from typing import Optional


def _read_os_release() -> dict:
    d: dict[str, str] = {}
    p = Path("/etc/os-release")
    if not p.exists():
        return d
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                d[k.strip()] = v.strip().strip('"')
    except Exception:
        pass
    return d


def is_debian_like() -> bool:
    info = _read_os_release()
    ident = (info.get("ID") or "").lower()
    like = (info.get("ID_LIKE") or "").lower()
    return "debian" in ident or "debian" in like


def in_conda_env() -> bool:
    return bool(os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV"))


def in_docker() -> bool:
    try:
        if Path("/.dockerenv").exists():
            return True
        # cgroup/containerd hint
        cg = Path("/proc/1/cgroup")
        if cg.exists() and "docker" in cg.read_text(errors="ignore"):
            return True
    except Exception:
        pass
    return False


def enforce_containment(task_name: str) -> None:
    """Exit if running on Debian without Conda/Docker containment.

    Use K3D_ALLOW_NATIVE=1 to override.
    """
    if os.getenv("K3D_ALLOW_NATIVE"):
        return
    if is_debian_like() and not (in_conda_env() or in_docker()):
        msg = (
            f"Refusing to run '{task_name}' on bare Debian without containment.\n"
            "Use a Conda env (recommended) or Docker/RAPIDS container.\n"
            "Override with K3D_ALLOW_NATIVE=1 if you know what you're doing.\n"
            "Docs: docs/LOCAL_ENV.md and docs/GPU_ACCEL.md\n"
        )
        raise SystemExit(msg)


def accel_log(msg: str) -> None:
    """Conditional accelerator logging when K3D_ACCEL_LOG is set."""
    if os.getenv("K3D_ACCEL_LOG"):
        print(f"[K3D-ACCEL] {msg}")

