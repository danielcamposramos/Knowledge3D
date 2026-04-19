"""⚠️  DEPRECATED - MOVED TO Old_Attempts/utils/ ⚠️

This file configures CuPy which is deprecated in the sovereign architecture.

MIGRATION PATH:
- Sovereign architecture uses NO CuPy
- All GPU operations via pure ctypes + libcuda.so
- See knowledge3d.cranium.sovereign.loader for memory management
- See knowledge3d.cranium.bridges.sovereign_bridges for all GPU operations

REASON FOR DEPRECATION:
- CuPy adds 500MB+ dependency
- Sovereign architecture: zero external GPU dependencies
- All PTX kernels precompiled, no NVRTC needed

⚠️  DO NOT USE - Use sovereign architecture instead!
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import cupy as cp


def ensure_nvrtc_include_path() -> None:
    """Set CUPY_NVRTC_INCLUDE_PATH if missing.

    On some Debian-based systems CUDA headers live under the `targets` directory
    shipped with the toolkit, which CuPy does not pick up automatically.  We
    detect a handful of known locations and configure the environment variable
    on the fly so NVRTC compiles succeed.
    """

    base = Path(cp.__file__).resolve().parents[1]
    root_candidates = [
        Path("/K3D/Knowledge3D.local/envs/k3d-cranium"),
        Path(os.environ.get("CONDA_PREFIX", sys.prefix)),
        base,
        Path("/usr/local/cuda"),
    ]

    candidates = [root / "targets" / "x86_64-linux" / "include" for root in root_candidates] + [Path("/usr/local/cuda/include")]

    for candidate in candidates:
        header = candidate / "cuda_fp16.h"
        if header.exists():
            os.environ["CUPY_NVRTC_INCLUDE_PATH"] = str(candidate)
            os.environ.setdefault("CUDA_PATH", str(candidate.parent.parent))
            os.environ.setdefault("CUPY_CUDA_PATH", str(candidate.parent.parent))
            try:
                cp.cuda.nvrtc.add_include_dirs([str(candidate)])
            except Exception:
                pass
            return

    # Fallback: if nothing found, keep existing setting (if any)


__all__ = ["ensure_nvrtc_include_path"]
