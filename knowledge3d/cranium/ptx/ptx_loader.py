"""
PTX/CUDA Kernel Loader Utility

Unified loader for .cu, .ptx files with NVRTC compilation support.
Used by morton_octree, modular_rpn_engine, geometry_ops, etc.

Author: K3D Core Team
License: Apache-2.0
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

_logger = logging.getLogger(__name__)


def load_cu_kernel(
    cu_path: str,
    cache_dir: Optional[Path] = None,  # retained for API compatibility (unused)
) -> cp.RawModule:
    """
    Load CUDA kernel from .cu source file with NVRTC compilation.

    Workflow:
    1. Read the CUDA source from disk
    2. Compile in-process via NVRTC (CuPy injects architecture flags automatically)
    3. Return a CuPy RawModule (CuPy handles kernel caching internally)

    Args:
        cu_path: Path to .cu source file
        cache_dir: Optional cache directory (default: same dir as .cu)

    Returns:
        CuPy RawModule with compiled kernels

    Raises:
        FileNotFoundError: If .cu file doesn't exist
        RuntimeError: If compilation fails
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy required for kernel loading. Install: pip install cupy-cuda12x")

    cu_file = Path(cu_path)
    if not cu_file.exists():
        raise FileNotFoundError(f"CUDA source not found: {cu_path}")

    _logger.info(f"Compiling CUDA source with CuPy NVRTC: {cu_path}")
    with open(cu_file, 'r', encoding='utf-8') as f:
        cu_source = f.read()
    try:
        options: list[str] = ["--std=c++14"]
        arch = os.getenv("K3D_CUDA_ARCH")
        if arch:
            options.append(f"--gpu-architecture={arch}")
        return cp.RawModule(
            code=cu_source,
            backend='nvrtc',
            options=tuple(options),
        )
    except Exception as e:
        _logger.error(f"CUDA compilation failed for {cu_path}: {e}")
        raise RuntimeError(f"Kernel compilation error: {e}") from e


def load_ptx_kernel(ptx_path: str) -> cp.RawModule:
    """
    Load pre-compiled PTX kernel directly.

    Args:
        ptx_path: Path to .ptx file

    Returns:
        CuPy RawModule
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy required. Install: pip install cupy-cuda12x")

    ptx_file = Path(ptx_path)
    if not ptx_file.exists():
        raise FileNotFoundError(f"PTX file not found: {ptx_path}")

    _logger.info(f"Loading PTX: {ptx_path}")

    return cp.RawModule(path=str(ptx_file))
