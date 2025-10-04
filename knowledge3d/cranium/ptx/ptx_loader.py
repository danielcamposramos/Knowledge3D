"""
PTX/CUDA Kernel Loader Utility

Unified loader for .cu, .ptx files with NVRTC compilation support.
Used by morton_octree, modular_rpn_engine, geometry_ops, etc.

Author: K3D Core Team
License: Apache-2.0
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

_logger = logging.getLogger(__name__)


def load_cu_kernel(cu_path: str, cache_dir: Optional[Path] = None) -> cp.RawModule:
    """
    Load CUDA kernel from .cu source file with NVRTC compilation.

    Workflow:
    1. Check for pre-compiled .ptx with same basename
    2. If .ptx exists and is newer, load directly
    3. Otherwise, compile .cu → .ptx via NVRTC
    4. Cache compiled .ptx for future runs

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

    # Determine PTX cache path
    if cache_dir is None:
        cache_dir = cu_file.parent

    ptx_file = cache_dir / cu_file.with_suffix('.ptx').name

    # Check if cached PTX is up-to-date
    if ptx_file.exists() and ptx_file.stat().st_mtime >= cu_file.stat().st_mtime:
        _logger.info(f"Loading pre-compiled PTX: {ptx_file}")
        with open(ptx_file, 'r') as f:
            ptx_code = f.read()
        return cp.RawModule(code=ptx_code, backend='nvrtc')

    # Compile .cu → PTX
    _logger.info(f"Compiling CUDA source: {cu_path}")

    with open(cu_file, 'r') as f:
        cu_source = f.read()

    try:
        # Compile with NVRTC (CuPy handles NVRTC internally)
        module = cp.RawModule(
            code=cu_source,
            backend='nvrtc',
            options=(
                '--gpu-architecture=compute_80',  # Adjust for target GPU
                '--use_fast_math',
                '--extra-device-vectorization'
            )
        )

        # Cache compiled PTX
        ptx_code = module.code.decode('utf-8') if isinstance(module.code, bytes) else module.code
        with open(ptx_file, 'w') as f:
            f.write(ptx_code)

        _logger.info(f"Cached compiled PTX: {ptx_file}")

        return module

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

    with open(ptx_file, 'r') as f:
        ptx_code = f.read()

    return cp.RawModule(code=ptx_code, backend='nvrtc')
