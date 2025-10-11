"""Sovereign GPU execution: Pure ctypes + libcuda.so, zero library dependencies.

This module implements the sovereign path for PTX kernel loading and execution:
- No CuPy, no cuda-python bindings
- Direct ctypes wrapper around libcuda.so (CUDA Driver API)
- Hand-authored PTX kernels loaded as static assets
- Python as pure I/O conduit, all math in PTX

Architecture:
    Python Bridge (I/O) -> Sovereign Loader (ctypes) -> Unified PTX (RPN+TRM)

Mandates:
- GPU sovereignty: All computation in PTX
- No version conflicts: Only stdlib ctypes + system libcuda.so
- No CPU fallbacks: Pure GPU execution
- <95µs latency: Zero-overhead launches
"""

from .loader import (
    load_ptx,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)

__all__ = [
    "load_ptx",
    "gpu_malloc",
    "gpu_free",
    "memcpy_htod",
    "memcpy_dtoh",
    "launch",
    "synchronize",
]
