"""Re-export sovereign loader for ptx_runtime modules.

This maintains Kimi's zero-copy strategy by providing direct access to sovereign
functions without forcing imports through the bridges layer.
"""
import ctypes
from knowledge3d.cranium.sovereign.loader import (
    load_ptx,
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)

def launch_kernel(ptx_path: str, grid: tuple, block: tuple, *args):
    """Convenience wrapper: load PTX and launch kernel in one call.

    This is Kimi's pattern for zero-copy inline kernel launches where
    PTX is loaded on-demand for small utility kernels.

    Args:
        ptx_path: Path to PTX file (can be relative)
        grid: Grid dimensions (x, y, z)
        block: Block dimensions (x, y, z)
        *args: Kernel arguments (will be wrapped in ctypes)

    Example:
        launch_kernel("path/to/kernel.ptx", (1,1,1), (64,1,1), d_ptr, count)
    """
    # Extract kernel name from PTX path (typically last part before .ptx)
    import os
    kernel_name = os.path.basename(ptx_path).replace('.ptx', '')

    # Load kernel
    kernel = load_ptx_file(ptx_path, kernel_name)

    # Launch with provided arguments
    launch(kernel, grid=grid, block=block, params=list(args))

__all__ = [
    "load_ptx",
    "load_ptx_file",
    "gpu_malloc",
    "gpu_free",
    "memcpy_htod",
    "memcpy_dtoh",
    "launch",
    "launch_kernel",
    "synchronize",
]
