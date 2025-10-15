"""Sovereign PTX Loader: Pure ctypes wrapper for CUDA Driver API.

Implements direct bindings to libcuda.so without any external dependencies.
This is the foundation for version-agnostic, library-independent GPU execution.

Based on Kimi's sovereign ideation from Step9 development chain.
"""
import ctypes
import os
from typing import List, Tuple, Optional


GPU_MEMORY_TARGET_GB = 3.5  # Updated target for sovereign GPU allocations

# Load CUDA Driver API library
# This is stable across CUDA versions and always present on systems with NVIDIA drivers
try:
    nvcuda = ctypes.CDLL("libcuda.so.1", use_errno=True)
except OSError:
    try:
        nvcuda = ctypes.CDLL("libcuda.so", use_errno=True)
    except OSError as e:
        raise RuntimeError(
            "Failed to load libcuda.so - ensure NVIDIA driver is installed"
        ) from e

# ==========================================
# CUDA Driver API Types
# ==========================================
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p

# ==========================================
# Error Handling
# ==========================================
def ck(res: int) -> None:
    """Check CUDA result and raise on error."""
    if res != 0:
        # Try to get error string
        err_str = ctypes.c_char_p()
        if nvcuda.cuGetErrorString(res, ctypes.byref(err_str)) == 0:
            msg = err_str.value.decode() if err_str.value else f"Unknown error {res}"
        else:
            msg = f"CUDA error {res}"
        raise RuntimeError(f"Sovereign loader error: {msg}")

# ==========================================
# One-Time Initialization
# ==========================================
_initialized = False
_device = None
_context = None

def _ensure_init():
    """Ensure CUDA is initialized (called automatically)."""
    global _initialized, _device, _context
    if not _initialized:
        # Initialize CUDA
        ck(nvcuda.cuInit(0))

        # Get device
        device = CUdevice()
        ck(nvcuda.cuDeviceGet(ctypes.byref(device), 0))
        _device = device

        ctx = CUcontext()
        res = nvcuda.cuCtxCreate(ctypes.byref(ctx), 0, device)
        if res != 0:
            if res == 2:  # CUDA_ERROR_OUT_OF_MEMORY
                ctx = CUcontext()
                ck(nvcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device))
                ck(nvcuda.cuCtxSetCurrent(ctx))
            else:
                ck(res)
        else:
            ck(nvcuda.cuCtxSetCurrent(ctx))
        _context = ctx
        _initialized = True

def _ensure_current_context():
    if not _initialized or _context is None:
        _ensure_init()
    else:
        ck(nvcuda.cuCtxSetCurrent(_context))

# ==========================================
# PTX Module Loading
# ==========================================
def load_ptx(ptx_source: bytes, entry_name: bytes) -> CUfunction:
    """Load PTX kernel from bytes and return function handle.

    Args:
        ptx_source: PTX code as bytes (can be from file.read())
        entry_name: Kernel entry point name as bytes (e.g., b"kernel_name")

    Returns:
        CUfunction handle for launching

    Example:
        >>> ptx = open("kernel.ptx", "rb").read()
        >>> kernel = load_ptx(ptx, b"my_kernel")
    """
    _ensure_current_context()

    module = CUmodule()
    ck(nvcuda.cuModuleLoadData(ctypes.byref(module), ptx_source))

    func = CUfunction()
    ck(nvcuda.cuModuleGetFunction(ctypes.byref(func), module, entry_name))

    return func


def load_module(ptx_source: bytes) -> CUmodule:
    """Load PTX as a CUDA module and return the module handle."""
    _ensure_current_context()
    module = CUmodule()
    ck(nvcuda.cuModuleLoadData(ctypes.byref(module), ptx_source))
    return module


def load_module_from_file(ptx_path: str) -> CUmodule:
    """Load PTX module from file path."""
    with open(ptx_path, "rb") as f:
        data = f.read()
    return load_module(data)


def get_function(module: CUmodule, entry_name: str) -> CUfunction:
    """Obtain a kernel function handle from an existing module."""
    _ensure_current_context()
    func = CUfunction()
    ck(nvcuda.cuModuleGetFunction(ctypes.byref(func), module, entry_name.encode()))
    return func


def get_global(module: CUmodule, symbol_name: str) -> Tuple[CUdeviceptr, int]:
    """Fetch a global/constant memory pointer from a module."""
    _ensure_current_context()
    device_ptr = CUdeviceptr()
    size = ctypes.c_size_t()
    ck(
        nvcuda.cuModuleGetGlobal(
            ctypes.byref(device_ptr),
            ctypes.byref(size),
            module,
            symbol_name.encode(),
        )
    )
    return device_ptr, size.value

def load_ptx_file(ptx_path: str, entry_name: str) -> CUfunction:
    """Load PTX kernel from file path.

    Args:
        ptx_path: Path to .ptx file
        entry_name: Kernel entry point name

    Returns:
        CUfunction handle for launching
    """
    with open(ptx_path, "rb") as f:
        ptx_data = f.read()
    return load_ptx(ptx_data, entry_name.encode())

# ==========================================
# Memory Management
# ==========================================
def gpu_malloc(size_bytes: int) -> CUdeviceptr:
    """Allocate GPU memory.

    Args:
        size_bytes: Number of bytes to allocate

    Returns:
        Device pointer (uint64)
    """
    _ensure_init()
    ptr = CUdeviceptr()
    ck(nvcuda.cuMemAlloc(ctypes.byref(ptr), size_bytes))
    return ptr

def gpu_free(ptr: CUdeviceptr) -> None:
    """Free GPU memory.

    Args:
        ptr: Device pointer from gpu_malloc
    """
    ck(nvcuda.cuMemFree(ptr))

def memcpy_htod(dst_device: CUdeviceptr, src_host: ctypes.c_void_p, size_bytes: int) -> None:
    """Copy from host to device.

    Args:
        dst_device: Destination device pointer
        src_host: Source host pointer (from numpy array.ctypes.data)
        size_bytes: Number of bytes to copy
    """
    ck(nvcuda.cuMemcpyHtoD(dst_device, src_host, size_bytes))

def memcpy_dtoh(dst_host: ctypes.c_void_p, src_device: CUdeviceptr, size_bytes: int) -> None:
    """Copy from device to host.

    Args:
        dst_host: Destination host pointer
        src_device: Source device pointer
        size_bytes: Number of bytes to copy
    """
    ck(nvcuda.cuMemcpyDtoH(dst_host, src_device, size_bytes))

# ==========================================
# Kernel Execution
# ==========================================
def launch(
    kernel: CUfunction,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    params: List,
    shared_mem: int = 0,
    stream: Optional[CUstream] = None
) -> None:
    """Launch CUDA kernel.

    Args:
        kernel: Kernel function handle from load_ptx
        grid: Grid dimensions (gridDim.x, gridDim.y, gridDim.z)
        block: Block dimensions (blockDim.x, blockDim.y, blockDim.z)
        params: List of kernel parameters (device pointers, scalars as ctypes)
        shared_mem: Shared memory size in bytes (default: 0)
        stream: CUDA stream (default: None = default stream)

    Example:
        >>> d_in = gpu_malloc(1024)
        >>> d_out = gpu_malloc(1024)
        >>> launch(kernel, (1,1,1), (256,1,1),
        ...        [d_in, d_out, ctypes.c_int(256)])
    """
    _ensure_init()

    # Convert grid/block to ctypes arrays
    grid_arr = (ctypes.c_uint * 3)(*grid)
    block_arr = (ctypes.c_uint * 3)(*block)

    # Convert params to void pointer array
    param_ptrs = (ctypes.c_void_p * len(params))(*[
        ctypes.cast(ctypes.byref(p), ctypes.c_void_p) if hasattr(p, '_type_')
        else ctypes.c_void_p(p)
        for p in params
    ])

    # Launch kernel
    ck(nvcuda.cuLaunchKernel(
        kernel,
        grid_arr[0], grid_arr[1], grid_arr[2],
        block_arr[0], block_arr[1], block_arr[2],
        shared_mem,
        stream or CUstream(),
        param_ptrs,
        None
    ))

def synchronize() -> None:
    """Synchronize device (wait for all kernels to complete)."""
    _ensure_init()
    ck(nvcuda.cuCtxSynchronize())

# ==========================================
# Cleanup
# ==========================================
def cleanup():
    """Clean up CUDA context (called automatically at exit)."""
    global _initialized, _context
    if _initialized and _context:
        nvcuda.cuCtxDestroy(_context)
        _initialized = False
        _context = None

import atexit
atexit.register(cleanup)

__all__ = [
    "load_ptx",
    "load_ptx_file",
    "load_module",
    "load_module_from_file",
    "get_function",
    "get_global",
    "gpu_malloc",
    "gpu_free",
    "memcpy_htod",
    "memcpy_dtoh",
    "launch",
    "synchronize",
]
