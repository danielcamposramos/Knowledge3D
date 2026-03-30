"""Sovereign PTX Loader: Pure ctypes wrapper for CUDA Driver API.

Implements direct bindings to libcuda.so without any external dependencies.
This is the foundation for version-agnostic, library-independent GPU execution.

Based on Kimi's sovereign ideation from Step9 development chain.
"""
import ctypes
import os
from typing import List, Tuple, Optional, Any


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

try:
    libcudart = ctypes.CDLL("libcudart.so")
    libcudart.cudaMalloc.restype = ctypes.c_int
    libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    libcudart.cudaFree.restype = ctypes.c_int
    libcudart.cudaFree.argtypes = [ctypes.c_void_p]
    libcudart.cudaMemcpy.restype = ctypes.c_int
    libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
except OSError:
    libcudart = None

try:
    _cuMemGetInfo = getattr(nvcuda, "cuMemGetInfo_v2")
except AttributeError:
    _cuMemGetInfo = getattr(nvcuda, "cuMemGetInfo", None)

try:
    _cuMemsetD32 = getattr(nvcuda, "cuMemsetD32_v2")
except AttributeError:  # pragma: no cover - legacy drivers
    _cuMemsetD32 = getattr(nvcuda, "cuMemsetD32")

try:
    _cuMemcpyDtoD_v2 = getattr(nvcuda, "cuMemcpyDtoD_v2")
except AttributeError:
    _cuMemcpyDtoD_v2 = None

try:
    _cuMemcpyDtoD_v1 = getattr(nvcuda, "cuMemcpyDtoD")
except AttributeError:
    _cuMemcpyDtoD_v1 = None

try:
    _cuMemcpyHtoD_v2 = getattr(nvcuda, "cuMemcpyHtoD_v2")
except AttributeError:
    _cuMemcpyHtoD_v2 = None

try:
    _cuMemcpyHtoD_v1 = getattr(nvcuda, "cuMemcpyHtoD")
except AttributeError:
    _cuMemcpyHtoD_v1 = None

try:
    _cuMemcpyDtoH_v2 = getattr(nvcuda, "cuMemcpyDtoH_v2")
except AttributeError:
    _cuMemcpyDtoH_v2 = None

try:
    _cuMemcpyDtoH_v1 = getattr(nvcuda, "cuMemcpyDtoH")
except AttributeError:
    _cuMemcpyDtoH_v1 = None

_cuMemcpyDtoD = _cuMemcpyDtoD_v2 or _cuMemcpyDtoD_v1
_cuMemcpyHtoD = _cuMemcpyHtoD_v2 or _cuMemcpyHtoD_v1
_cuMemcpyDtoH = _cuMemcpyDtoH_v2 or _cuMemcpyDtoH_v1

if _cuMemGetInfo is not None:
    _cuMemGetInfo.restype = ctypes.c_int
    _cuMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]

_cuMemsetD32.restype = ctypes.c_int
_cuMemsetD32.argtypes = [ctypes.c_uint64, ctypes.c_uint, ctypes.c_size_t]
if _cuMemcpyDtoD is not None:
    _cuMemcpyDtoD.restype = ctypes.c_int
    _cuMemcpyDtoD.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
if _cuMemcpyDtoD_v1 is not None:
    _cuMemcpyDtoD_v1.restype = ctypes.c_int
    _cuMemcpyDtoD_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
if _cuMemcpyDtoD_v2 is not None:
    _cuMemcpyDtoD_v2.restype = ctypes.c_int
    _cuMemcpyDtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
if _cuMemcpyHtoD is not None:
    _cuMemcpyHtoD.restype = ctypes.c_int
    _cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
if _cuMemcpyHtoD_v1 is not None:
    _cuMemcpyHtoD_v1.restype = ctypes.c_int
    _cuMemcpyHtoD_v1.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
if _cuMemcpyHtoD_v2 is not None:
    _cuMemcpyHtoD_v2.restype = ctypes.c_int
    _cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
if _cuMemcpyDtoH is not None:
    _cuMemcpyDtoH.restype = ctypes.c_int
    _cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
if _cuMemcpyDtoH_v1 is not None:
    _cuMemcpyDtoH_v1.restype = ctypes.c_int
    _cuMemcpyDtoH_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
if _cuMemcpyDtoH_v2 is not None:
    _cuMemcpyDtoH_v2.restype = ctypes.c_int
    _cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2

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

_cupy_allocations: List[Tuple[int, int, "Any"]] = []
_cudart_allocations: List[Tuple[int, int]] = []

def _find_cudart_allocation(address: int) -> Optional[Tuple[int, int]]:
    for start, size in _cudart_allocations:
        if start <= address < start + size:
            return start, size
    return None

def _find_cupy_allocation(address: int) -> Optional[Tuple[int, int, "Any"]]:
    for start, size, mem in _cupy_allocations:
        if start <= address < start + size:
            return start, size, mem
    return None

# ==========================================
# Error Handling
# ==========================================
def ck(res: int) -> None:
    """Check CUDA result and raise on error."""
    if res != 0:
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] raising error code {res}")
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
_init_pid = None  # Track which process initialized CUDA
_firewall_checked_pid = None  # Track sovereignty assertion per process


def _maybe_assert_sovereign_hot_path(current_pid: int) -> None:
    """Optionally run runtime sovereignty assertion once per process.

    This check is guarded behind `K3D_ENABLE_SOVEREIGN_FIREWALL=1` to keep
    existing non-Knowledgeverse paths stable while MVP hardening is phased in.
    """
    global _firewall_checked_pid
    if os.environ.get("K3D_ENABLE_SOVEREIGN_FIREWALL", "0") != "1":
        return
    if _firewall_checked_pid == current_pid:
        return

    from knowledge3d.knowledgeverse.sovereignty_firewall import SovereigntyFirewall

    SovereigntyFirewall.runtime_assert_hot_path()
    _firewall_checked_pid = current_pid

def _ensure_init():
    """Ensure CUDA is initialized (called automatically)."""
    global _initialized, _device, _context, _init_pid

    # CRITICAL: Detect if we're in a forked child process
    # CUDA contexts are NOT fork-safe and must be recreated per-process
    current_pid = os.getpid()
    if _initialized and _init_pid != current_pid:
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] Detected fork: parent PID={_init_pid}, current PID={current_pid}")
            print(f"[loader] Reinitializing CUDA context for child process")
        # Reset state - force reinitialization in this process
        _initialized = False
        _context = None
        _device = None

    if not _initialized:
        _maybe_assert_sovereign_hot_path(current_pid)

        # Initialize CUDA
        ck(nvcuda.cuInit(0))

        # Get device
        device = CUdevice()
        ck(nvcuda.cuDeviceGet(ctypes.byref(device), 0))
        _device = device

        ctx = CUcontext()
        
        # Check if we should force Primary Context (to share with PyTorch)
        use_primary = os.environ.get("K3D_USE_PRIMARY_CTX", "0") == "1"
        res = 201 if use_primary else nvcuda.cuCtxCreate(ctypes.byref(ctx), 0, device)
        
        if res != 0:
            if os.environ.get("K3D_RPN_DEBUG") and not use_primary:
                print(f"[loader] cuCtxCreate failed with code {res}")
            if res in (2, 201):  # out of memory or invalid context -> fall back to primary ctx
                set_flags_res = nvcuda.cuDevicePrimaryCtxSetFlags(device, 0)
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuDevicePrimaryCtxSetFlags -> {set_flags_res}")
                if set_flags_res not in (0, 708):  # 708: context already active
                    ck(set_flags_res)
                ctx = CUcontext()
                retain_res = nvcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device)
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuDevicePrimaryCtxRetain -> {retain_res}, ctx={ctx}")
                if retain_res != 0:
                    if os.environ.get("K3D_RPN_DEBUG"):
                        print(f"[loader] cuDevicePrimaryCtxRetain failed with code {retain_res}")
                    # Attempt to bootstrap via CuPy
                    try:
                        import cupy as _cupy  # type: ignore

                        _cupy.cuda.Device(0).use()
                        current_ctx = CUcontext()
                        if nvcuda.cuCtxGetCurrent(ctypes.byref(current_ctx)) == 0 and current_ctx:
                            ctx = current_ctx
                        else:
                            ck(retain_res)
                    except Exception as cupy_exc:  # pragma: no cover - debug path
                        if os.environ.get("K3D_RPN_DEBUG"):
                            print(f"[loader] CuPy bootstrap failed: {cupy_exc}")
                        ck(retain_res)
                set_res = nvcuda.cuCtxSetCurrent(ctx)
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuCtxSetCurrent -> {set_res}")
                ck(set_res)
                current = CUcontext()
                ck(nvcuda.cuCtxGetCurrent(ctypes.byref(current)))
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuCtxGetCurrent -> {current}")
                ctx = current
                try:
                    import cupy as _cupy  # type: ignore

                    _cupy.cuda.Device(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))).use()
                    if os.environ.get("K3D_RPN_DEBUG"):
                        print("[loader] CuPy context primed")
                    refreshed = CUcontext()
                    if nvcuda.cuCtxGetCurrent(ctypes.byref(refreshed)) == 0 and refreshed:
                        ctx = refreshed
                        if os.environ.get("K3D_RPN_DEBUG"):
                            print(f"[loader] context refreshed -> {ctx}")
                except Exception as cupy_exc:  # pragma: no cover - optional path
                    if os.environ.get("K3D_RPN_DEBUG"):
                        print(f"[loader] CuPy context bootstrap skipped: {cupy_exc}")
            else:
                ck(res)
        else:
            ck(nvcuda.cuCtxSetCurrent(ctx))
        _context = ctx
        _init_pid = current_pid  # Track which process owns this context
        _initialized = True

def _ensure_current_context():
    if not _initialized or _context is None:
        _ensure_init()
    else:
        res = nvcuda.cuCtxSetCurrent(_context)
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] cuCtxSetCurrent (ensure) -> {res}")
        ck(res)

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
    _ensure_current_context()
    ptr = CUdeviceptr()
    if os.environ.get("K3D_FORCE_CUPY_ALLOC"):
        res = 201
    else:
        res = nvcuda.cuMemAlloc(ctypes.byref(ptr), size_bytes)
    if os.environ.get("K3D_RPN_DEBUG"):
        print(f"[loader] cuMemAlloc({size_bytes}) -> {res}, ptr={ptr}")
    if res == 201:
        if libcudart is not None:
            runtime_ptr = ctypes.c_void_p()
            runtime_res = libcudart.cudaMalloc(ctypes.byref(runtime_ptr), size_bytes)
            if os.environ.get("K3D_RPN_DEBUG"):
                print(f"[loader] cudaMalloc({size_bytes}) -> {runtime_res}, ptr={runtime_ptr.value}")
            if runtime_res == 0 and runtime_ptr.value:
                _cudart_allocations.append((int(runtime_ptr.value), size_bytes))
                return CUdeviceptr(runtime_ptr.value)
            if runtime_res not in (0,):
                ck(runtime_res)
        # Fall back to CuPy-managed allocation while keeping sovereign interface.
        if os.environ.get("K3D_RPN_DEBUG"):
            print("[loader] Falling back to CuPy allocation")
        try:
            import cupy as _cupy  # type: ignore
        except Exception as cupy_exc:  # pragma: no cover - critical fallback
            if os.environ.get("K3D_RPN_DEBUG"):
                print(f"[loader] CuPy import failed during fallback: {cupy_exc}")
            ck(res)
        _cupy.cuda.Device(0).use()
        mem = _cupy.cuda.alloc(size_bytes)
        start = int(mem.ptr)
        _cupy_allocations.append((start, size_bytes, mem))
        return CUdeviceptr(start)
    ck(res)
    return ptr

def gpu_free(ptr: CUdeviceptr) -> None:
    """Free GPU memory.

    Args:
        ptr: Device pointer from gpu_malloc
    """
    key = int(ptr.value)
    alloc = _find_cupy_allocation(key)
    if alloc is not None:
        if os.environ.get("K3D_RPN_DEBUG"):
            print("[loader] Releasing CuPy-backed allocation")
        _cupy_allocations.remove(alloc)
        return
    cudart_alloc = _find_cudart_allocation(key)
    if cudart_alloc is not None:
        if os.environ.get("K3D_RPN_DEBUG"):
            print("[loader] Releasing cudaMalloc-backed allocation")
        if libcudart is not None:
            libcudart.cudaFree(ctypes.c_void_p(cudart_alloc[0]))
        _cudart_allocations.remove(cudart_alloc)
        return
    ck(nvcuda.cuMemFree(ptr))


def _coerce_host_pointer(value: object) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    try:
        return ctypes.c_void_p(int(value))
    except Exception:
        pass
    try:
        return ctypes.cast(value, ctypes.c_void_p)
    except Exception:
        pass
    inner = getattr(value, "_obj", None)
    if inner is not None:
        try:
            return ctypes.c_void_p(ctypes.addressof(inner))
        except Exception:
            pass
    raise TypeError(f"Cannot coerce host pointer from {type(value)!r}")


def memcpy_htod(dst_device: CUdeviceptr, src_host: ctypes.c_void_p, size_bytes: int) -> None:
    """Copy from host to device.

    Args:
        dst_device: Destination device pointer
        src_host: Source host pointer (from numpy array.ctypes.data)
        size_bytes: Number of bytes to copy
    """
    _ensure_current_context()
    key = int(dst_device.value)
    alloc = _find_cupy_allocation(key)
    if alloc is not None:
        try:
            import cupy as _cupy  # type: ignore
        except Exception as cupy_exc:
            if os.environ.get("K3D_RPN_DEBUG"):
                print(f"[loader] CuPy import failed during memcpy_htod fallback: {cupy_exc}")
        else:
            _cupy.cuda.Device(0).use()
            res = _cupy.cuda.runtime.memcpy(key, src_host.value, size_bytes, _cupy.cuda.runtime.memcpyHostToDevice)
            if os.environ.get("K3D_RPN_DEBUG"):
                print(f"[loader] runtime.memcpy HtoD -> {res}")
            if res is not None and res != 0:
                ck(res)
            return
    cudart_alloc = _find_cudart_allocation(key)
    if cudart_alloc is not None and libcudart is not None:
        res = libcudart.cudaMemcpy(ctypes.c_void_p(key), src_host, size_bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] cudaMemcpy HtoD -> {res}")
        if res != 0:
            ck(res)
        return
    if _cuMemcpyHtoD is None:
        raise RuntimeError("cuMemcpyHtoD not available in CUDA driver.")
    host_ptr = _coerce_host_pointer(src_host)
    res = _cuMemcpyHtoD(dst_device.value, host_ptr, ctypes.c_size_t(size_bytes))
    if (
        res == 201
        and _cuMemcpyHtoD_v2 is not None
        and _cuMemcpyHtoD_v1 is not None
        and _cuMemcpyHtoD is _cuMemcpyHtoD_v2
    ):
        if os.environ.get("K3D_RPN_DEBUG"):
            print("[loader] cuMemcpyHtoD_v2 invalid context, retrying v1")
        res = _cuMemcpyHtoD_v1(dst_device.value, host_ptr, ctypes.c_size_t(size_bytes))
    if os.environ.get("K3D_RPN_DEBUG"):
        print(f"[loader] cuMemcpyHtoD -> {res}")
    ck(res)

def memcpy_dtoh(dst_host: ctypes.c_void_p, src_device: CUdeviceptr, size_bytes: int) -> None:
    """Copy from device to host.

    Args:
        dst_host: Destination host pointer
        src_device: Source device pointer
        size_bytes: Number of bytes to copy
    """
    _ensure_current_context()
    key = int(src_device.value)
    alloc = _find_cupy_allocation(key)
    if alloc is not None:
        try:
            import cupy as _cupy  # type: ignore
        except Exception as cupy_exc:
            if os.environ.get("K3D_RPN_DEBUG"):
                print(f"[loader] CuPy import failed during memcpy_dtoh fallback: {cupy_exc}")
        else:
            _cupy.cuda.Device(0).use()
            res = _cupy.cuda.runtime.memcpy(dst_host.value, key, size_bytes, _cupy.cuda.runtime.memcpyDeviceToHost)
            if os.environ.get("K3D_RPN_DEBUG"):
                print(f"[loader] runtime.memcpy DtoH -> {res}")
            if res is not None and res != 0:
                ck(res)
            return
    cudart_alloc = _find_cudart_allocation(key)
    if cudart_alloc is not None and libcudart is not None:
        res = libcudart.cudaMemcpy(dst_host, ctypes.c_void_p(key), size_bytes, CUDA_MEMCPY_DEVICE_TO_HOST)
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[loader] cudaMemcpy DtoH -> {res}")
        if res != 0:
            ck(res)
        return
    if _cuMemcpyDtoH is None:
        raise RuntimeError("cuMemcpyDtoH not available in CUDA driver.")
    host_ptr = _coerce_host_pointer(dst_host)
    res = _cuMemcpyDtoH(host_ptr, src_device.value, ctypes.c_size_t(size_bytes))
    if (
        res == 201
        and _cuMemcpyDtoH_v2 is not None
        and _cuMemcpyDtoH_v1 is not None
        and _cuMemcpyDtoH is _cuMemcpyDtoH_v2
    ):
        if os.environ.get("K3D_RPN_DEBUG"):
            print("[loader] cuMemcpyDtoH_v2 invalid context, retrying v1")
        res = _cuMemcpyDtoH_v1(host_ptr, src_device.value, ctypes.c_size_t(size_bytes))
    if os.environ.get("K3D_RPN_DEBUG"):
        print(f"[loader] cuMemcpyDtoH -> {res}")
    ck(res)

def memcpy_dtod(dst_device: CUdeviceptr, src_device: CUdeviceptr, size_bytes: int) -> None:
    """Copy from device to device."""
    _ensure_current_context()
    if _cuMemcpyDtoD is None:
        raise RuntimeError("cuMemcpyDtoD not available in CUDA driver.")
    res = _cuMemcpyDtoD(dst_device.value, src_device.value, size_bytes)
    if (
        res == 201
        and _cuMemcpyDtoD_v2 is not None
        and _cuMemcpyDtoD_v1 is not None
        and _cuMemcpyDtoD is _cuMemcpyDtoD_v2
    ):
        if os.environ.get("K3D_RPN_DEBUG"):
            print("[loader] cuMemcpyDtoD_v2 invalid context, retrying v1")
        res = _cuMemcpyDtoD_v1(dst_device.value, src_device.value, size_bytes)
    if os.environ.get("K3D_RPN_DEBUG"):
        print(f"[loader] cuMemcpyDtoD -> {res}")
    ck(res)


def memset_d32(dst_device: CUdeviceptr, value: int, count: int) -> None:
    """Fill device memory with 32-bit value."""
    _ensure_current_context()
    res = _cuMemsetD32(dst_device.value, ctypes.c_uint(value), ctypes.c_size_t(count))
    if os.environ.get("K3D_RPN_DEBUG"):
        print(f"[loader] cuMemsetD32 -> {res} (count={count})")
    ck(res)

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
# CUDA Stream Management
# ==========================================
def create_stream() -> CUstream:
    """Create a CUDA stream for asynchronous execution.

    Returns:
        CUstream handle

    Example:
        stream = create_stream()
        launch(kernel, grid, block, params, stream=stream)
        stream_synchronize(stream)
        destroy_stream(stream)
    """
    _ensure_init()
    stream = CUstream()
    ck(nvcuda.cuStreamCreate(ctypes.byref(stream), 0))
    return stream


def destroy_stream(stream: CUstream) -> None:
    """Destroy a CUDA stream.

    Args:
        stream: Stream handle to destroy
    """
    if stream:
        ck(nvcuda.cuStreamDestroy(stream))


def stream_synchronize(stream: CUstream) -> None:
    """Wait for all operations in a stream to complete.

    Args:
        stream: Stream handle to synchronize
    """
    if stream:
        ck(nvcuda.cuStreamSynchronize(stream))


def get_vram_usage() -> tuple[int, int]:
    """
    Return (used_bytes, total_bytes) for the current CUDA device.

    Uses cuMemGetInfo (or cuMemGetInfo_v2) via the sovereign loader. Raises
    RuntimeError if the driver does not expose the information.
    """
    if _cuMemGetInfo is None:
        raise RuntimeError("cuMemGetInfo is not available on this CUDA driver")

    _ensure_current_context()
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    res = _cuMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    ck(res)
    used = total.value - free.value
    return used, total.value


def ensure_init() -> None:
    """Public entry point to initialize the CUDA context via the sovereign loader."""
    _ensure_init()

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

def cpu_to_gpu(dst: CUdeviceptr, array: Any) -> None:
    """Upload a CPU buffer (NumPy array or bytes-like) to GPU memory."""
    if array is None:
        raise ValueError("cpu_to_gpu requires a valid array")
    if hasattr(array, "ctypes") and hasattr(array, "dtype"):
        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - numpy required for array path
            raise RuntimeError("NumPy is required for cpu_to_gpu array uploads") from exc
        arr = np.ascontiguousarray(array, dtype=np.float32)
        memcpy_htod(dst, ctypes.c_void_p(arr.ctypes.data), arr.nbytes)
        return
    if isinstance(array, (bytes, bytearray)):
        buf = (ctypes.c_char * len(array)).from_buffer_copy(array)
        memcpy_htod(dst, ctypes.cast(buf, ctypes.c_void_p), len(array))
        return
    buf = memoryview(array)
    if not buf.c_contiguous:
        buf = memoryview(buf.tobytes())
    memcpy_htod(dst, ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf))), buf.nbytes)


def gpu_to_cpu_scalar(src: CUdeviceptr) -> float:
    """Download a single float32 scalar from GPU memory."""
    value = ctypes.c_float()
    memcpy_dtoh(ctypes.byref(value), src, ctypes.sizeof(value))
    return float(value.value)


def gpu_to_cpu_array(src: CUdeviceptr, shape: Any, dtype: Any = None) -> "Any":
    """Download a float array from GPU memory into a NumPy array."""
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - numpy required for array path
        raise RuntimeError("NumPy is required for gpu_to_cpu_array downloads") from exc
    dtype = np.float32 if dtype is None else dtype
    if isinstance(shape, int):
        shape = (shape,)
    arr = np.empty(tuple(shape), dtype=dtype)
    memcpy_dtoh(ctypes.c_void_p(arr.ctypes.data), src, arr.nbytes)
    return arr


def gpu_to_gpu_copy(dst: CUdeviceptr, src: CUdeviceptr, offset: int, size: int) -> None:
    """Copy a slice of GPU memory from src(+offset) into dst."""
    src_addr = int(src.value if hasattr(src, "value") else src)
    dst_addr = int(dst.value if hasattr(dst, "value") else dst)
    memcpy_dtod(CUdeviceptr(dst_addr), CUdeviceptr(src_addr + int(offset)), int(size))

__all__ = [
    "ensure_init",
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
    "memcpy_dtod",
    "launch",
    "synchronize",
    "get_vram_usage",
    "cpu_to_gpu",
    "gpu_to_cpu_scalar",
    "gpu_to_cpu_array",
    "gpu_to_gpu_copy",
]
