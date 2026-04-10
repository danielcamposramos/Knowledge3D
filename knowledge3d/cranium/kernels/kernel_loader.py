"""Thin Python compatibility layer over the sovereign CUDA loader.

This module exists so older kernel-facing tests and utilities can use a stable
`knowledge3d.cranium.kernels.kernel_loader` surface without bypassing the live
sovereign loader implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
from typing import Any

from knowledge3d.cranium.sovereign import loader

_PTX_KERNEL_CACHE: dict[tuple[str, str], loader.CUfunction] = {}
_ZERO_COPY_REGIONS: dict[int, tuple[ctypes.c_void_p, int, str, object | None]] = {}
_ZERO_COPY_MAX_BYTES = 0


def _is_recoverable_context_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "invalid context" in text
        or "invalid device context" in text
        or "illegal memory access" in text
    )


def _retry_context(op):
    try:
        return op()
    except RuntimeError as exc:
        if not _is_recoverable_context_error(exc):
            raise
        loader.cleanup()
        loader.ensure_init()
        return op()


def _coerce_host_ptr(value: Any) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), int):
        return ctypes.c_void_p(int(value.value))
    return ctypes.c_void_p(int(value))


def _as_device_ptr(value: Any) -> loader.CUdeviceptr:
    if isinstance(value, loader.CUdeviceptr):
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), int):
        return loader.CUdeviceptr(int(value.value))
    return loader.CUdeviceptr(int(value))


def _cached_kernel(ptx_code: str | bytes, kernel_name: str) -> loader.CUfunction:
    data = ptx_code.encode("utf-8") if isinstance(ptx_code, str) else bytes(ptx_code)
    cache_key = (hashlib.sha256(data).hexdigest(), kernel_name)
    if cache_key not in _PTX_KERNEL_CACHE:
        _PTX_KERNEL_CACHE[cache_key] = _retry_context(
            lambda: loader.load_ptx(data, kernel_name.encode("utf-8"))
        )
    return _PTX_KERNEL_CACHE[cache_key]


def gpu_malloc(size_bytes: int) -> loader.CUdeviceptr:
    return _retry_context(lambda: loader.gpu_malloc(int(size_bytes)))


def gpu_free(ptr: loader.CUdeviceptr) -> None:
    try:
        _retry_context(lambda: loader.gpu_free(_as_device_ptr(ptr)))
    except RuntimeError as exc:
        if not _is_recoverable_context_error(exc):
            raise
        loader.cleanup()


def gpu_memcpy_htod(dst_device: loader.CUdeviceptr, src_host: Any, size_bytes: int) -> None:
    _retry_context(
        lambda: loader.memcpy_htod(_as_device_ptr(dst_device), _coerce_host_ptr(src_host), int(size_bytes))
    )


def gpu_memcpy_dtoh(dst_host: Any, src_device: loader.CUdeviceptr, size_bytes: int) -> None:
    _retry_context(
        lambda: loader.memcpy_dtoh(_coerce_host_ptr(dst_host), _as_device_ptr(src_device), int(size_bytes))
    )


def gpu_memcpy_dtod(dst_device: loader.CUdeviceptr, src_device: loader.CUdeviceptr, size_bytes: int) -> None:
    _retry_context(
        lambda: loader.memcpy_dtod(_as_device_ptr(dst_device), _as_device_ptr(src_device), int(size_bytes))
    )


def gpu_synchronize() -> None:
    _retry_context(loader.synchronize)


def launch_kernel(
    ptx_code: str | bytes,
    kernel_name: str,
    grid_size: int,
    block_size: int,
    shared_mem_size: int,
    kernel_params: list[Any],
) -> None:
    kernel = _cached_kernel(ptx_code, kernel_name)
    _retry_context(
        lambda: loader.launch(
            kernel,
            grid=(int(grid_size), 1, 1),
            block=(int(block_size), 1, 1),
            params=kernel_params,
            shared_mem=int(shared_mem_size),
        )
    )


def call_c_function(ptx_code: str | bytes, function_name: str, args: list[Any]) -> Any:
    """Compatibility host-side hooks for zero-copy manager helpers.

    The zero-copy manager is host control-plane logic, not a PTX launch surface.
    These functions therefore route to the sovereign loader's pinned-memory
    primitives rather than pretending to dispatch nonexistent GPU entrypoints.
    """

    del ptx_code  # The legacy API passes it through; the control-plane path ignores it.

    if function_name == "zero_copy_initialize":
        global _ZERO_COPY_MAX_BYTES
        _ZERO_COPY_MAX_BYTES = int(args[0])
        return True

    if function_name == "zero_copy_create_region":
        region_id = int(args[0])
        region_size = int(args[1])
        region_name = str(args[2])
        if _ZERO_COPY_MAX_BYTES and region_size > _ZERO_COPY_MAX_BYTES:
            raise ValueError(
                f"Requested zero-copy region {region_size} exceeds initialized limit {_ZERO_COPY_MAX_BYTES}"
            )
        owner: object | None = None
        try:
            ptr = _retry_context(lambda: loader.pinned_host_alloc(region_size))
        except RuntimeError:
            owner = (ctypes.c_ubyte * region_size)()
            ptr = ctypes.c_void_p(ctypes.addressof(owner))
        _ZERO_COPY_REGIONS[region_id] = (ptr, region_size, region_name, owner)
        return True

    if function_name == "zero_copy_get_ptr":
        region_id = int(args[0])
        offset = int(args[1])
        ptr, size, _name, _owner = _ZERO_COPY_REGIONS[region_id]
        if offset < 0 or offset >= size:
            raise ValueError(f"Offset {offset} outside region {region_id} size {size}")
        return int(ptr.value) + offset

    if function_name == "zero_copy_free":
        region_id = int(args[0])
        region = _ZERO_COPY_REGIONS.pop(region_id, None)
        if region is None:
            return False
        if region[3] is None:
            try:
                _retry_context(lambda: loader.pinned_host_free(region[0]))
            except RuntimeError:
                loader.cleanup()
        return True

    raise NotImplementedError(f"Unsupported control-plane function: {function_name}")


__all__ = [
    "call_c_function",
    "gpu_free",
    "gpu_malloc",
    "gpu_memcpy_dtoh",
    "gpu_memcpy_dtod",
    "gpu_memcpy_htod",
    "gpu_synchronize",
    "launch_kernel",
]
