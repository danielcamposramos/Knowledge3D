"""Host bridge for zero-copy pinned memory primitives.

This is a control-plane helper for ingestion and test scaffolding. It wraps the
sovereign loader's pinned memory APIs; it is not part of the reasoning hot path.
"""

from __future__ import annotations

import ctypes

from knowledge3d.cranium.sovereign import loader

_HOST_BUFFER_FALLBACKS: dict[int, object] = {}


def cuMemHostAlloc_wrapper(size_bytes: int) -> ctypes.c_void_p | None:
    try:
        return loader.pinned_host_alloc(int(size_bytes))
    except Exception:
        owner = (ctypes.c_ubyte * max(1, int(size_bytes)))()
        ptr = ctypes.c_void_p(ctypes.addressof(owner))
        _HOST_BUFFER_FALLBACKS[int(ptr.value)] = owner
        return ptr


def cuMemFreeHost_wrapper(ptr: ctypes.c_void_p) -> bool:
    try:
        key = int(ptr.value or 0)
        owner = _HOST_BUFFER_FALLBACKS.pop(key, None)
        if owner is None:
            loader.pinned_host_free(ptr)
        return True
    except Exception:
        return False


def zero_copy_memcpy_async(dst_ptr, src_ptr, size_bytes: int) -> bool:
    size = int(size_bytes)
    try:
        if isinstance(dst_ptr, ctypes.c_void_p) and isinstance(src_ptr, ctypes.c_void_p):
            ctypes.memmove(dst_ptr, src_ptr, size)
            return True
        if isinstance(dst_ptr, ctypes.c_void_p):
            loader.memcpy_dtoh(dst_ptr, loader.CUdeviceptr(int(src_ptr)), size)
            return True
        if isinstance(src_ptr, ctypes.c_void_p):
            loader.memcpy_htod(loader.CUdeviceptr(int(dst_ptr)), src_ptr, size)
            return True
        loader.memcpy_dtod(loader.CUdeviceptr(int(dst_ptr)), loader.CUdeviceptr(int(src_ptr)), size)
        return True
    except Exception:
        return False


__all__ = [
    "cuMemFreeHost_wrapper",
    "cuMemHostAlloc_wrapper",
    "zero_copy_memcpy_async",
]
