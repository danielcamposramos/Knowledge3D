"""Zero-copy kernel/control-plane regression tests.

These tests cover the live Python/kernel support surface that the recent KIMI
audit revived: PTX compilation, region lifecycle, and host-side zero-copy
buffer helpers. The raw updater kernels remain tracked separately while the
driver-side illegal-memory-access issue is resolved.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.kernels import kernel_loader, ptx_compiler
from knowledge3d.cranium.kernels.zero_copy_memory_manager import (
    cuMemFreeHost_wrapper,
    cuMemHostAlloc_wrapper,
    zero_copy_memcpy_async,
)


ROOT = Path(__file__).resolve().parents[1]


def test_zero_copy_ptx_compilation_contains_expected_entries() -> None:
    ptx = ptx_compiler.compile_cuda_file(
        ROOT / "knowledge3d" / "cranium" / "kernels" / "galaxy_memory_updater_zero_copy.cu"
    )
    assert "update_star_embedding_kernel_zero_copy" in ptx
    assert "update_star_embedding_kernel_warp_level" in ptx
    assert "update_star_embedding_kernel_bank_optimized" in ptx


def test_zero_copy_memory_manager_source_compiles_for_control_plane() -> None:
    compiled = ptx_compiler.compile_cuda_file(
        ROOT / "knowledge3d" / "cranium" / "kernels" / "zero_copy_memory_manager.cu"
    )
    assert "zero_copy_alloc" in compiled
    assert "zero_copy_free" in compiled


def test_zero_copy_region_lifecycle() -> None:
    control_plane = ptx_compiler.compile_cuda_file(
        ROOT / "knowledge3d" / "cranium" / "kernels" / "zero_copy_memory_manager.cu"
    )

    assert kernel_loader.call_c_function(control_plane, "zero_copy_initialize", [1024 * 1024]) is True
    assert kernel_loader.call_c_function(
        control_plane, "zero_copy_create_region", [2, 64 * 1024, "test_galaxy_region"]
    ) is True

    base_ptr = kernel_loader.call_c_function(control_plane, "zero_copy_get_ptr", [2, 0])
    offset_ptr = kernel_loader.call_c_function(control_plane, "zero_copy_get_ptr", [2, 256])

    assert base_ptr != 0
    assert offset_ptr - base_ptr == 256
    assert kernel_loader.call_c_function(control_plane, "zero_copy_free", [2]) is True


def test_zero_copy_host_buffer_roundtrip() -> None:
    payload = np.arange(128, dtype=np.uint8)
    ptr = cuMemHostAlloc_wrapper(payload.nbytes)
    assert ptr is not None
    try:
        assert zero_copy_memcpy_async(ptr, ctypes.c_void_p(payload.ctypes.data), payload.nbytes) is True
        view = (ctypes.c_uint8 * payload.nbytes).from_address(int(ptr.value))
        roundtrip = np.ctypeslib.as_array(view).copy()
        assert np.array_equal(roundtrip, payload)
    finally:
        assert cuMemFreeHost_wrapper(ptr) is True
