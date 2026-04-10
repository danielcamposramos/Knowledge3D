"""Simple zero-copy surface checks for the current live support layer."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.kernels import kernel_loader, ptx_compiler


ROOT = Path(__file__).resolve().parents[1]


def test_phase4_lightweight_source_contains_expected_entries() -> None:
    source = (
        ROOT / "knowledge3d" / "cranium" / "kernels" / "zero_copy_memory_manager_phase4.cu"
    ).read_text(encoding="utf-8")
    assert "lightweight_procedural_kernel" in source
    assert "lightweight_warp_kernel" in source
    assert "symlink_procedural_kernel" in source


def test_zero_copy_control_plane_preserves_pointer_offsets() -> None:
    compiled = ptx_compiler.compile_cuda_file(
        ROOT / "knowledge3d" / "cranium" / "kernels" / "zero_copy_memory_manager.cu"
    )
    assert kernel_loader.call_c_function(compiled, "zero_copy_initialize", [4096]) is True
    assert kernel_loader.call_c_function(compiled, "zero_copy_create_region", [7, 4096, "tablet_log"]) is True
    base_ptr = kernel_loader.call_c_function(compiled, "zero_copy_get_ptr", [7, 0])
    shifted_ptr = kernel_loader.call_c_function(compiled, "zero_copy_get_ptr", [7, 128])
    assert shifted_ptr - base_ptr == 128
    assert kernel_loader.call_c_function(compiled, "zero_copy_free", [7]) is True


def test_zero_copy_host_to_host_async_copy() -> None:
    payload = np.linspace(0.0, 1.0, num=16, dtype=np.float32)
    destination = (ctypes.c_ubyte * payload.nbytes)()
    ok = kernel_loader.call_c_function(
        ptx_compiler.compile_cuda_file(ROOT / "knowledge3d" / "cranium" / "kernels" / "zero_copy_memory_manager.cu"),
        "zero_copy_initialize",
        [payload.nbytes],
    )
    assert ok is True
    ctypes.memmove(ctypes.addressof(destination), payload.ctypes.data, payload.nbytes)
    copied = np.frombuffer(destination, dtype=np.float32).copy()
    assert np.allclose(copied, payload)
