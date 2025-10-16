import ctypes
from typing import Iterable

import numpy as np
import pytest

from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.sovereign import loader


def _require_cuda() -> None:
    """Skip the test suite if a CUDA context cannot be created."""
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:  # pragma: no cover - hardware dependent
        pytest.skip(f"CUDA context unavailable: {exc}")


def _encode_pointer(ptr: ctypes.c_void_p, rows: int, cols: int) -> Iterable[float]:
    """Pack a device pointer + shape into four float32 scalars."""
    raw = int(ptr.value)
    lo = raw & 0xFFFFFFFF
    hi = (raw >> 32) & 0xFFFFFFFF
    lo_f = np.array([lo], dtype=np.uint32).view(np.float32)[0]
    hi_f = np.array([hi], dtype=np.uint32).view(np.float32)[0]
    return float(rows), float(cols), float(lo_f), float(hi_f)


def _upload(array: np.ndarray) -> ctypes.c_void_p:
    ptr = loader.gpu_malloc(array.nbytes)
    loader.memcpy_htod(ptr, array.ctypes.data_as(ctypes.c_void_p), array.nbytes)
    return ptr


def _download(ptr: ctypes.c_void_p, length: int) -> np.ndarray:
    host = np.zeros(length, dtype=np.float32)
    loader.memcpy_dtoh(host.ctypes.data_as(ctypes.c_void_p), ptr, host.nbytes)
    return host


def test_trm_vec_add3_opcode():
    _require_cuda()

    engine = AdvancedRPNEngine()
    engine.reset_instance(0)

    a = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    b = np.linspace(-1.0, 0.5, 512, dtype=np.float32)
    c = np.ones(512, dtype=np.float32) * 0.25
    dest = np.zeros(512, dtype=np.float32)

    ptr_a = _upload(a)
    ptr_b = _upload(b)
    ptr_c = _upload(c)
    ptr_dest = _upload(dest)

    try:
        op_codes = np.array([0x03, 0x03, 0x03, 0x03, 0x62], dtype=np.uint16)
        scalars = np.array(
            [
                *_encode_pointer(ptr_a, 512, 1),
                *_encode_pointer(ptr_b, 512, 1),
                *_encode_pointer(ptr_c, 512, 1),
                *_encode_pointer(ptr_dest, 512, 1),
            ],
            dtype=np.float32,
        )

        engine.execute_program(0, op_codes, scalars=scalars)

        result = _download(ptr_dest, 512)
        expected = a + b + c
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    finally:
        loader.gpu_free(ptr_a)
        loader.gpu_free(ptr_b)
        loader.gpu_free(ptr_c)
        loader.gpu_free(ptr_dest)


def test_trm_matvec_512x1024_opcode():
    _require_cuda()

    engine = AdvancedRPNEngine()
    engine.reset_instance(0)

    weights = np.arange(1024 * 512, dtype=np.float32).reshape(1024, 512) * 1e-4
    vector = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    dest = np.zeros(1024, dtype=np.float32)

    ptr_w = _upload(weights.ravel())
    ptr_v = _upload(vector)
    ptr_dest = _upload(dest)

    try:
        op_codes = np.array([0x03, 0x03, 0x03, 0x60], dtype=np.uint16)
        scalars = np.array(
            [
                *_encode_pointer(ptr_v, 512, 1),
                *_encode_pointer(ptr_w, 1024, 512),
                *_encode_pointer(ptr_dest, 1024, 1),
            ],
            dtype=np.float32,
        )

        engine.execute_program(0, op_codes, scalars=scalars)

        result = _download(ptr_dest, 1024)
        expected = weights @ vector
        np.testing.assert_allclose(result, expected, rtol=2e-3, atol=1e-3)
    finally:
        loader.gpu_free(ptr_w)
        loader.gpu_free(ptr_v)
        loader.gpu_free(ptr_dest)


def test_trm_matvec_1024x512_opcode():
    _require_cuda()

    engine = AdvancedRPNEngine()
    engine.reset_instance(0)

    weights = np.arange(512 * 1024, dtype=np.float32).reshape(512, 1024) * 5e-5
    vector = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    dest = np.zeros(512, dtype=np.float32)

    ptr_w = _upload(weights.ravel())
    ptr_v = _upload(vector)
    ptr_dest = _upload(dest)

    try:
        op_codes = np.array([0x03, 0x03, 0x03, 0x61], dtype=np.uint16)
        scalars = np.array(
            [
                *_encode_pointer(ptr_v, 1024, 1),
                *_encode_pointer(ptr_w, 512, 1024),
                *_encode_pointer(ptr_dest, 512, 1),
            ],
            dtype=np.float32,
        )

        engine.execute_program(0, op_codes, scalars=scalars)

        result = _download(ptr_dest, 512)
        expected = weights @ vector
        np.testing.assert_allclose(result, expected, rtol=2e-3, atol=1e-3)
    finally:
        loader.gpu_free(ptr_w)
        loader.gpu_free(ptr_v)
        loader.gpu_free(ptr_dest)


def test_trm_swiglu_1024_opcode():
    _require_cuda()

    engine = AdvancedRPNEngine()
    engine.reset_instance(0)

    vector = np.linspace(-3.0, 3.0, 1024, dtype=np.float32)
    dest = np.zeros(1024, dtype=np.float32)

    ptr_v = _upload(vector)
    ptr_dest = _upload(dest)

    try:
        op_codes = np.array([0x03, 0x03, 0x64], dtype=np.uint16)
        scalars = np.array(
            [
                *_encode_pointer(ptr_v, 1024, 1),
                *_encode_pointer(ptr_dest, 1024, 1),
            ],
            dtype=np.float32,
        )

        engine.execute_program(0, op_codes, scalars=scalars)

        result = _download(ptr_dest, 1024)
        sigmoid = 1.0 / (1.0 + np.exp(-vector))
        expected = vector * sigmoid
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    finally:
        loader.gpu_free(ptr_v)
        loader.gpu_free(ptr_dest)
