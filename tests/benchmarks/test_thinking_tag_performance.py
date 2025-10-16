from __future__ import annotations

import ctypes
import time

import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc
from knowledge3d.cranium.sovereign import loader


@pytest.mark.gpu
def test_thinking_tag_parallel_rpn_benchmark() -> None:
    """Benchmark ThinkingTag RPN bridge against a naïve legacy loop."""
    bridge = ThinkingTagRPNBridge()
    rng = np.random.default_rng(321)

    input_dim = 512
    hidden1 = 256
    hidden2 = 256
    output_dim = 100

    input_vec = rng.standard_normal(input_dim).astype(np.float32)
    weights = {
        "W1": rng.standard_normal((hidden1, input_dim)).astype(np.float32),
        "W2": rng.standard_normal((hidden2, hidden1)).astype(np.float32),
        "W3": rng.standard_normal((output_dim, hidden2)).astype(np.float32),
    }
    mask = np.clip(rng.random(hidden2, dtype=np.float32), 0.0, 1.0)

    # GPU warm-up
    for _ in range(5):
        bridge.execute_temporal(input_vec, weights, mask=mask)

    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        bridge.execute_temporal(input_vec, weights, mask=mask)
    gpu_ms = (time.perf_counter() - start) / runs * 1000

    def legacy_forward(x: np.ndarray, w: dict[str, np.ndarray], m: np.ndarray) -> np.ndarray:
        h1 = np.zeros(w["W1"].shape[0], dtype=np.float32)
        for i in range(w["W1"].shape[0]):
            acc = 0.0
            row = w["W1"][i]
            for j in range(row.shape[0]):
                acc += row[j] * x[j]
            h1[i] = acc if acc > 0 else 0.0

        h2 = np.zeros(w["W2"].shape[0], dtype=np.float32)
        for i in range(w["W2"].shape[0]):
            acc = 0.0
            row = w["W2"][i]
            for j in range(row.shape[0]):
                acc += row[j] * h1[j]
            acc *= m[i]
            h2[i] = acc if acc > 0 else 0.0

        out = np.zeros(w["W3"].shape[0], dtype=np.float32)
        for i in range(w["W3"].shape[0]):
            acc = 0.0
            row = w["W3"][i]
            for j in range(row.shape[0]):
                acc += row[j] * h2[j]
            out[i] = 1.0 / (1.0 + np.exp(-acc))
        return out

    legacy_forward(input_vec, weights, mask)
    start = time.perf_counter()
    for _ in range(3):
        legacy_forward(input_vec, weights, mask)
    legacy_ms = (time.perf_counter() - start) / 3 * 1000

    speedup = legacy_ms / gpu_ms if gpu_ms > 0 else float("inf")
    print(
        "\nThinkingTag temporal stage benchmark:\n"
        f"  GPU bridge:  {gpu_ms:.3f} ms\n"
        f"  Legacy loop: {legacy_ms:.3f} ms\n"
        f"  Speedup:     {speedup:.1f}×"
    )


@pytest.mark.gpu
def test_op_matvec_f32_latency_regression() -> None:
    """Ensure the optimized OP_MATVEC_F32 stays within the Phase 1C latency budget."""
    bridge = ThinkingTagRPNBridge()
    op_device = None
    scalars_device = None
    try:
        rng = np.random.default_rng(42)
        M, K = 256, 512
        matrix = rng.standard_normal((M, K)).astype(np.float32)
        vector = rng.standard_normal(K).astype(np.float32)

        matrix_tensor = bridge._upload_matrix(matrix)
        vector_ptr = bridge._upload_vector(vector)
        dest_ptr = bridge._get_vector_buffer(M)

        op_codes: list[int] = []
        scalars: list[float] = []

        def encode_pointer_literal(ptr_val: int, rows: int, cols: int = 1) -> tuple[float, float, float, float]:
            lo_bits = np.array([ptr_val & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]
            hi_bits = np.array([(ptr_val >> 32) & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]
            return float(rows), float(cols), float(lo_bits), float(hi_bits)

        def append_pointer(ptr: loader.CUdeviceptr, rows: int, cols: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(encode_pointer_literal(int(ptr.value), rows, cols))

        append_pointer(dest_ptr, M, 1)
        append_pointer(matrix_tensor.ptr, M, K)
        append_pointer(vector_ptr, K, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        op_device = loader.gpu_malloc(op_np.nbytes)
        scalars_device = loader.gpu_malloc(scalars_np.nbytes)
        loader.memcpy_htod(op_device, op_np.ctypes.data_as(ctypes.c_void_p), op_np.nbytes)
        loader.memcpy_htod(scalars_device, scalars_np.ctypes.data_as(ctypes.c_void_p), scalars_np.nbytes)

        bridge.engine.reset_instance(0)
        for _ in range(20):
            bridge.engine.execute_prebuilt(0, op_device, scalars_device, len(op_np))

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            bridge.engine.execute_prebuilt(0, op_device, scalars_device, len(op_np))
        avg_us = (time.perf_counter() - start) / iterations * 1e6

        print(f"\nOP_MATVEC_F32 ({M}x{K}) latency: {avg_us:.2f} us (target < 120 us)")
        assert avg_us < 120.0, f"OP_MATVEC_F32 latency {avg_us:.2f} us exceeds regression budget"
    finally:
        if op_device is not None:
            loader.gpu_free(op_device)
        if scalars_device is not None:
            loader.gpu_free(scalars_device)
        bridge.cleanup()
