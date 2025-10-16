import ctypes
import numpy as np
import pytest

from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc


def _encode_pointer(ptr_value: int, rows: int, cols: int = 1) -> list[float]:
    """Encode a 64-bit device pointer into float components expected by OP_POINTER_LITERAL."""
    lo_bits = np.array([ptr_value & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]
    hi_bits = np.array([(ptr_value >> 32) & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]
    return [float(rows), float(cols), lo_bits, hi_bits]

def _make_engine() -> AdvancedRPNEngine:
    try:
        return AdvancedRPNEngine()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"GPU path unavailable: {exc}")


def test_tier2_memcpy_cooperative():
    engine = _make_engine()
    length = 256
    host_src = np.linspace(0.0, 1.0, length, dtype=np.float32)
    host_dst = np.zeros_like(host_src)

    d_src = loader.gpu_malloc(host_src.nbytes)
    d_dst = loader.gpu_malloc(host_dst.nbytes)

    try:
        loader.memcpy_htod(
            d_src,
            host_src.ctypes.data_as(ctypes.c_void_p),
            host_src.nbytes,
        )

        op_codes = np.array(
            [
                ropc.OP_POINTER_LITERAL,
                ropc.OP_POINTER_LITERAL,
                ropc.OP_MEMCPY_F32,
            ],
            dtype=np.uint16,
        )
        scalars = np.array(
            _encode_pointer(int(d_dst.value), length)
            + _encode_pointer(int(d_src.value), length),
            dtype=np.float32,
        )

        engine.reset_instance(0)
        engine.execute_program(0, op_codes, scalars)

        loader.memcpy_dtoh(
            host_dst.ctypes.data_as(ctypes.c_void_p),
            d_dst,
            host_dst.nbytes,
        )

        np.testing.assert_allclose(host_dst, host_src, rtol=1e-5, atol=1e-6)
    finally:
        loader.gpu_free(d_src)
        loader.gpu_free(d_dst)
        engine.cleanup()


def test_tier2_fill_cooperative():
    engine = _make_engine()
    length = 128
    fill_value = 3.14159
    host_result = np.zeros(length, dtype=np.float32)

    d_tensor = loader.gpu_malloc(host_result.nbytes)

    try:
        op_codes = np.array(
            [
                ropc.OP_POINTER_LITERAL,
                0x0000,  # scalar literal
                ropc.OP_FILL_F32,
            ],
            dtype=np.uint16,
        )
        scalars = np.array(
            _encode_pointer(int(d_tensor.value), length) + [fill_value],
            dtype=np.float32,
        )

        engine.reset_instance(0)
        engine.execute_program(0, op_codes, scalars)

        loader.memcpy_dtoh(
            host_result.ctypes.data_as(ctypes.c_void_p),
            d_tensor,
            host_result.nbytes,
        )
        np.testing.assert_allclose(
            host_result,
            np.full_like(host_result, fill_value),
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        loader.gpu_free(d_tensor)
        engine.cleanup()


def test_tier2_reduce_sum_and_max():
    engine = _make_engine()
    length = 512
    host_data = np.random.default_rng(123).standard_normal(length).astype(np.float32)

    d_tensor = loader.gpu_malloc(host_data.nbytes)

    try:
        loader.memcpy_htod(
            d_tensor,
            host_data.ctypes.data_as(ctypes.c_void_p),
            host_data.nbytes,
        )

        pointer_scalars = np.array(
            _encode_pointer(int(d_tensor.value), length),
            dtype=np.float32,
        )

        # Reduce sum
        engine.reset_instance(0)
        sum_stack = engine.execute_program(
            0,
            np.array([ropc.OP_POINTER_LITERAL, ropc.OP_REDUCE_SUM_F32], dtype=np.uint16),
            pointer_scalars,
        )
        assert sum_stack.shape[0] >= 1
        sum_result = float(sum_stack[sum_stack.shape[0] - 1, 0])
        np.testing.assert_allclose(sum_result, host_data.sum(), rtol=1e-5, atol=1e-5)

        # Reduce max
        engine.reset_instance(0)
        max_stack = engine.execute_program(
            0,
            np.array([ropc.OP_POINTER_LITERAL, ropc.OP_REDUCE_MAX_F32], dtype=np.uint16),
            pointer_scalars,
        )
        assert max_stack.shape[0] >= 1
        max_result = float(max_stack[max_stack.shape[0] - 1, 0])
        np.testing.assert_allclose(max_result, host_data.max(), rtol=1e-5, atol=1e-5)
    finally:
        loader.gpu_free(d_tensor)
        engine.cleanup()


def test_tier2_matvec_sigmoid_entropy():
    engine = _make_engine()
    rng = np.random.default_rng(42)
    input_dim = 8
    hidden_dim = 6

    host_input = rng.standard_normal(input_dim).astype(np.float32)
    host_weight = rng.standard_normal((hidden_dim, input_dim)).astype(np.float32)
    host_output = np.zeros(hidden_dim, dtype=np.float32)

    d_input = loader.gpu_malloc(host_input.nbytes)
    d_weight = loader.gpu_malloc(host_weight.nbytes)
    d_output = loader.gpu_malloc(host_output.nbytes)

    try:
        loader.memcpy_htod(d_input, host_input.ctypes.data_as(ctypes.c_void_p), host_input.nbytes)
        loader.memcpy_htod(d_weight, host_weight.ctypes.data_as(ctypes.c_void_p), host_weight.nbytes)

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows: int, cols: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer(int(ptr.value), rows, cols))

        append_pointer(d_output, hidden_dim, 1)
        append_pointer(d_weight, hidden_dim, input_dim)
        append_pointer(d_input, input_dim, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_output, hidden_dim, 1)
        op_codes.append(ropc.OP_VECTOR_SIGMOID)

        append_pointer(d_output, hidden_dim, 1)
        op_codes.append(ropc.OP_ENTROPY_SUM)

        op_codes_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        engine.reset_instance(0)
        stack = engine.execute_program(0, op_codes_np, scalars_np)

        loader.memcpy_dtoh(
            host_output.ctypes.data_as(ctypes.c_void_p),
            d_output,
            host_output.nbytes,
        )

        expected = 1.0 / (1.0 + np.exp(-host_weight @ host_input))
        np.testing.assert_allclose(host_output, expected, rtol=1e-5, atol=1e-5)

        entropy_stack = float(stack[-1, 0])
        entropy_expected = float(-np.sum(np.clip(expected, 1e-6, 1.0) * np.log(np.clip(expected, 1e-6, 1.0))))
        np.testing.assert_allclose(entropy_stack, entropy_expected, rtol=1e-5, atol=1e-5)
    finally:
        loader.gpu_free(d_input)
        loader.gpu_free(d_weight)
        loader.gpu_free(d_output)
        engine.cleanup()


def test_tier2_vector_relu_and_mul():
    engine = _make_engine()
    rng = np.random.default_rng(7)
    length = 32
    host_vec = rng.standard_normal(length).astype(np.float32)
    host_mask = rng.random(length, dtype=np.float32)

    d_vec = loader.gpu_malloc(host_vec.nbytes)
    d_mask = loader.gpu_malloc(host_mask.nbytes)

    try:
        loader.memcpy_htod(d_vec, host_vec.ctypes.data_as(ctypes.c_void_p), host_vec.nbytes)
        loader.memcpy_htod(d_mask, host_mask.ctypes.data_as(ctypes.c_void_p), host_mask.nbytes)

        op_codes = []
        scalars = []

        def append(ptr):
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer(int(ptr.value), length))

        append(d_vec)
        op_codes.append(ropc.OP_VECTOR_RELU)

        append(d_vec)
        append(d_mask)
        op_codes.append(ropc.OP_VECTOR_MUL_F32)

        op_codes_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        engine.reset_instance(0)
        engine.execute_program(0, op_codes_np, scalars_np)

        result = np.zeros(length, dtype=np.float32)
        loader.memcpy_dtoh(result.ctypes.data_as(ctypes.c_void_p), d_vec, result.nbytes)

        expected = np.maximum(host_vec, 0.0) * host_mask
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    finally:
        loader.gpu_free(d_vec)
        loader.gpu_free(d_mask)
        engine.cleanup()
