from __future__ import annotations

import ctypes
import os

import pytest

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


STATE_STRIDE = 1040
STATE_COUNT = 18
ERROR_OFFSET = 8
STACK0_OFFSET = 16


def _pack_fact(predicate_mask: int, context_id: int, cluster_id: int, ethical_code: int) -> float:
    handle = (predicate_mask & 0xFF) | ((context_id & 0xFF) << 8) | ((cluster_id & 0xF) << 16) | ((ethical_code & 0x3) << 20)
    return float(handle)


def _pack_alpha(required_mask: int, required_context: int, required_cluster: int, ethical_policy: int) -> float:
    return _pack_fact(required_mask, required_context, required_cluster, ethical_policy)


def _pack_token(binding_mask: int, join_key: int, context_id: int, cluster_id: int) -> float:
    handle = (binding_mask & 0xFF) | ((join_key & 0xFF) << 8) | ((context_id & 0xF) << 16) | ((cluster_id & 0xF) << 20)
    return float(handle)


def _pack_activation(priority: int, payload: int, depth: int) -> float:
    handle = (priority & 0xFF) | ((payload & 0xFF) << 8) | ((depth & 0xFF) << 16)
    return float(handle)


def _unpack_token(value: float) -> tuple[int, int, int, int]:
    handle = int(round(value))
    return handle & 0xFF, (handle >> 8) & 0xFF, (handle >> 16) & 0xF, (handle >> 20) & 0xF


def _unpack_agenda(value: float) -> tuple[int, int, int]:
    handle = int(round(value))
    return handle & 0xFF, (handle >> 8) & 0xFF, (handle >> 16) & 0xFF


@pytest.fixture(scope="module")
def batch3_rete_kernel(tmp_path_factory: pytest.TempPathFactory):
    ptx_text = compile_cuda_file(
        "knowledge3d/cranium/kernels/modular_rpn_kernel.cu",
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    ptx_path = tmp_path_factory.mktemp("batch3_rete") / "modular_rpn_kernel_batch3_rete.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    module = loader.load_module_from_file(str(ptx_path))
    return loader.get_function(module, "modular_rpn_geometric_kernel")


def _run_scalar_program(kernel, opcodes: list[int], scalars: list[float]) -> tuple[int, float]:
    state_bytes = STATE_COUNT * STATE_STRIDE
    state_buffer = loader.gpu_malloc(state_bytes)
    try:
        zero_state = ctypes.create_string_buffer(state_bytes)
        loader.memcpy_htod(state_buffer, ctypes.cast(zero_state, ctypes.c_void_p), state_bytes)

        opcode_arr = (ctypes.c_uint16 * len(opcodes))(*opcodes)
        scalar_count = max(1, len(scalars))
        scalar_arr = (ctypes.c_float * scalar_count)(*([*scalars] if scalars else [0.0]))
        vector_arr = (ctypes.c_float * 3)(0.0, 0.0, 0.0)

        d_opcodes = loader.gpu_malloc(ctypes.sizeof(opcode_arr))
        d_scalars = loader.gpu_malloc(ctypes.sizeof(scalar_arr))
        d_vectors = loader.gpu_malloc(ctypes.sizeof(vector_arr))
        try:
            loader.memcpy_htod(d_opcodes, ctypes.cast(opcode_arr, ctypes.c_void_p), ctypes.sizeof(opcode_arr))
            loader.memcpy_htod(d_scalars, ctypes.cast(scalar_arr, ctypes.c_void_p), ctypes.sizeof(scalar_arr))
            loader.memcpy_htod(d_vectors, ctypes.cast(vector_arr, ctypes.c_void_p), ctypes.sizeof(vector_arr))

            loader.launch(
                kernel,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_uint32(0),
                    ctypes.c_uint64(d_opcodes.value),
                    ctypes.c_uint64(d_scalars.value),
                    ctypes.c_uint64(d_vectors.value),
                    ctypes.c_uint64(state_buffer.value),
                    ctypes.c_uint32(len(opcodes)),
                ],
            )
            loader.synchronize()

            error_host = ctypes.c_uint32()
            result_host = ctypes.c_float()
            loader.memcpy_dtoh(
                ctypes.byref(error_host),
                loader.CUdeviceptr(state_buffer.value + ERROR_OFFSET),
                ctypes.sizeof(error_host),
            )
            loader.memcpy_dtoh(
                ctypes.byref(result_host),
                loader.CUdeviceptr(state_buffer.value + STACK0_OFFSET),
                ctypes.sizeof(result_host),
            )
            return int(error_host.value), float(result_host.value)
        finally:
            loader.gpu_free(d_opcodes)
            loader.gpu_free(d_scalars)
            loader.gpu_free(d_vectors)
    finally:
        loader.gpu_free(state_buffer)


def test_rete_alpha_test_filters_by_mask_context_and_ethics(batch3_rete_kernel) -> None:
    fact = _pack_fact(0b0011, 7, 2, 1)
    alpha = _pack_alpha(0b0001, 7, 2, 1)
    error, result = _run_scalar_program(batch3_rete_kernel, [0x00, 0x00, 0xE0], [fact, alpha])
    assert error == 0
    binding_mask, join_key, context_id, cluster_id = _unpack_token(result)
    assert binding_mask == 0b0011
    assert join_key == 7
    assert context_id == 7
    assert cluster_id == 2


def test_rete_beta_join_emits_joined_token_only_when_both_sides_match(batch3_rete_kernel) -> None:
    left = _pack_token(0b0001, 9, 3, 1)
    right = _pack_token(0b0100, 9, 3, 1)
    error, result = _run_scalar_program(batch3_rete_kernel, [0x00, 0x00, 0xE1], [left, right])
    assert error == 0
    binding_mask, join_key, context_id, cluster_id = _unpack_token(result)
    assert binding_mask == 0b0101
    assert join_key == 9
    assert context_id == 3
    assert cluster_id == 1


def test_agenda_insert_preserves_priority_and_rejects_overflow(batch3_rete_kernel) -> None:
    activation = _pack_activation(12, 44, 31)
    error, result = _run_scalar_program(batch3_rete_kernel, [0x00, 0xE2], [activation])
    assert error == 0
    priority, payload, depth = _unpack_agenda(result)
    assert priority == 12
    assert payload == 44
    assert depth == 32

    overflow = _pack_activation(12, 44, 32)
    error, result = _run_scalar_program(batch3_rete_kernel, [0x00, 0xE2], [overflow])
    assert error == 0
    assert result == 0.0
