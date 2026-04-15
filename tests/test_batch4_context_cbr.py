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


def _pack_alpha(required_mask: int, required_context: int, required_cluster: int, ethical_policy: int, heuristic_floor: int) -> float:
    handle = int(_pack_fact(required_mask, required_context, required_cluster, ethical_policy))
    handle |= (heuristic_floor & 0x3) << 22
    return float(handle)


def _pack_case(case_id: int, anchor: int, context_id: int, ethical_code: int, flags: int = 0) -> float:
    handle = (case_id & 0x3F)
    handle |= (anchor & 0xFF) << 6
    handle |= (context_id & 0x3F) << 14
    handle |= (ethical_code & 0x3) << 20
    handle |= (flags & 0x3) << 22
    return float(handle)


def _pack_rebind(symbol_mask: int, anchor_bias: int, context_override: int, flags: int) -> float:
    handle = (symbol_mask & 0xFF)
    handle |= (anchor_bias & 0xFF) << 8
    handle |= (context_override & 0x3F) << 16
    handle |= (flags & 0x3) << 22
    return float(handle)


def _pack_constraint(anchor_floor: int, revise_delta: int, required_context: int, ethical_policy: int, conflict_code: int) -> float:
    handle = (anchor_floor & 0xFF)
    handle |= (revise_delta & 0x3F) << 8
    handle |= (required_context & 0x3F) << 14
    handle |= (ethical_policy & 0x3) << 20
    handle |= (conflict_code & 0x3) << 22
    return float(handle)


def _unpack_token(value: float) -> tuple[int, int, int, int]:
    handle = int(round(value))
    return handle & 0xFF, (handle >> 8) & 0xFF, (handle >> 16) & 0xF, (handle >> 20) & 0xF


def _unpack_case(value: float) -> tuple[int, int, int, int, int]:
    handle = int(round(value))
    return handle & 0x3F, (handle >> 6) & 0xFF, (handle >> 14) & 0x3F, (handle >> 20) & 0x3, (handle >> 22) & 0x3


@pytest.fixture(scope="module")
def batch4_kernel(tmp_path_factory: pytest.TempPathFactory):
    ptx_text = compile_cuda_file(
        "knowledge3d/cranium/kernels/modular_rpn_kernel.cu",
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    ptx_path = tmp_path_factory.mktemp("batch4_context_cbr") / "modular_rpn_kernel_batch4.ptx"
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
        scalar_arr = (ctypes.c_float * max(1, len(scalars)))(*([*scalars] if scalars else [0.0]))
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
            loader.memcpy_dtoh(ctypes.byref(error_host), loader.CUdeviceptr(state_buffer.value + ERROR_OFFSET), ctypes.sizeof(error_host))
            loader.memcpy_dtoh(ctypes.byref(result_host), loader.CUdeviceptr(state_buffer.value + STACK0_OFFSET), ctypes.sizeof(result_host))
            return int(error_host.value), float(result_host.value)
        finally:
            loader.gpu_free(d_opcodes)
            loader.gpu_free(d_scalars)
            loader.gpu_free(d_vectors)
    finally:
        loader.gpu_free(state_buffer)

def test_ctx_switch_accepts_matching_context_with_heuristic_filter(batch4_kernel) -> None:
    fact = _pack_fact(0b0011, 7, 2, 1)
    alpha = _pack_alpha(0b0001, 0, 2, 1, 2)
    error, result = _run_scalar_program(batch4_kernel, [0x00, 0xB6, 0x00, 0x00, 0xE0], [7.0, fact, alpha])
    assert error == 0
    binding_mask, join_key, context_id, cluster_id = _unpack_token(result)
    assert binding_mask == 0b0011
    assert join_key == 7
    assert context_id == 7
    assert cluster_id == 2


def test_ctx_switch_keeps_global_visible_and_rejects_mismatch(batch4_kernel) -> None:
    global_fact = _pack_fact(0b0001, 0, 2, 1)
    mismatch_fact = _pack_fact(0b0001, 5, 2, 1)
    alpha = _pack_alpha(0b0001, 0, 2, 1, 1)

    error, result = _run_scalar_program(batch4_kernel, [0x00, 0xB6, 0x00, 0x00, 0xE0], [9.0, global_fact, alpha])
    assert error == 0
    assert int(round(result)) != 0

    error, result = _run_scalar_program(batch4_kernel, [0x00, 0xB6, 0x00, 0x00, 0xE0], [9.0, mismatch_fact, alpha])
    assert error == 0
    assert result == 0.0


def test_case_fetch_returns_nearest_valid_case_handle(batch4_kernel) -> None:
    query = _pack_case(33, 120, 7, 1)
    far_case = _pack_case(3, 40, 7, 1)
    near_case = _pack_case(7, 122, 7, 1)
    error, result = _run_scalar_program(batch4_kernel, [0x00, 0x00, 0x00, 0x100], [query, far_case, near_case])
    assert error == 0
    case_id, anchor, context_id, ethical_code, flags = _unpack_case(result)
    assert case_id == 7
    assert anchor == 122
    assert context_id == 7
    assert ethical_code == 1
    assert flags == 0


def test_case_rebind_rewrites_symbols_without_mutating_source_case(batch4_kernel) -> None:
    source_case = _pack_case(11, 40, 7, 1, 2)
    rebind = _pack_rebind(0x0F, 5, 9, 3)
    error, result = _run_scalar_program(batch4_kernel, [0x00, 0x00, 0x101], [source_case, rebind])
    assert error == 0
    src = _unpack_case(source_case)
    rebound = _unpack_case(result)
    assert src == (11, 40, 7, 1, 2)
    assert rebound != src
    assert rebound[2] == 9
    assert rebound[3] == 1


def test_case_revise_and_retain_hint_follow_constraints(batch4_kernel) -> None:
    bound_case = _pack_case(8, 90, 7, 1, 0)
    constraint = _pack_constraint(80, 5, 7, 1, 0)
    error, revised = _run_scalar_program(batch4_kernel, [0x00, 0x00, 0x102], [bound_case, constraint])
    assert error == 0
    case_id, anchor, context_id, ethical_code, flags = _unpack_case(revised)
    assert case_id == 8
    assert anchor == 95
    assert context_id == 7
    assert ethical_code == 1
    assert flags == 1

    error, retain_hint = _run_scalar_program(batch4_kernel, [0x00, 0x103], [revised])
    assert error == 0
    hint = _unpack_case(retain_hint)
    assert hint[0] == 8
    assert hint[4] == 2


def test_casuistry_gate_rejects_forbidden_and_accepts_defeasible_with_conflict_resolution(batch4_kernel) -> None:
    forbidden_case = _pack_case(5, 88, 3, 0, 0)
    safe_constraint = _pack_constraint(40, 1, 3, 0, 2)
    error, result = _run_scalar_program(batch4_kernel, [0x00, 0x00, 0x102], [forbidden_case, safe_constraint])
    assert error == 0
    assert result == 0.0

    defeasible_case = _pack_case(6, 88, 3, 2, 0)
    error, result = _run_scalar_program(batch4_kernel, [0x00, 0x00, 0x102], [defeasible_case, safe_constraint])
    assert error == 0
    case_id, anchor, context_id, ethical_code, flags = _unpack_case(result)
    assert case_id == 6
    assert anchor == 89
    assert context_id == 3
    assert ethical_code == 2
    assert flags == 1
