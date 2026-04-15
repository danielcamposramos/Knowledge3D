from __future__ import annotations

import ctypes
import os

import pytest

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader
from knowledge3d.ingestion.star_crafter import build_foundational_star_crafter_outputs


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


STATE_STRIDE = 1040
STATE_COUNT = 18
ERROR_OFFSET = 8
STACK0_OFFSET = 16

TERM_F = 1
TERM_G = 2
TERM_H = 3
TERM_A = 7


def _pack_rule(lhs: int, rhs: int) -> float:
    return float((lhs & 0xFFF) | ((rhs & 0xFFF) << 12))


@pytest.fixture(scope="module")
def batch3_kernel(tmp_path_factory: pytest.TempPathFactory):
    ptx_text = compile_cuda_file(
        "knowledge3d/cranium/kernels/modular_rpn_kernel.cu",
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    ptx_path = tmp_path_factory.mktemp("batch3_superposition") / "modular_rpn_kernel_batch3.ptx"
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


def test_star_crafter_contains_reasoning_kbo_precedence_star() -> None:
    rows = build_foundational_star_crafter_outputs()
    by_id = {str(row.get("id")): row for row in rows}
    assert "reasoning_kbo_precedence" in by_id
    metadata = dict(by_id["reasoning_kbo_precedence"].get("metadata") or {})
    assert metadata.get("precedence_ranks", {}).get("f") > metadata.get("precedence_ranks", {}).get("g")
    assert metadata.get("symbol_ids", {}).get("a") == TERM_A


def test_torder_compare_uses_precedence_surface(batch3_kernel) -> None:
    error, result = _run_scalar_program(batch3_kernel, [0x00, 0x00, 0xC2], [float(TERM_F), float(TERM_G)])
    assert error == 0
    assert result == 1.0


def test_trewrite_only_fires_when_ordered(batch3_kernel) -> None:
    rule_fg = _pack_rule(TERM_F, TERM_G)
    error, result = _run_scalar_program(batch3_kernel, [0x00, 0x00, 0xC5], [float(TERM_F), rule_fg])
    assert error == 0
    assert result == float(TERM_G)


def test_trewrite_rejects_non_oriented_rule(batch3_kernel) -> None:
    rule_gf = _pack_rule(TERM_G, TERM_F)
    error, result = _run_scalar_program(batch3_kernel, [0x00, 0x00, 0xC5], [float(TERM_G), rule_gf])
    assert error == 0
    assert result == 0.0


def test_tsuperpos_emits_critical_pair_handle_for_overlap(batch3_kernel) -> None:
    rule_fg = _pack_rule(TERM_F, TERM_G)
    error, result = _run_scalar_program(batch3_kernel, [0x00, 0x00, 0xC4], [rule_fg, float(TERM_F)])
    assert error == 0
    assert result > 0.0


def test_tsuperpos_returns_zero_for_non_overlapping_target(batch3_kernel) -> None:
    rule_fg = _pack_rule(TERM_F, TERM_G)
    error, result = _run_scalar_program(batch3_kernel, [0x00, 0x00, 0xC4], [rule_fg, float(TERM_A)])
    assert error == 0
    assert result == 0.0
