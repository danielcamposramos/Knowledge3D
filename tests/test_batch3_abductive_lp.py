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


def _pack_horn_rule(head_symbol: int, body_mask: int, ic_mask: int = 0) -> float:
    handle = (head_symbol & 0xFF) | ((body_mask & 0xFF) << 8) | ((ic_mask & 0xFF) << 16)
    return float(handle)


@pytest.fixture(scope="module")
def batch3_alp_kernel(tmp_path_factory: pytest.TempPathFactory):
    ptx_text = compile_cuda_file(
        "knowledge3d/cranium/kernels/modular_rpn_kernel.cu",
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    ptx_path = tmp_path_factory.mktemp("batch3_abductive_lp") / "modular_rpn_kernel_batch3_alp.ptx"
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


def test_alpchain_resolves_goal_against_horn_rule(batch3_alp_kernel) -> None:
    goal_mask = float((1 << 3) | (1 << 0))
    horn_rule = _pack_horn_rule(3, (1 << 1) | (1 << 2))
    error, result = _run_scalar_program(batch3_alp_kernel, [0x00, 0x00, 0xB7], [goal_mask, horn_rule])
    assert error == 0
    assert int(result) == ((1 << 0) | (1 << 1) | (1 << 2))


def test_icheck_blocks_integrity_constraint_on_horn_surface(batch3_alp_kernel) -> None:
    assumption_pool = float((1 << 1) | (1 << 2))
    horn_rule_with_ic = _pack_horn_rule(3, (1 << 0), (1 << 2))
    error, result = _run_scalar_program(batch3_alp_kernel, [0x00, 0x00, 0xA5], [assumption_pool, horn_rule_with_ic])
    assert error == 0
    assert result == 0.0


def test_abdres_grows_assumption_pool_exactly_once(batch3_alp_kernel) -> None:
    residual_goal_mask = float((1 << 0) | (1 << 1) | (1 << 2))
    kb_support_mask = float((1 << 0) | (1 << 1))
    error, result = _run_scalar_program(batch3_alp_kernel, [0x00, 0x00, 0xA6], [residual_goal_mask, kb_support_mask])
    assert error == 0
    assert int(result) == (1 << 2)
