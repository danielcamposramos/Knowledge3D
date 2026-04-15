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


def _pack_opinion(belief: int, disbelief: int, uncertainty: int, status: int = 0) -> float:
    handle = (belief & 0x7F) | ((disbelief & 0x7F) << 7) | ((uncertainty & 0x7F) << 14) | ((status & 0x3) << 21)
    return float(handle)


def _unpack_opinion(value: float) -> tuple[int, int, int, int]:
    handle = int(round(value))
    belief = handle & 0x7F
    disbelief = (handle >> 7) & 0x7F
    uncertainty = (handle >> 14) & 0x7F
    status = (handle >> 21) & 0x3
    return belief, disbelief, uncertainty, status


@pytest.fixture(scope="module")
def batch3_subjective_kernel(tmp_path_factory: pytest.TempPathFactory):
    ptx_text = compile_cuda_file(
        "knowledge3d/cranium/kernels/modular_rpn_kernel.cu",
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    ptx_path = tmp_path_factory.mktemp("batch3_subjective_logic") / "modular_rpn_kernel_batch3_subjective.ptx"
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


def test_ebelief_shifts_belief_toward_supported_hypothesis(batch3_subjective_kernel) -> None:
    prior = _pack_opinion(32, 16, 64)
    error, result = _run_scalar_program(batch3_subjective_kernel, [0x00, 0x00, 0x00, 0xB0], [prior, 24.0, 0.0])
    assert error == 0
    belief, disbelief, uncertainty, _ = _unpack_opinion(result)
    assert belief > 32
    assert disbelief <= 16
    assert uncertainty < 64


def test_ebelief_uncertainty_shrinks_with_consistent_evidence(batch3_subjective_kernel) -> None:
    prior = _pack_opinion(24, 8, 80)
    error, result = _run_scalar_program(batch3_subjective_kernel, [0x00, 0x00, 0x00, 0xB0], [prior, 12.0, 0.0])
    assert error == 0
    _, _, uncertainty, _ = _unpack_opinion(result)
    assert uncertainty < 80


def test_ebelief_contradictory_evidence_stays_defeasible(batch3_subjective_kernel) -> None:
    prior = _pack_opinion(72, 8, 24)
    error, result = _run_scalar_program(batch3_subjective_kernel, [0x00, 0x00, 0x00, 0xB0], [prior, 20.0, 18.0])
    assert error == 0
    belief, disbelief, uncertainty, status = _unpack_opinion(result)
    assert status == 2
    assert disbelief > 8
    assert uncertainty >= 16
    assert belief < 96
