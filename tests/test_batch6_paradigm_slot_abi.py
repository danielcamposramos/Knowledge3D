from __future__ import annotations

import ctypes
from pathlib import Path

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file


class ReasoningTickIO(ctypes.Structure):
    _fields_ = [
        ("galaxy_atlas", ctypes.c_uint64),
        ("phys_lane_id", ctypes.c_uint32),
        ("tick_seed", ctypes.c_uint32),
        ("paradigm_slot", ctypes.c_uint32),
        ("query_handle", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("ethical_trit", ctypes.c_int8),
        ("_pad0", ctypes.c_uint8 * 3),
    ]


class ReasoningLaneOutput(ctypes.Structure):
    _fields_ = [
        ("halt_flag", ctypes.c_uint32),
        ("result_handle", ctypes.c_uint32),
        ("belief_q15", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * 48),
    ]


def test_batch6_reasoning_abi_stays_byte_stable() -> None:
    assert ctypes.sizeof(ReasoningTickIO) == 32
    assert ctypes.sizeof(ReasoningLaneOutput) == 64


def test_batch6_reasoning_slot_constants_compile_to_expected_values(tmp_path: Path) -> None:
    header = (Path.cwd() / "knowledge3d/cranium/cuda/reasoning_tick_io.cuh").resolve()
    wrapper = tmp_path / "batch6_reasoning_slot_probe.cu"
    wrapper.write_text(
        "\n".join(
            [
                '#include "' + str(header) + '"',
                'extern "C" __global__ void batch6_reasoning_slot_probe(unsigned int* out) {',
                '  if (threadIdx.x != 0) return;',
                '  out[0] = sizeof(ReasoningTickIO);',
                '  out[1] = sizeof(ReasoningLaneOutput);',
                '  out[2] = REASONING_SLOT_TABLEAUX;',
                '  out[3] = REASONING_SLOT_RESOLUTION;',
                '  out[4] = REASONING_SLOT_ALPCHAIN;',
                '  out[5] = REASONING_SLOT_DPLL;',
                '  out[6] = REASONING_SLOT_CTX_SWITCH;',
                '  out[7] = REASONING_SLOT_SUBSUME;',
                '  out[8] = REASONING_SLOT_UNIFY;',
                '}',
            ]
        ),
        encoding="utf-8",
    )
    ptx = compile_cuda_file(
        wrapper,
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    assert "batch6_reasoning_slot_probe" in ptx

