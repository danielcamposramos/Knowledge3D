from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

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


def test_reasoning_tick_io_ctypes_layout_matches_spec() -> None:
    assert ctypes.sizeof(ReasoningTickIO) == 32
    assert ctypes.sizeof(ReasoningLaneOutput) == 64


def test_reasoning_tick_io_header_compiles_under_nvcc(tmp_path: Path) -> None:
    header = (Path.cwd() / "knowledge3d/cranium/cuda/reasoning_tick_io.cuh").resolve()
    wrapper = tmp_path / "batch5_reasoning_tick_io_probe.cu"
    wrapper.write_text(
        "\n".join(
            [
                '#include "' + str(header) + '"',
                'extern "C" __global__ void batch5_reasoning_tick_io_probe(ReasoningLaneOutput* out) {',
                '  const ReasoningTickIO io = {nullptr, 0u, 0u, REASONING_SLOT_CBR, 0u, 0u, 0, {0u, 0u, 0u}};',
                '  if (threadIdx.x == 0) {',
                '    out[0].halt_flag = sizeof(io);',
                '    out[0].result_handle = sizeof(ReasoningLaneOutput);',
                '  }',
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
    assert "batch5_reasoning_tick_io_probe" in ptx
