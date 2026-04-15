from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import SwarmTickControl
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader
from tests._batch5_helpers import reference_assign


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


@pytest.fixture(scope="module")
def assign_kernel(tmp_path_factory: pytest.TempPathFactory):
    source = tmp_path_factory.mktemp("batch5_assign") / "batch5_assign_probe.cu"
    selector = (Path.cwd() / "knowledge3d/cranium/cuda/n_selector.cu").resolve()
    source.write_text(
        "\n".join(
            [
                '#include "' + str(selector) + '"',
                'extern "C" __global__ void batch5_assign_probe(const SwarmTickControl* control, uint32_t* out, uint32_t count) {',
                '  const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;',
                '  if (idx < count) {',
                '    out[idx] = swarm_assign_paradigm_slot(control, idx);',
                '  }',
                '}',
            ]
        ),
        encoding="utf-8",
    )
    ptx_text = compile_cuda_file(source, arch="sm_86", use_fast_math=False, extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"])
    ptx_path = tmp_path_factory.mktemp("batch5_assign_ptx") / "batch5_assign_probe.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    module = loader.load_module_from_file(str(ptx_path))
    return loader.get_function(module, "batch5_assign_probe")


@pytest.mark.parametrize("mask", [0b0, 0b10, 0b110, 0b11110])
def test_swarm_assign_paradigm_slot_matches_reference(mask: int, assign_kernel) -> None:
    control = SwarmTickControl()
    control.vram_free_mib = 4096
    control.t_remaining_us = 20_000
    control.n_cand_frustum = 16
    control.h_belief_q10 = 0
    control.n_floor = 4
    control.n_hard_max = 16
    control.sleep_calibration_n_hint = 0
    control.paradigm_mask = mask

    count = 32
    out = (ctypes.c_uint32 * count)()
    d_control = loader.gpu_malloc(ctypes.sizeof(control))
    d_out = loader.gpu_malloc(ctypes.sizeof(out))
    try:
        loader.memcpy_htod(d_control, ctypes.byref(control), ctypes.sizeof(control))
        loader.launch(
            assign_kernel,
            grid=(1, 1, 1),
            block=(count, 1, 1),
            params=[
                ctypes.c_uint64(d_control.value),
                ctypes.c_uint64(d_out.value),
                ctypes.c_uint32(count),
            ],
        )
        loader.synchronize()
        loader.memcpy_dtoh(ctypes.byref(out), d_out, ctypes.sizeof(out))
    finally:
        loader.gpu_free(d_control)
        loader.gpu_free(d_out)

    observed = [int(value) for value in out]
    expected = [reference_assign(mask, idx) for idx in range(count)]
    assert observed == expected
