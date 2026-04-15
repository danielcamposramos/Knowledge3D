from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def _pack_activation(priority: int, payload: int, depth: int = 0) -> int:
    return (priority & 0xFF) | ((payload & 0xFF) << 8) | ((depth & 0xFF) << 16)


@pytest.fixture(scope="module")
def rete_agenda_kernel(tmp_path_factory: pytest.TempPathFactory):
    source = tmp_path_factory.mktemp("batch6_rete_agenda") / "batch6_rete_agenda_probe.cu"
    rete = (Path.cwd() / "knowledge3d/cranium/cuda/rpn_rete.cu").resolve()
    source.write_text(
        "\n".join(
            [
                '#include <math.h>',
                'struct StackValue { float x; float y; float z; float w; };',
                'static constexpr unsigned int kErrorNone = 0u;',
                '__device__ __forceinline__ bool pop_scalar(StackValue*, unsigned int&, float&, unsigned int&) { return false; }',
                '__device__ __forceinline__ void push(StackValue*, unsigned int&, StackValue, unsigned int&) {}',
                '__device__ __forceinline__ StackValue make_scalar(float value) { StackValue out = {value, 0.0f, 0.0f, 0.0f}; return out; }',
                '__device__ __forceinline__ bool rpn_context_allows_star(unsigned int active_context, unsigned int star_context) { return active_context == 0u || star_context == 0u || active_context == star_context; }',
                '#include "' + str(rete) + '"',
                'extern "C" __global__ void batch6_rete_agenda_probe(const uint32_t* activations, uint32_t* out, uint32_t* accepted, uint32_t count) {',
                '  if (threadIdx.x != 0 || blockIdx.x != 0) return;',
                '  uint32_t agenda[32];',
                '  uint32_t agenda_count = 0u;',
                '  uint32_t ok = 0u;',
                '  for (uint32_t i = 0u; i < 32u; ++i) agenda[i] = 0u;',
                '  for (uint32_t i = 0u; i < count; ++i) {',
                '    ok += k3d_rete_agenda_insert_top32(agenda, agenda_count, activations[i]) ? 1u : 0u;',
                '  }',
                '  for (uint32_t i = 0u; i < 32u; ++i) out[i] = agenda[i];',
                '  accepted[0] = ok;',
                '}',
            ]
        ),
        encoding="utf-8",
    )
    ptx_text = compile_cuda_file(source, arch="sm_86", use_fast_math=False, extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"])
    ptx_path = tmp_path_factory.mktemp("batch6_rete_agenda_ptx") / "batch6_rete_agenda_probe.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    module = loader.load_module_from_file(str(ptx_path))
    return loader.get_function(module, "batch6_rete_agenda_probe")


def test_rete_agenda_keeps_top32_in_priority_order(rete_agenda_kernel) -> None:
    activations = (_pack_activation(priority, priority) for priority in range(1, 65))
    host_in = (ctypes.c_uint32 * 64)(*activations)
    host_out = (ctypes.c_uint32 * 32)()
    host_accepted = (ctypes.c_uint32 * 1)()

    d_in = loader.gpu_malloc(ctypes.sizeof(host_in))
    d_out = loader.gpu_malloc(ctypes.sizeof(host_out))
    d_accepted = loader.gpu_malloc(ctypes.sizeof(host_accepted))
    try:
        loader.memcpy_htod(d_in, ctypes.byref(host_in), ctypes.sizeof(host_in))
        loader.memset_d32(d_out, 0, 32)
        loader.memset_d32(d_accepted, 0, 1)
        loader.launch(
            rete_agenda_kernel,
            grid=(1, 1, 1),
            block=(1, 1, 1),
            params=[
                ctypes.c_uint64(d_in.value),
                ctypes.c_uint64(d_out.value),
                ctypes.c_uint64(d_accepted.value),
                ctypes.c_uint32(64),
            ],
        )
        loader.synchronize()
        loader.memcpy_dtoh(ctypes.byref(host_out), d_out, ctypes.sizeof(host_out))
        loader.memcpy_dtoh(ctypes.byref(host_accepted), d_accepted, ctypes.sizeof(host_accepted))
    finally:
        loader.gpu_free(d_in)
        loader.gpu_free(d_out)
        loader.gpu_free(d_accepted)

    observed = [int(value) for value in host_out]
    expected_priorities = list(range(64, 32, -1))
    observed_priorities = [value & 0xFF for value in observed]
    assert host_accepted[0] == 64
    assert observed_priorities == expected_priorities
    assert all(((value >> 8) & 0xFF) == (value & 0xFF) for value in observed)
