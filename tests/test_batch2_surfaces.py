from __future__ import annotations

import os
import ctypes

import pytest

from knowledge3d.ingestion.star_crafter import build_foundational_star_crafter_outputs
from knowledge3d.cranium.bridges.sleep_perf_consumer_bridge import (
    SleepPerfConsumerBridge,
    SwarmPerfCalibration,
)
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_sleep_perf_consumer_kernel_compiles() -> None:
    ptx = compile_cuda_file(
        "knowledge3d/cranium/cuda/sleep_perf_consumer.cu",
        arch="sm_86",
        use_fast_math=False,
    )
    assert ".entry k3d_sleep_perf_consume" in ptx


def test_modular_rpn_kernel_compiles_with_halt_opcodes() -> None:
    ptx = compile_cuda_file(
        "knowledge3d/cranium/kernels/modular_rpn_kernel.cu",
        arch="sm_86",
        use_fast_math=False,
        extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
    )
    assert "modular_rpn_geometric_kernel" in ptx


def test_star_crafter_contains_swarm_perf_calibration_star() -> None:
    rows = build_foundational_star_crafter_outputs()
    by_id = {str(row.get("id")): row for row in rows}
    assert "swarm_perf_calibration" in by_id
    row = by_id["swarm_perf_calibration"]
    metadata = dict(row.get("metadata") or {})
    assert metadata.get("calibration_schema", {}).get("n_hint") == "u32"


class _LanePerf(ctypes.Structure):
    _fields_ = [
        ("n_active", ctypes.c_uint32),
        ("entropy_input", ctypes.c_uint32),
        ("belief_delta", ctypes.c_float),
        ("cycles_consumed", ctypes.c_uint32),
        ("specialist_id", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
        ("_tail_pad", ctypes.c_uint32 * 3),
    ]


def test_sleep_perf_consumer_aggregates_synthetic_ring() -> None:
    bridge = SleepPerfConsumerBridge()
    ring = (_LanePerf * 4)()
    ring[0] = _LanePerf(5, 11, ctypes.c_float(0.95), 10, 0, (0, 0, 0))
    ring[1] = _LanePerf(5, 12, ctypes.c_float(0.90), 10, 1, (0, 0, 0))
    ring[2] = _LanePerf(2, 9, ctypes.c_float(0.10), 10, 2, (0, 0, 0))
    ring[3] = _LanePerf(3, 7, ctypes.c_float(0.20), 10, 3, (0, 0, 0))

    d_ring = loader.gpu_malloc(ctypes.sizeof(ring))
    calibration_host, calibration_device = loader.mapped_host_alloc(ctypes.sizeof(SwarmPerfCalibration))
    calibration = SwarmPerfCalibration.from_address(int(calibration_host.value))
    try:
        loader.memcpy_htod(d_ring, ctypes.cast(ring, ctypes.c_void_p), ctypes.sizeof(ring))
        bridge.consume(
            perf_ring_ptr=d_ring,
            ring_size=4,
            ring_head=4,
            calibration_ptr=calibration_device,
        )
        assert int(calibration.sample_count_total) == 4
        assert int(calibration.n_hint) == 5
        assert int(calibration.bucket_samples[4]) == 2
        assert int(calibration.utility_peak_q20) > 0
    finally:
        loader.gpu_free(d_ring)
        loader.mapped_host_free(calibration_host)
