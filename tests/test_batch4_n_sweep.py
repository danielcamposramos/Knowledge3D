from __future__ import annotations

import ctypes
import os

import pytest

from knowledge3d.cranium.bridges.sleep_perf_consumer_bridge import (
    SleepPerfConsumerBridge,
    SwarmPerfCalibration,
)
from knowledge3d.cranium.sovereign import loader


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


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


def test_n_sweep_utility_curve_is_non_decreasing_on_perf_ring() -> None:
    bridge = SleepPerfConsumerBridge()
    ring = (_LanePerf * 8)()
    ring[0] = _LanePerf(9, 10, ctypes.c_float(0.09), 10, 0, (0, 0, 0))
    ring[1] = _LanePerf(9, 11, ctypes.c_float(0.10), 10, 1, (0, 0, 0))
    ring[2] = _LanePerf(18, 12, ctypes.c_float(0.20), 10, 2, (0, 0, 0))
    ring[3] = _LanePerf(18, 13, ctypes.c_float(0.21), 10, 3, (0, 0, 0))
    ring[4] = _LanePerf(36, 14, ctypes.c_float(0.35), 10, 4, (0, 0, 0))
    ring[5] = _LanePerf(36, 15, ctypes.c_float(0.36), 10, 5, (0, 0, 0))
    ring[6] = _LanePerf(72, 16, ctypes.c_float(0.50), 10, 6, (0, 0, 0))
    ring[7] = _LanePerf(72, 17, ctypes.c_float(0.51), 10, 7, (0, 0, 0))

    d_ring = loader.gpu_malloc(ctypes.sizeof(ring))
    calibration_host, calibration_device = loader.mapped_host_alloc(ctypes.sizeof(SwarmPerfCalibration))
    calibration = SwarmPerfCalibration.from_address(int(calibration_host.value))
    try:
        loader.memcpy_htod(d_ring, ctypes.cast(ring, ctypes.c_void_p), ctypes.sizeof(ring))
        bridge.consume(
            perf_ring_ptr=d_ring,
            ring_size=8,
            ring_head=8,
            calibration_ptr=calibration_device,
        )
        utilities: dict[int, list[float]] = {}
        for sample in ring:
            utilities.setdefault(int(sample.n_active), []).append(abs(float(sample.belief_delta)) / float(sample.cycles_consumed))
        avgs = {n_active: sum(values) / len(values) for n_active, values in utilities.items()}
        assert avgs[9] <= avgs[18] <= avgs[36] <= avgs[72]
        assert int(calibration.sample_count_total) == 8
        assert int(calibration.utility_peak_q20) > 0
    finally:
        loader.gpu_free(d_ring)
        loader.mapped_host_free(calibration_host)
