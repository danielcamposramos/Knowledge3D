from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_temporal_frame_kernel_is_deterministic():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.temporal_frame_kernels import TemporalFrameKernels

    kernels = TemporalFrameKernels()
    seed = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    times = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float32)

    frames_a = kernels.generate_frames(seed, width=32, height=32, time_points=times)
    frames_b = kernels.generate_frames(seed, width=32, height=32, time_points=times)

    assert np.array_equal(frames_a, frames_b)


@pytest.mark.cuda
def test_temporal_frame_kernel_has_temporal_coherence():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.temporal_frame_kernels import TemporalFrameKernels

    kernels = TemporalFrameKernels()
    seed = np.random.default_rng(1).standard_normal(64).astype(np.float32)
    times = np.asarray([0.0, 0.01], dtype=np.float32)
    frames = kernels.generate_frames(seed, width=64, height=64, time_points=times)

    diff = np.mean(np.abs(frames[1].astype(np.float32) - frames[0].astype(np.float32)))
    assert diff < 50.0
    assert frames.shape == (2, 64, 64, 3)
