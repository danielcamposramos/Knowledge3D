from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_spectrogram_to_rgba_maps_negative_neutral_positive():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.signal_visualization_kernels import SignalVisualizationKernels

    spectrogram = np.asarray([[-1, 0, 1]], dtype=np.int32)
    kernels = SignalVisualizationKernels()
    out = kernels.spectrogram_to_rgba(spectrogram)

    assert out.shape == (1, 3, 4)
    assert np.allclose(out[0, 0], [0.14, 0.46, 0.88, 1.0], atol=1e-5)
    assert np.allclose(out[0, 1], [0.08, 0.08, 0.10, 1.0], atol=1e-5)
    assert np.allclose(out[0, 2], [1.0, 0.72, 0.18, 1.0], atol=1e-5)
