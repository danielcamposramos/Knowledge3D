from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_heightfield_to_vertices_generates_expected_positions():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.signal_surface_kernels import SignalSurfaceKernels

    heightfield = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )
    kernels = SignalSurfaceKernels()
    vertices = kernels.heightfield_to_vertices(heightfield, time_scale=1.0, frequency_scale=1.0)

    assert vertices.shape == (4, 3)
    assert np.allclose(vertices[0], [-0.5, 0.0, 0.5], atol=1e-5)
    assert np.allclose(vertices[1], [0.5, 1.0, 0.5], atol=1e-5)
    assert np.allclose(vertices[2], [-0.5, 2.0, -0.5], atol=1e-5)
    assert np.allclose(vertices[3], [0.5, 3.0, -0.5], atol=1e-5)


@pytest.mark.cuda
def test_heightfield_to_normals_emits_unit_vectors():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.signal_surface_kernels import SignalSurfaceKernels

    heightfield = np.asarray(
        [
            [0.0, 0.2, 0.4],
            [0.0, 0.2, 0.4],
            [0.0, 0.2, 0.4],
        ],
        dtype=np.float32,
    )
    kernels = SignalSurfaceKernels()
    normals = kernels.heightfield_to_normals(heightfield, time_scale=1.0, frequency_scale=1.0)

    assert normals.shape == (9, 3)
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-4)
    center = normals[4]
    assert center[0] < 0.0
    assert center[1] > 0.0
