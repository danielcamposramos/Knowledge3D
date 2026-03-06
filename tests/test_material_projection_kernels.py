from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_sample_preview_matches_expected_texels():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.material_projection_kernels import MaterialProjectionKernels

    preview = np.asarray(
        [
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    coords = np.asarray(
        [
            [0.01, 0.01],  # bottom-left -> blue
            [0.99, 0.99],  # top-right -> green
            [0.99, 0.01],  # bottom-right -> white
        ],
        dtype=np.float32,
    )

    kernels = MaterialProjectionKernels()
    out = kernels.sample_preview(preview, coords, np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), 1.0)

    assert out.shape == (3, 4)
    assert np.allclose(out[0], [0.0, 0.0, 1.0, 1.0], atol=1e-5)
    assert np.allclose(out[1], [0.0, 1.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(out[2], [1.0, 1.0, 1.0, 1.0], atol=1e-5)


@pytest.mark.cuda
def test_blend_triplanar_combines_planes_by_weight():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.material_projection_kernels import MaterialProjectionKernels

    yz = np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    xz = np.asarray([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    xy = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    weights = np.asarray([[0.25, 0.5, 0.25]], dtype=np.float32)

    kernels = MaterialProjectionKernels()
    out = kernels.blend_triplanar(yz, xz, xy, weights)

    assert out.shape == (1, 4)
    assert np.allclose(out[0, :3], [0.25, 0.5, 0.25], atol=1e-5)
    assert np.isclose(out[0, 3], 1.0, atol=1e-5)
