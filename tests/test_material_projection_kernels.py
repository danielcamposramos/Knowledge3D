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


@pytest.mark.cuda
def test_project_triplanar_matches_sample_plus_blend():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.material_projection_kernels import MaterialProjectionKernels

    preview = np.asarray(
        [
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    vertices = np.asarray(
        [
            [0.10, 0.15, 0.80],
            [0.85, 0.90, 0.20],
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            [0.25, 0.50, 0.25],
            [0.60, 0.20, 0.20],
        ],
        dtype=np.float32,
    )
    mins = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    extents = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    kernels = MaterialProjectionKernels()
    yz = kernels.sample_preview(preview, vertices[:, [1, 2]], mins[[1, 2]], extents[[1, 2]], 1.0)
    xz = kernels.sample_preview(preview, vertices[:, [0, 2]], mins[[0, 2]], extents[[0, 2]], 1.0)
    xy = kernels.sample_preview(preview, vertices[:, [0, 1]], mins[[0, 1]], extents[[0, 1]], 1.0)
    blended = kernels.blend_triplanar(yz, xz, xy, weights)
    fused = kernels.project_triplanar(preview, vertices, weights, mins, extents, 1.0)

    assert fused.shape == blended.shape == (2, 4)
    assert np.allclose(fused, blended, atol=1e-5)
