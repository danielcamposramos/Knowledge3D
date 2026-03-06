from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_transform_kernels_use_sovereign_ptx_bbox_and_crop():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.drawing_transform_kernels import crop_gpu, find_bbox_gpu

    grid = np.zeros((10, 12), dtype=np.int32)
    grid[2:7, 4:9] = 3

    bbox = find_bbox_gpu(grid, color=3)
    region = crop_gpu(grid, 2, 4, 5, 5)

    assert bbox == (2, 4, 6, 8)
    assert region.shape == (5, 5)
    assert np.all(region == 3)


@pytest.mark.cuda
def test_geometry_prep_returns_profile_metadata():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.geometry_prep import GeometryPrep

    grid = np.zeros((12, 12), dtype=np.int32)
    grid[3:10, 5] = 1
    grid[7:10, 6] = 1

    prep = GeometryPrep()
    profile = prep.prepare_profile(grid, color=1, pad=1)

    assert profile.bbox.min_y == 2
    assert profile.bbox.min_x == 4
    assert profile.bbox.max_y == 10
    assert profile.bbox.max_x == 7
    assert profile.region.shape == (9, 4)
    assert profile.nonzero_count == 10
    assert profile.mask_density > 0.0
    assert len(profile.column_fill) == profile.region.shape[1]
    assert len(profile.top_contour) == profile.region.shape[1]
    assert len(profile.bottom_contour) == profile.region.shape[1]
    assert len(profile.left_contour) == profile.region.shape[0]
    assert len(profile.right_contour) == profile.region.shape[0]
    assert len(profile.top_contour_smoothed) == profile.region.shape[1]
    assert len(profile.bottom_contour_smoothed) == profile.region.shape[1]
    assert len(profile.left_contour_smoothed) == profile.region.shape[0]
    assert len(profile.right_contour_smoothed) == profile.region.shape[0]
    assert max(profile.column_fill) >= 7
    assert max(profile.row_fill) >= 2


@pytest.mark.cuda
def test_geometry_prep_warmup_is_idempotent():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.geometry_prep import GeometryPrep

    prep = GeometryPrep()
    first = prep.warmup_runtime()
    second = prep.warmup_runtime()

    assert first["status"] == "ready"
    assert float(first["total_warmup_ms"]) > 0.0
    assert first == second
