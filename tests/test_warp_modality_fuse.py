import os
from pathlib import Path

import numpy as np
import pytest


def test_warp_modality_fuse_weights(tmp_path) -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - GPU unavailable
        pytest.skip("CUDA device not available")

    base = Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "ptx"
    module = cupy.RawModule(path=str(base / "warp_modality_fuse.ptx"))
    kernel = module.get_function("warp_modality_fuse")

    text = cupy.ones((1, 4), dtype=cupy.float32)
    image = cupy.full((1, 4), 2.0, dtype=cupy.float32)
    audio = cupy.full((1, 4), 3.0, dtype=cupy.float32)
    video = cupy.full((1, 4), 4.0, dtype=cupy.float32)
    morton = cupy.zeros((1,), dtype=cupy.uint32)
    out = cupy.zeros((1, 4), dtype=cupy.float32)

    dim = np.int32(4)
    nodes = np.int32(1)
    block = 32
    grid = (1,)
    kernel(grid, (block,), (out, text, image, audio, video, morton, dim, nodes))
    cupy.cuda.runtime.deviceSynchronize()

    expected = 0.4 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0 + 0.1 * 4.0
    assert np.allclose(out.get(), expected, atol=1e-6)


def test_warp_modality_fuse_lod_bias(tmp_path) -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - GPU unavailable
        pytest.skip("CUDA device not available")

    base = Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "ptx"
    module = cupy.RawModule(path=str(base / "warp_modality_fuse.ptx"))
    kernel = module.get_function("warp_modality_fuse")

    text = cupy.ones((1, 4), dtype=cupy.float32)
    image = cupy.ones((1, 4), dtype=cupy.float32)
    audio = cupy.zeros((1, 4), dtype=cupy.float32)
    video = cupy.zeros((1, 4), dtype=cupy.float32)
    morton = cupy.full((1,), 4, dtype=cupy.uint32)
    out = cupy.zeros((1, 4), dtype=cupy.float32)

    dim = np.int32(4)
    nodes = np.int32(1)
    kernel((1,), (32,), (out, text, image, audio, video, morton, dim, nodes))
    cupy.cuda.runtime.deviceSynchronize()

    lod_scale = 1.0 / (1.0 + 0.125 * 4.0)
    expected = (0.4 + 0.3) * lod_scale
    assert np.allclose(out.get(), expected, atol=1e-6)
