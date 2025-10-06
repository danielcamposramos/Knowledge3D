
from pathlib import Path

import numpy as np
import pytest

PTX_DIR = Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "ptx"


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - GPU unavailable
        pytest.skip("CUDA device not available")
    return cupy


def _load_kernel(cupy, filename: str, entry: str):
    module = cupy.RawModule(path=str(PTX_DIR / filename))
    return module.get_function(entry)


def test_warp_modality_fuse_weights() -> None:
    cupy = _require_gpu()

    kernel = _load_kernel(cupy, "warp_modality_fuse.ptx", "warp_modality_fuse")

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


def test_warp_modality_fuse_lod_bias() -> None:
    cupy = _require_gpu()

    kernel = _load_kernel(cupy, "warp_modality_fuse.ptx", "warp_modality_fuse")

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


def test_warp_modality_fuse_simd_consistency() -> None:
    cupy = _require_gpu()

    scalar_kernel = _load_kernel(cupy, "warp_modality_fuse.ptx", "warp_modality_fuse")
    simd_kernel = _load_kernel(cupy, "warp_modality_fuse_simd.ptx", "warp_modality_fuse_simd")

    nodes, dim = 4, 64
    text = cupy.random.random_sample((nodes, dim)).astype(cupy.float32)
    image = cupy.random.random_sample((nodes, dim)).astype(cupy.float32)
    audio = cupy.random.random_sample((nodes, dim)).astype(cupy.float32)
    video = cupy.random.random_sample((nodes, dim)).astype(cupy.float32)
    morton = cupy.arange(nodes, dtype=cupy.uint32)

    block = 128
    total = nodes * dim
    grid = (max(1, (total + block - 1) // block),)

    out_scalar = cupy.zeros((nodes, dim), dtype=cupy.float32)
    scalar_kernel(grid, (block,), (out_scalar, text, image, audio, video, morton, np.int32(dim), np.int32(nodes)))

    out_simd = cupy.zeros_like(out_scalar)
    simd_kernel(grid, (block,), (out_simd, text, image, audio, video, morton, np.int32(dim), np.int32(nodes)))

    cupy.cuda.runtime.deviceSynchronize()

    assert np.allclose(out_scalar.get(), out_simd.get(), atol=1e-5)
