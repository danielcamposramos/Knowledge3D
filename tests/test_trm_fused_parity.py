import ctypes
from pathlib import Path

import numpy as np
import pytest

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


@pytest.mark.gpu
@pytest.mark.parametrize("n_steps", [1, 3])
def test_trm_fused_matches_ptx(n_steps: int):
    """Fused PTX kernel must match legacy PTX outputs."""
    _ensure_cuda()

    rng = np.random.default_rng(123)
    q = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    z = rng.standard_normal(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    trm_ptx = TRMLauncher(use_rpn=False, use_fused=False)
    y_ptx, z_ptx = trm_ptx.refine(q, y, z, W1, W2, W3, W4, n_steps=n_steps, eps=0.0)
    trm_ptx.cleanup()

    trm_fused = TRMLauncher(use_rpn=False, use_fused=True)
    y_fused, z_fused = trm_fused.refine(q, y, z, W1, W2, W3, W4, n_steps=n_steps, eps=0.0)
    trm_fused.cleanup()

    y_error = np.linalg.norm(y_ptx - y_fused)
    z_error = np.linalg.norm(z_ptx - z_fused)

    print(f"steps={n_steps} | y L2 error={y_error:.2e} | z L2 error={z_error:.2e}")

    assert y_error < 1e-5, f"y mismatch for n_steps={n_steps}: {y_error}"
    assert z_error < 1e-5, f"z mismatch for n_steps={n_steps}: {z_error}"


@pytest.mark.gpu
def test_trm_recursive_fused_matches_stepwise_fused_loop():
    """Recursive fused kernel must match the old repeated trm_step_fused loop exactly."""
    _ensure_cuda()

    rng = np.random.default_rng(321)
    q = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    z = rng.standard_normal(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    trm_fused = TRMLauncher(use_rpn=False, use_fused=True)
    try:
        y_recursive, z_recursive = trm_fused.refine(q, y, z, W1, W2, W3, W4, n_steps=3, eps=0.0)
    finally:
        trm_fused.cleanup()

    kernel = loader.load_ptx_file(
        str(Path("knowledge3d/cranium/ptx/trm_step_fused.ptx")),
        "trm_step_fused",
    )

    d_q = loader.gpu_malloc(q.nbytes)
    d_y = loader.gpu_malloc(y.nbytes)
    d_z = loader.gpu_malloc(z.nbytes)
    d_W1 = loader.gpu_malloc(W1.nbytes)
    d_W2 = loader.gpu_malloc(W2.nbytes)
    d_W3 = loader.gpu_malloc(W3.nbytes)
    d_W4 = loader.gpu_malloc(W4.nbytes)
    d_z_new = loader.gpu_malloc(512 * 4)
    d_y_new = loader.gpu_malloc(512 * 4)
    d_workspace = loader.gpu_malloc((512 + 1024 + 512 + 1024) * 4)

    try:
        for arr, device_ptr in [
            (q, d_q),
            (y, d_y),
            (z, d_z),
            (W1, d_W1),
            (W2, d_W2),
            (W3, d_W3),
            (W4, d_W4),
        ]:
            loader.memcpy_htod(device_ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)

        y_step = np.zeros(512, dtype=np.float32)
        z_step = np.zeros(512, dtype=np.float32)

        for _ in range(3):
            loader.launch(
                kernel,
                grid=(1, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_y.value),
                    ctypes.c_uint64(d_z.value),
                    ctypes.c_uint64(d_W1.value),
                    ctypes.c_uint64(d_W2.value),
                    ctypes.c_uint64(d_W3.value),
                    ctypes.c_uint64(d_W4.value),
                    ctypes.c_uint64(d_z_new.value),
                    ctypes.c_uint64(d_y_new.value),
                    ctypes.c_uint64(d_workspace.value),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(z_step.ctypes.data_as(ctypes.c_void_p), d_z_new, z_step.nbytes)
            loader.memcpy_dtoh(y_step.ctypes.data_as(ctypes.c_void_p), d_y_new, y_step.nbytes)
            loader.memcpy_htod(d_z, z_step.ctypes.data_as(ctypes.c_void_p), z_step.nbytes)
            loader.memcpy_htod(d_y, y_step.ctypes.data_as(ctypes.c_void_p), y_step.nbytes)

    finally:
        for ptr in [d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, d_z_new, d_y_new, d_workspace]:
            loader.gpu_free(ptr)

    np.testing.assert_allclose(y_recursive, y_step, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(z_recursive, z_step, rtol=0.0, atol=0.0)
