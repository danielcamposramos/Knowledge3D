import ctypes

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
def test_trm_runtime_tick_matches_recursive_oracle():
    """The embodied fused tick must preserve the recursive kernel's query fast-lane outputs."""
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
    d_q = loader.gpu_malloc(q.nbytes)
    d_y_runtime = loader.gpu_malloc(y.nbytes)
    d_z_runtime = loader.gpu_malloc(z.nbytes)
    d_W1 = loader.gpu_malloc(W1.nbytes)
    d_W2 = loader.gpu_malloc(W2.nbytes)
    d_W3 = loader.gpu_malloc(W3.nbytes)
    d_W4 = loader.gpu_malloc(W4.nbytes)
    d_z_new_runtime = loader.gpu_malloc(512 * 4)
    d_y_new_runtime = loader.gpu_malloc(512 * 4)
    d_y_oracle = loader.gpu_malloc(y.nbytes)
    d_z_oracle = loader.gpu_malloc(z.nbytes)
    d_workspace_oracle = loader.gpu_malloc(4096 * 4)
    d_steps_oracle = loader.gpu_malloc(ctypes.sizeof(ctypes.c_int32))
    d_drift_oracle = loader.gpu_malloc(ctypes.sizeof(ctypes.c_float))
    y_runtime = np.zeros(512, dtype=np.float32)
    z_runtime = np.zeros(512, dtype=np.float32)
    y_oracle = np.zeros(512, dtype=np.float32)
    z_oracle = np.zeros(512, dtype=np.float32)
    steps_oracle = ctypes.c_int32()
    drift_oracle = ctypes.c_float()

    try:
        for arr, device_ptr in [
            (q, d_q),
            (y, d_y_runtime),
            (z, d_z_runtime),
            (W1, d_W1),
            (W2, d_W2),
            (W3, d_W3),
            (W4, d_W4),
            (y, d_y_oracle),
            (z, d_z_oracle),
        ]:
            loader.memcpy_htod(device_ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)

        runtime_meta = trm_fused.run_query_tick(
            q_ptr=d_q,
            y_ptr=d_y_runtime,
            z_ptr=d_z_runtime,
            W1_ptr=d_W1,
            W2_ptr=d_W2,
            W3_ptr=d_W3,
            W4_ptr=d_W4,
            z_new_ptr=d_z_new_runtime,
            y_new_ptr=d_y_new_runtime,
            workspace_ptr=trm_fused.d_workspace,
            max_steps=3,
            epsilon=0.0,
            reset_runtime=True,
        )

        loader.launch(
            trm_fused.kernel_recursive_fused,
            grid=(1, 1, 1),
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(d_q.value),
                ctypes.c_uint64(d_y_oracle.value),
                ctypes.c_uint64(d_z_oracle.value),
                ctypes.c_uint64(d_W1.value),
                ctypes.c_uint64(d_W2.value),
                ctypes.c_uint64(d_W3.value),
                ctypes.c_uint64(d_W4.value),
                ctypes.c_uint64(d_workspace_oracle.value),
                ctypes.c_uint64(d_steps_oracle.value),
                ctypes.c_uint64(d_drift_oracle.value),
                ctypes.c_int32(3),
                ctypes.c_float(0.0),
            ],
        )
        loader.synchronize()

        loader.memcpy_dtoh(y_runtime.ctypes.data_as(ctypes.c_void_p), d_y_new_runtime, y_runtime.nbytes)
        loader.memcpy_dtoh(z_runtime.ctypes.data_as(ctypes.c_void_p), d_z_new_runtime, z_runtime.nbytes)
        loader.memcpy_dtoh(y_oracle.ctypes.data_as(ctypes.c_void_p), d_y_oracle, y_oracle.nbytes)
        loader.memcpy_dtoh(z_oracle.ctypes.data_as(ctypes.c_void_p), d_z_oracle, z_oracle.nbytes)
        loader.memcpy_dtoh(ctypes.byref(steps_oracle), d_steps_oracle, ctypes.sizeof(steps_oracle))
        loader.memcpy_dtoh(ctypes.byref(drift_oracle), d_drift_oracle, ctypes.sizeof(drift_oracle))
    finally:
        trm_fused.cleanup()
        for ptr in [
            d_q,
            d_y_runtime,
            d_z_runtime,
            d_W1,
            d_W2,
            d_W3,
            d_W4,
            d_z_new_runtime,
            d_y_new_runtime,
            d_y_oracle,
            d_z_oracle,
            d_workspace_oracle,
            d_steps_oracle,
            d_drift_oracle,
        ]:
            loader.gpu_free(ptr)

    np.testing.assert_allclose(y_runtime, y_oracle, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(z_runtime, z_oracle, rtol=0.0, atol=1e-5)
    assert runtime_meta["steps"] == steps_oracle.value
    assert runtime_meta["drift"] == pytest.approx(float(drift_oracle.value), rel=0.0, abs=1e-5)
