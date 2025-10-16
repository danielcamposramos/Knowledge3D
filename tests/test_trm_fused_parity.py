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
