import numpy as np
import pytest

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


def _require_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:  # pragma: no cover - hardware dependent
        pytest.skip(f"CUDA context unavailable: {exc}")


def _run_launcher(use_rpn: bool, q, y, z, W1, W2, W3, W4, n_steps: int, eps: float):
    launcher = TRMLauncher(use_rpn=use_rpn)
    try:
        return launcher.refine(q, y, z, W1, W2, W3, W4, n_steps=n_steps, eps=eps)
    finally:
        launcher.cleanup()


@pytest.mark.parametrize("n_steps", [2, 4])
def test_trm_rpn_matches_ptx(n_steps: int):
    _require_cuda()

    rng = np.random.default_rng(42)
    q = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    z = rng.standard_normal(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    y_ptx, z_ptx = _run_launcher(False, q, y, z, W1, W2, W3, W4, n_steps, 1e-4)
    y_rpn, z_rpn = _run_launcher(True, q, y, z, W1, W2, W3, W4, n_steps, 1e-4)

    np.testing.assert_allclose(y_rpn, y_ptx, rtol=2e-3, atol=1e-3)
    np.testing.assert_allclose(z_rpn, z_ptx, rtol=2e-3, atol=1e-3)
