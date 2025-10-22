from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


def _expand_to_512(vec: np.ndarray) -> np.ndarray:
    if vec.shape[0] >= 512:
        return vec[:512]
    pad_width = 512 - vec.shape[0]
    return np.pad(vec, (0, pad_width))


@pytest.fixture(scope="module")
def trm():
    try:
        return TRMLauncher(use_fused=True)
    except Exception as exc:  # pragma: no cover - GPU availability guard
        pytest.skip(f"TRMLauncher unavailable: {exc}")


@pytest.fixture(scope="module")
def trm_weights():
    rng = np.random.default_rng(seed=42)
    W1 = rng.normal(scale=0.02, size=(1024, 512)).astype(np.float32)
    W2 = rng.normal(scale=0.02, size=(512, 1024)).astype(np.float32)
    W3 = rng.normal(scale=0.02, size=(1024, 512)).astype(np.float32)
    W4 = rng.normal(scale=0.02, size=(512, 1024)).astype(np.float32)
    return W1, W2, W3, W4


class TestTRMReasoning:
    def test_six_recursion_convergence(self, trm, trm_weights, rpn_engine):
        question_text = (
            "If all roses are flowers and some flowers fade quickly, "
            "can we conclude that some roses fade quickly?"
        )
        q_emb = rpn_engine.embed_sentence(question_text).astype(np.float32)
        q = _expand_to_512(q_emb).astype(np.float32)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)
        y_out, _ = trm.refine(q, y, z, *trm_weights, n_steps=6, eps=1e-4)

        assert np.isfinite(y_out).all(), "TRM output contains NaN/Inf"
        assert np.linalg.norm(y_out) > 1e-3, "TRM output too weak"

    @pytest.mark.parametrize("harmonics", [3, 6, 9])
    def test_tesla_harmonics(self, trm, trm_weights, rpn_engine, harmonics):
        question_text = "Solve the equation 2x + 5 = 17."
        q_emb = rpn_engine.embed_sentence(question_text).astype(np.float32)
        q = _expand_to_512(q_emb).astype(np.float32)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)
        y_out, _ = trm.refine(q, y, z, *trm_weights, n_steps=harmonics, eps=1e-4)

        assert np.isfinite(y_out).all()
        assert np.linalg.norm(y_out) > 1e-3, f"Weak output for n={harmonics}"
