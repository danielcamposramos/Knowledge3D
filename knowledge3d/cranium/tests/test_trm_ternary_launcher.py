import numpy as np
import pytest

from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.sovereign.trm_ternary_launcher import TRMTernaryLauncher


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32) * 0.01
    W2 = rng.standard_normal((512, 1024), dtype=np.float32) * 0.01
    W3 = rng.standard_normal((1024, 512), dtype=np.float32) * 0.01
    W4 = rng.standard_normal((512, 1024), dtype=np.float32) * 0.01
    return W1, W2, W3, W4


def test_trm_ternary_amplify(weights):
    base = TRMLauncher()
    ternary = TRMTernaryLauncher()
    W1, W2, W3, W4 = weights
    q = np.ones(512, dtype=np.float32) * 0.5
    y = np.zeros(512, dtype=np.float32)
    z = np.zeros(512, dtype=np.float32)

    y_base, z_base = base.refine(q, y, z, W1, W2, W3, W4, n_steps=6)
    y_mod, z_mod = ternary.refine(q, y, z, W1, W2, W3, W4, n_steps=6)

    # Positive dot => trit=+1 => amplify
    assert np.allclose(y_mod, y_base * 2.0)
    assert np.allclose(z_mod, z_base * 2.0)


def test_trm_ternary_dampen(weights):
    base = TRMLauncher()
    ternary = TRMTernaryLauncher()
    W1, W2, W3, W4 = weights
    q = -np.ones(512, dtype=np.float32) * 0.5  # negative dot -> repel
    y = np.zeros(512, dtype=np.float32)
    z = np.zeros(512, dtype=np.float32)

    y_base, z_base = base.refine(q, y, z, W1, W2, W3, W4, n_steps=6)
    y_mod, z_mod = ternary.refine(q, y, z, W1, W2, W3, W4, n_steps=6)

    # Repel path may short-circuit to zeros (skip) or dampen
    assert np.linalg.norm(y_mod) <= np.linalg.norm(y_base) * 0.2
    assert np.linalg.norm(z_mod) <= np.linalg.norm(z_base) * 0.2


def test_trm_ternary_batch(weights):
    ternary = TRMTernaryLauncher()
    W1, W2, W3, W4 = weights
    q_batch = np.stack([
        np.ones(512, dtype=np.float32) * 0.25,    # attract
        -np.ones(512, dtype=np.float32) * 0.25,   # repel
    ], axis=0)
    y_batch = np.zeros_like(q_batch)
    z_batch = np.zeros_like(q_batch)

    y_out, z_out = ternary.refine_batch(q_batch, y_batch, z_batch, W1, W2, W3, W4, n_steps=6)

    # First amplified, second dampened
    assert np.linalg.norm(y_out[0]) > np.linalg.norm(y_out[1]) or np.allclose(y_out[1], 0.0)
    assert np.linalg.norm(z_out[0]) > np.linalg.norm(z_out[1]) or np.allclose(z_out[1], 0.0)
