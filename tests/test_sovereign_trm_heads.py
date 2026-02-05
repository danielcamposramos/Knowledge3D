"""Test Sovereign TRM heads against PyTorch reference."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.sovereign_trm import SovereignTRM


def _save_weights(tmpdir: str, arrays: dict) -> None:
    for name, array in arrays.items():
        np.save(os.path.join(tmpdir, f"{name}.npy"), array)


def test_sovereign_trm_heads():
    vocab_size = 8
    embedding_dim = 4
    hidden_dim = 6
    rule_vocab_size = vocab_size + 3
    conf_hidden_dim = hidden_dim // 2

    rng = np.random.RandomState(42)
    arrays = {
        "embedding": rng.randn(vocab_size, embedding_dim).astype(np.float32),
        "lstm_weight_ih": rng.randn(4 * hidden_dim, embedding_dim).astype(np.float32),
        "lstm_weight_hh": rng.randn(4 * hidden_dim, hidden_dim).astype(np.float32),
        "lstm_bias_ih": rng.randn(4 * hidden_dim).astype(np.float32),
        "lstm_bias_hh": rng.randn(4 * hidden_dim).astype(np.float32),
        "rule_head_weight": rng.randn(rule_vocab_size, hidden_dim).astype(np.float32),
        "rule_head_bias": rng.randn(rule_vocab_size).astype(np.float32),
        "confidence_head_0_weight": rng.randn(conf_hidden_dim, hidden_dim).astype(np.float32),
        "confidence_head_0_bias": rng.randn(conf_hidden_dim).astype(np.float32),
        "confidence_head_2_weight": rng.randn(1, conf_hidden_dim).astype(np.float32),
        "confidence_head_2_bias": rng.randn(1).astype(np.float32),
    }

    trm = SovereignTRM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_weights(tmpdir, arrays)
        trm.load_weights(tmpdir)

    hidden = rng.randn(hidden_dim).astype(np.float32)
    hidden_ptr = loader.gpu_malloc(hidden.nbytes)
    loader.cpu_to_gpu(hidden_ptr, hidden)

    try:
        logits_ptr = trm._rule_head(hidden_ptr)
        try:
            logits = loader.gpu_to_cpu_array(logits_ptr, rule_vocab_size)
        finally:
            trm._free_ptr(logits_ptr)

        pt_rule_head = nn.Linear(hidden_dim, rule_vocab_size)
        pt_rule_head.weight.data = torch.tensor(arrays["rule_head_weight"])
        pt_rule_head.bias.data = torch.tensor(arrays["rule_head_bias"])
        with torch.no_grad():
            expected_logits = pt_rule_head(torch.tensor(hidden)).numpy()

        np.testing.assert_allclose(logits, expected_logits, rtol=1e-4, atol=1e-5)

        pt_confidence = nn.Sequential(
            nn.Linear(hidden_dim, conf_hidden_dim),
            nn.ReLU(),
            nn.Linear(conf_hidden_dim, 1),
            nn.Sigmoid(),
        )
        pt_confidence[0].weight.data = torch.tensor(arrays["confidence_head_0_weight"])
        pt_confidence[0].bias.data = torch.tensor(arrays["confidence_head_0_bias"])
        pt_confidence[2].weight.data = torch.tensor(arrays["confidence_head_2_weight"])
        pt_confidence[2].bias.data = torch.tensor(arrays["confidence_head_2_bias"])
        with torch.no_grad():
            expected_conf = float(pt_confidence(torch.tensor(hidden)).item())

        conf_value = trm._confidence_head(hidden_ptr)
        np.testing.assert_allclose(conf_value, expected_conf, rtol=1e-4, atol=1e-5)
    finally:
        trm._free_ptr(hidden_ptr)
        trm.cleanup()
