"""Integration tests for Sovereign TRM inference loop."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from knowledge3d.cranium.sovereign_trm import BOS_ID, RULE_OFFSET, SovereignTRM


def _save_weights(tmpdir: str, arrays: dict) -> None:
    for name, array in arrays.items():
        np.save(os.path.join(tmpdir, f"{name}.npy"), array)


def test_sovereign_trm_inference_single_rule():
    vocab_size = 6
    embedding_dim = 4
    hidden_dim = 6
    rule_vocab_size = vocab_size + 3
    conf_hidden_dim = hidden_dim // 2

    arrays = {
        "embedding": np.zeros((vocab_size, embedding_dim), dtype=np.float32),
        "lstm_weight_ih": np.zeros((4 * hidden_dim, embedding_dim), dtype=np.float32),
        "lstm_weight_hh": np.zeros((4 * hidden_dim, hidden_dim), dtype=np.float32),
        "lstm_bias_ih": np.zeros(4 * hidden_dim, dtype=np.float32),
        "lstm_bias_hh": np.zeros(4 * hidden_dim, dtype=np.float32),
        "rule_head_weight": np.zeros((rule_vocab_size, hidden_dim), dtype=np.float32),
        "rule_head_bias": np.zeros(rule_vocab_size, dtype=np.float32),
        "confidence_head_0_weight": np.zeros((conf_hidden_dim, hidden_dim), dtype=np.float32),
        "confidence_head_0_bias": np.zeros(conf_hidden_dim, dtype=np.float32),
        "confidence_head_2_weight": np.zeros((1, conf_hidden_dim), dtype=np.float32),
        "confidence_head_2_bias": np.zeros(1, dtype=np.float32),
    }
    arrays["rule_head_bias"][RULE_OFFSET] = 1.0

    trm = SovereignTRM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_weights(tmpdir, arrays)
        trm.load_weights(tmpdir)

    rules, confidences = trm.infer([BOS_ID], max_rules=1)

    assert rules == [0]
    assert len(confidences) == 1
    assert abs(confidences[0] - 0.5) < 1e-5

    trm.cleanup()
