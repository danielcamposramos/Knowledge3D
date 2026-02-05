"""Test Sovereign TRM LSTM layer against PyTorch reference."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from knowledge3d.cranium.sovereign_trm import SovereignTRM
from knowledge3d.cranium.sovereign import loader


def test_lstm_single_step():
    """Test single LSTM forward step matches PyTorch."""
    vocab_size = 256
    embedding_dim = 256
    hidden_dim = 512

    pt_embedding = nn.Embedding(vocab_size, embedding_dim)
    pt_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    pt_lstm.eval()

    trm = SovereignTRM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "embedding.npy"), pt_embedding.weight.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, "lstm_weight_ih.npy"), pt_lstm.weight_ih_l0.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, "lstm_weight_hh.npy"), pt_lstm.weight_hh_l0.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, "lstm_bias_ih.npy"), pt_lstm.bias_ih_l0.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, "lstm_bias_hh.npy"), pt_lstm.bias_hh_l0.detach().cpu().numpy())

        np.save(
            os.path.join(tmpdir, "rule_head_weight.npy"),
            np.zeros((vocab_size + 3, hidden_dim), dtype=np.float32),
        )
        np.save(os.path.join(tmpdir, "rule_head_bias.npy"), np.zeros(vocab_size + 3, dtype=np.float32))
        np.save(
            os.path.join(tmpdir, "confidence_head_0_weight.npy"),
            np.zeros((hidden_dim // 2, hidden_dim), dtype=np.float32),
        )
        np.save(
            os.path.join(tmpdir, "confidence_head_0_bias.npy"),
            np.zeros(hidden_dim // 2, dtype=np.float32),
        )
        np.save(
            os.path.join(tmpdir, "confidence_head_2_weight.npy"),
            np.zeros((1, hidden_dim // 2), dtype=np.float32),
        )
        np.save(os.path.join(tmpdir, "confidence_head_2_bias.npy"), np.zeros(1, dtype=np.float32))

        trm.load_weights(tmpdir)

    token_id = 42

    with torch.no_grad():
        pt_input = torch.tensor([[token_id]])
        pt_emb = pt_embedding(pt_input)
        _, (pt_h, _) = pt_lstm(pt_emb)
        pt_h_np = pt_h[0, 0].cpu().numpy()

    trm.reset_lstm_state()
    sov_h = trm._lstm_step(token_id)
    sov_h_np = loader.gpu_to_cpu_array(sov_h, hidden_dim)

    assert sov_h_np.shape == pt_h_np.shape
    np.testing.assert_allclose(sov_h_np, pt_h_np, rtol=1e-4, atol=1e-5)

    trm.cleanup()


def test_lstm_sequence():
    """Placeholder for sequence-level LSTM verification."""
    pytest.skip("Sequence test not implemented for Phase 2.")
