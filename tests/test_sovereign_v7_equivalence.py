"""Validate Sovereign TRM against a PyTorch V7 checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from knowledge3d.cranium.sovereign_trm import BOS_ID, RULE_OFFSET, SovereignTRM


def _resolve_key(state_dict: Dict[str, Any], name: str) -> Optional[str]:
    if name in state_dict:
        return name
    matches = [key for key in state_dict.keys() if key.endswith(name)]
    if not matches:
        return None
    matches.sort(key=len)
    return matches[0]


def _load_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


@pytest.mark.skipif(
    not Path("/K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_lstm_confidence_final.pt").exists()
    or not Path("/K3D/Knowledge3D.local/checkpoints/v7_sovereign").exists(),
    reason="Missing V7 checkpoint or converted Sovereign weights.",
)
def test_sovereign_v7_equivalence():
    checkpoint = torch.load(
        "/K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_lstm_confidence_final.pt",
        map_location="cpu",
    )
    state_dict = _load_state_dict(checkpoint)

    embedding_key = _resolve_key(state_dict, "embedding.weight")
    if embedding_key is None:
        pytest.skip("Checkpoint missing embedding weights.")

    embedding_weight = state_dict[embedding_key]
    vocab_size, embedding_dim = embedding_weight.shape
    rule_head_weight = state_dict[_resolve_key(state_dict, "rule_head.weight")]
    hidden_dim = int(checkpoint.get("hidden_dim", 0)) or int(rule_head_weight.shape[1])

    model = nn.Module()
    model.embedding = nn.Embedding(vocab_size, embedding_dim)
    model.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    rule_vocab_size = int(rule_head_weight.shape[0])
    model.rule_head = nn.Linear(hidden_dim, rule_vocab_size)
    conf_hidden_dim = max(1, hidden_dim // 2)
    model.confidence_head = nn.Sequential(
        nn.Linear(hidden_dim, conf_hidden_dim),
        nn.ReLU(),
        nn.Linear(conf_hidden_dim, 1),
        nn.Sigmoid(),
    )

    model.embedding.weight.data = embedding_weight
    model.lstm.weight_ih_l0.data = state_dict[_resolve_key(state_dict, "lstm.weight_ih_l0")]
    model.lstm.weight_hh_l0.data = state_dict[_resolve_key(state_dict, "lstm.weight_hh_l0")]
    model.lstm.bias_ih_l0.data = state_dict[_resolve_key(state_dict, "lstm.bias_ih_l0")]
    model.lstm.bias_hh_l0.data = state_dict[_resolve_key(state_dict, "lstm.bias_hh_l0")]
    model.rule_head.weight.data = rule_head_weight
    model.rule_head.bias.data = state_dict[_resolve_key(state_dict, "rule_head.bias")]
    model.confidence_head[0].weight.data = state_dict[_resolve_key(state_dict, "confidence_head.0.weight")]
    model.confidence_head[0].bias.data = state_dict[_resolve_key(state_dict, "confidence_head.0.bias")]
    model.confidence_head[2].weight.data = state_dict[_resolve_key(state_dict, "confidence_head.2.weight")]
    model.confidence_head[2].bias.data = state_dict[_resolve_key(state_dict, "confidence_head.2.bias")]
    model.eval()

    problem_tokens = [BOS_ID, 42, 15, 3]
    decode_token = BOS_ID
    token_seq = problem_tokens + [decode_token]

    with torch.no_grad():
        token_tensor = torch.tensor([token_seq], dtype=torch.long)
        embedded = model.embedding(token_tensor)
        outputs, _ = model.lstm(embedded)
        last_hidden = outputs[:, -1, :]
        logits = model.rule_head(last_hidden)
        confidence = model.confidence_head(last_hidden)
        pt_next_id = int(torch.argmax(logits[0]).item())
        pt_conf = float(confidence[0, 0].item())

    trm = SovereignTRM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    trm.load_weights("/K3D/Knowledge3D.local/checkpoints/v7_sovereign")
    try:
        sov_rules, sov_conf = trm.infer(problem_tokens, max_rules=1)
    finally:
        trm.cleanup()

    assert sov_rules
    assert sov_conf
    assert 0.0 <= sov_conf[0] <= 1.0
    if pt_next_id >= RULE_OFFSET:
        assert sov_rules[0] == (pt_next_id - RULE_OFFSET)
    np.testing.assert_allclose(sov_conf[0], pt_conf, rtol=1e-2, atol=1e-3)
