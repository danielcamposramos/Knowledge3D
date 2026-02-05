"""Smoke test Sovereign TRM with real V7 LSTM weights."""

from __future__ import annotations

import json
from pathlib import Path

import os
import pytest

from knowledge3d.cranium.sovereign_trm import BOS_ID, SovereignTRM


def _load_metadata(checkpoint_dir: Path) -> dict:
    meta_path = checkpoint_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not Path("/K3D/Knowledge3D.local/checkpoints/v7_sovereign").exists(),
    reason="Converted V7 sovereign checkpoint not found.",
)
def test_sovereign_trm_v7_real():
    if os.environ.get("K3D_RUN_LONG_TESTS") != "1":
        pytest.skip("Set K3D_RUN_LONG_TESTS=1 to run the full sovereign inference test.")
    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/v7_sovereign")
    meta = _load_metadata(checkpoint_dir)
    vocab_size = int(meta.get("vocab_size", 0)) or 256
    embedding_dim = int(meta.get("embedding_dim", 0)) or 256
    hidden_dim = int(meta.get("hidden_dim", 0)) or 512

    trm = SovereignTRM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    trm.load_weights(str(checkpoint_dir))
    try:
        problem = "Find f'(1) where f(x) = (3x-4)/(2x+3)"
        tokens = [BOS_ID] + list(problem.encode("utf-8", errors="ignore")[:8])
        rules, confidences = trm.infer(tokens, max_rules=8)
        assert len(rules) == len(confidences)
        for conf in confidences:
            assert 0.0 <= conf <= 1.0
    finally:
        trm.cleanup()
