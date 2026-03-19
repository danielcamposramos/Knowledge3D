import pytest

torch = pytest.importorskip("torch")

from knowledge3d.training.math_benchmarks.navigation_model import (
    CONTROL_TOKENS,
    NavigationSeqModel,
)


def test_control_token_vocab_enabled():
    model = NavigationSeqModel(embedding_dim=8, hidden_dim=16, vocab_size=5, enable_control_tokens=True)
    assert model.vocab_size == 5 + len(CONTROL_TOKENS)
    assert model.control_token_offset == 5
    registry = [f"rule_{i}" for i in range(5)]
    assert model.decode_token(5, registry) == "<CONFIDENT>"
    assert model.decode_token(6, registry) == "<UNCERTAIN>"
    assert model.decode_token(7, registry) == "<VERIFY>"


def test_control_token_vocab_disabled():
    model = NavigationSeqModel(embedding_dim=8, hidden_dim=16, vocab_size=5)
    assert model.vocab_size == 5
    registry = [f"rule_{i}" for i in range(5)]
    assert model.decode_token(2, registry) == "rule_2"
