import torch

from knowledge3d.training.math_benchmarks.navigation_model_with_confidence import (
    NavigationModelWithConfidence,
)


def test_confidence_head_shapes():
    model = NavigationModelWithConfidence(
        embedding_dim=8,
        hidden_dim=16,
        base_vocab_size=5,
        vocab_size=32,
    )
    token_inputs = torch.zeros((2, 4), dtype=torch.long)
    rule_logits, confidence = model.forward_with_teacher_forcing(token_inputs)
    assert rule_logits.shape == (2, 4, model.rule_vocab_size)
    assert confidence.shape == (2, 4, 1)
    assert torch.all(confidence >= 0.0)
    assert torch.all(confidence <= 1.0)
