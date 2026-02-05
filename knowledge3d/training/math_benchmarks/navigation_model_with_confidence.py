"""
Navigation model with confidence head for Phase 5.1.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from knowledge3d.training.math_benchmarks.navigation_model import RULE_OFFSET


class NavigationModelWithConfidence(nn.Module):
    """
    Navigation model with dual heads: rule logits + confidence scores.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        base_vocab_size: int,
        vocab_size: int,
    ):
        super().__init__()
        self.base_vocab_size = int(base_vocab_size)
        self.vocab_size = int(vocab_size)
        self.rule_vocab_size = self.base_vocab_size + RULE_OFFSET

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.rule_head = nn.Linear(hidden_dim, self.rule_vocab_size)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, token_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_vecs = self.embedding(token_inputs)
        outputs, _ = self.lstm(token_vecs)
        rule_logits = self.rule_head(outputs)
        confidence = self.confidence_head(outputs)
        return rule_logits, confidence

    def forward_with_teacher_forcing(
        self,
        token_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(token_inputs)


__all__ = ["NavigationModelWithConfidence"]
