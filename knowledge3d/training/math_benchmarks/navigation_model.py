"""
Navigation specialist model and shared token constants.
"""

from __future__ import annotations

from typing import Dict, List

from torch import nn


PAD_ID = 0
BOS_ID = 1
RULE_OFFSET = 2
CONTROL_TOKENS = ["<CONFIDENT>", "<UNCERTAIN>", "<VERIFY>"]


class NavigationSeqModel(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        vocab_size: int,
        hidden_dim: int,
        enable_control_tokens: bool = False,
    ):
        super().__init__()
        self.enable_control_tokens = bool(enable_control_tokens)
        self.base_vocab_size = int(vocab_size)
        if self.enable_control_tokens:
            self.control_token_offset = self.base_vocab_size
            self.vocab_size = self.base_vocab_size + len(CONTROL_TOKENS)
        else:
            self.control_token_offset = None
            self.vocab_size = self.base_vocab_size

        self.encoder = nn.Linear(embedding_dim, hidden_dim)
        self.token_embed = nn.Embedding(self.vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, embeddings, token_inputs):
        hidden = nn.functional.tanh(self.encoder(embeddings)).unsqueeze(0)
        token_vecs = self.token_embed(token_inputs)
        outputs, _ = self.gru(token_vecs, hidden)
        return self.output(outputs)

    def decode_token(self, token_id: int, rule_registry: List[str]) -> str:
        if self.enable_control_tokens and self.control_token_offset is not None:
            if token_id >= self.control_token_offset:
                idx = token_id - self.control_token_offset
                if 0 <= idx < len(CONTROL_TOKENS):
                    return CONTROL_TOKENS[idx]
                return f"unknown_{token_id}"
        if 0 <= token_id < len(rule_registry):
            return rule_registry[token_id]
        return f"unknown_{token_id}"

    def control_token_map(self) -> Dict[str, int]:
        if not self.enable_control_tokens or self.control_token_offset is None:
            return {}
        return {
            token: self.control_token_offset + idx
            for idx, token in enumerate(CONTROL_TOKENS)
        }


__all__ = [
    "NavigationSeqModel",
    "PAD_ID",
    "BOS_ID",
    "RULE_OFFSET",
    "CONTROL_TOKENS",
]
