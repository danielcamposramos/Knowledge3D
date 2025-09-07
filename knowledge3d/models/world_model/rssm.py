from __future__ import annotations

import torch
import torch.nn as nn


class RSSM(nn.Module):
    """Tiny recurrent model that predicts next 3D point from history.

    This is a minimal stand-in world model (frontier base). It consumes a
    sequence of 3D observations and predicts the next 3D vector.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_size=3, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3] -> predicts y: [B, 3] for next-step
        out, h = self.rnn(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y

