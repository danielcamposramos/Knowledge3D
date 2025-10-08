from __future__ import annotations

from typing import List

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class ThinkingTagEmbedder(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_tags: int = 100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_tags)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding):
        x = self.relu(self.fc1(embedding))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x)

    def predict_thinking_tags(self, embedding: List[float], tag_names: List[str]) -> List[str]:
        if torch is None:
            return []
        with torch.no_grad():
            import torch as _t
            emb = _t.tensor(embedding, dtype=_t.float32).unsqueeze(0)
            probs = self.forward(emb).squeeze(0).tolist()
            out: List[str] = []
            for i, p in enumerate(probs):
                if i < len(tag_names) and p > 0.5:
                    out.append(tag_names[i])
            return out

