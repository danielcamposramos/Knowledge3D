from __future__ import annotations

from typing import List

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class _DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def predict_shape(self, embedding: List[float]) -> str:
        # simple heuristic fallback
        n = len(embedding)
        if n >= 256: return 'icosahedron'
        if n >= 128: return 'cube'
        return 'tetrahedron'


if torch is None:
    ShapeRecognizer = _DummyModel  # type: ignore
else:
    class ShapeRecognizer(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_classes: int = 14):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()

        def forward(self, embedding):
            x = self.relu(self.fc1(embedding))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

        def predict_shape(self, embedding: List[float]) -> str:
            with torch.no_grad():
                import torch as _t
                import numpy as _np
                # pad/trim to input dim
                d = self.fc1.in_features
                v = list(embedding or [])
                if len(v) < d: v = v + [0.0]*(d-len(v))
                if len(v) > d: v = v[:d]
                emb_tensor = _t.tensor([v], dtype=_t.float32)
                logits = self.forward(emb_tensor)
                idx = int(_t.argmax(logits, dim=1).item())
                return self.idx_to_shape(idx)

        def idx_to_shape(self, idx: int) -> str:
            shapes = [
                'tetrahedron', 'cube', 'octahedron', 'icosahedron', 'dodecahedron',
                'triangular_prism', 'pentagonal_prism', 'rhombic_dodecahedron',
                'truncated_icosahedron', 'snub_dodecahedron', 'great_rhombicuboctahedron',
                'omnitruncated_icosahedron', 'hypersphere_projection', 'fractal_tree'
            ]
            return shapes[idx] if 0 <= idx < len(shapes) else 'unknown'
