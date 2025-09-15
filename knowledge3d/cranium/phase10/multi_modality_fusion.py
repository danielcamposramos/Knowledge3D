from __future__ import annotations

from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore
    F = None  # type: ignore


if torch is None:
    class MultiModalityFusion:  # type: ignore
        def __init__(self, input_dim: int = 512):
            self.input_dim = input_dim
        def forward(self, input_data: Dict) -> List[float]:
            # Fallback: naive concatenation of scalars if present
            feats: List[float] = []
            for k in ("text", "image", "audio", "video", "spatial"):
                v = input_data.get(k)
                if isinstance(v, (list, tuple)):
                    feats.extend([float(x) for x in v][: self.input_dim])
            if len(feats) < self.input_dim:
                feats += [0.0] * (self.input_dim - len(feats))
            return feats[: self.input_dim]
else:
    class _Attn(nn.Module):  # simple feature attention (not sequence MHA)
        def __init__(self, d: int):
            super().__init__()
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
        def forward(self, x):  # x: [B, N, D]
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            att = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (x.shape[-1] ** 0.5), dim=-1)  # [B,N,N]
            out = torch.matmul(att, v)  # [B,N,D]
            return out.mean(dim=1)  # [B,D]

    class _Proj(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.lin = nn.Linear(in_dim, out_dim)
            self.act = nn.ReLU()
        def forward(self, x):
            return self.act(self.lin(x))

    class MultiModalityFusion(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            # Lazy-initialize projections on first use based on input shapes
            self.text_proj: Optional[nn.Module] = None
            self.audio_proj: Optional[nn.Module] = None
            self.video_proj: Optional[nn.Module] = None
            self.spatial_proj: Optional[nn.Module] = None
            self.image_conv: Optional[nn.Module] = None
            self.attn = _Attn(self.input_dim)

        def _ensure_proj(self, key: str, x):
            if key == "image":
                if self.image_conv is None:
                    # Expect [B, C=3, H, W]; stride to coarse tokens then flatten
                    self.image_conv = nn.Conv2d(3, self.input_dim, kernel_size=16, stride=16)
                return self.image_conv(x)
            else:
                in_dim = x.shape[-1]
                proj_attr = {
                    "text": "text_proj",
                    "audio": "audio_proj",
                    "video": "video_proj",
                    "spatial": "spatial_proj",
                }[key]
                if getattr(self, proj_attr) is None:
                    setattr(self, proj_attr, _Proj(in_dim, self.input_dim))
                return getattr(self, proj_attr)(x)

        def forward(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
            feats = []
            for key in ("text", "image", "audio", "video", "spatial"):
                if key not in input_data or input_data[key] is None:
                    continue
                x = input_data[key]
                # Normalize shapes to [B, D]
                if key == "image":
                    x = self._ensure_proj("image", x)  # [B, D, H', W']
                    x = torch.flatten(x, start_dim=2).mean(dim=2)  # [B, D]
                else:
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    if x.dim() == 2:
                        x = self._ensure_proj(key, x)
                    else:
                        x = x.view(x.shape[0], -1)
                        x = self._ensure_proj(key, x)
                feats.append(x)
            if not feats:
                # Return zeros if nothing provided
                return torch.zeros((1, self.input_dim), dtype=torch.float32)
            # Stack as feature tokens [B, N, D]
            tokens = torch.stack(feats, dim=1) if len(feats) > 1 else feats[0].unsqueeze(1)
            fused = self.attn(tokens)  # [B, D]
            return fused

