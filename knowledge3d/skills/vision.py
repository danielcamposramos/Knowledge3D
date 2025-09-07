from __future__ import annotations

"""
Minimal vision skill for K3D.

Provides a best-effort `embed_image(path) -> List[float]` that prefers
OpenCLIP/CLIP when available and otherwise returns a deterministic small
hash-based vector. The function avoids large downloads or heavyweight
dependencies by degrading gracefully.
"""

import hashlib
from pathlib import Path
from typing import List


def _hash_vec(text: str, dims: int = 32) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals: List[float] = []
    i = 0
    while len(vals) < dims:
        b = h[i % len(h)]
        vals.append((b / 255.0) - 0.5)
        i += 1
    return vals


def embed_image(path: str) -> List[float]:
    p = Path(path)
    # Try OpenCLIP
    try:  # pragma: no cover
        import torch  # type: ignore
        import open_clip  # type: ignore
        from PIL import Image  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat.squeeze(0).detach().cpu().float().tolist()
        # Reduce to 32 dims by hashing buckets for consistency
        if len(vec) > 32:
            step = max(1, len(vec) // 32)
            buckets = [sum(vec[i : i + step]) / step for i in range(0, len(vec), step)]
            return buckets[:32]
        return vec
    except Exception:
        pass
    # Try CLIP
    try:  # pragma: no cover
        import torch  # type: ignore
        import clip  # type: ignore
        from PIL import Image  # type: ignore

        model, preprocess = clip.load("ViT-B/32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat.squeeze(0).detach().cpu().float().tolist()
        if len(vec) > 32:
            step = max(1, len(vec) // 32)
            buckets = [sum(vec[i : i + step]) / step for i in range(0, len(vec), step)]
            return buckets[:32]
        return vec
    except Exception:
        pass
    # Fallback deterministic embedding
    return _hash_vec(p.as_posix(), 32)

