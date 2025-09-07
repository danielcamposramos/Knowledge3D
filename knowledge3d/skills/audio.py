from __future__ import annotations

"""
Minimal audio skill for K3D.

Provides a best-effort `embed_audio(path) -> List[float]` that prefers
torchaudio (MFCC/log-mel) when available and otherwise returns a
deterministic small hash-based vector.
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


def embed_audio(path: str) -> List[float]:
    p = Path(path)
    try:  # pragma: no cover
        import torch  # type: ignore
        import torchaudio  # type: ignore

        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(dim=0, keepdim=True)  # mono
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(wav)
        feat = spec.mean(dim=-1).squeeze(0)
        vec = feat.detach().cpu().float().tolist()
        if len(vec) > 32:
            step = max(1, len(vec) // 32)
            buckets = [sum(vec[i : i + step]) / step for i in range(0, len(vec), step)]
            return buckets[:32]
        return vec
    except Exception:
        pass
    return _hash_vec(p.as_posix(), 32)

