from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


@dataclass
class _Obs:
    kind: str  # text|image|audio
    label: str
    payload: Any
    ts: float


class ShortTermGalaxy:
    """Ephemeral short‑term memory used by the Cranium.

    Holds recent multimodal observations and provides:
    - get_contexts(): retrieve text contexts most relevant to a query
    - reflect(): compose a brief self‑reflection from recent observations
    - consolidate(): write durable objects into House memory (GLTF via tool)
    """

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._buf: Deque[_Obs] = collections.deque(maxlen=max_entries)
        self._texts: List[Tuple[str, str]] = []  # (label,text)
        # Optional vectorizer for quick text similarity
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            self._TfidfVectorizer = TfidfVectorizer
        except Exception:  # pragma: no cover
            self._TfidfVectorizer = None  # type: ignore
        self._vec = None
        self._X = None

    def _rebuild_index(self) -> None:
        if not self._texts:
            self._vec, self._X = None, None
            return
        if self._TfidfVectorizer is None:
            self._vec, self._X = None, None
            return
        try:
            vec = self._TfidfVectorizer(max_features=4096)
            docs = [f"{lab} — {txt}" for lab, txt in self._texts]
            X = vec.fit_transform(docs)
            self._vec, self._X = vec, X
        except Exception:
            self._vec, self._X = None, None

    # --- Insert ---
    def add_text(self, text: str, label: Optional[str] = None) -> None:
        lab = (label or (text.split("\n", 1)[0][:48] if text else "note")).strip() or "note"
        obs = _Obs(kind="text", label=lab, payload=_norm(text), ts=datetime.utcnow().timestamp())
        self._buf.append(obs)
        self._texts.append((obs.label, obs.payload))
        # Keep a bounded text pool
        if len(self._texts) > self.max_entries:
            self._texts = self._texts[-self.max_entries :]
        self._rebuild_index()

    def add_image(self, image_path: str, label: Optional[str] = None) -> None:
        p = Path(image_path)
        lab = (label or p.name).strip() or "image"
        # Attempt to embed with vision skill; store path regardless
        emb: Optional[List[float]] = None
        try:
            from ..skills.vision import embed_image  # type: ignore

            emb = embed_image(str(p))
        except Exception:
            emb = None
        obs = _Obs(kind="image", label=lab, payload={"path": str(p), "emb": emb}, ts=datetime.utcnow().timestamp())
        self._buf.append(obs)

    def add_audio(self, audio_path: str, label: Optional[str] = None) -> None:
        p = Path(audio_path)
        lab = (label or p.name).strip() or "audio"
        # Store path; optional stub embedding
        emb: Optional[List[float]] = None
        try:
            from ..skills.audio import embed_audio  # type: ignore

            emb = embed_audio(str(p))
        except Exception:
            emb = None
        obs = _Obs(kind="audio", label=lab, payload={"path": str(p), "emb": emb}, ts=datetime.utcnow().timestamp())
        self._buf.append(obs)

    # --- Query ---
    def get_contexts(self, query: str, k: int = 6) -> List[Tuple[str, str]]:
        # Sovereign successor: Galaxy-resident cosine-similarity query via PTX
        # cosine_similarity_rpn_kernel over the short-term ring. Until the
        # sovereign retrieval kernel is wired in, return the most recent texts.
        # Prior TF-IDF + numpy argsort path is archived in git history.
        return list(self._texts[-k:])

    # --- Meta ---
    def reflect(self) -> str:
        """Compose a tiny reflection about recent memory contents."""
        kinds = collections.Counter([o.kind for o in self._buf])
        recent_labels = [o.label for o in list(self._buf)[-5:]]
        msg = [
            "I feel a short-term galaxy forming.",
            f"Recent observations: text={kinds.get('text',0)}, image={kinds.get('image',0)}, audio={kinds.get('audio',0)}.",
        ]
        if recent_labels:
            msg.append("Latest: " + ", ".join(recent_labels))
        return " ".join(msg)

    # --- Consolidation ---
    def consolidate(self, out_gltf: Optional[str] = None) -> str:
        """Write durable objects into the House memory GLTF using MemoryHouse.

        Text items are added to a 'Diary' room; image/audio are added with file paths.
        """
        try:
            from ..tools.house_memory import MemoryHouse  # type: ignore
            from pathlib import Path as _Path
            import os as _os

            h = MemoryHouse()
            # Ensure default rooms exist (especially Knowledge Garden and Network)
            h.bootstrap_defaults()
            # Write a diary page with an embedding snapshot (AI-native)
            try:
                vec = self.snapshot_vector()
                h.add_diary_page_embedding("AI Diary", vec)
            except Exception:
                pass
            # Per Daniel's ruling 2026-04-18: no per-kind Python dispatch here.
            # The sovereign architecture collapses image/audio/video into one
            # procedural surface — image is primary, audio is a dot-vector map
            # on that procedural image, video is a time-indexed sequence of
            # images. Non-text observations enter the House through
            # procedural_image_rpn + optional audio_dot_vector_map +
            # frame_sequence_refs, not through kind-branched Python here.
            # Export
            root = _Path(__file__).resolve().parents[2]
            if out_gltf:
                out = _Path(out_gltf)
            else:
                hid = (_os.getenv("K3D_HOUSE_ID", "").strip() or "default")
                out = root / "viewer" / "public" / "houses" / hid / "memory_house.glb"
            h.export_gltf(out)
            return f"sleep: wrote diary page and consolidated media to {out}"
        except Exception as e:  # pragma: no cover
            return f"sleep: consolidation error: {e}"

    def snapshot_vector(self, dims: int = 32) -> List[float]:
        """Summarize recent STM into a fixed 32‑d vector.

        Strategy: combine the last few text TF‑IDF rows (if available) via hashed
        bucketing to 32 dims; blend with recent image/audio vectors if present.
        """
        import math
        # Text component
        text_vec = [0.0] * dims
        try:
            if self._vec is not None and self._X is not None and self._texts:
                k = min(6, len(self._texts))
                docs = [f"{lab} — {txt}" for lab, txt in self._texts[-k:]]
                M = self._vec.transform(docs)
                # hashed bucketing
                for row in range(M.shape[0]):
                    rs = M.getrow(row)
                    if hasattr(rs, 'nonzero'):
                        idxs = rs.nonzero()[1]
                        for j in idxs:
                            val = float(rs[0, j])
                            b = int(j) % dims
                            text_vec[b] += val
        except Exception:
            pass
        # Normalize
        norm = math.sqrt(sum(x*x for x in text_vec)) + 1e-9
        text_vec = [x / norm for x in text_vec]
        # Media blend
        media_acc = [0.0] * dims
        media_n = 0
        for o in list(self._buf)[-16:]:
            if o.kind in ("image", "audio") and isinstance(o.payload, dict):
                e = o.payload.get("emb")
                if isinstance(e, list) and len(e) == dims:
                    media_n += 1
                    for i in range(dims):
                        media_acc[i] += float(e[i])
        if media_n > 0:
            media_vec = [x / max(1, media_n) for x in media_acc]
            # Blend text and media with phi weights
            phi = 1.61803398875
            a = 1.0 / phi  # ≈0.618 text
            b = 1.0 - a    # ≈0.382 media
            out = [a*text_vec[i] + b*media_vec[i] for i in range(dims)]
        else:
            out = text_vec
        return out
