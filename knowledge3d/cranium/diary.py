from __future__ import annotations

"""
DiaryPolicy — decides if and when to write AI diary pages.

Signals considered:
- Novelty: cosine distance between current STM snapshot and last diary page.
- Event confidence: e.g., navigation TF‑IDF similarity score.
- Reflection events: favored for insight logging.

Thresholds (env‑configurable):
- K3D_DIARY_NOVELTY (default 0.382) — minimum cosine distance to count as novel.
- K3D_DIARY_GOOD (default 0.618) — confidence threshold for “good feeling”.
- K3D_DIARY_BAD  (default 0.382) — confidence threshold for “bad feeling”.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(y * y for y in b)) + 1e-9
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


def _envf(name: str, default: float) -> float:
    try:
        import os
        v = os.getenv(name, "").strip()
        return float(v) if v else default
    except Exception:
        return default


@dataclass
class DiaryPolicy:
    novelty_thr: float = _envf("K3D_DIARY_NOVELTY", 0.382)  # 1 - 1/phi
    good_thr: float = _envf("K3D_DIARY_GOOD", 0.618)
    bad_thr: float = _envf("K3D_DIARY_BAD", 0.382)

    def should_write(self, vec32: List[float], last_vec32: Optional[List[float]], event: str, meta: Dict[str, object]) -> bool:
        # Always normalize to 32 dims
        v = (vec32 + [0.0] * 32)[:32]
        # Reflection events are good candidates; still require minimal novelty
        if event.startswith("reflect"):
            if last_vec32 is None:
                return True
            dist = 1.0 - _cosine(v, (last_vec32 + [0.0] * 32)[:32])
            return dist >= (self.novelty_thr * 0.5)
        # Navigation: prefer high confidence discoveries or contradictions
        if event == "navigate":
            score = float(meta.get("score", 0.0)) if isinstance(meta.get("score"), (int, float)) else None
            if score is not None:
                if score >= self.good_thr:
                    return True
                if score <= self.bad_thr:
                    return True
            # Otherwise rely on novelty vs. last page
            if last_vec32 is None:
                return True
            dist = 1.0 - _cosine(v, (last_vec32 + [0.0] * 32)[:32])
            return dist >= self.novelty_thr
        # Brain sleep: always write a summary page
        if event == "sleep":
            return True
        # Default: require novelty
        if last_vec32 is None:
            return True
        dist = 1.0 - _cosine(v, (last_vec32 + [0.0] * 32)[:32])
        return dist >= self.novelty_thr

    def feeling(self, event: str, meta: Dict[str, object]) -> Optional[str]:
        # Map simple confidence signals to feelings for humans
        if event == "navigate":
            try:
                score = float(meta.get("score", 0.0))
            except Exception:
                score = 0.0
            if score >= self.good_thr:
                return "good"
            if score <= self.bad_thr:
                return "bad"
        return None

