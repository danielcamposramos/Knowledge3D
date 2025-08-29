from __future__ import annotations

import re
import unicodedata
from typing import Tuple


_WS_RE = re.compile(r"\s+")
_BRACKETS_RE = re.compile(r"\s*\[\s*([^\]]+)\s*\]")


def normalize_text(s: str) -> str:
    # Unicode normalize + lowercase
    t = unicodedata.normalize("NFKC", s).strip().lower()
    # collapse whitespace
    t = _WS_RE.sub(" ", t)
    # normalize number lists inside brackets: [a, b, c] -> [a,b,c]
    def _tighten(m):
        inner = m.group(1)
        inner = ",".join([p.strip() for p in inner.split(",")])
        return f"[{inner}]"
    t = _BRACKETS_RE.sub(_tighten, t)
    return t


def parse_coords(t: str) -> Tuple[float, float, float] | None:
    # Expect pattern teleport to [x,y,z]
    m = re.search(r"teleport\s+to\s*\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]", t)
    if not m:
        return None
    x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return (x, y, z)

