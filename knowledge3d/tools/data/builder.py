from __future__ import annotations

"""Build Galaxy stars/GLBs for meaning themes."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np  # type: ignore

from knowledge3d.tools.data.utils import ensure_symlink

SSD_BASE = Path("/K3D/Knowledge3D.local/datasets")
GALAXY_DIR = Path("viewer/public/galaxy/working")
GALAXY_DIR.mkdir(parents=True, exist_ok=True)

ZONE_BY_THEME = {
    "galaxy_geometry": "Zone 1 (Entrance)",
    "house_zone5": "Zone 5 (Knowledge Garden)",
    "house_zone7": "Zone 7 (Mirror Room)",
}


def build_theme_glbs(theme: str, max_files: int = 200) -> None:
    """Convert curated assets into Galaxy star JSONs.

    Each asset becomes a star with a hashed embedding (512-d floats) so that
    the meaning trainer can mutate it later.
    """

    theme_path = SSD_BASE / theme
    if not theme_path.exists():
        raise ValueError(f"Theme path not found: {theme_path}")

    zone = ZONE_BY_THEME.get(theme, "Zone 1 (Entrance)")
    processed = 0
    for modality_path in sorted(theme_path.glob("*/")):
        modality = modality_path.name
        for dataset_path in sorted(modality_path.glob("*/")):
            files = sorted(f for f in dataset_path.iterdir() if f.is_file())
            for file_path in files:
                if processed >= max_files:
                    break
                star_id = slugify(f"star_{theme}_{modality}_{file_path.stem}")
                star_path = GALAXY_DIR / f"{star_id}.json"
                if star_path.exists():
                    continue
                embedding = hashed_embedding(f"{theme}|{modality}|{file_path.stem}")
                star = {
                    "type": "star",
                    "id": star_id,
                    "name": f"{theme}:{modality}:{file_path.stem}",
                    "created_at": None,
                    "honesty_score": 0.5,
                    "embedding": embedding,
                    "modality": modality,
                    "source_file": str(file_path),
                    "zone_placement": zone,
                }
                star_path.write_text(json.dumps(star, ensure_ascii=False, indent=2), encoding="utf-8")
                processed += 1
            if processed >= max_files:
                break
    print(f"🌟 Built {processed} stars for theme '{theme}' → {GALAXY_DIR}")


def hashed_embedding(text: str, dim: int = 512) -> Iterable[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    while len(vals) < dim:
        for byte in h:
            vals.append(((byte / 255.0) * 2.0) - 1.0)
            if len(vals) >= dim:
                break
        h = hashlib.sha256(h).digest()
    return [float(v) for v in vals[:dim]]


def slugify(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in value)


__all__ = ["build_theme_glbs"]
