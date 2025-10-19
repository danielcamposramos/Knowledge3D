import pickle
from pathlib import Path

import numpy as np
import pytest

from knowledge3d.cranium.sleep.glyph_consolidator import GlyphConsolidator


def _make_feature(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=128).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


@pytest.fixture()
def font_db_path(tmp_path: Path) -> Path:
    feature_a = _make_feature(1)
    feature_a_variant = _make_feature(1)  # identical to feature_a for merge
    feature_b = _make_feature(2)
    feature_c = _make_feature(3)

    db = {
        "FontA": {
            "font_path": "/fonts/font_a.ttf",
            "glyphs": {
                "A": {
                    "visual_features": feature_a,
                    "embedding": feature_a,
                    "confidence": 1.0,
                },
                "B": {
                    "visual_features": feature_b,
                    "embedding": feature_b,
                    "confidence": 1.0,
                },
            },
            "is_symbol_font": False,
        },
        "FontB": {
            "font_path": "/fonts/font_b.ttf",
            "glyphs": {
                "A": {
                    "visual_features": feature_a_variant,
                    "embedding": feature_a_variant,
                    "confidence": 0.9,
                },
                "C": {
                    "visual_features": feature_c,
                    "embedding": feature_c,
                    "confidence": 1.0,
                },
            },
            "is_symbol_font": False,
        },
    }

    path = tmp_path / "font_db.pkl"
    with path.open("wb") as handle:
        pickle.dump(db, handle)
    return path


def test_glyph_consolidation(font_db_path: Path, tmp_path: Path):
    consolidator = GlyphConsolidator(font_db_path)

    metrics_path = tmp_path / "glyph_metrics.jsonl"
    result = consolidator.consolidate(similarity_threshold=0.99, metrics_path=metrics_path)

    assert result.glyphs_before == 4
    assert result.glyphs_after == 3  # "A" merged across fonts
    assert 20 <= result.reduction_pct <= 40
    assert result.clusters_per_char["A"] == 1
    assert result.backup_path.exists()
    assert metrics_path.exists()

    with font_db_path.open("rb") as handle:
        new_db = pickle.load(handle)

    # Ensure only one "A" glyph remains and it corresponds to the highest-confidence font
    assert "FontA" in new_db
    assert "A" in new_db["FontA"]["glyphs"]
    assert "A" not in new_db.get("FontB", {}).get("glyphs", {})

    # Confirm each char has unique representative
    for char, count in result.clusters_per_char.items():
        surviving_fonts = [
            font_name
            for font_name, font_data in new_db.items()
            if char in font_data.get("glyphs", {})
        ]
        assert len(surviving_fonts) == count
