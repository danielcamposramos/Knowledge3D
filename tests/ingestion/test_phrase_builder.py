import json
from pathlib import Path

from knowledge3d.ingestion.atomic.phrase_builder import build_phrase_stars


def _write_tmp(tmp_path: Path, lines):
    p = tmp_path / "phrases.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return p


def test_phrase_builder_idiom(tmp_path):
    lines = [
        {"lang": "en", "phrase": "kick the bucket", "meaning": "die", "usage": "idiom", "sense": "default"},
    ]
    path = _write_tmp(tmp_path, lines)
    stars = build_phrase_stars(path)
    assert len(stars) == 1
    star = stars[0]
    assert star["phrase_id"].startswith("PHRASE_en_kick_the_bucket")
    assert len(star["word_refs"]) == 3
