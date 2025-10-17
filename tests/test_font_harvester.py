import json
from pathlib import Path

import numpy as np

from knowledge3d.ingestion.fonts.font_harvester import FontGlyphHarvester


class StubVisualIngestor:
    def ingest_glyph(self, char: str, font_path: Path, language: str):
        base = (ord(char) % 5) + 1
        embedding = np.full(128, base, dtype=np.float32)
        return {
            "character": char,
            "font_family": font_path.stem,
            "position_3d": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "embedding_128": embedding,
            "language": language,
        }


class StubTextIngestor:
    def __init__(self):
        self.calls = []

    def ingest_sentence(self, lang: str, sentence: str):
        self.calls.append((lang, sentence))
        embedding = np.full(128, len(self.calls), dtype=np.float32)
        node = (sentence, np.array([0.01, 0.02, 0.03], dtype=np.float32), 0)
        return {"embedding_128": embedding, "nodes": [node]}


class StubSwarmProcessor:
    def fuse_multimodal_embedding(self, *, text_emb, visual_emb, language, include_diagnostics=False):
        fused = (np.asarray(text_emb) + np.asarray(visual_emb)) / 2.0
        return {
            "refined_embedding": fused.astype(np.float32),
            "position_3d": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "diagnostics": None,
            "language": language,
            "modalities_used": ["text", "visual"],
        }


def test_harvest_single_font(tmp_path):
    font_file = tmp_path / "fake.ttf"
    font_file.write_bytes(b"")  # Placeholder file

    harvester = FontGlyphHarvester(
        visual_ingestor=StubVisualIngestor(),
        text_ingestor=StubTextIngestor(),
        swarm_processor=StubSwarmProcessor(),
        output_path=tmp_path / "fonts.json",
    )

    result = harvester.harvest_font_glyphs(font_file, characters="AB")
    assert result["glyph_count"] == 2
    assert result["glyphs"][0]["char"] == "A"
    assert len(result["glyphs"][0]["visual_embedding"]) == 128


def test_harvest_font_directory(tmp_path):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts" / "one.ttf").write_bytes(b"")
    (tmp_path / "fonts" / "two.otf").write_bytes(b"")

    output_path = tmp_path / "out" / "glyphs.json"

    harvester = FontGlyphHarvester(
        visual_ingestor=StubVisualIngestor(),
        text_ingestor=StubTextIngestor(),
        swarm_processor=StubSwarmProcessor(),
        output_path=output_path,
    )

    summary = harvester.harvest_font_directory(tmp_path / "fonts", max_fonts=1, characters="A")
    assert summary["font_count"] == 1

    payload = json.loads(output_path.read_text())
    assert payload["font_count"] == 1
    assert payload["fonts"][0]["glyph_count"] == 1
