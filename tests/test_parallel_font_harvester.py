import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

from knowledge3d.ingestion.fonts.parallel_font_harvester import ParallelFontHarvester


def stub_render_worker(task: Tuple[str, str, str]) -> Dict[str, object]:
    char, font_path, language = task
    # Produce a simple 2×2 glyph array encoded as uint8 values.
    glyph_array = np.full((2, 2), ord(char) % 255, dtype=np.uint8)
    return {
        "char": char,
        "font_path": font_path,
        "language": language,
        "glyph_array": glyph_array,
    }


def stub_gpu_batch_processor(batch):
    results = []
    for item in batch:
        value = float(ord(item["char"]) % 5)
        results.append(
            {
                "char": item["char"],
                "font_path": item["font_path"],
                "visual_embedding": [value],
                "text_embedding": [value + 1],
                "fused_embedding": [value + 2],
                "position_3d": [0.0, 0.0, 0.0],
            }
        )
    return results


def test_parallel_font_harvester_with_stubs(tmp_path):
    # Create dummy font files so the harvester discovers them.
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    for name in ("one.ttf", "two.ttf"):
        (font_dir / name).write_bytes(b"fake-font")

    output_path = tmp_path / "glyphs.json"

    harvester = ParallelFontHarvester(
        num_workers=0,
        batch_size=3,
        render_worker=stub_render_worker,
        gpu_batch_processor=stub_gpu_batch_processor,
        characters="AB",
    )

    metrics = harvester.harvest_fonts_parallel(
        font_dir=font_dir,
        output_path=output_path,
        language="en",
    )

    assert metrics["glyph_count"] == 4.0
    data = json.loads(output_path.read_text())
    assert data["font_count"] == 2
    assert len(data["glyphs"]) == 4
