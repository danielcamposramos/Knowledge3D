from pathlib import Path

import numpy as np
import pytest

from knowledge3d.ingestion.language.sovereign_visual_pipeline import SovereignVisualIngestor


@pytest.mark.gpu
def test_sovereign_visual_glyph_ingestion() -> None:
    ingestor = SovereignVisualIngestor()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.exists():
        pytest.skip(f"Font not available: {font_path}")

    result = ingestor.ingest_glyph("A", str(font_path), "en")

    assert result["character"] == "A"
    assert result["embedding_128"].shape == (128,)
    assert result["position_3d"].shape == (3,)
    assert np.all((result["position_3d"] >= 0.0) & (result["position_3d"] <= 1.0))

    print(
        "\nVisual ingestion"
        f"\n  Position: {result['position_3d']}"
    )
