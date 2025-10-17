import tempfile

import numpy as np
import pytest
import soundfile as sf

from knowledge3d.ingestion.language.sovereign_audio_pipeline import SovereignAudioIngestor


@pytest.mark.gpu
def test_sovereign_audio_phoneme_ingestion() -> None:
    ingestor = SovereignAudioIngestor()

    sr = ingestor.target_sr
    duration = ingestor.clip_duration
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        result = ingestor.ingest_phoneme(tmp.name, "/a/", "en")

    assert result["phoneme"] == "/a/"
    assert result["embedding_128"].shape == (128,)
    assert result["position_3d"].shape == (3,)
    assert result["formants"].shape == (3,)
    assert np.all((result["position_3d"] >= 0.0) & (result["position_3d"] <= 1.0))

    print(
        "\nAudio ingestion"
        f"\n  Formants: {result['formants']}"
        f"\n  Position: {result['position_3d']}"
    )
