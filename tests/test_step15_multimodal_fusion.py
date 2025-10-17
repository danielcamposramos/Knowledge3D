import numpy as np
import pytest

from knowledge3d.ingestion.language.swarm_integration import (
    LanguageSwarmProcessor,
    SovereignLanguageSwarmProcessor,
)


@pytest.mark.gpu
def test_multimodal_fusion_single_modality() -> None:
    processor = LanguageSwarmProcessor()
    text_emb = np.random.randn(128).astype(np.float32)
    result = processor.fuse_multimodal_embedding(text_emb=text_emb, language="en")
    assert result["refined_embedding"].shape == (128,)
    assert result["position_3d"].shape == (3,)
    assert result["modalities_used"] == ["text"]


@pytest.mark.gpu
def test_multimodal_fusion_all_modalities() -> None:
    processor = SovereignLanguageSwarmProcessor()
    text_emb = np.random.randn(128).astype(np.float32)
    audio_emb = np.random.randn(128).astype(np.float32)
    visual_emb = np.random.randn(128).astype(np.float32)
    result = processor.fuse_multimodal_embedding(
        text_emb=text_emb,
        audio_emb=audio_emb,
        visual_emb=visual_emb,
        language="en",
    )
    assert result["refined_embedding"].shape == (128,)
    assert result["position_3d"].shape == (3,)
    assert set(result["modalities_used"]) == {"text", "audio", "visual"}
