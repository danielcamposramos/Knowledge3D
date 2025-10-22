from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

EMBEDDINGS_PATH = Path("/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl")


@pytest.fixture(scope="session")
def rpn_engine():
    """Load consolidated RPN embeddings for generalization tests."""
    if not EMBEDDINGS_PATH.exists():
        pytest.skip(
            f"RPN embeddings not found at {EMBEDDINGS_PATH}. "
            "Run ingestion + sleep consolidation first."
        )
    engine = RPNEmbeddingEngine()
    engine.load_embeddings(EMBEDDINGS_PATH)
    return engine
