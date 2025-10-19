"""
Integration test validating PDF → Galaxy pipeline.

This ensures the ingestion bridge produces fused embeddings and a valid
Galaxy position after wiring AtomicFissionFusion and GraphCrystallizer.
"""

from __future__ import annotations

import numpy as np

from knowledge3d.ingestion.documents.pdf_multimodal_ingestor import PDFMultiModalIngestor


def test_pdf_to_galaxy_integration():
    ingestor = PDFMultiModalIngestor()
    pdf_path = (
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/"
        "How to think/Algorithmic.Thinking.BASE.pdf"
    )

    page = ingestor.bridge.ingest_pdf_page(pdf_path, page_num=0)

    assert "galaxy_position" in page
    assert page["galaxy_position"].shape == (3,)
    assert np.all(np.isfinite(page["galaxy_position"]))

    assert "embeddings" in page
    embeddings = page["embeddings"]
    assert embeddings.ndim == 2 and embeddings.shape[1] == 128
    assert np.all(np.isfinite(embeddings))
