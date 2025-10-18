from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge
from knowledge3d.ingestion.documents.pdf_ingestor import PDFIngestor
from knowledge3d.ingestion.documents.pdf_multimodal_ingestor import PDFMultiModalIngestor


class StubTextIngestor:
    def __init__(self):
        self.calls = []

    def ingest_sentence(self, lang: str, sentence: str):
        self.calls.append((lang, sentence))
        idx = len(self.calls)
        embedding = np.full(128, idx, dtype=np.float32)
        return {"embedding_128": embedding}


class StubSwarmProcessor:
    def __init__(self):
        self.calls = []

    def process_language_embedding(self, embedding_128, *, modality, language, include_diagnostics=False):
        self.calls.append((modality, language))
        embedding = np.asarray(embedding_128, dtype=np.float32)
        position = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        return SimpleNamespace(
            refined_embedding=embedding,
            position_3d=position,
            diagnostics=None,
            modality=modality,
            language=language,
        )


class FakePDFIngestor(PDFIngestor):
    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self._text = text

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        return self._text


def test_ingest_single_pdf(tmp_path):
    text_ingestor = StubTextIngestor()
    swarm = StubSwarmProcessor()
    fake_text = "This is a sovereign embedding test sentence. Another sentence carries insight."

    ingestor = FakePDFIngestor(
        fake_text,
        text_ingestor=text_ingestor,
        swarm_processor=swarm,
        output_root=tmp_path,
    )

    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.write_bytes(b"")  # existence only; extraction is stubbed

    summary = ingestor.ingest_pdf(dummy_pdf, language="en")
    assert summary["sentence_count"] == 2
    assert text_ingestor.calls[0][1].startswith("This is")
    assert swarm.calls == [("text", "en"), ("text", "en")]


def test_ingest_pdf_directory(tmp_path):
    text_ingestor = StubTextIngestor()
    swarm = StubSwarmProcessor()
    fake_text = "First sentence with depth. Second clause expands thought."

    ingestor = FakePDFIngestor(
        fake_text,
        text_ingestor=text_ingestor,
        swarm_processor=swarm,
        output_root=tmp_path,
    )

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc1.pdf").write_bytes(b"")
    (pdf_dir / "doc2.pdf").write_bytes(b"")

    result = ingestor.ingest_pdf_directory(pdf_dir, language="en", output_dir=tmp_path / "out")
    assert result["pdf_count"] == 2
    assert result["total_sentences"] == 4

    # Ensure per-document artefacts are written
    outputs = sorted((tmp_path / "out").glob("*.json"))
    assert len(outputs) == 2


# --------------------------------------------------------------------------- #
# Phase C1 prototype tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def dummy_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 720), "Sovereign PDF bridge test page.")
    page.insert_text((72, 680), "Second line anchors layout graph.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_bridge_ingest_page(dummy_pdf: Path):
    bridge = PDFIngestionBridge()
    result = bridge.ingest_pdf_page(dummy_pdf)

    assert "galaxy_position" in result
    assert result["galaxy_position"].shape == (3,)
    assert np.all(np.isfinite(result["galaxy_position"]))

    assert "layout_graph" in result
    layout = result["layout_graph"]
    assert isinstance(layout.get("nodes"), list)
    assert layout["node_count"] == len(layout["nodes"])
    assert any(node["type"] == 1.0 for node in layout["nodes"])
    text_nodes = [node for node in layout["nodes"] if node["type"] == 1.0]
    assert text_nodes and "Sovereign" in text_nodes[0]["text_sample"]

    assert "embeddings" in result
    embeddings = result["embeddings"]
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 128
    assert layout["is_scanned"] is False
    assert result["object_count"] == layout["node_count"]

    # GPU buffers should be released after ingestion
    assert bridge.allocated_buffers == []


def test_multimodal_ingestor_single_page(dummy_pdf: Path, tmp_path: Path):
    ingestor = PDFMultiModalIngestor()
    result = ingestor.ingest_pdf(dummy_pdf, output_glb=str(tmp_path / "out.glb"))

    assert len(result["pages"]) == 1
    assert result["total_objects"] >= 0
    assert result["total_time_ms"] >= 0.0
    assert result["pages"][0]["layout_graph"]["is_scanned"] is False

    # JSON sidecar placeholder should be created
    json_sidecar = tmp_path / "out.json"
    assert json_sidecar.exists()
