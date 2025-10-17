from pathlib import Path
from types import SimpleNamespace

import numpy as np

from knowledge3d.ingestion.documents.pdf_ingestor import PDFIngestor


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
