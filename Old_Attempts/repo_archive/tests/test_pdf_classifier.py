from pathlib import Path

from knowledge3d.ingestion.pdf_classifier import PDFKnowledgeClassifier


class _DummyResponse:
    def __init__(self, output: str, returncode: int = 0, stderr: str = "") -> None:
        self.output = output
        self.returncode = returncode
        self.stderr = stderr


class _DummyOllama:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0
        self.last_prompt = ""

    def query(self, *, model: str, prompt: str, timeout: float):
        self.calls += 1
        self.last_prompt = prompt
        return _DummyResponse(self.response_text)


def test_pdf_classifier_caches_page_decision(tmp_path: Path):
    ollama = _DummyOllama(
        '{"classification":"knowledge","confidence":0.93,"reason":"theorem page","context_needed":[],"knowledge_type":"theorem"}'
    )
    classifier = PDFKnowledgeClassifier(
        ollama=ollama,  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        model="dummy-model",
        timeout=1.0,
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 dummy")

    first = classifier.classify_page(
        pdf_path=pdf_path,
        page_num=1,
        total_pages=1,
        page_text="Theorem 1.1: ...",
    )
    second = classifier.classify_page(
        pdf_path=pdf_path,
        page_num=1,
        total_pages=1,
        page_text="Theorem 1.1: ...",
    )

    assert first["classification"] == "knowledge"
    assert second["classification"] == "knowledge"
    assert ollama.calls == 1


def test_pdf_classifier_sanitizes_null_bytes_in_prompt(tmp_path: Path):
    ollama = _DummyOllama(
        '{"classification":"knowledge","confidence":0.90,"reason":"clean","context_needed":[],"knowledge_type":"definition"}'
    )
    classifier = PDFKnowledgeClassifier(
        ollama=ollama,  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        model="dummy-model",
        timeout=1.0,
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 dummy")

    classifier.classify_page(
        pdf_path=pdf_path,
        page_num=1,
        total_pages=1,
        page_text="A\x00B\x01C theorem text",
        context_pages={2: "ctx\x00text"},
        force_reprocess=True,
    )

    assert "\x00" not in ollama.last_prompt
    assert "\x01" not in ollama.last_prompt
