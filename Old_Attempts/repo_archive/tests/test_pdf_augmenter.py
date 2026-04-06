from pathlib import Path

from knowledge3d.ingestion.pdf_augmenter import PDFKnowledgeAugmenter


class _DummyResponse:
    def __init__(self, output: str, returncode: int = 0, stderr: str = "") -> None:
        self.output = output
        self.returncode = returncode
        self.stderr = stderr


class _DummyOllama:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def query(self, *, model: str, prompt: str, timeout: float):
        return _DummyResponse(self.response_text)


def test_pdf_augmenter_generates_payload_rows():
    ollama = _DummyOllama(
        """
        {
          "summary":"Linear equation derivation from constraints",
          "entities":[{"type":"definition","name":"linear_equation","content":"ax+b=c"}],
          "relationships":[{"from":"a","relation":"constrains","to":"x"}],
          "cross_modal":{"has_equations":true,"has_figures":false,"has_algorithms":false,"notation":["a","b","c","x"]},
          "embedding_text":"linear equation isolate variable solve",
          "galaxy_hints":{"target_galaxy":"Math","pattern_type":"equation","difficulty":"introductory"}
        }
        """
    )
    augmenter = PDFKnowledgeAugmenter(
        ollama=ollama,  # type: ignore[arg-type]
        model="dummy-model",
        timeout=1.0,
    )

    augmented = augmenter.augment_page(
        pdf_path=Path("paper.pdf"),
        page_num=3,
        total_pages=10,
        page_text="Given ax+b=c, isolate x.",
        classification={"classification": "knowledge", "knowledge_type": "definition"},
    )
    rows = augmenter.to_payload_rows(
        pdf_path=Path("paper.pdf"),
        page_num=3,
        augmented=augmented,
        classification={"classification": "knowledge", "knowledge_type": "definition"},
    )

    assert augmented["galaxy_hints"]["target_galaxy"] == "Math"
    assert len(rows) == 2
    assert rows[0]["galaxy"] == "Math"
    assert rows[1]["galaxy"] == "Grammar"
    assert rows[0]["entry"]["metadata"]["augmented_by_ollama"] is True
    assert rows[0]["entry"]["metadata"]["symlink"] == "character_galaxy|word_galaxy"

