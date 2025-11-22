from pathlib import Path


def test_drawing_grammar_spec_exists():
    spec = Path("docs/research/DRAWING_GRAMMAR_SPEC.md")
    assert spec.exists()
    text = spec.read_text(encoding="utf-8")
    assert "Primitives" in text and "Books/Collections" in text
