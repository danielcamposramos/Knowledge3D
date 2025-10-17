import pytest

from knowledge3d.cranium.sovereign.loader import get_vram_usage
from knowledge3d.ingestion.wikipedia import SovereignWikipediaIngestor


@pytest.mark.gpu
def test_wikipedia_multilingual_benchmark() -> None:
    articles = [
        ("en", "Artificial_intelligence"),
        ("en", "Machine_learning"),
        ("en", "Deep_learning"),
        ("en", "Natural_language_processing"),
        ("en", "Computer_vision"),
        ("pt", "Inteligência_artificial"),
        ("pt", "Aprendizado_de_máquina"),
        ("es", "Inteligencia_artificial"),
        ("es", "Aprendizaje_automático"),
        ("en", "Neural_network"),
    ]

    ingestor = SovereignWikipediaIngestor(languages=["en", "pt", "es"])

    successes = 0
    peak_vram = 0
    total_latency = 0.0

    for idx, (lang, title) in enumerate(articles, start=1):
        try:
            result = ingestor.ingest_article(title, lang, max_sentences=30)
        except Exception as exc:
            print(f"[{idx}/10] {lang}:{title} FAILED -> {exc}")
            continue

        successes += 1
        total_latency += result["total_latency_s"]
        peak_vram = max(peak_vram, result["vram_after_bytes"])

        print(
            f"[{idx}/10] {lang}:{title} -> "
            f"{len(result['sentences'])} sentences, "
            f"{result['total_latency_s']:.2f}s total, "
            f"{result['per_sentence_latency_ms']:.2f}ms per sentence"
        )
        assert result["total_latency_s"] < 5.0, "Article exceeded 5s budget"

    _, total_bytes = get_vram_usage()
    print(f"{successes}/10 articles ingested, peak VRAM {peak_vram/1e9:.2f}GB / {total_bytes/1e9:.2f}GB")

    assert successes >= 8, "At least 8 articles should ingest successfully"
    assert peak_vram < 8e9, "VRAM usage exceeded 8GB budget"
    if successes:
        avg_latency = total_latency / successes
        assert avg_latency < 5.0, "Average article latency exceeded 5s"
