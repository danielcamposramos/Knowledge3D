"""
Sovereign Wikipedia ingestion pipeline.

Fetches Wikipedia articles, converts sentences to sovereign embeddings, and
processes them through the specialised swarm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import wikipediaapi

from knowledge3d.cranium.sovereign.loader import get_vram_usage
from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
from knowledge3d.ingestion.language.swarm_integration import LanguageSwarmProcessor


def _split_sentences(text: str, max_sentences: int) -> List[str]:
    raw = [segment.strip() for segment in text.split(".") if len(segment.strip()) > 8]
    return raw[:max_sentences]


@dataclass
class SovereignWikipediaIngestor:
    """
    Sovereign ingestion wrapper for Wikipedia articles.
    """

    languages: Sequence[str] = ("en", "pt", "es")
    max_sentences: int = 100

    def __post_init__(self) -> None:
        self._wiki_clients: Dict[str, wikipediaapi.Wikipedia] = {
            lang: wikipediaapi.Wikipedia(
                language=lang,
                user_agent="Knowledge3D/1.0 (Sovereign AI Research)",
            )
            for lang in self.languages
        }
        self._text_ingestor = SovereignTextIngestor(languages=self.languages)
        self._swarm = LanguageSwarmProcessor()

    def ingest_article(self, title: str, lang: str = "en", max_sentences: int | None = None) -> Dict[str, object]:
        """
        Ingest a single Wikipedia article.
        """
        client = self._wiki_clients.get(lang)
        if client is None:
            raise ValueError(f"Language '{lang}' is not configured")

        page = client.page(title)
        if not page.exists():
            raise ValueError(f"Article '{title}' not found in language '{lang}'")

        limit = max_sentences if max_sentences is not None else self.max_sentences
        sentences = _split_sentences(page.text, limit)
        results: List[Dict[str, object]] = []

        start = time.perf_counter()
        vram_before, total_vram = self._safe_vram_read()

        for sentence in sentences:
            sentence_info = self._text_ingestor.ingest_sentence(lang, sentence)
            swarm_info = self._swarm.fuse_multimodal_embedding(
                text_emb=sentence_info["embedding_128"],
                language=lang,
                include_diagnostics=False,
            )

            results.append(
                {
                    "text": sentence,
                    "nodes": sentence_info["nodes"],
                    "edges": sentence_info["edges"],
                    "position_3d": swarm_info["position_3d"],
                    "embedding": swarm_info["refined_embedding"],
                }
            )

        total_latency = time.perf_counter() - start
        vram_after, _ = self._safe_vram_read()

        per_sentence_ms = (
            (total_latency / len(results)) * 1000.0 if results else 0.0
        )

        return {
            "title": title,
            "language": lang,
            "sentences": results,
            "total_latency_s": total_latency,
            "per_sentence_latency_ms": per_sentence_ms,
            "vram_before_bytes": vram_before,
            "vram_after_bytes": vram_after,
            "total_vram_bytes": total_vram,
        }

    @staticmethod
    def _safe_vram_read() -> Tuple[int, int]:
        try:
            return get_vram_usage()
        except RuntimeError:
            return 0, int(12e9)


__all__ = ["SovereignWikipediaIngestor"]
