"""
Lexicon ingestion pipeline for sovereign text embeddings.

Provides helpers to embed structured vocabularies (WordNet, dictionary lists)
using the RPN-powered text ingestion stack and route them through the specialised
swarm for Galaxy placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Sequence
import time

import numpy as np

from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
from knowledge3d.ingestion.language.swarm_integration import (
    SovereignLanguageSwarmProcessor,
    SwarmResult,
)


def _ensure_list(value: Iterable[str]) -> List[str]:
    if isinstance(value, list):
        return value
    return list(value)


@dataclass
class LexiconIngestor:
    """
    High-level API to ingest lexicon-style resources into the Galaxy.

    Parameters
    ----------
    text_ingestor:
        Optional pre-configured :class:`SovereignTextIngestor`. Supplying a stub
        enables lightweight unit testing without GPU calls.
    swarm_processor:
        Optional :class:`SovereignLanguageSwarmProcessor` instance.
    output_root:
        Root directory where JSON artefacts are stored.
    """

    text_ingestor: SovereignTextIngestor | None = None
    swarm_processor: SovereignLanguageSwarmProcessor | None = None
    output_root: Path = Path("/K3D/Knowledge3D.local/house_zone7/lexicons/")

    def __post_init__(self) -> None:
        self.text_ingestor = self.text_ingestor or SovereignTextIngestor()
        self.swarm_processor = self.swarm_processor or SovereignLanguageSwarmProcessor()
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public ingestion routines
    # ------------------------------------------------------------------ #
    def ingest_wordnet_en(
        self,
        *,
        limit: int | None = None,
        output_path: str | Path | None = None,
        wordnet_module=None,
    ) -> dict:
        """
        Ingest English WordNet synsets.

        Parameters
        ----------
        limit:
            Optional hard cap on the number of synsets processed (useful for tests).
        output_path:
            Explicit output JSON path. Defaults to `<output_root>/wordnet_en.json`.
        wordnet_module:
            Optional module-like object providing `all_synsets()`. Allows injection
            of fakes during tests.
        """
        wn = wordnet_module or self._import_wordnet()
        synsets = list(wn.all_synsets())
        if limit is not None:
            synsets = synsets[:limit]

        results = []
        start = time.perf_counter()

        for synset in synsets:
            definition = synset.definition() or synset.name()
            examples = getattr(synset, "examples", lambda: [])() or []
            lemma = synset.name().split(".")[0]

            sentence_payload = self.text_ingestor.ingest_sentence("en", definition)
            swarm_result = self._route_through_swarm(
                sentence_payload["embedding_128"],
                modality="text",
                language="en",
            )

            results.append(
                {
                    "synset": synset.name(),
                    "lemma": lemma,
                    "definition": definition,
                    "examples": examples,
                    "position_3d": swarm_result.position_3d.tolist(),
                    "embedding": swarm_result.refined_embedding.tolist(),
                }
            )

        total_time = time.perf_counter() - start
        payload = {
            "language": "en",
            "source": "wordnet",
            "synset_count": len(results),
            "total_time_s": total_time,
            "synsets": results,
        }

        output_file = Path(output_path) if output_path else self.output_root / "wordnet_en.json"
        self._write_json(output_file, payload)
        return {"output_path": str(output_file), "synset_count": len(results), "total_time_s": total_time}

    def ingest_simple_vocabulary(
        self,
        lang: str,
        tokens: Sequence[str],
        *,
        output_path: str | Path | None = None,
        label: str | None = None,
    ) -> dict:
        """
        Ingest a plain list of vocabulary tokens.
        """
        tokens_list = _ensure_list(tokens)
        if not tokens_list:
            raise ValueError("tokens must not be empty")

        results = []
        start = time.perf_counter()

        for token in tokens_list:
            sentence_payload = self.text_ingestor.ingest_sentence(lang, token)
            swarm_result = self._route_through_swarm(
                sentence_payload["embedding_128"],
                modality="text",
                language=lang,
            )

            results.append(
                {
                    "token": token,
                    "position_3d": swarm_result.position_3d.tolist(),
                    "embedding": swarm_result.refined_embedding.tolist(),
                }
            )

        total_time = time.perf_counter() - start
        payload = {
            "language": lang,
            "token_count": len(results),
            "total_time_s": total_time,
            "label": label,
            "tokens": results,
        }

        if output_path is None:
            safe_label = label or f"{lang}_vocabulary"
            output_file = self.output_root / f"{safe_label}.json"
        else:
            output_file = Path(output_path)

        self._write_json(output_file, payload)
        return {"output_path": str(output_file), "token_count": len(results), "total_time_s": total_time}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _route_through_swarm(self, embedding_128: np.ndarray, *, modality: str, language: str) -> SwarmResult:
        embedding_128 = np.asarray(embedding_128, dtype=np.float32).reshape(128)
        result = self.swarm_processor.process_language_embedding(
            embedding_128,
            modality=modality,
            language=language,
            include_diagnostics=False,
        )
        # Ensure arrays are float32 for consistent serialisation
        result.position_3d = np.asarray(result.position_3d, dtype=np.float32)
        result.refined_embedding = np.asarray(result.refined_embedding, dtype=np.float32)
        return result

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _import_wordnet():
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "NLTK WordNet corpus is required. Install NLTK and run nltk.download('wordnet')."
            ) from exc
        try:
            wn.ensure_loaded()  # type: ignore[attr-defined]
        except Exception:
            # Older NLTK versions expose LazyCorpusLoader which loads on demand.
            pass
        return wn


__all__ = ["LexiconIngestor"]
