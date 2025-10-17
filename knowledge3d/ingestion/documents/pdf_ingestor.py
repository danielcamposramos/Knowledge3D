"""
PDF ingestion pipeline for the sovereign text stack.

Extracts text, generates RPN embeddings, and routes them through the specialised
swarm to obtain Galaxy-ready coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import time

import numpy as np

from knowledge3d.ingestion.language.sovereign_text_pipeline import SovereignTextIngestor
from knowledge3d.ingestion.language.swarm_integration import (
    SovereignLanguageSwarmProcessor,
    SwarmResult,
)


@dataclass
class PDFIngestor:
    """
    High-level PDF ingestion wrapper.

    Parameters
    ----------
    text_ingestor:
        Optional :class:`SovereignTextIngestor` instance (allows dependency injection).
    swarm_processor:
        Optional :class:`SovereignLanguageSwarmProcessor`.
    output_root:
        Directory where document summaries are written.
    """

    text_ingestor: SovereignTextIngestor | None = None
    swarm_processor: SovereignLanguageSwarmProcessor | None = None
    output_root: Path = Path("/K3D/Knowledge3D.local/house_zone7/documents/")
    default_max_sentences: int = 500

    def __post_init__(self) -> None:
        self.text_ingestor = self.text_ingestor or SovereignTextIngestor()
        self.swarm_processor = self.swarm_processor or SovereignLanguageSwarmProcessor()
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_pdf(
        self,
        pdf_path: str | Path,
        *,
        language: str = "en",
        max_sentences: int | None = None,
        output_path: str | Path | None = None,
    ) -> Dict[str, object]:
        """
        Ingest a single PDF file and return embedded sentence data.
        """
        pdf_path = Path(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)

        if not text.strip():
            return {
                "pdf_path": str(pdf_path),
                "sentences": [],
                "total_time_s": 0.0,
                "error": "no_text_extracted",
            }

        sentences = self._split_into_sentences(text)
        limit = max_sentences or self.default_max_sentences
        sentences = sentences[:limit]

        processed = []
        start = time.perf_counter()

        for sentence in sentences:
            try:
                payload = self.text_ingestor.ingest_sentence(language, sentence)
                swarm_result = self._route_through_swarm(
                    payload["embedding_128"],
                    modality="text",
                    language=language,
                )
                processed.append(
                    {
                        "text": sentence,
                        "position_3d": swarm_result.position_3d.tolist(),
                        "embedding": swarm_result.refined_embedding.tolist(),
                    }
                )
            except Exception as exc:  # pragma: no cover - ingestion robustness
                print(f"[PDFIngestor] Failed sentence '{sentence[:32]}…': {exc}")

        total_time = time.perf_counter() - start
        summary = {
            "pdf_path": str(pdf_path),
            "language": language,
            "total_time_s": total_time,
            "sentence_count": len(processed),
            "sentences": processed,
        }

        if output_path is not None:
            self._write_json(Path(output_path), summary)

        return summary

    def ingest_pdf_directory(
        self,
        directory: str | Path,
        *,
        language: str = "en",
        max_sentences: int | None = None,
        output_dir: str | Path | None = None,
    ) -> Dict[str, object]:
        """
        Ingest all PDFs within `directory`. Returns aggregate statistics.
        """
        directory = Path(directory)
        pdf_files = sorted(directory.glob("*.pdf"))

        aggregate = {
            "pdf_count": 0,
            "total_sentences": 0,
            "total_time_s": 0.0,
            "documents": [],
        }

        for pdf_file in pdf_files:
            result = self.ingest_pdf(
                pdf_file,
                language=language,
                max_sentences=max_sentences,
                output_path=None,
            )
            aggregate["documents"].append(result)
            aggregate["pdf_count"] += 1
            aggregate["total_sentences"] += result.get("sentence_count", 0)
            aggregate["total_time_s"] += result.get("total_time_s", 0.0)

            if output_dir is not None:
                target = Path(output_dir) / f"{pdf_file.stem}.json"
                self._write_json(target, result)

        return aggregate

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract plain text from a PDF file.

        Separated into its own method so tests can override behaviour without
        touching the underlying logic.
        """
        try:
            import PyPDF2  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyPDF2 is required for PDF ingestion. Install via `pip install PyPDF2`.") from exc

        text_fragments: List[str] = []
        with pdf_path.open("rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_fragments.append(page_text)
        return "\n".join(text_fragments)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Lightweight sentence splitter (period-based). Real pipeline can swap in
        a more advanced tokenizer if required.
        """
        raw_sentences = [segment.strip() for segment in text.replace("\r", " ").split(".")]
        return [sent for sent in raw_sentences if len(sent) > 10]

    def _route_through_swarm(self, embedding_128: np.ndarray, *, modality: str, language: str) -> SwarmResult:
        embedding_128 = np.asarray(embedding_128, dtype=np.float32).reshape(128)
        result = self.swarm_processor.process_language_embedding(
            embedding_128,
            modality=modality,
            language=language,
            include_diagnostics=False,
        )
        result.position_3d = np.asarray(result.position_3d, dtype=np.float32)
        result.refined_embedding = np.asarray(result.refined_embedding, dtype=np.float32)
        return result

    def _write_json(self, path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["PDFIngestor"]
