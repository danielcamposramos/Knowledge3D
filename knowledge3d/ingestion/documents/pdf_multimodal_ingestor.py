"""
High-level wrapper for the Phase C1 PDF ingestion pipeline.

Delegates per-page processing to `PDFIngestionBridge` and aggregates results.
GLB serialisation is a TODO for Phase C2; the prototype stores a lightweight
JSON sidecar so downstream tooling can inspect outcomes today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge


class PDFMultiModalIngestor:
    """Multi-page orchestrator for sovereign PDF ingestion."""

    def __init__(self) -> None:
        self.bridge = PDFIngestionBridge()

    def ingest_pdf(self, pdf_path: str | Path, output_glb: str | None = None) -> Dict[str, object]:
        pdf_path = Path(pdf_path)

        pages: List[Dict[str, object]] = []
        total_objects = 0
        total_time_ms = 0.0

        for page_num in self._enumerate_pages(pdf_path):
            result = self.bridge.ingest_pdf_page(pdf_path, page_num=page_num)
            pages.append(result)
            total_objects += int(result.get("object_count", 0))
            total_time_ms += float(result.get("processing_time_ms", 0.0))

        payload = {
            "pages": pages,
            "total_objects": total_objects,
            "total_time_ms": total_time_ms,
            "glb_path": None,
        }

        if output_glb is not None:
            self._write_placeholder_glb(pages, output_glb)
            payload["glb_path"] = output_glb

        return payload

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _enumerate_pages(self, pdf_path: Path) -> List[int]:
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as doc:
                return list(range(len(doc)))
        except Exception:
            # Fallback: assume single page if PyMuPDF is unavailable.
            return [0]

    def _write_placeholder_glb(self, pages: List[Dict[str, object]], output_path: str) -> None:
        """
        Temporary JSON serialisation until the Galaxy-native GLB writer lands.
        """
        serialisable = []
        for page in pages:
            serialisable.append(
                {
                    "galaxy_position": list(map(float, page.get("galaxy_position", [0.0, 0.0, 0.0]))),
                    "object_count": int(page.get("object_count", 0)),
                    "processing_time_ms": float(page.get("processing_time_ms", 0.0)),
                }
            )

        json_path = Path(output_path).with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump({"pages": serialisable}, handle, indent=2)
