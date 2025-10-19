#!/usr/bin/env python3
"""
Knowledge Ingestion - Full Corpus

Ingest all knowledge sources into K3D Galaxy:
1. Local PDF libraries
2. ArXiv papers dataset
3. GitHub code dataset

Leverages Phase B/C/D pipeline with automatic sleep consolidation.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge


class KnowledgeIngestor:
    """Orchestrate ingestion of all knowledge sources."""

    def __init__(self) -> None:
        self.bridge = PDFIngestionBridge()
        self.rpn_engine = self.bridge.rpn_engine
        self.metrics_path = Path(
            "/K3D/Knowledge3D.local/logs/ingestion_metrics.jsonl"
        )
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def log_metrics(self, source: str, metrics: dict) -> None:
        """Append ingestion metrics to JSONL log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _print_banner(self, title: str) -> None:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Local PDFs
    # ------------------------------------------------------------------ #
    def ingest_local_pdfs(self) -> None:
        """Ingest local PDF libraries."""
        self._print_banner("STEP 1: Ingesting Local PDF Libraries")

        base_path = Path(
            "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries"
        )
        pdf_files = list(base_path.rglob("*.pdf")) + list(base_path.rglob("*.PDF"))

        print(f"Found {len(pdf_files)} PDFs")

        try:
            import fitz  # type: ignore
        except ImportError:
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF ingestion. Install via `pip install pymupdf`."
            )

        total_pages = 0
        total_objects = 0
        start_time = time.time()

        for idx, pdf_path in enumerate(pdf_files, 1):
            try:
                doc = fitz.open(str(pdf_path))
                num_pages = len(doc)
                print(f"\n[{idx}/{len(pdf_files)}] {pdf_path.name} ({num_pages} pages)")

                for page_num in range(num_pages):
                    result = self.bridge.ingest_pdf_page(str(pdf_path), page_num)
                    total_objects += len(result["objects"])
                    total_pages += 1

                    if (page_num + 1) % 10 == 0 or (page_num + 1) == num_pages:
                        method = result.get("method", "structured")
                        print(
                            f"  Page {page_num + 1:>4}/{num_pages:<4} "
                            f"→ {len(result['objects']):3d} objects ({method})"
                        )

                doc.close()

            except Exception as exc:  # pragma: no cover - ingestion resiliency
                print(f"  ERROR ingesting {pdf_path.name}: {exc}")
                continue

        elapsed = time.time() - start_time
        ms_per_page = (elapsed * 1000) / total_pages if total_pages else 0.0

        metrics = {
            "pdfs": len(pdf_files),
            "pages": total_pages,
            "objects": total_objects,
            "elapsed_seconds": elapsed,
            "ms_per_page": ms_per_page,
        }
        self.log_metrics("local_pdfs", metrics)

        print(
            f"\n✅ Local PDFs complete: {total_pages} pages in {elapsed:.1f}s "
            f"({ms_per_page:.1f} ms/page)"
        )

    # ------------------------------------------------------------------ #
    # ArXiv dataset
    # ------------------------------------------------------------------ #
    def ingest_arxiv_papers(self) -> None:
        """Ingest ArXiv papers dataset from HuggingFace."""
        self._print_banner("STEP 2: Ingesting ArXiv Papers Dataset")

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise RuntimeError(
                "huggingface-datasets is required. Install via `pip install datasets`."
            )

        try:
            print("Downloading dataset preview (first 10 entries)...")
            dataset = load_dataset("nick007x/arxiv-papers", split="train[:10]")
            print(f"Loaded {len(dataset)} sample papers")
            print("\nDataset features:", dataset.features)
            print("\nSample entry:")
            print(dataset[0])

            metrics = {
                "papers_sampled": len(dataset),
                "status": "analysis_complete",
            }
            self.log_metrics("arxiv_papers", metrics)
            print(
                "⚠️  Full ingestion strategy requires tailoring to dataset structure."
            )

        except Exception as exc:
            print(f"ERROR accessing ArXiv dataset: {exc}")
            metrics = {"error": str(exc)}
            self.log_metrics("arxiv_papers", metrics)

    # ------------------------------------------------------------------ #
    # GitHub dataset
    # ------------------------------------------------------------------ #
    def ingest_github_code(self) -> None:
        """Ingest GitHub code dataset from HuggingFace."""
        self._print_banner("STEP 3: Ingesting GitHub Code Dataset")

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise RuntimeError(
                "huggingface-datasets is required. Install via `pip install datasets`."
            )

        try:
            print("Downloading dataset preview (first 10 entries)...")
            dataset = load_dataset("nick007x/github-code-2025", split="train[:10]")
            print(f"Loaded {len(dataset)} sample files")
            print("\nDataset features:", dataset.features)
            print("\nSample entry:")
            print(dataset[0])

            metrics = {
                "code_sampled": len(dataset),
                "status": "analysis_complete",
            }
            self.log_metrics("github_code", metrics)
            print(
                "⚠️  Full ingestion strategy requires tailoring to dataset structure."
            )

        except Exception as exc:
            print(f"ERROR accessing GitHub dataset: {exc}")
            metrics = {"error": str(exc)}
            self.log_metrics("github_code", metrics)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_all(self) -> None:
        """Run all ingestion steps."""
        self._print_banner("K3D KNOWLEDGE INGESTION - Full Corpus")
        print(f"Start time: {datetime.now().isoformat()}")

        scheduler_state = "Active"
        if not getattr(self.bridge, "sleep_scheduler", None):
            scheduler_state = "Inactive"
        print(f"Sleep scheduler: {scheduler_state}")

        start_time = time.time()

        # Step 1: Local PDFs
        self.ingest_local_pdfs()

        # Step 2: ArXiv papers
        self.ingest_arxiv_papers()

        # Step 3: GitHub code
        self.ingest_github_code()

        elapsed = time.time() - start_time

        self._print_banner("INGESTION COMPLETE")
        print(f"Total time: {elapsed / 3600:.1f} hours")
        print(f"Metrics log: {self.metrics_path}")
        print("Sleep consolidation will trigger ~5 minutes after ingestion finishes.")


if __name__ == "__main__":
    KnowledgeIngestor().run_all()
