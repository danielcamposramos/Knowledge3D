#!/usr/bin/env python3
"""
Phase G Knowledge Ingestion - Full AGI Integration

Complete knowledge ingestion pipeline with:
1. Adaptive variable-dimension embeddings (64-2048D)
2. Trained Phase G specialists (multimodal, OCR, speech, router)
3. Galaxy star creation (knowledge storage in 3D space)
4. Shadow weights (safe model updates)
5. Two sleep cycles (model logic vs knowledge consolidation)

Key Innovation:
- Single phrase → 64D (256× faster than 2048D!)
- Full page → 512D (16× faster than 2048D!)
- Complex document → 1024D or 2048D

Usage:
    python scripts/ingest_knowledge_phase_g.py --library echosystems
    python scripts/ingest_knowledge_phase_g.py --library echosystems --max-pdfs 10
    python scripts/ingest_knowledge_phase_g.py --sample-only  # Test on 5 PDFs
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge


class PhaseGKnowledgeIngestor:
    """
    Phase G knowledge ingestion orchestrator.

    Extends base ingestion with adaptive dimensions and specialist integration.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Phase G ingestor.

        Args:
            checkpoint_dir: Phase G specialist checkpoints directory
        """
        print("[PhaseG] Initializing Phase G Knowledge Ingestor...")

        # Initialize Phase G bridge with specialist loading
        self.bridge = PhaseGPDFIngestionBridge(
            phase_g_checkpoint_dir=checkpoint_dir
        )

        # Metrics
        self.metrics_path = Path(
            "/K3D/Knowledge3D.local/logs/phase_g_ingestion_metrics.jsonl"
        )
        self.failed_log_path = Path(
            "/K3D/Knowledge3D.local/logs/phase_g_ingestion_failed.jsonl"
        )
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.failed_log_path.parent.mkdir(parents=True, exist_ok=True)

        print("[PhaseG] Phase G Knowledge Ingestor ready")

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

    def log_failed_pages(self, failures: list[dict]) -> None:
        """Record problematic PDF pages."""
        if not failures:
            return
        with self.failed_log_path.open("a", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item) + "\n")

    def _print_banner(self, title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    # ------------------------------------------------------------------ #
    # PDF Ingestion
    # ------------------------------------------------------------------ #
    def ingest_pdf_library(self, library_path: Path, max_pdfs: Optional[int] = None) -> dict:
        """
        Ingest PDF library with Phase G pipeline.

        Args:
            library_path: Path to PDF library
            max_pdfs: Maximum PDFs to ingest (None = all)

        Returns:
            Metrics dictionary
        """
        self._print_banner(f"Ingesting PDF Library: {library_path}")

        # Find PDFs
        pdf_files = list(library_path.rglob("*.pdf")) + list(library_path.rglob("*.PDF"))

        if max_pdfs is not None:
            pdf_files = pdf_files[:max_pdfs]

        print(f"Found {len(pdf_files)} PDFs to ingest")

        # Import PyMuPDF
        try:
            import fitz  # type: ignore
            fitz.TOOLS.mupdf_display_errors(False)
        except ImportError:
            raise RuntimeError(
                "PyMuPDF (fitz) is required. Install via: pip install pymupdf"
            )

        # Ingestion statistics
        total_pages = 0
        total_objects = 0
        failed_pages: list[dict] = []
        dimension_stats = {}  # Track dimension usage
        specialist_stats = {}  # Track specialist usage

        start_time = time.time()

        for idx, pdf_path in enumerate(pdf_files, 1):
            doc = None
            try:
                doc = fitz.open(str(pdf_path))
                num_pages = len(doc)

                print(f"\n[{idx}/{len(pdf_files)}] {pdf_path.name} ({num_pages} pages)")

                page_errors = 0
                for page_num in range(num_pages):
                    try:
                        result = self.bridge.ingest_pdf_page(str(pdf_path), page_num)

                        # Update statistics
                        object_count = int(result.get("object_count", 0))
                        total_objects += object_count
                        total_pages += 1

                        # Track dimension usage
                        emb_dim = result.get("embedding_dimension", 0)
                        dimension_stats[emb_dim] = dimension_stats.get(emb_dim, 0) + 1

                        # Track specialist usage
                        specialist = result.get("specialist_used", "unknown")
                        specialist_stats[specialist] = specialist_stats.get(specialist, 0) + 1

                        # Progress logging
                        if (page_num + 1) % 10 == 0 or (page_num + 1) == num_pages:
                            method = result.get("method", "structured")
                            specialist_used = result.get("specialist_used", "base")
                            print(
                                f"  Page {page_num + 1:>4}/{num_pages:<4} "
                                f"→ {object_count:3d} objects | "
                                f"{emb_dim:>4}D | {specialist_used:>10} | {method}"
                            )

                    except Exception as page_exc:
                        page_errors += 1
                        failure = {
                            "timestamp": datetime.now().isoformat(),
                            "pdf": str(pdf_path),
                            "page": page_num,
                            "error": str(page_exc),
                        }
                        failed_pages.append(failure)
                        print(f"    ERROR page {page_num + 1}/{num_pages}: {page_exc}")

                        if page_errors >= 5:
                            print("    Too many errors; skipping remaining pages.")
                            break

            except Exception as exc:
                print(f"  ERROR ingesting {pdf_path.name}: {exc}")
                failed_pages.append({
                    "timestamp": datetime.now().isoformat(),
                    "pdf": str(pdf_path),
                    "page": None,
                    "error": str(exc),
                })
            finally:
                if doc is not None:
                    doc.close()

                # Periodic save
                if idx % 10 == 0:
                    self.bridge.save_galaxy_stars()
                    self.bridge.adaptive_rpn.save_all(
                        Path("/K3D/Knowledge3D.local/checkpoints/phase_g/embeddings/adaptive_rpn")
                    )

        # Final save
        self.bridge.save_galaxy_stars()
        self.bridge.adaptive_rpn.save_all(
            Path("/K3D/Knowledge3D.local/checkpoints/phase_g/embeddings/adaptive_rpn")
        )

        elapsed = time.time() - start_time
        ms_per_page = (elapsed * 1000) / total_pages if total_pages else 0.0

        # Compile metrics
        metrics = {
            "pdfs": len(pdf_files),
            "pages": total_pages,
            "objects": total_objects,
            "elapsed_seconds": elapsed,
            "ms_per_page": ms_per_page,
            "failed_pages": len(failed_pages),
            "dimension_usage": dimension_stats,
            "specialist_usage": specialist_stats,
        }

        self.log_metrics(f"pdf_library_{library_path.name}", metrics)
        self.log_failed_pages(failed_pages)

        # Print summary
        print(f"\n✅ Ingestion complete: {total_pages} pages in {elapsed:.1f}s ({ms_per_page:.1f} ms/page)")

        print("\nDimension Usage:")
        for dim, count in sorted(dimension_stats.items()):
            percentage = (count / total_pages) * 100 if total_pages > 0 else 0
            print(f"  {dim:>4}D: {count:>5} pages ({percentage:>5.1f}%)")

        print("\nSpecialist Usage:")
        for specialist, count in sorted(specialist_stats.items()):
            percentage = (count / total_pages) * 100 if total_pages > 0 else 0
            print(f"  {specialist:>12}: {count:>5} pages ({percentage:>5.1f}%)")

        if failed_pages:
            print(f"\n⚠️  Logged {len(failed_pages)} failures to {self.failed_log_path}")

        return metrics

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_full_ingestion(self, library_name: str = "echosystems", max_pdfs: Optional[int] = None):
        """
        Run full knowledge ingestion.

        Args:
            library_name: Library identifier ('echosystems', 'k3d', etc.)
            max_pdfs: Maximum PDFs to process (None = all)
        """
        self._print_banner("PHASE G KNOWLEDGE INGESTION")
        print(f"Start time: {datetime.now().isoformat()}")
        print(f"Library: {library_name}")
        print(f"Max PDFs: {max_pdfs if max_pdfs else 'unlimited'}")

        start_time = time.time()

        # Select library
        if library_name == "echosystems":
            library_path = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries")
        elif library_name == "k3d":
            library_path = Path("/K3D/datasets")
        else:
            raise ValueError(f"Unknown library: {library_name}")

        if not library_path.exists():
            raise FileNotFoundError(f"Library not found: {library_path}")

        # Ingest library
        metrics = self.ingest_pdf_library(library_path, max_pdfs)

        elapsed = time.time() - start_time

        # Print Phase G statistics
        self.bridge.print_phase_g_stats()

        self._print_banner("INGESTION COMPLETE")
        print(f"Total time: {elapsed / 3600:.1f} hours")
        print(f"Metrics log: {self.metrics_path}")
        print("\nKnowledge stored in Galaxy (3D space) - ready for sleep consolidation!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase G Knowledge Ingestion")
    parser.add_argument(
        "--library",
        type=str,
        default="echosystems",
        choices=["echosystems", "k3d"],
        help="PDF library to ingest"
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Maximum number of PDFs to process"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Process only 5 PDFs as a sample test"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/K3D/Knowledge3D.local/checkpoints/phase_g/current",
        help="Phase G specialist checkpoint directory"
    )

    args = parser.parse_args()

    # Handle sample mode
    max_pdfs = 5 if args.sample_only else args.max_pdfs

    # Initialize ingestor
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    ingestor = PhaseGKnowledgeIngestor(checkpoint_dir=checkpoint_dir)

    # Run ingestion
    ingestor.run_full_ingestion(
        library_name=args.library,
        max_pdfs=max_pdfs
    )


if __name__ == "__main__":
    main()
