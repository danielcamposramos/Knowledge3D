"""
Benchmark harness for the Phase C1 PDF ingestion pipeline.

The script compares per-page timings across a handful of sample PDFs. It keeps
outputs lightweight so the benchmark can run inside the repo without external
dependencies. Real datasets should live in Knowledge3D.local according to the
large-asset policy.
"""

import sys
import time
from pathlib import Path

from knowledge3d.ingestion.documents.pdf_multimodal_ingestor import PDFMultiModalIngestor


def benchmark_phase_c(pdf_paths) -> None:
    ingestor = PDFMultiModalIngestor()

    print("=" * 60)
    print("Phase C1 PDF Ingestion Benchmark")
    print("=" * 60)

    for path in pdf_paths:
        pdf_path = Path(path)
        if not pdf_path.exists():
            print(f"[SKIP] {pdf_path} not found")
            continue

        start = time.perf_counter()
        result = ingestor.ingest_pdf(pdf_path)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0

        page_count = len(result.get("pages", []))
        per_page = elapsed_ms / page_count if page_count else 0.0

        print(f"\n[PDF] {pdf_path}")
        print(f"  pages............: {page_count}")
        print(f"  total objects....: {result.get('total_objects', 0)}")
        print(f"  total time (ms)..: {elapsed_ms:.2f}")
        print(f"  per page (ms)....: {per_page:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        samples = sys.argv[1:]
    else:
        samples = [
            "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/Algorithmic.Thinking.BASE.pdf",
            "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/APOLLO.PDF",
        ]
    benchmark_phase_c(samples)
