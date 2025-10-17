#!/usr/bin/env python3
"""
Batch ingest Daniel's curated PDF corpus using sovereign RPN embeddings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from knowledge3d.ingestion.documents.pdf_ingestor import PDFIngestor


PRIORITY_DIRECTORIES: Sequence[Path] = [
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/"),
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Teach/"),
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Academic Research/"),
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Self Reflection/"),
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Understand Time/"),
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Eloquence/"),
    Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/"),
]

DEFAULT_OUTPUT_ROOT = Path("/K3D/Knowledge3D.local/house_zone7/documents/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint for the corpus (default: en)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=500,
        help="Maximum sentences per document (default: 500)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where per-document summaries are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    pdf_ingestor = PDFIngestor(output_root=output_root)

    total_docs = 0
    total_sentences = 0
    total_time = 0.0

    for directory in PRIORITY_DIRECTORIES:
        if not directory.exists():
            print(f"[WARN] Skipping missing directory: {directory}")
            continue

        print("=" * 72)
        print(f"Ingesting directory: {directory}")

        per_directory_output = output_root / directory.name
        per_directory_output.mkdir(parents=True, exist_ok=True)

        result = pdf_ingestor.ingest_pdf_directory(
            directory,
            language=args.language,
            max_sentences=args.max_sentences,
            output_dir=per_directory_output,
        )

        docs = result["pdf_count"]
        sentences = result["total_sentences"]
        elapsed = result["total_time_s"]

        print(f"  Documents: {docs}")
        print(f"  Sentences: {sentences}")
        print(f"  Time: {elapsed:.2f}s\n")

        total_docs += docs
        total_sentences += sentences
        total_time += elapsed

        # Persist incremental learning after each directory
        pdf_ingestor.text_ingestor.save_learned_embeddings()

    if total_docs == 0:
        print("No documents processed.")
        return

    avg_per_doc = total_time / max(total_docs, 1)
    avg_per_sentence = (total_time / max(total_sentences, 1)) * 1000.0

    print("=" * 72)
    print("FULL CORPUS INGESTION COMPLETE")
    print(f"Total documents: {total_docs}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per document: {avg_per_doc:.2f}s")
    print(f"Average per sentence: {avg_per_sentence:.2f}ms")


if __name__ == "__main__":
    main()
