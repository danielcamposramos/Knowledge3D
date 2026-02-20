#!/usr/bin/env python3
"""
Fundamental Full PDF Ingestion - Overnight Batch Processor

Purpose: Process ALL PDFs from database with automatic batching, checkpointing, and progress tracking
Usage: Run in tmux session for overnight processing

    tmux new -s k3d_pdf_ingestion
    python3 scripts/fundamental_overnight_ingestion.py
    # Detach: Ctrl+b then d
    # Reattach: tmux attach -t k3d_pdf_ingestion
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def find_all_pdfs(database_root: Path) -> List[Path]:
    """Find all PDFs in database."""
    return sorted(database_root.rglob("*.pdf"))


def process_pdf_batch(
    pdf_paths: List[Path],
    batch_num: int,
    output_dir: Path,
    cache_dir: Path,
    log_dir: Path,
    classifier_model: str,
    augmenter_model: str,
    python_bin: str,
) -> Dict[str, Any]:
    """Process a batch of PDFs."""
    batch_start = datetime.now()

    # Create temp directory for this batch
    temp_dir = output_dir / f"batch_{batch_num:04d}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    batch_output = temp_dir / "payload.jsonl"
    batch_report = temp_dir / "report.json"
    batch_log = log_dir / f"batch_{batch_num:04d}.log"

    print(f"[{datetime.now().isoformat()}] Batch {batch_num}: Processing {len(pdf_paths)} PDFs")

    # Create temp file list for this batch
    pdf_list_file = temp_dir / "pdf_list.txt"
    pdf_list_file.write_text("\n".join(str(p) for p in pdf_paths))

    # Build command
    cmd = [
        python_bin,
        "scripts/fundamental_ingest_pdfs.py",
        "--pdf-list", str(pdf_list_file),
        "--max-pages-per-pdf", "0",
        "--classifier-model", classifier_model,
        "--augmenter-model", augmenter_model,
        "--ollama-timeout", "180.0",
        "--cache-dir", str(cache_dir),
        "--payload-output", str(batch_output),
        "--report-output", str(batch_report),
    ]

    # Run ingestion
    try:
        with open(batch_log, "w") as log_file:
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                env={"PYTHONPATH": "."},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=7200,  # 2 hour timeout per batch
            )

        success = result.returncode == 0

        # Read report
        if batch_report.exists():
            report = json.loads(batch_report.read_text())
        else:
            report = {}

        # Count entries
        entries = 0
        if batch_output.exists():
            entries = len(batch_output.read_text().strip().split("\n"))

        elapsed = (datetime.now() - batch_start).total_seconds()

        return {
            "batch_num": batch_num,
            "success": success,
            "pdfs_count": len(pdf_paths),
            "entries_generated": entries,
            "elapsed_seconds": elapsed,
            "output_file": str(batch_output),
            "report_file": str(batch_report),
            "log_file": str(batch_log),
        }

    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().isoformat()}] Batch {batch_num}: TIMEOUT")
        return {
            "batch_num": batch_num,
            "success": False,
            "error": "timeout",
            "pdfs_count": len(pdf_paths),
        }

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Batch {batch_num}: ERROR - {e}")
        return {
            "batch_num": batch_num,
            "success": False,
            "error": str(e),
            "pdfs_count": len(pdf_paths),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("/mnt/arquivos/0 ChatGPTs/DataBase"),
        help="Root directory containing PDFs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../Knowledge3D.local/fundamental_augmentation/overnight_pdfs"),
        help="Output directory for payloads",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("../Knowledge3D.local/pdf_cache"),
        help="Cache directory for page classifications",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of PDFs per batch",
    )
    parser.add_argument(
        "--classifier-model",
        default="deepseek-r1:14b",
        help="Ollama model for page classification",
    )
    parser.add_argument(
        "--augmenter-model",
        default="qwen2.5:14b",
        help="Ollama model for knowledge augmentation",
    )
    parser.add_argument(
        "--python-bin",
        default="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python",
        help="Python binary path",
    )
    parser.add_argument(
        "--resume-from-batch",
        type=int,
        default=0,
        help="Resume from specific batch number (0 = start fresh)",
    )

    args = parser.parse_args()

    # Setup directories
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    print("Finding PDFs...")
    all_pdfs = find_all_pdfs(args.database_root)
    print(f"Found {len(all_pdfs)} PDFs")

    # Split into batches
    batches = []
    for i in range(0, len(all_pdfs), args.batch_size):
        batches.append(all_pdfs[i:i + args.batch_size])

    print(f"Total batches: {len(batches)} (batch size: {args.batch_size})")

    # Process batches
    start_time = time.time()
    results = []

    for batch_num, pdf_batch in enumerate(batches, start=1):
        # Skip if resuming
        if batch_num < args.resume_from_batch:
            print(f"Skipping batch {batch_num} (resume point: {args.resume_from_batch})")
            continue

        # Process batch
        result = process_pdf_batch(
            pdf_paths=pdf_batch,
            batch_num=batch_num,
            output_dir=output_root,
            cache_dir=cache_dir,
            log_dir=log_dir,
            classifier_model=args.classifier_model,
            augmenter_model=args.augmenter_model,
            python_bin=args.python_bin,
        )

        results.append(result)

        # Progress update
        completed_pdfs = batch_num * args.batch_size
        if completed_pdfs > len(all_pdfs):
            completed_pdfs = len(all_pdfs)

        elapsed = time.time() - start_time
        progress_pct = 100 * completed_pdfs / len(all_pdfs)

        if completed_pdfs > 0:
            avg_time_per_pdf = elapsed / completed_pdfs
            remaining_pdfs = len(all_pdfs) - completed_pdfs
            eta_seconds = avg_time_per_pdf * remaining_pdfs
            eta_hours = eta_seconds / 3600

        print(f"\n{'='*60}")
        print(f"Progress: {completed_pdfs}/{len(all_pdfs)} PDFs ({progress_pct:.1f}%)")
        print(f"Batches: {batch_num}/{len(batches)}")
        print(f"Elapsed: {elapsed/3600:.2f} hours")
        if completed_pdfs > 0:
            print(f"ETA: {eta_hours:.2f} hours remaining")
        print(f"{'='*60}\n")

        # Save checkpoint
        checkpoint = {
            "last_batch": batch_num,
            "completed_pdfs": completed_pdfs,
            "total_pdfs": len(all_pdfs),
            "results": results,
        }
        checkpoint_file = output_root / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2))

        # Cooldown between batches
        if batch_num < len(batches):
            print("Cooling down for 10 seconds...")
            time.sleep(10)

    # Merge all payloads
    print("\nMerging batch payloads...")
    merged_output = output_root / "full_pdf_payloads_overnight.jsonl"

    with open(merged_output, "w") as merged:
        for batch_dir in sorted(output_root.glob("batch_*")):
            payload_file = batch_dir / "payload.jsonl"
            if payload_file.exists():
                merged.write(payload_file.read_text())

    total_entries = len(merged_output.read_text().strip().split("\n"))

    # Generate summary
    total_elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    summary = {
        "execution_date": datetime.now().isoformat(),
        "total_time_seconds": total_elapsed,
        "total_time_hours": total_elapsed / 3600,
        "total_pdfs_attempted": len(all_pdfs),
        "total_batches": len(batches),
        "successful_batches": successful,
        "failed_batches": failed,
        "total_entries_generated": total_entries,
        "classifier_model": args.classifier_model,
        "augmenter_model": args.augmenter_model,
        "merged_payload": str(merged_output),
        "batch_results": results,
    }

    summary_file = output_root / "overnight_ingestion_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print("Ingestion Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Successful batches: {successful}/{len(batches)}")
    print(f"Failed batches: {failed}")
    print(f"Total entries: {total_entries}")
    print(f"Merged output: {merged_output}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}\n")

    print("Next steps:")
    print(f"1. Ingest to Galaxy:")
    print(f"   PYTHONPATH=. {args.python_bin} scripts/fundamental_ingest_payloads.py \\")
    print(f"     --payload {merged_output} \\")
    print(f"     --storage-root ../Knowledge3D.local \\")
    print(f"     --report ../Knowledge3D.local/results/overnight_pdf_ingestion_report.json")


if __name__ == "__main__":
    main()
