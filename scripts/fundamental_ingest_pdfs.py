#!/usr/bin/env python3
"""Fundamental PDF ingestion with intelligent classification + augmentation.

PURPOSE:
  Construct foundational Galaxy payloads from legacy PDF documents.
  This is a construction/ingestion tool, not a runtime solver.

WHEN TO USE:
  - Initial/foundational world population from PDFs.
  - Controlled re-ingestion when classifier/augmenter model improves.

NOT FOR:
  - PTX hot-path inference.
  - Runtime daemon command routing.

PIPELINE:
  1) Extract PDF pages.
  2) Classify pages with Ollama (knowledge/non_knowledge/ambiguous).
  3) Augment knowledge pages to Galaxy-ready rows.
  4) Persist payload + report.
  5) Optional ingestion via `scripts/fundamental_ingest_payloads.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from knowledge3d.ingestion.ollama_manager import OllamaModelManager
from knowledge3d.ingestion.pdf_augmenter import PDFKnowledgeAugmenter
from knowledge3d.ingestion.pdf_classifier import PDFKnowledgeClassifier


def _extract_pdf_pages(pdf_path: Path, *, max_pages: int = 0) -> dict[int, str]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for intelligent PDF ingestion. "
            "Install it in the active environment."
        ) from exc

    pages: dict[int, str] = {}
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        limit = total if max_pages <= 0 else min(total, int(max_pages))
        for i in range(limit):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            if not text:
                blocks = page.get_text("blocks") if hasattr(page, "get_text") else []
                if isinstance(blocks, list):
                    parts: list[str] = []
                    for block in blocks:
                        if isinstance(block, (list, tuple)) and len(block) >= 5:
                            parts.append(str(block[4]).strip())
                    text = "\n".join(part for part in parts if part).strip()
            pages[i + 1] = text
    return pages


def _iter_pdf_paths(*, pdf: Path | None, pdf_dir: Path | None, pattern: str, limit: int) -> list[Path]:
    out: list[Path] = []
    if pdf is not None:
        if pdf.exists() and pdf.suffix.lower() == ".pdf":
            out.append(pdf)
        return out
    if pdf_dir is None or not pdf_dir.exists():
        return out
    for path in sorted(pdf_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() == ".pdf":
            out.append(path)
            if limit > 0 and len(out) >= limit:
                break
    return out


def _classification_label(decision: dict[str, Any]) -> str:
    return str(decision.get("resolved_classification") or decision.get("classification") or "").strip().lower()


def _stage_root(payload_output: Path, explicit_stage_dir: Path | None) -> Path:
    if explicit_stage_dir is not None:
        return explicit_stage_dir
    return payload_output.parent / f".{payload_output.stem}_stage"


def _stage_manifest_path(stage_root: Path) -> Path:
    return stage_root / "manifest.json"


def _load_stage_manifest(stage_root: Path) -> dict[str, Any]:
    path = _stage_manifest_path(stage_root)
    if not path.exists():
        return {"pdfs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"pdfs": {}}
    if not isinstance(payload, dict):
        return {"pdfs": {}}
    pdfs = payload.get("pdfs")
    if not isinstance(pdfs, dict):
        payload["pdfs"] = {}
    return payload


def _save_stage_manifest(stage_root: Path, payload: dict[str, Any]) -> None:
    path = _stage_manifest_path(stage_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _pdf_sha(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _pdf_stage_dir(stage_root: Path, pdf_sha: str) -> Path:
    return stage_root / pdf_sha


def _page_stage_path(pdf_stage_dir: Path, page_num: int) -> Path:
    return pdf_stage_dir / f"page_{int(page_num):05d}.json"


def _existing_staged_pages(pdf_stage_dir: Path) -> list[int]:
    if not pdf_stage_dir.exists():
        return []
    out: list[int] = []
    for path in pdf_stage_dir.glob("page_*.json"):
        stem = path.stem
        try:
            page_num = int(stem.split("_", 1)[1])
        except Exception:
            continue
        out.append(page_num)
    return sorted(set(out))


def _write_stage_page(
    *,
    pdf_stage_dir: Path,
    page_num: int,
    pdf_path: Path,
    total_pages: int,
    decision: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "pdf": str(pdf_path),
        "page_num": int(page_num),
        "total_pages": int(total_pages),
        "decision": decision,
        "rows": rows,
    }
    pdf_stage_dir.mkdir(parents=True, exist_ok=True)
    target = _page_stage_path(pdf_stage_dir, page_num)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)


def _iter_stage_records(stage_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pdf_dir in sorted(stage_root.glob("*")):
        if not pdf_dir.is_dir():
            continue
        for page_file in sorted(pdf_dir.glob("page_*.json")):
            try:
                payload = json.loads(page_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    records.sort(key=lambda rec: (str(rec.get("pdf", "")), int(rec.get("page_num", 0))))
    return records


def _rebuild_payload_from_stage(*, stage_root: Path, payload_output: Path) -> tuple[Counter[str], Counter[str], dict[str, dict[str, Any]], int]:
    by_galaxy: Counter[str] = Counter()
    by_classification: Counter[str] = Counter()
    per_pdf: dict[str, dict[str, Any]] = {}
    total_rows = 0

    payload_output.parent.mkdir(parents=True, exist_ok=True)
    with payload_output.open("w", encoding="utf-8") as handle:
        for record in _iter_stage_records(stage_root):
            pdf_key = str(record.get("pdf", ""))
            page_num = int(record.get("page_num", 0) or 0)
            total_pages = int(record.get("total_pages", 0) or 0)
            decision = record.get("decision")
            rows = record.get("rows")
            if not isinstance(decision, dict):
                decision = {}
            if not isinstance(rows, list):
                rows = []

            label = _classification_label(decision) or "unknown"
            by_classification[label] += 1
            stats = per_pdf.setdefault(
                pdf_key,
                {
                    "pdf": pdf_key,
                    "pages_total": total_pages,
                    "knowledge_pages": 0,
                    "rows_generated": 0,
                    "_seen_pages": set(),
                },
            )
            if total_pages > 0:
                stats["pages_total"] = max(int(stats["pages_total"]), total_pages)
            if page_num > 0:
                stats["_seen_pages"].add(page_num)
            if label == "knowledge":
                stats["knowledge_pages"] = int(stats["knowledge_pages"]) + 1

            for row in rows:
                if not isinstance(row, dict):
                    continue
                galaxy = str(row.get("galaxy", "")).strip()
                if galaxy:
                    by_galaxy[galaxy] += 1
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_rows += 1
                stats["rows_generated"] = int(stats["rows_generated"]) + 1

    return by_galaxy, by_classification, per_pdf, total_rows


def _ingest_payload(payload_path: Path, *, storage_root: Path, ingest_report: Path) -> int:
    cmd = [
        sys.executable,
        "scripts/fundamental_ingest_payloads.py",
        "--storage-root",
        str(storage_root),
        "--payload",
        str(payload_path),
        "--report",
        str(ingest_report),
    ]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=Path, help="Single PDF path to ingest")
    src.add_argument("--pdf-dir", type=Path, help="Directory of PDFs to ingest")
    parser.add_argument("--pattern", default="**/*.pdf", help="Glob pattern when using --pdf-dir")
    parser.add_argument("--limit-pdfs", type=int, default=0, help="Limit number of PDFs from --pdf-dir")
    parser.add_argument("--max-pages-per-pdf", type=int, default=0, help="0 means all pages")
    parser.add_argument("--cache-dir", type=Path, default=Path("../Knowledge3D.local/pdf_cache"))
    parser.add_argument("--classifier-model", default="qwen2.5:32b")
    parser.add_argument("--augmenter-model", default="qwen2.5:32b")
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument(
        "--payload-output",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/external_payloads/pdf_intelligent_payload.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/external_payloads/pdf_intelligent_report.json"),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Optional directory for per-page resumable staging (defaults near payload output).",
    )
    parser.add_argument(
        "--disable-resume-last-page",
        action="store_true",
        help="By default reruns reprocess last staged page to overwrite partial attempts.",
    )
    parser.add_argument(
        "--payload-checkpoint-interval-pdfs",
        type=int,
        default=25,
        help="Rebuild payload output every N processed PDFs (0 disables periodic checkpoints).",
    )
    parser.add_argument("--storage-root", type=Path, default=Path("../Knowledge3D.local"))
    parser.add_argument("--ingest", action="store_true", help="Run single-world ingestion after payload generation")
    parser.add_argument(
        "--ingest-report",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/external_payloads/pdf_intelligent_ingest_report.json"),
    )
    args = parser.parse_args()

    pdf_paths = _iter_pdf_paths(pdf=args.pdf, pdf_dir=args.pdf_dir, pattern=args.pattern, limit=max(0, int(args.limit_pdfs)))
    if not pdf_paths:
        print("[pdf-ingest] no PDF files found")
        return 1

    stage_root = _stage_root(args.payload_output, args.stage_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_stage_manifest(stage_root)
    manifest_pdfs = manifest.setdefault("pdfs", {})
    if not isinstance(manifest_pdfs, dict):
        manifest["pdfs"] = {}
        manifest_pdfs = manifest["pdfs"]

    with OllamaModelManager(default_timeout=float(args.ollama_timeout)) as ollama:
        classifier = PDFKnowledgeClassifier(
            ollama=ollama,
            cache_dir=args.cache_dir,
            model=args.classifier_model,
            timeout=float(args.ollama_timeout),
        )
        augmenter = PDFKnowledgeAugmenter(
            ollama=ollama,
            model=args.augmenter_model,
            timeout=float(args.ollama_timeout),
        )

        checkpoint_interval = max(0, int(args.payload_checkpoint_interval_pdfs))
        processed_pdfs = 0
        for pdf_path in pdf_paths:
            pages = _extract_pdf_pages(pdf_path, max_pages=max(0, int(args.max_pages_per_pdf)))
            pdf_sha = _pdf_sha(pdf_path)
            pdf_stage = _pdf_stage_dir(stage_root, pdf_sha)
            if not pages:
                manifest_pdfs[str(pdf_path)] = {
                    "sha256": pdf_sha,
                    "pages_total": 0,
                    "resume_from_page": 1,
                }
                _save_stage_manifest(stage_root, manifest)
                continue

            staged_pages = _existing_staged_pages(pdf_stage)
            if staged_pages:
                last_done = staged_pages[-1]
                resume_from = last_done if not bool(args.disable_resume_last_page) else (last_done + 1)
                resume_from = max(1, resume_from)
            else:
                resume_from = 1
            manifest_pdfs[str(pdf_path)] = {
                "sha256": pdf_sha,
                "pages_total": int(len(pages)),
                "resume_from_page": int(resume_from),
            }
            _save_stage_manifest(stage_root, manifest)

            decisions = classifier.classify_pdf_pages(
                pdf_path=pdf_path,
                pages=pages,
                force_reprocess=bool(args.force_reprocess),
            )

            total_pages = len(pages)
            for page_num, decision in sorted(decisions.items(), key=lambda item: item[0]):
                stage_page = _page_stage_path(pdf_stage, int(page_num))
                if int(page_num) < int(resume_from) and stage_page.exists():
                    continue

                label = _classification_label(decision)
                rows: list[dict[str, Any]] = []
                context: dict[int, str] = {}
                for adj in (page_num - 1, page_num + 1):
                    if adj in pages:
                        context[adj] = pages[adj]
                if label == "knowledge":
                    augmented = augmenter.augment_page(
                        pdf_path=pdf_path,
                        page_num=page_num,
                        total_pages=total_pages,
                        page_text=pages.get(page_num, ""),
                        classification=decision,
                        context_pages=context,
                    )
                    rows = augmenter.to_payload_rows(
                        pdf_path=pdf_path,
                        page_num=page_num,
                        augmented=augmented,
                        classification=decision,
                    )
                _write_stage_page(
                    pdf_stage_dir=pdf_stage,
                    page_num=int(page_num),
                    pdf_path=pdf_path,
                    total_pages=total_pages,
                    decision=decision,
                    rows=rows,
                )

            processed_pdfs += 1
            if checkpoint_interval > 0 and processed_pdfs % checkpoint_interval == 0:
                _rebuild_payload_from_stage(
                    stage_root=stage_root,
                    payload_output=args.payload_output,
                )

    by_galaxy, by_classification, per_pdf, payload_rows = _rebuild_payload_from_stage(
        stage_root=stage_root,
        payload_output=args.payload_output,
    )
    pdf_stats = []
    for pdf in pdf_paths:
        key = str(pdf)
        stat = per_pdf.get(
            key,
            {"pdf": key, "pages_total": 0, "knowledge_pages": 0, "rows_generated": 0, "_seen_pages": set()},
        )
        pages_total = int(stat.get("pages_total", 0) or 0)
        seen_pages = stat.get("_seen_pages", set())
        if isinstance(seen_pages, set):
            pages_total = max(pages_total, len(seen_pages))
        pdf_stats.append(
            {
                "pdf": key,
                "pages_total": pages_total,
                "knowledge_pages": int(stat.get("knowledge_pages", 0) or 0),
                "rows_generated": int(stat.get("rows_generated", 0) or 0),
            }
        )

    report = {
        "pdf_count": len(pdf_paths),
        "payload_rows": int(payload_rows),
        "pdfs": pdf_stats,
        "classification_counts": dict(sorted(by_classification.items())),
        "rows_by_galaxy": dict(sorted(by_galaxy.items())),
        "models": {
            "classifier": args.classifier_model,
            "augmenter": args.augmenter_model,
        },
        "cache_dir": str(args.cache_dir),
        "stage_dir": str(stage_root),
        "force_reprocess": bool(args.force_reprocess),
        "resume_last_page": not bool(args.disable_resume_last_page),
        "ingest_requested": bool(args.ingest),
    }

    if args.ingest:
        rc = _ingest_payload(args.payload_output, storage_root=args.storage_root, ingest_report=args.ingest_report)
        report["ingest_returncode"] = rc
        report["ingest_report"] = str(args.ingest_report)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[pdf-ingest] pdfs={len(pdf_paths)} rows={len(all_rows)} payload={args.payload_output}")
    print(f"[pdf-ingest] report={args.report_output}")
    for galaxy, count in sorted(by_galaxy.items()):
        print(f"[pdf-ingest] {galaxy}: {count}")

    return int(report.get("ingest_returncode", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
