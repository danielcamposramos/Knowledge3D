#!/usr/bin/env python3
"""Canonical PDF OCR/eligibility preflight for ordered base-knowledge ingestion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - handled by _require_fitz
    fitz = None


ROOT_SLUG_OVERRIDES = {
    "encyclopedias": "encyclopedias",
    "echosystems default libraries": "default_libraries",
}
ELIGIBLE_TYPES = {"vector", "mixed", "scanned_with_ocr"}


def _require_fitz() -> Any:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF preflight. "
            "Run this script inside /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python."
        )
    return fitz


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    override = ROOT_SLUG_OVERRIDES.get(lowered)
    if override:
        return override
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "root"


def root_slug(index: int, root: Path) -> str:
    return f"{int(index):02d}_{_slugify(root.name)}"


def discover_pdf_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.pdf") if path.is_file() and path.suffix.lower() == ".pdf")


def count_source_inventory(root: Path) -> dict[str, int]:
    counts = {"pdf": 0, "json": 0, "other": 0}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            counts["pdf"] += 1
        elif suffix == ".json":
            counts["json"] += 1
        else:
            counts["other"] += 1
    return counts


def analyze_pdf_page(page: Any) -> dict[str, Any]:
    rect = page.rect
    page_area = rect.width * rect.height

    images = page.get_images(full=True)
    image_coverage = 0.0
    for img_ref in images:
        try:
            xref = img_ref[0]
            bbox_list = page.get_image_rects(xref)
            for bbox in bbox_list:
                img_area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
                image_coverage += img_area
        except Exception:
            continue

    image_coverage_ratio = image_coverage / page_area if page_area > 0 else 0.0
    text = page.get_text("text").strip()
    char_count = len(text)
    blocks = page.get_text("dict").get("blocks", [])
    text_block_count = sum(1 for block in blocks if block.get("type") == 0)

    return {
        "page_area": page_area,
        "image_count": len(images),
        "image_coverage_ratio": image_coverage_ratio,
        "char_count": char_count,
        "text_block_count": text_block_count,
        "has_images": len(images) > 0,
        "has_text": char_count > 50,
    }


def classify_pdf(pdf_path: Path, sample_pages: int = 5) -> dict[str, Any]:
    fitz_mod = _require_fitz()

    try:
        doc = fitz_mod.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return {
                "path": str(pdf_path),
                "type": "error",
                "pages": 0,
                "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
                "error": "No pages",
            }

        total_pages = len(doc)
        pages_to_sample = min(max(1, int(sample_pages)), total_pages)
        if total_pages <= pages_to_sample:
            sample_indices = list(range(total_pages))
        else:
            raw_indices = [0, total_pages - 1, total_pages // 2, total_pages // 4, (3 * total_pages) // 4]
            sample_indices = []
            for idx in raw_indices:
                if 0 <= idx < total_pages and idx not in sample_indices:
                    sample_indices.append(idx)
                if len(sample_indices) >= pages_to_sample:
                    break

        page_analyses = [analyze_pdf_page(doc[idx]) for idx in sample_indices]
        doc.close()

        avg_image_coverage = sum(page["image_coverage_ratio"] for page in page_analyses) / len(page_analyses)
        avg_char_count = sum(page["char_count"] for page in page_analyses) / len(page_analyses)
        pages_with_images = sum(1 for page in page_analyses if page["has_images"])
        pages_with_text = sum(1 for page in page_analyses if page["has_text"])

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        size_per_page_kb = (pdf_path.stat().st_size / total_pages) / 1024
        is_scanned = avg_image_coverage > 0.5 and size_per_page_kb > 50
        has_ocr_text = pages_with_text >= max(1, int(round(len(page_analyses) * 0.8)))

        if is_scanned and has_ocr_text:
            pdf_type = "scanned_with_ocr"
        elif is_scanned and not has_ocr_text:
            pdf_type = "scanned_no_text"
        elif not is_scanned and has_ocr_text:
            pdf_type = "vector"
        else:
            pdf_type = "mixed"

        return {
            "path": str(pdf_path),
            "type": pdf_type,
            "pages": total_pages,
            "file_size_mb": file_size_mb,
            "size_per_page_kb": size_per_page_kb,
            "avg_image_coverage": avg_image_coverage,
            "avg_char_count": avg_char_count,
            "pages_with_images": pages_with_images,
            "pages_with_text": pages_with_text,
            "sampled_pages": len(page_analyses),
        }
    except Exception as exc:
        return {
            "path": str(pdf_path),
            "type": "error",
            "pages": 0,
            "file_size_mb": pdf_path.stat().st_size / (1024 * 1024) if pdf_path.exists() else 0.0,
            "error": str(exc)[:200],
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _sorted_eligible_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [record for record in records if str(record.get("type", "")).strip().lower() in ELIGIBLE_TYPES]
    return sorted(
        eligible,
        key=lambda record: (-int(record.get("pages", 0) or 0), str(record.get("path", ""))),
    )


def analyze_root(root: Path, *, output_dir: Path, sample_pages: int = 5) -> dict[str, Any]:
    inventory = count_source_inventory(root)
    pdf_paths = discover_pdf_paths(root)
    records = [classify_pdf(path, sample_pages=sample_pages) for path in pdf_paths]

    type_counts: dict[str, int] = {}
    for record in records:
        pdf_type = str(record.get("type", "error")).strip().lower() or "error"
        type_counts[pdf_type] = type_counts.get(pdf_type, 0) + 1

    eligible_records = _sorted_eligible_entries(records)
    ocr_needed = sorted(
        str(record.get("path", ""))
        for record in records
        if str(record.get("type", "")).strip().lower() == "scanned_no_text"
    )
    extraction_errors = sorted(
        (
            f"{record.get('path', '')}\t{record.get('error', '')}"
            if str(record.get("error", "")).strip()
            else str(record.get("path", ""))
        )
        for record in records
        if str(record.get("type", "")).strip().lower() == "error"
    )

    all_inventory_path = output_dir / "all_pdf_inventory.json"
    eligible_path = output_dir / "eligible_pdfs.txt"
    ocr_needed_path = output_dir / "ocr_needed_pdfs.txt"
    errors_path = output_dir / "extraction_errors.txt"
    summary_path = output_dir / "summary.json"

    _write_json(all_inventory_path, records)
    _write_lines(eligible_path, [str(record["path"]) for record in eligible_records])
    _write_lines(ocr_needed_path, ocr_needed)
    _write_lines(errors_path, extraction_errors)

    summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "inventory": inventory,
        "discovered_pdf_count": len(pdf_paths),
        "eligible_pdf_count": len(eligible_records),
        "ocr_needed_count": len(ocr_needed),
        "error_count": len(extraction_errors),
        "sample_pages": int(sample_pages),
        "type_counts": dict(sorted(type_counts.items())),
        "artifacts": {
            "all_pdf_inventory": str(all_inventory_path),
            "eligible_pdfs": str(eligible_path),
            "ocr_needed_pdfs": str(ocr_needed_path),
            "extraction_errors": str(errors_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        type=Path,
        required=True,
        help="Source root to preflight. Repeat to preserve ingestion order.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/results/base_knowledge_ingest"),
        help="Root directory for preflight outputs.",
    )
    parser.add_argument("--sample-pages", type=int, default=5, help="Number of pages to sample per PDF.")
    args = parser.parse_args()

    try:
        _require_fitz()
    except RuntimeError as exc:
        print(f"[pdf-preflight] {exc}", file=sys.stderr)
        return 2

    summaries = []
    for index, root in enumerate(args.roots, start=1):
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"[pdf-preflight] missing root: {root}", file=sys.stderr)
            return 1
        output_dir = args.results_root / root_slug(index, root) / "preflight"
        summary = analyze_root(root, output_dir=output_dir, sample_pages=max(1, int(args.sample_pages)))
        summaries.append(summary)
        print(
            f"[pdf-preflight] root={root} pdf={summary['discovered_pdf_count']} "
            f"eligible={summary['eligible_pdf_count']} ocr_needed={summary['ocr_needed_count']} "
            f"errors={summary['error_count']} out={output_dir}"
        )

    aggregate_path = args.results_root / "summary.json"
    _write_json(aggregate_path, {"roots": summaries})
    print(f"[pdf-preflight] aggregate_summary={aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
