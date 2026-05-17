#!/usr/bin/env python3
"""Fundamental PDF ingestion with unified proceduralization.

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
  2) Submit each page/chunk to the canonical knowledge proceduralizer.
  3) Preserve receipt + staged payload rows with resumable per-page checkpoints.
  4) Persist payload + report.
  5) Optional ingestion via `scripts/fundamental_ingest_payloads.py`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge3d.ingestion.ollama_manager import OllamaModelManager
from knowledge3d.ingestion.proceduralizer_contract import extract_json_object
from knowledge3d.knowledgeverse.proceduralizer_stargate import (
    build_row_enrichment_context,
    bundle_to_payload_rows,
    load_external_enrichment_context,
    second_pass_enrich_payload_rows,
)
from knowledge3d.tools.knowledge_proceduralizer import proceduralize_text_content, receipt_is_usable


RETRYABLE_PAGE_FAILURE_CODES = {"timeout", "invalid_json"}
OCR_DEFAULT_MODEL = "qwen3-vl:235b-instruct-cloud"
OCR_RETRY_MODEL = "qwen3-vl:235b-cloud"
OCR_PAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_text": {"type": "string"},
        "image_descriptions": {"type": "array", "items": {"type": "string"}},
        "diagram_descriptions": {"type": "array", "items": {"type": "string"}},
        "table_descriptions": {"type": "array", "items": {"type": "string"}},
        "ocr_quality": {"type": "string", "enum": ["good", "partial", "poor"]},
        "needs_review": {"type": "boolean"},
    },
    "required": [
        "page_text",
        "image_descriptions",
        "diagram_descriptions",
        "table_descriptions",
        "ocr_quality",
        "needs_review",
    ],
    "additionalProperties": False,
}
OCR_SYSTEM_PROMPT = """You are the K3D PDF OCR reconstruction stage.

You operate at ingestion time only. Reconstruct the content of a single PDF page.

Requirements:
- Extract all readable text faithfully.
- Preserve equations, captions, and headings in plain text.
- Describe images, diagrams, maps, charts, tables, and figures compactly and concretely.
- If text is unclear, keep the readable portion and set needs_review=true.
- Return strict JSON only. No markdown fences. No commentary.

Schema:
{
  "page_text": "full reconstructed page text",
  "image_descriptions": ["short factual image descriptions"],
  "diagram_descriptions": ["short factual diagram or chart descriptions"],
  "table_descriptions": ["short factual table descriptions"],
  "ocr_quality": "good|partial|poor",
  "needs_review": false
}
"""


def _is_cloud_model(model_name: str | None) -> bool:
    name = str(model_name or "").strip().lower()
    return bool(name) and "cloud" in name


def _normalize_ocr_retry_model(
    *,
    primary_model: str,
    retry_model: str | None,
    legacy_retry_model: str | None,
) -> str | None:
    primary = str(primary_model or "").strip()
    candidates = [str(retry_model or "").strip(), str(legacy_retry_model or "").strip()]
    for candidate in candidates:
        if not candidate or candidate == primary:
            continue
        if _is_cloud_model(candidate):
            return candidate
    return None


def _ocr_payload_requires_retry(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True
    page_text = str(payload.get("page_text") or "").strip()
    status = str(payload.get("status") or "").strip().lower()
    ocr_quality = str(payload.get("ocr_quality") or "").strip().lower()
    model = str(payload.get("model") or "").strip()
    if status and status != "completed":
        return True
    if not page_text:
        return True
    if bool(payload.get("needs_review")):
        return True
    if ocr_quality and ocr_quality != "good":
        return True
    if model and not _is_cloud_model(model):
        return True
    return False


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


def _read_path_list(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    out: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Path(line))
    return out


def _path_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def _ocr_artifact_root(payload_output: Path, explicit_ocr_dir: Path | None) -> Path:
    if explicit_ocr_dir is not None:
        return explicit_ocr_dir
    return payload_output.parent / "ocr"


def _ocr_pdf_dir(ocr_root: Path, pdf_sha: str) -> Path:
    return ocr_root / pdf_sha


def _ocr_page_path(ocr_pdf_dir: Path, page_num: int) -> Path:
    return ocr_pdf_dir / f"page_{int(page_num):05d}.json"


def _is_retryable_failure(receipt: Any) -> bool:
    status = str(getattr(receipt, "status", "") or "").strip().lower()
    failure_code = str(getattr(receipt, "failure_code", "") or "").strip().lower()
    if failure_code in RETRYABLE_PAGE_FAILURE_CODES:
        return True
    return status == "invalid_json"


def _compose_ocr_page_text(payload: dict[str, Any]) -> str:
    page_text = str(payload.get("page_text") or "").strip()
    sections: list[str] = []
    if page_text:
        sections.append(page_text)
    for label, key in (
        ("Image descriptions", "image_descriptions"),
        ("Diagram descriptions", "diagram_descriptions"),
        ("Table descriptions", "table_descriptions"),
    ):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        cleaned = [str(item).strip() for item in values if str(item or "").strip()]
        if not cleaned:
            continue
        sections.append(label + ":\n" + "\n".join(f"- {item}" for item in cleaned))
    return "\n\n".join(section for section in sections if section).strip()


def _write_reconstructed_pdf(output_path: Path, page_texts: dict[int, str]) -> None:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to write reconstructed OCR PDFs."
        ) from exc

    doc = fitz.open()
    try:
        for page_num in sorted(page_texts):
            base_text = str(page_texts.get(page_num) or "").strip()
            if not base_text:
                base_text = "[No OCR text reconstructed]"
            wrapped = textwrap.wrap(base_text, width=100, replace_whitespace=False, drop_whitespace=False)
            if not wrapped:
                wrapped = [base_text]
            chunk_size = 58
            for chunk_index in range(0, len(wrapped), chunk_size):
                segment = "\n".join(wrapped[chunk_index : chunk_index + chunk_size]).strip()
                page = doc.new_page(width=595, height=842)
                header = f"Source page {page_num}"
                if chunk_index:
                    header += f" (segment {chunk_index // chunk_size + 1})"
                page.insert_textbox(
                    fitz.Rect(36, 24, 559, 818),
                    f"{header}\n\n{segment}",
                    fontsize=9,
                    fontname="courier",
                    align=fitz.TEXT_ALIGN_LEFT,
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path)
    finally:
        doc.close()


def _ocr_page_with_ollama(
    *,
    pdf_path: Path,
    page_num: int,
    page: Any,
    ollama: OllamaModelManager,
    primary_model: str,
    fallback_model: str | None,
    timeout: float,
    artifact_dir: Path,
) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required for OCR page rendering.") from exc

    pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    image_bytes = pixmap.tobytes("png")
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    models = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models.append(fallback_model)
    attempts: list[dict[str, Any]] = []
    artifact_dir.mkdir(parents=True, exist_ok=True)
    response_path = artifact_dir / f"page_{int(page_num):05d}.response.txt"
    request_path = artifact_dir / f"page_{int(page_num):05d}.request.json"
    request_payload = {
        "pdf": str(pdf_path),
        "page_num": int(page_num),
        "models": models,
        "schema": OCR_PAGE_JSON_SCHEMA,
    }
    request_path.write_text(json.dumps(request_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_outputs: list[str] = []
    for model_name in models:
        result = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PDF source: {pdf_path.name}\n"
                        f"Page number: {int(page_num)}\n"
                        "Reconstruct the page text and describe visual content."
                    ),
                    "images": [image_b64],
                },
            ],
            timeout=timeout,
            temperature=0.0,
            options={"num_predict": 4096},
            response_format=OCR_PAGE_JSON_SCHEMA,
        )
        raw_outputs.append(str(result.output or result.stderr or ""))
        payload = extract_json_object(result.output or "")
        attempts.append(
            {
                "model": model_name,
                "returncode": int(result.returncode),
                "stderr": str(result.stderr or ""),
                "raw_length": len(str(result.output or "")),
            }
        )
        if result.returncode == 0 and isinstance(payload, dict):
            normalized = {
                "page_text": str(payload.get("page_text") or "").strip(),
                "image_descriptions": [str(item).strip() for item in list(payload.get("image_descriptions") or []) if str(item or "").strip()],
                "diagram_descriptions": [str(item).strip() for item in list(payload.get("diagram_descriptions") or []) if str(item or "").strip()],
                "table_descriptions": [str(item).strip() for item in list(payload.get("table_descriptions") or []) if str(item or "").strip()],
                "ocr_quality": str(payload.get("ocr_quality") or "partial").strip().lower() or "partial",
                "needs_review": bool(payload.get("needs_review")),
                "status": "completed",
                "model": model_name,
                "attempts": attempts,
            }
            response_path.write_text("\n\n".join(raw_outputs), encoding="utf-8")
            return normalized
    response_path.write_text("\n\n".join(raw_outputs), encoding="utf-8")
    return {
        "page_text": "",
        "image_descriptions": [],
        "diagram_descriptions": [],
        "table_descriptions": [],
        "ocr_quality": "poor",
        "needs_review": True,
        "status": "failed",
        "model": models[-1],
        "attempts": attempts,
    }


def _extract_pdf_pages_with_ocr(
    pdf_path: Path,
    *,
    max_pages: int,
    ocr_root: Path,
    ollama: OllamaModelManager,
    ocr_model: str,
    ocr_fallback_model: str | None,
    ocr_timeout: float,
) -> tuple[dict[int, str], set[int]]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for OCR-backed PDF reconstruction."
        ) from exc

    pdf_sha = _pdf_sha(pdf_path)
    pdf_ocr_dir = _ocr_pdf_dir(ocr_root, pdf_sha)
    pdf_ocr_dir.mkdir(parents=True, exist_ok=True)
    page_texts: dict[int, str] = {}
    regenerated_pages: set[int] = set()
    summary_records: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        total = len(doc)
        limit = total if max_pages <= 0 else min(total, int(max_pages))
        for i in range(limit):
            page_num = i + 1
            page_path = _ocr_page_path(pdf_ocr_dir, page_num)
            payload: dict[str, Any]
            if page_path.exists():
                try:
                    cached = json.loads(page_path.read_text(encoding="utf-8"))
                except Exception:
                    cached = {}
                payload = cached if isinstance(cached, dict) else {}
            else:
                payload = {}
            if _ocr_payload_requires_retry(payload):
                payload = _ocr_page_with_ollama(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    page=doc.load_page(i),
                    ollama=ollama,
                    primary_model=ocr_model,
                    fallback_model=ocr_fallback_model,
                    timeout=ocr_timeout,
                    artifact_dir=pdf_ocr_dir,
                )
                page_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                regenerated_pages.add(page_num)
            reconstructed = _compose_ocr_page_text(payload)
            page_texts[page_num] = reconstructed
            summary_records.append(
                {
                    "page_num": page_num,
                    "status": str(payload.get("status") or ""),
                    "ocr_quality": str(payload.get("ocr_quality") or ""),
                    "needs_review": bool(payload.get("needs_review")),
                    "model": str(payload.get("model") or ""),
                    "text_chars": len(reconstructed),
                }
            )

    reconstructed_text = "\n\n".join(
        f"[page {page_num}]\n{page_texts[page_num]}".strip() for page_num in sorted(page_texts)
    ).strip()
    (pdf_ocr_dir / "reconstructed.txt").write_text(reconstructed_text + ("\n" if reconstructed_text else ""), encoding="utf-8")
    _write_reconstructed_pdf(pdf_ocr_dir / "reconstructed.pdf", page_texts)
    (pdf_ocr_dir / "summary.json").write_text(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "page_count": len(page_texts),
                "ocr_model": ocr_model,
                "ocr_retry_model": ocr_fallback_model or "",
                "artifacts": {
                    "reconstructed_text": str(pdf_ocr_dir / "reconstructed.txt"),
                    "reconstructed_pdf": str(pdf_ocr_dir / "reconstructed.pdf"),
                },
                "pages": summary_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return page_texts, regenerated_pages


def _load_pdf_list(pdf_list: Path, *, limit: int) -> list[Path]:
    out: list[Path] = []
    if not pdf_list.exists():
        return out
    for raw_line in pdf_list.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if path.is_file() and path.suffix.lower() == ".pdf":
            out.append(path)
            if limit > 0 and len(out) >= limit:
                break
    return out


def _build_adjacent_context(pages: dict[int, str], page_num: int) -> list[str]:
    context: list[str] = []
    for adj in (page_num - 1, page_num + 1):
        text = str(pages.get(adj) or "").strip()
        if not text:
            continue
        context.append(f"[page {adj}] {text[:1200]}")
    return context


def _proceduralize_pdf_page(
    *,
    pdf_path: Path,
    page_num: int,
    total_pages: int,
    page_text: str,
    pages: dict[int, str],
    provider: str,
    model: str | None,
    model_profile: str,
    timeout: float,
    capture_dir: Path | None,
    ollama: OllamaModelManager,
    retry_attempts: int,
    retry_timeout_multiplier: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    final_receipt = None
    final_request = None
    final_timeout = float(timeout)
    max_attempts = max(1, int(retry_attempts))
    for attempt_index in range(max_attempts):
        current_timeout = float(timeout) * (float(retry_timeout_multiplier) ** attempt_index)
        receipt, request = proceduralize_text_content(
            content=str(page_text or ""),
            source_id=f"{pdf_path.stem}_p{int(page_num):04d}",
            domain_hint="General",
            source_path=f"{pdf_path}#page={int(page_num)}",
            context_chunks=_build_adjacent_context(pages, page_num),
            model=model,
            timeout=current_timeout,
            capture_dir=capture_dir,
            provider=provider,
            model_profile=model_profile,
            ollama=ollama,
            source_kind="pdf",
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "timeout_seconds": round(current_timeout, 3),
                "status": str(getattr(receipt, "status", "") or ""),
                "failure_code": str(getattr(receipt, "failure_code", "") or ""),
                "schema_ok": bool(getattr(receipt, "schema_ok", False)),
                "model": str(getattr(receipt, "model", "") or model or ""),
            }
        )
        final_receipt = receipt
        final_request = request
        final_timeout = current_timeout
        if str(getattr(receipt, "failure_code", "") or "").strip() == "plan_limit_consumed":
            break
        if receipt_is_usable(receipt) or not _is_retryable_failure(receipt):
            break
    assert final_receipt is not None
    assert final_request is not None
    decision = _receipt_to_decision(final_receipt.to_dict())
    decision["attempt_count"] = len(attempts)
    decision["final_timeout_seconds"] = round(final_timeout, 3)
    rows = bundle_to_payload_rows(final_receipt.parsed_bundle, final_request) if receipt_is_usable(final_receipt) else []
    return (
        decision,
        rows,
        final_receipt.to_dict(),
        final_receipt.parsed_bundle.to_dict(),
        attempts,
        str(getattr(final_receipt, "failure_code", "") or "").strip(),
    )


def _load_pdf_pages_for_ingest(
    pdf_path: Path,
    *,
    max_pages: int,
    ocr_needed_paths: set[str],
    ocr_root: Path,
    ollama: OllamaModelManager,
    ocr_model: str,
    ocr_fallback_model: str | None,
    ocr_timeout: float,
) -> tuple[dict[int, str], str, set[int]]:
    if _path_key(pdf_path) in ocr_needed_paths:
        pages, regenerated_pages = _extract_pdf_pages_with_ocr(
            pdf_path,
            max_pages=max_pages,
            ocr_root=ocr_root,
            ollama=ollama,
            ocr_model=ocr_model,
            ocr_fallback_model=ocr_fallback_model,
            ocr_timeout=ocr_timeout,
        )
        return pages, "ocr_reconstructed", regenerated_pages
    return _extract_pdf_pages(pdf_path, max_pages=max_pages), "direct_text", set()


def _delete_stage_pages(pdf_stage_dir: Path, page_numbers: set[int]) -> int:
    removed = 0
    for page_num in sorted(page_numbers):
        path = _page_stage_path(pdf_stage_dir, int(page_num))
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _retry_failed_stage_pages(
    *,
    stage_root: Path,
    pdf_paths: list[Path],
    storage_root: Path,
    provider: str,
    model_profile: str,
    model: str | None,
    timeout: float,
    capture_dir: Path | None,
    ollama: OllamaModelManager,
    retry_attempts: int,
    retry_timeout_multiplier: float,
    ocr_needed_paths: set[str],
    ocr_root: Path,
    ocr_model: str,
    ocr_fallback_model: str | None,
    ocr_timeout: float,
) -> dict[str, Any]:
    retryable_records: list[dict[str, Any]] = []
    allowed_pdfs = {_path_key(path): path for path in pdf_paths}
    for record in _iter_stage_records(stage_root):
        pdf_key = _path_key(str(record.get("pdf", "")))
        decision = record.get("decision")
        if pdf_key not in allowed_pdfs or not isinstance(decision, dict):
            continue
        reason = str(decision.get("reason") or "").strip().lower()
        if reason not in RETRYABLE_PAGE_FAILURE_CODES:
            continue
        retryable_records.append(record)

    stats = {
        "retryable_stage_pages": len(retryable_records),
        "retried_stage_pages": 0,
        "recovered_stage_pages": 0,
        "retry_stage_rows_added": 0,
        "stopped_due_to_plan_limit": False,
        "retry_after_utc": "",
    }
    if not retryable_records:
        return stats

    pages_cache: dict[str, tuple[dict[int, str], Path]] = {}
    for record in retryable_records:
        pdf_path = allowed_pdfs[_path_key(str(record.get("pdf", "")))]
        pdf_key = _path_key(pdf_path)
        if pdf_key not in pages_cache:
            pages_cache[pdf_key] = _load_pdf_pages_for_ingest(
                pdf_path,
                max_pages=0,
                ocr_needed_paths=ocr_needed_paths,
                ocr_root=ocr_root,
                ollama=ollama,
                ocr_model=ocr_model,
                ocr_fallback_model=ocr_fallback_model,
                ocr_timeout=ocr_timeout,
            )[0], pdf_path
        pages = pages_cache[pdf_key][0]
        page_num = int(record.get("page_num", 0) or 0)
        page_text = str(pages.get(page_num) or "").strip()
        if not page_text:
            continue
        total_pages = len(pages)
        pdf_stage_dir = _pdf_stage_dir(stage_root, _pdf_sha(pdf_path))
        previous_rows = record.get("rows")
        previous_row_count = len(previous_rows) if isinstance(previous_rows, list) else 0
        decision, rows, receipt_dict, bundle_dict, attempts, failure_code = _proceduralize_pdf_page(
            pdf_path=pdf_path,
            page_num=page_num,
            total_pages=total_pages,
            page_text=page_text,
            pages=pages,
            provider=provider,
            model=model,
            model_profile=model_profile,
            timeout=timeout,
            capture_dir=capture_dir,
            ollama=ollama,
            retry_attempts=retry_attempts,
            retry_timeout_multiplier=retry_timeout_multiplier,
        )
        _write_stage_page(
            pdf_stage_dir=pdf_stage_dir,
            page_num=page_num,
            pdf_path=pdf_path,
            total_pages=total_pages,
            decision=decision,
            rows=rows,
            receipt=receipt_dict,
            bundle=bundle_dict,
            attempts=attempts,
        )
        stats["retried_stage_pages"] = int(stats["retried_stage_pages"]) + 1
        if receipt_dict.get("schema_ok"):
            stats["recovered_stage_pages"] = int(stats["recovered_stage_pages"]) + 1
        stats["retry_stage_rows_added"] = int(stats["retry_stage_rows_added"]) + max(0, len(rows) - previous_row_count)
        if failure_code == "plan_limit_consumed":
            stats["stopped_due_to_plan_limit"] = True
            stats["retry_after_utc"] = str(receipt_dict.get("retry_after_utc") or "")
            break
    return stats


def _iter_pdf_paths(
    *,
    pdf: Path | None,
    pdf_dir: Path | None,
    pdf_list: Path | None,
    pattern: str,
    limit: int,
) -> list[Path]:
    out: list[Path] = []
    if pdf is not None:
        if pdf.exists() and pdf.suffix.lower() == ".pdf":
            out.append(pdf)
        return out
    if pdf_list is not None:
        return _load_pdf_list(pdf_list, limit=limit)
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


def _empty_page_decision(*, provider: str, model: str | None) -> dict[str, Any]:
    return {
        "classification": "non_knowledge",
        "resolved_classification": "non_knowledge",
        "confidence": 1.0,
        "reason": "empty_page",
        "context_needed": [],
        "knowledge_type": None,
        "provider": provider,
        "model": model,
    }


def _receipt_to_decision(receipt: dict[str, Any]) -> dict[str, Any]:
    bundle = receipt.get("parsed_bundle") if isinstance(receipt, dict) else {}
    if not isinstance(bundle, dict):
        bundle = {}
    action = str(bundle.get("ingest_action") or "").strip().lower()
    if action == "augment" and bool(receipt.get("schema_ok")):
        classification = "knowledge"
        knowledge_type = "summary"
    elif action == "reject" or not bool(receipt.get("schema_ok")):
        classification = "ambiguous"
        knowledge_type = None
    elif action == "needs_context":
        classification = "ambiguous"
        knowledge_type = None
    else:
        classification = "non_knowledge"
        knowledge_type = None
    return {
        "classification": classification,
        "resolved_classification": classification,
        "confidence": 1.0 if bool(receipt.get("schema_ok")) else 0.35,
        "reason": str(receipt.get("failure_code") or action or receipt.get("status") or "proceduralizer"),
        "context_needed": [] if action != "needs_context" else ["adjacent_pages"],
        "knowledge_type": knowledge_type,
        "provider": receipt.get("provider"),
        "model": receipt.get("model"),
    }


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
    receipt: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "pdf": str(pdf_path),
        "page_num": int(page_num),
        "total_pages": int(total_pages),
        "decision": decision,
        "rows": rows,
    }
    if receipt is not None:
        payload["receipt"] = receipt
    if bundle is not None:
        payload["bundle"] = bundle
    if attempts:
        payload["attempts"] = attempts
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


def _rebuild_payload_from_stage(
    *,
    stage_root: Path,
    payload_output: Path,
    storage_root: Path | None = None,
) -> tuple[Counter[str], Counter[str], dict[str, dict[str, Any]], int]:
    by_galaxy: Counter[str] = Counter()
    by_classification: Counter[str] = Counter()
    per_pdf: dict[str, dict[str, Any]] = {}
    total_rows = 0
    records = _iter_stage_records(stage_root)
    external_context = load_external_enrichment_context(
        (storage_root / "checkpoints" / "galaxy_consolidated_latest.json") if storage_root is not None else None
    )
    rows_by_pdf: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pdf_key = str(record.get("pdf", ""))
        rows = record.get("rows")
        if not isinstance(rows, list):
            continue
        rows_by_pdf.setdefault(pdf_key, []).extend(row for row in rows if isinstance(row, dict))
    contexts_by_pdf = {
        pdf_key: build_row_enrichment_context(rows, external_context=external_context)
        for pdf_key, rows in rows_by_pdf.items()
    }

    payload_output.parent.mkdir(parents=True, exist_ok=True)
    with payload_output.open("w", encoding="utf-8") as handle:
        for record in records:
            pdf_key = str(record.get("pdf", ""))
            page_num = int(record.get("page_num", 0) or 0)
            total_pages = int(record.get("total_pages", 0) or 0)
            decision = record.get("decision")
            rows = record.get("rows")
            if not isinstance(decision, dict):
                decision = {}
            if not isinstance(rows, list):
                rows = []
            rows = second_pass_enrich_payload_rows(rows, context=contexts_by_pdf.get(pdf_key))

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


def _skip_sources_output(payload_output: Path, explicit_skip_output: Path | None) -> Path:
    if explicit_skip_output is not None:
        return explicit_skip_output
    return payload_output.parent / f"{payload_output.stem}_skipped_sources.jsonl"


def _can_skip_pdf_processing_from_manifest(
    *,
    manifest_entry: dict[str, Any] | None,
    pdf_sha: str,
    force_reprocess: bool,
) -> bool:
    if bool(force_reprocess):
        return False
    if not isinstance(manifest_entry, dict):
        return False
    if str(manifest_entry.get("sha256") or "").strip() != str(pdf_sha).strip():
        return False
    if str(manifest_entry.get("status") or "").strip().lower() != "staged_complete":
        return False
    try:
        pages_total = int(manifest_entry.get("pages_total", 0) or 0)
        resume_from_page = int(manifest_entry.get("resume_from_page", 0) or 0)
    except Exception:
        return False
    return pages_total > 0 and resume_from_page >= (pages_total + 1)


def _append_skipped_source(
    *,
    skip_output: Path,
    pdf_path: Path,
    phase: str,
    error: Exception,
    pages_total: int | None = None,
) -> None:
    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf_path),
        "phase": str(phase),
        "error_type": error.__class__.__name__,
        "error_message": str(error),
    }
    if pages_total is not None:
        entry["pages_total"] = int(pages_total)
    skip_output.parent.mkdir(parents=True, exist_ok=True)
    with skip_output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


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
    src.add_argument(
        "--pdf-list",
        type=Path,
        help="Newline-delimited PDF list processed exactly in file order; preferred for ordered large-batch ingest.",
    )
    parser.add_argument("--pattern", default="**/*.pdf", help="Glob pattern when using --pdf-dir")
    parser.add_argument(
        "--limit-pdfs",
        type=int,
        default=0,
        help="Limit number of PDFs from --pdf-dir or --pdf-list (0 means no limit).",
    )
    parser.add_argument("--max-pages-per-pdf", type=int, default=0, help="0 means all pages")
    parser.add_argument("--cache-dir", type=Path, default=Path("/K3D/Knowledge3D.local/pdf_cache"))
    parser.add_argument("--provider", default="ollama", help="Canonical transport provider; ollama is default.")
    parser.add_argument("--model-profile", default="quality", help="Proceduralizer model profile.")
    parser.add_argument("--model", default=None, help="Optional explicit model override.")
    parser.add_argument(
        "--classifier-model",
        default="qwen2.5:32b",
        help="Legacy compatibility override; use --model or --model-profile for canonical runs.",
    )
    parser.add_argument(
        "--augmenter-model",
        default="qwen2.5:32b",
        help="Legacy compatibility override; use --model or --model-profile for canonical runs.",
    )
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--capture-dir", type=Path, default=None, help="Optional request/response capture directory.")
    parser.add_argument(
        "--payload-output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/external_payloads/pdf_intelligent_payload.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/external_payloads/pdf_intelligent_report.json"),
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
    parser.add_argument("--storage-root", type=Path, default=Path("/K3D/Knowledge3D.local"))
    parser.add_argument("--ingest", action="store_true", help="Run single-world ingestion after payload generation")
    parser.add_argument(
        "--ingest-report",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/external_payloads/pdf_intelligent_ingest_report.json"),
    )
    parser.add_argument(
        "--skip-sources-output",
        type=Path,
        default=None,
        help="JSONL output file listing skipped PDFs (encrypted/corrupt/failed extraction).",
    )
    parser.add_argument(
        "--ocr-needed-list",
        type=Path,
        default=None,
        help="Optional newline-delimited list of PDFs that require OCR reconstruction before ingestion.",
    )
    parser.add_argument(
        "--ocr-artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for OCR reconstruction artifacts and reconstructed PDFs.",
    )
    parser.add_argument(
        "--ocr-model",
        default=OCR_DEFAULT_MODEL,
        help="Primary OCR/vision model used for scanned PDFs.",
    )
    parser.add_argument(
        "--ocr-retry-model",
        default=OCR_RETRY_MODEL,
        help="Secondary cloud OCR/vision model used when the primary OCR model fails.",
    )
    parser.add_argument(
        "--ocr-fallback-model",
        default="",
        help="Deprecated alias for --ocr-retry-model. Non-cloud values are ignored.",
    )
    parser.add_argument(
        "--ocr-timeout",
        type=float,
        default=240.0,
        help="Timeout in seconds for OCR/vision page reconstruction calls.",
    )
    parser.add_argument(
        "--page-retry-attempts",
        type=int,
        default=3,
        help="Total proceduralizer attempts per page before advancing.",
    )
    parser.add_argument(
        "--page-retry-timeout-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to timeout on each transient retry attempt.",
    )
    parser.add_argument(
        "--repair-retryable-stage-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before final payload rebuild, rerun staged pages that previously ended in timeout or invalid_json.",
    )
    args = parser.parse_args()

    pdf_paths = _iter_pdf_paths(
        pdf=args.pdf,
        pdf_dir=args.pdf_dir,
        pdf_list=args.pdf_list,
        pattern=args.pattern,
        limit=max(0, int(args.limit_pdfs)),
    )
    if not pdf_paths:
        print("[pdf-ingest] no PDF files found")
        return 1

    ocr_needed_paths = {_path_key(path) for path in _read_path_list(args.ocr_needed_list)}
    if ocr_needed_paths:
        seen_keys = {_path_key(path) for path in pdf_paths}
        for ocr_path in _read_path_list(args.ocr_needed_list):
            key = _path_key(ocr_path)
            if key in seen_keys:
                continue
            if ocr_path.is_file() and ocr_path.suffix.lower() == ".pdf":
                pdf_paths.append(ocr_path)
                seen_keys.add(key)

    stage_root = _stage_root(args.payload_output, args.stage_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    skip_output = _skip_sources_output(args.payload_output, args.skip_sources_output)
    ocr_root = _ocr_artifact_root(args.payload_output, args.ocr_artifact_dir)
    ocr_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_stage_manifest(stage_root)
    manifest_pdfs = manifest.setdefault("pdfs", {})
    if not isinstance(manifest_pdfs, dict):
        manifest["pdfs"] = {}
        manifest_pdfs = manifest["pdfs"]
    skipped_sources_count = 0
    stop_due_to_plan_limit = False
    retry_after_utc = ""

    explicit_model = str(args.model or "").strip() or None
    legacy_override = None
    if explicit_model is None:
        if str(args.augmenter_model).strip() and str(args.augmenter_model).strip() != "qwen2.5:32b":
            legacy_override = str(args.augmenter_model).strip()
        elif str(args.classifier_model).strip() and str(args.classifier_model).strip() != "qwen2.5:32b":
            legacy_override = str(args.classifier_model).strip()
    resolved_model = explicit_model or legacy_override
    resolved_ocr_retry_model = _normalize_ocr_retry_model(
        primary_model=str(args.ocr_model).strip() or OCR_DEFAULT_MODEL,
        retry_model=getattr(args, "ocr_retry_model", None),
        legacy_retry_model=getattr(args, "ocr_fallback_model", None),
    )

    with OllamaModelManager(default_timeout=float(args.ollama_timeout)) as ollama:

        checkpoint_interval = max(0, int(args.payload_checkpoint_interval_pdfs))
        processed_pdfs = 0
        for pdf_path in pdf_paths:
            pdf_sha = _pdf_sha(pdf_path)
            existing_manifest_entry = manifest_pdfs.get(str(pdf_path))
            if _can_skip_pdf_processing_from_manifest(
                manifest_entry=existing_manifest_entry if isinstance(existing_manifest_entry, dict) else None,
                pdf_sha=pdf_sha,
                force_reprocess=bool(args.force_reprocess),
            ):
                processed_pdfs += 1
                if checkpoint_interval > 0 and processed_pdfs % checkpoint_interval == 0:
                    _rebuild_payload_from_stage(
                        stage_root=stage_root,
                        payload_output=args.payload_output,
                        storage_root=args.storage_root,
                    )
                continue
            try:
                pages, extraction_mode, refreshed_ocr_pages = _load_pdf_pages_for_ingest(
                    pdf_path,
                    max_pages=max(0, int(args.max_pages_per_pdf)),
                    ocr_needed_paths=ocr_needed_paths,
                    ocr_root=ocr_root,
                    ollama=ollama,
                    ocr_model=str(args.ocr_model).strip() or OCR_DEFAULT_MODEL,
                    ocr_fallback_model=resolved_ocr_retry_model,
                    ocr_timeout=float(args.ocr_timeout),
                )
            except Exception as exc:
                _append_skipped_source(
                    skip_output=skip_output,
                    pdf_path=pdf_path,
                    phase="extract_pages_or_ocr",
                    error=exc,
                )
                manifest_pdfs[str(pdf_path)] = {
                    "sha256": pdf_sha,
                    "pages_total": 0,
                    "resume_from_page": 1,
                    "status": "skipped",
                    "skip_reason": f"{exc.__class__.__name__}: {exc}",
                }
                _save_stage_manifest(stage_root, manifest)
                skipped_sources_count += 1
                continue
            pdf_stage = _pdf_stage_dir(stage_root, pdf_sha)
            if not pages:
                manifest_pdfs[str(pdf_path)] = {
                    "sha256": pdf_sha,
                    "pages_total": 0,
                    "resume_from_page": 1,
                    "status": "skipped_empty",
                }
                _save_stage_manifest(stage_root, manifest)
                continue

            staged_pages = _existing_staged_pages(pdf_stage)
            if refreshed_ocr_pages:
                _delete_stage_pages(pdf_stage, refreshed_ocr_pages)
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
                "extraction_mode": extraction_mode,
            }
            _save_stage_manifest(stage_root, manifest)

            total_pages = len(pages)
            for page_num, page_text in sorted(pages.items(), key=lambda item: item[0]):
                stage_page = _page_stage_path(pdf_stage, int(page_num))
                if int(page_num) < int(resume_from) and stage_page.exists():
                    continue

                if not str(page_text or "").strip():
                    _write_stage_page(
                        pdf_stage_dir=pdf_stage,
                        page_num=int(page_num),
                        pdf_path=pdf_path,
                        total_pages=total_pages,
                        decision=_empty_page_decision(
                            provider=str(args.provider).strip().lower(),
                            model=resolved_model,
                        ),
                        rows=[],
                    )
                    manifest_pdfs[str(pdf_path)]["resume_from_page"] = int(page_num) + 1
                    _save_stage_manifest(stage_root, manifest)
                    continue

                decision, rows, receipt_dict, bundle_dict, attempts, failure_code = _proceduralize_pdf_page(
                    pdf_path=pdf_path,
                    page_num=int(page_num),
                    total_pages=total_pages,
                    page_text=str(page_text or ""),
                    pages=pages,
                    provider=str(args.provider).strip().lower(),
                    model=resolved_model,
                    model_profile=str(args.model_profile).strip().lower(),
                    timeout=float(args.ollama_timeout),
                    capture_dir=args.capture_dir,
                    ollama=ollama,
                    retry_attempts=max(1, int(args.page_retry_attempts)),
                    retry_timeout_multiplier=max(1.0, float(args.page_retry_timeout_multiplier)),
                )
                _write_stage_page(
                    pdf_stage_dir=pdf_stage,
                    page_num=int(page_num),
                    pdf_path=pdf_path,
                    total_pages=total_pages,
                    decision=decision,
                    rows=rows,
                    receipt=receipt_dict,
                    bundle=bundle_dict,
                    attempts=attempts,
                )
                manifest_pdfs[str(pdf_path)]["resume_from_page"] = int(page_num) + 1
                _save_stage_manifest(stage_root, manifest)
                if failure_code == "plan_limit_consumed":
                    stop_due_to_plan_limit = True
                    retry_after_utc = str(receipt_dict.get("retry_after_utc") or "").strip()
                    manifest_pdfs[str(pdf_path)]["status"] = "stopped_plan_limit"
                    manifest_pdfs[str(pdf_path)]["retry_after_utc"] = retry_after_utc
                    _save_stage_manifest(stage_root, manifest)
                    break

            processed_pdfs += 1
            if not stop_due_to_plan_limit:
                manifest_pdfs[str(pdf_path)]["status"] = "staged_complete"
                manifest_pdfs[str(pdf_path)]["resume_from_page"] = int(total_pages) + 1
                _save_stage_manifest(stage_root, manifest)
            if checkpoint_interval > 0 and processed_pdfs % checkpoint_interval == 0:
                _rebuild_payload_from_stage(
                    stage_root=stage_root,
                    payload_output=args.payload_output,
                    storage_root=args.storage_root,
                )
            if stop_due_to_plan_limit:
                break

    repair_stats = {
        "retryable_stage_pages": 0,
        "retried_stage_pages": 0,
        "recovered_stage_pages": 0,
        "retry_stage_rows_added": 0,
        "stopped_due_to_plan_limit": False,
        "retry_after_utc": "",
    }
    if bool(args.repair_retryable_stage_pages) and not stop_due_to_plan_limit:
        with OllamaModelManager(default_timeout=float(args.ollama_timeout)) as repair_ollama:
            repair_stats = _retry_failed_stage_pages(
                stage_root=stage_root,
                pdf_paths=pdf_paths,
                storage_root=args.storage_root,
                provider=str(args.provider).strip().lower(),
                model_profile=str(args.model_profile).strip().lower(),
                model=resolved_model,
                timeout=float(args.ollama_timeout),
                capture_dir=args.capture_dir,
                ollama=repair_ollama,
                retry_attempts=max(1, int(args.page_retry_attempts)),
                retry_timeout_multiplier=max(1.0, float(args.page_retry_timeout_multiplier)),
                ocr_needed_paths=ocr_needed_paths,
                ocr_root=ocr_root,
                ocr_model=str(args.ocr_model).strip() or OCR_DEFAULT_MODEL,
                ocr_fallback_model=resolved_ocr_retry_model,
                ocr_timeout=float(args.ocr_timeout),
            )
        if bool(repair_stats.get("stopped_due_to_plan_limit")):
            stop_due_to_plan_limit = True
            retry_after_utc = str(repair_stats.get("retry_after_utc") or "").strip()

    by_galaxy, by_classification, per_pdf, payload_rows = _rebuild_payload_from_stage(
        stage_root=stage_root,
        payload_output=args.payload_output,
        storage_root=args.storage_root,
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
        "source_mode": "pdf" if args.pdf is not None else ("pdf_list" if args.pdf_list is not None else "pdf_dir"),
        "payload_rows": int(payload_rows),
        "skipped_sources_count": int(skipped_sources_count),
        "skipped_sources_output": str(skip_output),
        "pdfs": pdf_stats,
        "classification_counts": dict(sorted(by_classification.items())),
        "rows_by_galaxy": dict(sorted(by_galaxy.items())),
        "models": {
            "provider": args.provider,
            "model_profile": args.model_profile,
            "model": resolved_model,
            "classifier": args.classifier_model,
            "augmenter": args.augmenter_model,
        },
        "cache_dir": str(args.cache_dir),
        "stage_dir": str(stage_root),
        "capture_dir": str(args.capture_dir) if args.capture_dir is not None else "",
        "ocr_artifact_dir": str(ocr_root),
        "ocr_needed_list": str(args.ocr_needed_list) if args.ocr_needed_list is not None else "",
        "ocr_needed_pdf_count": len(ocr_needed_paths),
        "ocr_model": str(args.ocr_model).strip(),
        "ocr_retry_model": resolved_ocr_retry_model or "",
        "ocr_timeout": float(args.ocr_timeout),
        "page_retry_attempts": int(args.page_retry_attempts),
        "page_retry_timeout_multiplier": float(args.page_retry_timeout_multiplier),
        "repair_retryable_stage_pages": bool(args.repair_retryable_stage_pages),
        "repair_stats": repair_stats,
        "force_reprocess": bool(args.force_reprocess),
        "resume_last_page": not bool(args.disable_resume_last_page),
        "ingest_requested": bool(args.ingest),
        "stopped_due_to_plan_limit": bool(stop_due_to_plan_limit),
        "retry_after_utc": retry_after_utc,
    }

    if args.ingest:
        rc = _ingest_payload(args.payload_output, storage_root=args.storage_root, ingest_report=args.ingest_report)
        report["ingest_returncode"] = rc
        report["ingest_report"] = str(args.ingest_report)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[pdf-ingest] pdfs={len(pdf_paths)} rows={payload_rows} payload={args.payload_output}")
    print(f"[pdf-ingest] report={args.report_output}")
    print(f"[pdf-ingest] skipped_sources={skipped_sources_count} file={skip_output}")
    for galaxy, count in sorted(by_galaxy.items()):
        print(f"[pdf-ingest] {galaxy}: {count}")

    if stop_due_to_plan_limit:
        return 75
    return int(report.get("ingest_returncode", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
