"""LLM-driven PDF page knowledge classifier with persistent page-decision cache."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from knowledge3d.ingestion.ollama_manager import OllamaModelManager


CLASSIFICATION_SYSTEM_PROMPT = """You classify academic PDF pages.
Return strict JSON with keys:
classification: knowledge|non_knowledge|ambiguous
confidence: 0..1
reason: short text
context_needed: list|null
knowledge_type: definition|theorem|experiment|algorithm|summary|null

Classify as non_knowledge for publishing metadata, acknowledgements, references-only pages.
Classify as ambiguous when fragment lacks context (continued proof/table without caption).
"""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _coerce_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


@dataclass
class PageDecision:
    classification: str
    confidence: float
    reason: str
    context_needed: list[str] | None
    knowledge_type: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "context_needed": list(self.context_needed or []),
            "knowledge_type": self.knowledge_type,
        }


class PDFKnowledgeClassifier:
    """Classify PDF pages into knowledge/non-knowledge/ambiguous with cache memory."""

    def __init__(
        self,
        *,
        ollama: OllamaModelManager,
        cache_dir: str | Path,
        model: str = "qwen2.5:32b",
        timeout: float = 90.0,
    ) -> None:
        self.ollama = ollama
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.timeout = float(timeout)

    def classify_page(
        self,
        *,
        pdf_path: str | Path,
        page_num: int,
        total_pages: int,
        page_text: str,
        context_pages: Mapping[int, str] | None = None,
        force_reprocess: bool = False,
    ) -> dict[str, Any]:
        pdf_obj = Path(pdf_path)
        sanitized_page_text = self._sanitize_text(page_text, max_chars=6000)
        cache = self._load_cache(pdf_obj)
        key = str(int(page_num))
        existing = cache.get("page_decisions", {}).get(key)
        if not force_reprocess and isinstance(existing, dict):
            confidence = float(existing.get("confidence", 0.0) or 0.0)
            if bool(existing.get("skip_in_future")):
                return dict(existing)
            if confidence >= 0.7:
                return dict(existing)

        context_blob = ""
        if context_pages:
            lines: list[str] = []
            for pnum, ptxt in sorted(context_pages.items(), key=lambda item: item[0]):
                safe_context = self._sanitize_text(ptxt, max_chars=1200)
                lines.append(f"[Page {int(pnum)}]\n{safe_context}")
            if lines:
                context_blob = "\n\nContext pages:\n" + "\n\n".join(lines)

        request_prompt = (
            f"{CLASSIFICATION_SYSTEM_PROMPT}\n\n"
            f"PDF: {pdf_obj.name}\n"
            f"Page: {int(page_num)} / {int(total_pages)}\n"
            f"Page text:\n{sanitized_page_text}\n"
            f"{context_blob}\n"
            "Return JSON only."
        )
        response = self.ollama.query(model=self.model, prompt=request_prompt, timeout=self.timeout)
        parsed = _coerce_json_object(response.output)

        decision = self._normalize_decision(parsed)
        payload = decision.as_dict()
        payload.update(
            {
                "processed_at": _now_iso(),
                "model": self.model,
                "skip_in_future": decision.classification == "non_knowledge" and decision.confidence >= 0.8,
                "context_used": sorted([str(int(k)) for k in (context_pages or {}).keys()]),
            }
        )

        page_decisions = cache.setdefault("page_decisions", {})
        page_decisions[key] = payload
        cache.setdefault("pdf_metadata", {})
        cache["pdf_metadata"].update(
            {
                "filename": pdf_obj.name,
                "sha256": self._pdf_sha(pdf_obj),
                "total_pages": int(total_pages),
                "processed_at": _now_iso(),
                "ollama_model": self.model,
            }
        )
        self._save_cache(pdf_obj, cache)
        return payload

    def classify_pdf_pages(
        self,
        *,
        pdf_path: str | Path,
        pages: Mapping[int, str],
        force_reprocess: bool = False,
    ) -> dict[int, dict[str, Any]]:
        total_pages = len(pages)
        decisions: dict[int, dict[str, Any]] = {}
        for page_num, page_text in sorted(pages.items(), key=lambda item: item[0]):
            decisions[int(page_num)] = self.classify_page(
                pdf_path=pdf_path,
                page_num=int(page_num),
                total_pages=total_pages,
                page_text=self._sanitize_text(page_text, max_chars=12000),
                context_pages=None,
                force_reprocess=force_reprocess,
            )

        # Second pass: resolve ambiguous pages with nearby context.
        for page_num, decision in list(decisions.items()):
            if str(decision.get("classification", "")).strip().lower() != "ambiguous":
                continue
            neighbors: dict[int, str] = {}
            for adj in (page_num - 1, page_num + 1):
                if adj in pages:
                    neighbors[adj] = pages[adj]
            if not neighbors:
                continue
            resolved = self.classify_page(
                pdf_path=pdf_path,
                page_num=page_num,
                total_pages=total_pages,
                page_text=self._sanitize_text(pages[page_num], max_chars=12000),
                context_pages=neighbors,
                force_reprocess=True,
            )
            resolved["resolved_classification"] = resolved.get("classification")
            decisions[page_num] = resolved
        return decisions

    def _sanitize_text(self, text: str, *, max_chars: int) -> str:
        raw = str(text or "")
        if "\x00" in raw:
            raw = raw.replace("\x00", "")
        raw = "".join(ch if (ch >= " " or ch in "\n\t") else " " for ch in raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = raw.encode("utf-8", errors="ignore").decode("utf-8")
        out = raw.strip()
        if max_chars > 0:
            out = out[: int(max_chars)]
        return out

    def _normalize_decision(self, payload: dict[str, Any]) -> PageDecision:
        cls = str(payload.get("classification", "ambiguous")).strip().lower()
        if cls not in {"knowledge", "non_knowledge", "ambiguous"}:
            cls = "ambiguous"
        confidence = payload.get("confidence", 0.5)
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except Exception:
            conf = 0.5
        reason = str(payload.get("reason", "model_response")).strip() or "model_response"
        ctx = payload.get("context_needed")
        if isinstance(ctx, list):
            context_needed = [str(item) for item in ctx if str(item).strip()]
        else:
            context_needed = None
        ktype_raw = payload.get("knowledge_type")
        knowledge_type = str(ktype_raw).strip().lower() if isinstance(ktype_raw, str) and ktype_raw.strip() else None
        return PageDecision(
            classification=cls,
            confidence=conf,
            reason=reason,
            context_needed=context_needed,
            knowledge_type=knowledge_type,
        )

    def _cache_path(self, pdf_path: Path) -> Path:
        return self.cache_dir / f"{self._pdf_sha(pdf_path)}.json"

    def _load_cache(self, pdf_path: Path) -> dict[str, Any]:
        path = self._cache_path(pdf_path)
        if not path.exists():
            return {"pdf_metadata": {}, "page_decisions": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("pdf_metadata", {})
                payload.setdefault("page_decisions", {})
                return payload
        except Exception:
            pass
        return {"pdf_metadata": {}, "page_decisions": {}}

    def _save_cache(self, pdf_path: Path, payload: dict[str, Any]) -> None:
        path = self._cache_path(pdf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _pdf_sha(self, pdf_path: Path) -> str:
        data = pdf_path.read_bytes()
        return hashlib.sha256(data).hexdigest()
