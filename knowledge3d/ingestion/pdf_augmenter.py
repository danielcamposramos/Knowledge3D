"""LLM-driven PDF page augmenter for Galaxy-ready payload rows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from knowledge3d.ingestion.ollama_manager import OllamaModelManager


AUGMENTATION_SYSTEM_PROMPT = """You augment academic knowledge for K3D Galaxy ingestion.
Return strict JSON with keys:
summary: compact summary text
entities: list of {"type","name","content"}
relationships: list of {"from","relation","to"}
cross_modal: {"has_equations","has_figures","has_algorithms","notation"}
embedding_text: compact semantic search text
galaxy_hints: {"target_galaxy","pattern_type","difficulty"}

target_galaxy must be one of: Math, Grammar, Reality, Drawing, Word, Character, 3DObjects, Audio.
Avoid long narrative and avoid chain-of-thought.
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


class PDFKnowledgeAugmenter:
    """Augment knowledge pages and produce galaxy payload rows."""

    def __init__(
        self,
        *,
        ollama: OllamaModelManager,
        model: str = "qwen2.5:32b",
        timeout: float = 120.0,
    ) -> None:
        self.ollama = ollama
        self.model = model
        self.timeout = float(timeout)

    def augment_page(
        self,
        *,
        pdf_path: str | Path,
        page_num: int,
        total_pages: int,
        page_text: str,
        classification: Mapping[str, Any] | None = None,
        context_pages: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        pdf_obj = Path(pdf_path)
        cls = dict(classification or {})
        cls_type = str(cls.get("knowledge_type", "")).strip().lower() or "summary"
        cls_label = str(cls.get("resolved_classification") or cls.get("classification") or "knowledge")

        context_blob = ""
        if context_pages:
            parts: list[str] = []
            for pnum, ptxt in sorted(context_pages.items(), key=lambda item: item[0]):
                parts.append(f"[Page {int(pnum)}]\n{str(ptxt)[:1200]}")
            if parts:
                context_blob = "\n\nContext pages:\n" + "\n\n".join(parts)

        prompt = (
            f"{AUGMENTATION_SYSTEM_PROMPT}\n\n"
            f"PDF: {pdf_obj.name}\n"
            f"Page: {int(page_num)} / {int(total_pages)}\n"
            f"Classification: {cls_label}\n"
            f"Knowledge type: {cls_type}\n"
            f"Page text:\n{(page_text or '')[:7000]}\n"
            f"{context_blob}\n"
            "Return JSON only."
        )
        response = self.ollama.query(model=self.model, prompt=prompt, timeout=self.timeout)
        parsed = _coerce_json_object(response.output)
        if not parsed:
            parsed = {
                "summary": (page_text or "")[:320],
                "entities": [],
                "relationships": [],
                "cross_modal": {
                    "has_equations": False,
                    "has_figures": False,
                    "has_algorithms": False,
                    "notation": [],
                },
                "embedding_text": (page_text or "")[:1024],
                "galaxy_hints": {"target_galaxy": "Word", "pattern_type": "summary", "difficulty": "intermediate"},
            }
        return self._normalize_augmentation(parsed)

    def to_payload_rows(
        self,
        *,
        pdf_path: str | Path,
        page_num: int,
        augmented: Mapping[str, Any],
        classification: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pdf_obj = Path(pdf_path)
        aug = dict(augmented)
        hints = aug.get("galaxy_hints")
        hints_map = hints if isinstance(hints, dict) else {}
        target = self._normalize_galaxy_name(str(hints_map.get("target_galaxy", "Word")))
        pattern_type = str(hints_map.get("pattern_type", "summary")).strip().lower() or "summary"
        summary = str(aug.get("summary", "")).strip()
        embedding_text = str(aug.get("embedding_text", "")).strip()
        entities = aug.get("entities") if isinstance(aug.get("entities"), list) else []
        relationships = aug.get("relationships") if isinstance(aug.get("relationships"), list) else []
        cross_modal = aug.get("cross_modal") if isinstance(aug.get("cross_modal"), dict) else {}

        base_id = f"{pdf_obj.stem}_p{int(page_num):04d}"
        metadata = {
            "source": "pdf_intelligent_augmentation",
            "source_pdf": pdf_obj.name,
            "page_num": int(page_num),
            "knowledge_type": str((classification or {}).get("knowledge_type", "summary")).strip().lower() or "summary",
            "classification": str((classification or {}).get("resolved_classification") or (classification or {}).get("classification") or "knowledge"),
            "entities": entities,
            "relationships": relationships,
            "cross_modal": cross_modal,
            "embedding_text": embedding_text,
            "augmented_by_ollama": True,
            "generated": False,
            "symlink": "character_galaxy|word_galaxy",
            "processed_at": _now_iso(),
        }

        primary = {
            "galaxy": target,
            "entry": {
                "id": f"{target.lower()}_{base_id}",
                "name": summary or f"{pdf_obj.stem} page {int(page_num)}",
                "domain": target.lower(),
                "category": f"pdf_{pattern_type}",
                "rpn_program": self._rpn_from_entities(entities, fallback=f"PDF_PAGE_{int(page_num)} KNOWLEDGE_TOKENIZE"),
                "metadata": metadata,
            },
        }
        support = {
            "galaxy": "Grammar",
            "entry": {
                "id": f"grammar_{base_id}",
                "name": f"PDF grammar bridge {pdf_obj.stem} p{int(page_num)}",
                "domain": "grammar",
                "category": "pdf_reasoning_bridge",
                "rpn_program": "QUERY CONTEXT ALIGN COMPOSE",
                "metadata": {
                    **metadata,
                    "target_galaxy": target,
                    "symlink": f"{target.lower()}_galaxy",
                },
            },
        }
        return [primary, support]

    def _normalize_augmentation(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = str(payload.get("summary", "")).strip()
        entities = payload.get("entities")
        relationships = payload.get("relationships")
        cross_modal = payload.get("cross_modal")
        embedding_text = str(payload.get("embedding_text", "")).strip()
        hints = payload.get("galaxy_hints")

        if not isinstance(entities, list):
            entities = []
        if not isinstance(relationships, list):
            relationships = []
        if not isinstance(cross_modal, dict):
            cross_modal = {
                "has_equations": False,
                "has_figures": False,
                "has_algorithms": False,
                "notation": [],
            }
        if not isinstance(hints, dict):
            hints = {"target_galaxy": "Word", "pattern_type": "summary", "difficulty": "intermediate"}

        hints = {
            "target_galaxy": self._normalize_galaxy_name(str(hints.get("target_galaxy", "Word"))),
            "pattern_type": str(hints.get("pattern_type", "summary")).strip().lower() or "summary",
            "difficulty": str(hints.get("difficulty", "intermediate")).strip().lower() or "intermediate",
        }
        if not embedding_text:
            embedding_text = summary[:1024]

        return {
            "summary": summary,
            "entities": entities,
            "relationships": relationships,
            "cross_modal": cross_modal,
            "embedding_text": embedding_text,
            "galaxy_hints": hints,
        }

    def _normalize_galaxy_name(self, raw: str) -> str:
        key = raw.strip().lower()
        mapping = {
            "math": "Math",
            "grammar": "Grammar",
            "reality": "Reality",
            "drawing": "Drawing",
            "word": "Word",
            "character": "Character",
            "3dobjects": "3DObjects",
            "audio": "Audio",
        }
        return mapping.get(key, "Word")

    def _rpn_from_entities(self, entities: list[Any], *, fallback: str) -> str:
        names: list[str] = []
        for item in entities[:8]:
            if not isinstance(item, dict):
                continue
            n = str(item.get("name", "")).strip()
            if not n:
                continue
            safe = "_".join(part for part in n.replace("/", " ").split() if part)
            if safe:
                names.append(safe[:32])
        if not names:
            return fallback
        return " ".join(names) + " KNOWLEDGE_COMPOSE"

