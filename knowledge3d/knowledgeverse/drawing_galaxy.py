"""Knowledgeverse-integrated Drawing Galaxy wrapper."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

from knowledge3d.training.arc_agi.drawing_galaxy import (
    DrawingGalaxy as LegacyDrawingGalaxy,
)
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingItem
from .foundational_drawing_bootstrap import default_foundational_drawing_entries

_legacy_drawing_module = importlib.import_module("drawing_galaxy")
TRANSFORMATION_RULES = getattr(_legacy_drawing_module, "TRANSFORMATION_RULES", {})


class DrawingGalaxy(LegacyDrawingGalaxy):
    """Wrap legacy DrawingGalaxy with Knowledgeverse-compatible interfaces."""

    def __init__(self, knowledgeverse: Any = None):
        self.knowledgeverse = knowledgeverse
        self.name = "Drawing"
        self._extra_entries: list[dict[str, Any]] = []
        self._extra_entry_ids: set[str] = set()
        super().__init__()
        self._bootstrap_foundational_drawing_knowledge()

    @property
    def entries(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in self.primitives.values():
            items.append({"type": "primitive", "id": item.item_id, "payload": item.payload})
        for item in self.strokes.values():
            items.append({"type": "stroke", "id": item.item_id, "payload": item.payload})
        for item in self.shapes.values():
            items.append({"type": "shape", "id": item.item_id, "payload": item.payload})
        for item in self.scenes.values():
            items.append({"type": "scene", "id": item.item_id, "payload": item.payload})
        for item in self.collections.values():
            items.append({"type": "collection", "id": item.item_id, "payload": item.payload})
        for rule_id, rpn_program in self.transformations.items():
            items.append({"type": "transformation", "id": rule_id, "rpn_program": rpn_program})
        items.extend(self._extra_entries)
        return items

    def add_entry(self, entry: dict[str, Any], *, record_event: bool = True) -> None:
        """Add a generic manager entry into this specialized galaxy."""
        entry_type = str(entry.get("type", "")).lower()
        if entry_type == "shape":
            payload = dict(entry.get("payload", {}))
            shape_id = str(entry.get("id") or payload.get("id") or f"shape_{len(self.shapes)}")
            if "procedural_programs" in payload:
                rpn_program = str(
                    payload.get("procedural_programs", {}).get("composition", "NOP")
                )
                self.add_shape(shape_id, rpn_program, source=payload.get("discovered_from"))
            else:
                payload["id"] = shape_id
                self.shapes[shape_id] = DrawingItem(shape_id, "shape", payload)
                self._rebuild_shape_indices()
        elif entry_type == "transformation" and "id" in entry and "rpn_program" in entry:
            self.transformations[str(entry["id"])] = str(entry["rpn_program"])
        else:
            normalized = dict(entry)
            entry_id = str(normalized.get("id", ""))
            if entry_id and entry_id in self._extra_entry_ids:
                return
            self._extra_entries.append(normalized)
            if entry_id:
                self._extra_entry_ids.add(entry_id)

        if record_event:
            self._log_discovery_event(
                event_type="drawing_discovery",
                payload={"entry_type": entry_type or "generic", "entry_id": entry.get("id", "")},
            )

    def _bootstrap_foundational_drawing_knowledge(self) -> None:
        """Load foundational procedural drawing primitives once per instance."""
        loaded = 0
        for entry in default_foundational_drawing_entries():
            before = len(self._extra_entries)
            self.add_entry(entry, record_event=False)
            if len(self._extra_entries) > before:
                loaded += 1
        for entry in self._load_optional_enrichment_entries():
            before = len(self._extra_entries)
            self.add_entry(entry, record_event=False)
            if len(self._extra_entries) > before:
                loaded += 1
        if loaded:
            self._log_discovery_event(
                event_type="drawing_foundational_bootstrap",
                payload={
                    "loaded_entries": loaded,
                    "specialist": "visual",
                    "galaxy": "Drawing",
                    "confidence": 1.0,
                    "verification": "default_bootstrap_loaded",
                },
            )

    def _load_optional_enrichment_entries(self) -> list[dict[str, Any]]:
        """Load LLM-enriched drawing entries from local ingestion artifacts.

        This is ingestion-time only and never required for hot-path execution.
        """
        base = Path("../Knowledge3D.local/datasets/foundational_drawing_sources")
        default_paths = [
            base / "drawing_enrichment_curated.jsonl",
            base / "vision_enrichment_merged.jsonl",
            base / "vision_enrichment_llama32v_crossmodal.jsonl",
            base / "ollama_enrichment.jsonl",
        ]
        explicit = os.environ.get("K3D_DRAWING_ENRICHED_JSONL")
        enabled = os.environ.get("K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT", "0") == "1"
        if explicit:
            paths = [Path(p.strip()) for p in explicit.split(",") if p.strip()]
        elif enabled:
            paths = default_paths
        else:
            return []
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if "id" not in row or "rpn_program" not in row:
                    continue
                row_id = str(row.get("id", ""))
                if not row_id or row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                row.setdefault("type", "foundational_primitive")
                row.setdefault("domain", "drawing")
                row.setdefault("category", "llm_enrichment")
                metadata = row.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata.setdefault("source", "ollama_enrichment")
                    metadata.setdefault("symlink", "drawing_galaxy")
                out.append(row)
        return out

    def add_shape(
        self,
        shape_id: str,
        rpn_program: str,
        source: dict[str, Any] | None = None,
    ) -> None:
        super().add_shape(shape_id=shape_id, rpn_program=rpn_program, source=source)
        self._log_discovery_event(
            event_type="drawing_discovery",
            payload={
                "shape_id": shape_id,
                "rpn_program": rpn_program,
                "source": source or {},
                "specialist": "visual",
                "galaxy": "Drawing",
                "confidence": 0.85,
                "verification": "shape_registered",
            },
        )

    def summary(self) -> dict[str, int]:
        data = super().summary()
        data["transformations"] = len(self.transformations)
        data["extra_entries"] = len(self._extra_entries)
        return data

    def _log_discovery_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.knowledgeverse is None:
            return
        try:
            self.knowledgeverse.log_event(event_type=event_type, event_data=payload)
        except Exception:
            # Discovery logging should not break core galaxy operations.
            return


__all__ = ["DrawingGalaxy", "DrawingItem", "TRANSFORMATION_RULES"]
