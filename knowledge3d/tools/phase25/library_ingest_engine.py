"""Phase 25 library ingestion utilities.

This module brings the Architect's curated JSON libraries into the
Galaxy working directory by generating fused star descriptors from the
source texts. It depends on the Phase 18 ``MeaningClusterTrainer`` for
embedding generation so the new stars integrate with the existing
multimodal training toolchain.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

try:  # Lazy import with fallback inside ``trainer`` property.
    from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore
except Exception:  # pragma: no cover
    MeaningClusterTrainer = None  # type: ignore


class LibraryIngestEngine:
    """Ingest curated JSON corpora into Galaxy "stars"."""

    DEFAULT_BASE_PATH = (
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries"
    )

    def __init__(self, base_path: str = DEFAULT_BASE_PATH) -> None:
        self.base_path = Path(base_path)
        self.galaxy_working_dir = Path("viewer/public/galaxy/working")
        self.galaxy_working_dir.mkdir(parents=True, exist_ok=True)
        self._trainer: MeaningClusterTrainer | None = None

    @property
    def trainer(self) -> "MeaningClusterTrainer":
        """Instantiate the MeaningClusterTrainer on first access."""
        if self._trainer is None:
            if MeaningClusterTrainer is None:  # pragma: no cover - defensive
                raise ImportError(
                    "MeaningClusterTrainer is unavailable; ensure phase18 tooling is installed."
                )
            self._trainer = MeaningClusterTrainer()
        return self._trainer

    def ingest_library(self) -> None:
        """Ingest the Architect's curated categories."""
        print("📚 Ingesting Architect's Library — Algorithmic Soul Installation...")
        categories = [
            ("Advanced Maths", "JSON", "advanced_math", "Zone 1 (Entrance)"),
            ("How to think", "JSON", "algorithmic_thinking", "Zone 2 (Study)"),
            ("Self Reflection", "JSON", "self_reflection", "Zone 7 (Mirror Room)"),
        ]
        for category_name, leaf, tag, zone in categories:
            category_path = self.base_path / category_name / leaf
            self.ingest_category(category_path, tag, zone)

    def ingest_category(self, path: Path, category: str, zone: str) -> None:
        """Ingest all JSON files under ``path`` for a given category."""
        if not path.exists():
            print(f"⚠️  Path not found: {path}")
            return

        for filepath in sorted(path.glob("*.json")):
            try:
                text = self._load_text(filepath)
                if not text.strip():
                    print(f"⚠️  No textual content found in {filepath}")
                    continue

                embedding = self.trainer.generate_text_embedding(text)
                star_id = self._build_star_id(category, filepath.stem)
                star_data = self._build_star_payload(
                    star_id=star_id,
                    name=f"{category}: {filepath.stem}",
                    embedding=embedding,
                    zone=zone,
                    category=category,
                    source=filepath,
                    source_text=text,
                )

                star_path = self.galaxy_working_dir / f"{star_id}.json"
                with star_path.open("w", encoding="utf-8") as handle:
                    json.dump(star_data, handle, ensure_ascii=False, indent=2)

                print(f"🌟 Ingested: {star_id} → {star_path}")
            except Exception as exc:
                print(f"⚠️  Failed to ingest {filepath}: {exc}")

    def extract_text(self, data: Any) -> str:
        """Extract textual content from arbitrary JSON payloads."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, (int, float, bool)):
            return str(data)
        if isinstance(data, list):
            return " ".join(filter(None, (self.extract_text(item) for item in data)))
        if isinstance(data, dict):
            return " ".join(
                filter(None, (self.extract_text(value) for value in data.values()))
            )
        return str(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_text(self, filepath: Path) -> str:
        with filepath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self.extract_text(data)

    def _build_star_id(self, category: str, stem: str) -> str:
        sanitized = self._sanitize_identifier(stem)
        return f"star_{category}_{sanitized}"

    def _sanitize_identifier(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
        return cleaned.strip("_") or "item"

    def _build_star_payload(
        self,
        *,
        star_id: str,
        name: str,
        embedding: Iterable[Union[int, float]],
        zone: str,
        category: str,
        source: Path,
        source_text: str,
    ) -> Dict[str, Any]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        text_preview = source_text.strip()
        if len(text_preview) > 4000:
            text_preview = text_preview[:4000]
        return {
            "type": "star",
            "id": star_id,
            "name": name,
            "created_at": timestamp,
            "honesty_score": 1.0,
            "embedding": list(embedding),
            "modality_fusion": ["text"],
            "zone_placement": zone,
            "source_file": str(source),
            "source_text": text_preview,
            "ptx_kernel": f"generate_{category}_kernel",
            "tags": [category, "architect_curated", "algorithmic_soul"],
        }
