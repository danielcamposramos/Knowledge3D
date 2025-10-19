"""
Glyph consolidation for sleep-time refinement.

The glyph harvester gathered ~124K character instances across ~2K fonts. Many
of those glyphs are visually indistinguishable (e.g. Helvetica vs Arial "A").
This consolidator merges near-identical glyph embeddings so the OCR fallback
works with a leaner database.

The current implementation uses a CPU-based greedy clustering routine per
character class:

1. Group glyphs by character (e.g. all "A" glyphs together).
2. Sort by confidence and font name so the most reliable glyph becomes the
   cluster representative.
3. For each glyph, compare against existing cluster representatives using
   cosine similarity. If the similarity exceeds the threshold (default 0.98),
   the glyph joins that cluster; otherwise a new cluster is created.
4. Only the representative glyph for each cluster is retained in the final
   database; duplicates are dropped.

The consolidator writes a backup of the original database before saving the
reduced version. Run inside the CPU or GPU consolidation environment described
in `envs/k3d-cranium.yml`.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

DEFAULT_FONT_DB_PATH = Path("/K3D/Knowledge3D.local/font_db.pkl")


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


@dataclass
class GlyphConsolidationResult:
    glyphs_before: int
    glyphs_after: int
    reduction_pct: float
    clusters_per_char: Dict[str, int]
    backup_path: Path

    def to_dict(self) -> Dict[str, object]:
        return {
            "glyphs_before": self.glyphs_before,
            "glyphs_after": self.glyphs_after,
            "reduction_pct": self.reduction_pct,
            "clusters_per_char": self.clusters_per_char,
            "backup_path": str(self.backup_path),
        }


class GlyphConsolidator:
    """Consolidate glyphs in the harvested font database."""

    def __init__(self, font_db_path: Path | str = DEFAULT_FONT_DB_PATH):
        self.font_db_path = Path(font_db_path)
        if not self.font_db_path.exists():
            raise FileNotFoundError(f"Font database not found: {self.font_db_path}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def consolidate(
        self,
        similarity_threshold: float = 0.98,
        min_retention_ratio: float = 0.5,
        metrics_path: Path | None = None,
    ) -> GlyphConsolidationResult:
        """
        Run greedy clustering per character class and persist the reduced DB.

        Parameters
        ----------
        similarity_threshold:
            Cosine similarity threshold for considering two glyphs identical.
            0.98 → very strict (only tiny visual differences permitted).
        metrics_path:
            Optional JSONL file to append consolidation stats.
        """
        font_db = self._load_database()
        total_glyphs_before = self._count_glyphs(font_db)

        char_groups = self._group_by_character(font_db)
        clusters_per_char: Dict[str, int] = {}
        consolidated_fonts = {
            font_name: {
                key: copy.deepcopy(value)
                for key, value in font_data.items()
                if key != "glyphs"
            }
            for font_name, font_data in font_db.items()
        }
        for font_data in consolidated_fonts.values():
            font_data["glyphs"] = {}

        for char, entries in char_groups.items():
            clusters, retained = self._cluster_char_entries(
                entries,
                similarity_threshold=similarity_threshold,
                min_retention_ratio=min_retention_ratio,
            )
            clusters_per_char[char] = len(clusters)

            for rep_entry in retained:
                font_name = rep_entry["font"]
                if font_name not in consolidated_fonts:
                    consolidated_fonts[font_name] = {
                        "font_path": rep_entry.get("font_path"),
                        "glyphs": {},
                        "is_symbol_font": rep_entry.get("is_symbol_font", False),
                    }
                payload = copy.deepcopy(rep_entry["glyph_data"])
                consolidated_fonts[font_name]["glyphs"][char] = payload

        # Drop fonts with no surviving glyphs
        consolidated_fonts = {
            font_name: font_data
            for font_name, font_data in consolidated_fonts.items()
            if font_data.get("glyphs")
        }

        total_glyphs_after = self._count_glyphs(consolidated_fonts)
        reduction_pct = (
            ((total_glyphs_before - total_glyphs_after) / max(total_glyphs_before, 1)) * 100.0
        )

        backup_path = self._write_backup()
        self._save_database(consolidated_fonts)

        result = GlyphConsolidationResult(
            glyphs_before=total_glyphs_before,
            glyphs_after=total_glyphs_after,
            reduction_pct=reduction_pct,
            clusters_per_char=clusters_per_char,
            backup_path=backup_path,
        )

        if metrics_path is not None:
            self._append_metrics(metrics_path, result)

        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_database(self) -> Dict[str, Dict]:
        import pickle

        with self.font_db_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Malformed glyph database at {self.font_db_path}")
        return payload

    def _save_database(self, payload: Dict[str, Dict]) -> None:
        import pickle

        with self.font_db_path.open("wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def _count_glyphs(font_db: Dict[str, Dict]) -> int:
        return sum(len(font.get("glyphs", {})) for font in font_db.values())

    def _group_by_character(
        self,
        font_db: Dict[str, Dict],
    ) -> Dict[str, List[Dict[str, object]]]:
        groups: Dict[str, List[Dict[str, object]]] = {}
        for font_name, font_data in font_db.items():
            glyphs = font_data.get("glyphs", {})
            for char, glyph_data in glyphs.items():
                visual = glyph_data.get("visual_features")
                if visual is None:
                    continue
                vector = _normalize(np.asarray(visual, dtype=np.float32))
                entry = {
                    "font": font_name,
                    "font_path": font_data.get("font_path"),
                    "is_symbol_font": font_data.get("is_symbol_font", False),
                    "glyph_data": glyph_data,
                    "vector": vector,
                    "confidence": float(glyph_data.get("confidence", 1.0)),
                }
                groups.setdefault(char, []).append(entry)

        for char_entries in groups.values():
            char_entries.sort(
                key=lambda e: (-e["confidence"], e["font"])
            )
        return groups

    @staticmethod
    def _cluster_char_entries(
        entries: List[Dict[str, object]],
        similarity_threshold: float,
        min_retention_ratio: float,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        clusters: List[Dict[str, object]] = []
        retained: List[Dict[str, object]] = []
        min_cluster_target = max(1, math.ceil(len(entries) * min_retention_ratio))

        for entry in entries:
            vector = entry["vector"]
            assigned = False
            if len(clusters) >= min_cluster_target:
                for cluster in clusters:
                    similarity = float(np.dot(vector, cluster["rep_vector"]))
                    if similarity >= similarity_threshold:
                        cluster["members"].append(entry)
                        assigned = True
                        break

            if not assigned:
                cluster = {
                    "rep": entry,
                    "rep_vector": vector,
                    "members": [entry],
                }
                clusters.append(cluster)
                retained.append(entry)

        return clusters, retained

    def _write_backup(self) -> Path:
        backup_path = self.font_db_path.with_name(
            f"{self.font_db_path.stem}_pre_consolidation{self.font_db_path.suffix}"
        )
        shutil.copy2(self.font_db_path, backup_path)
        return backup_path

    @staticmethod
    def _append_metrics(metrics_path: Path, result: GlyphConsolidationResult) -> None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result.to_dict(), default=float)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
