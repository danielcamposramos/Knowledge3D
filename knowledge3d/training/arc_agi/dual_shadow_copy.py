"""
DualShadowCopy: records successful discoveries across Drawing + Grammar galaxies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, GrammarRule
from knowledge3d.training.arc_agi.content_deduplicator import ContentDeduplicator
from knowledge3d.training.arc_agi.quality_scorer import QualityScorer
from knowledge3d.training.arc_agi.semantic_context import SemanticContext


class DualShadowCopy:
    """Store successes and commit to both galaxies when appropriate."""

    def __init__(self, drawing_galaxy: DrawingGalaxy, grammar_galaxy: GrammarGalaxy, *, staged: bool = False):
        self.drawing = drawing_galaxy
        self.grammar = grammar_galaxy
        self.library: List[Dict] = []
        self.staged = staged
        self._pending: List[Dict] = []
        self.deduplicator = ContentDeduplicator()
        self.quality_scorer = QualityScorer()
        self.semantic_context = SemanticContext()
        # Pattern/task confidence tracking for calibration.
        self._pattern_confidence: Dict[str, float] = {}
        self._pattern_counts: Dict[str, int] = {}
        self._task_history: Dict[str, Dict[str, float]] = {}

    def record(
        self,
        task_signature: Dict,
        program: str,
        program_type: str,
        score: float,
        *,
        input_grid=None,
        output_grid=None,
        task_id: str = "",
    ) -> None:
        """
        Record discovery with deduplication + quality filtering.

        - Compute quality score; skip if below threshold.
        - Deduplicate by content hash; only persist first occurrence.
        """
        quality_score = self.quality_scorer.score_quality(program, score)
        if quality_score < 0.45:
            return

        prog_hash, is_new = self.deduplicator.add_or_reference(
            program=program,
            program_type=program_type,
            score=quality_score,
            context=task_signature,
        )

        if not is_new:
            return

        context = None
        if input_grid is not None and output_grid is not None:
            context = self.semantic_context.record_context(
                program, input_grid, output_grid, task_id or str(task_signature), quality_score
            )

        complexity = self.quality_scorer.get_complexity_level(program)
        entry = {
            "hash": prog_hash,
            "task_signature": dict(task_signature),
            "program": program,
            "program_type": program_type,
            "quality_score": float(quality_score),
            "complexity": complexity,
            "semantic_context": context or {},
        }

        if self.staged:
            self.library.append(entry)
            self._pending.append(entry)
        else:
            self.library.append(entry)
            self._commit_entry(entry)

    def _commit_entry(self, entry: Dict) -> None:
        program_type = entry["program_type"]
        program = entry["program"]
        signature = entry["task_signature"]

        if program_type == "visual":
            shape_id = f"DISCOVERED_SHAPE_{len(self.drawing.shapes)}"
            self.drawing.add_shape(shape_id, program, source=signature)
        elif program_type == "transformation":
            rule_id = f"DISCOVERED_RULE_{len(self.grammar.rules)}"
            self.grammar.rules[rule_id] = GrammarRule(
                rule_id=rule_id,
                language="drawing",
                domain="drawing",
                pattern="discovered",
                rpn_program=program,
                examples=[signature],
                description="Discovered transformation from ARC task",
                semantics=entry.get("semantic_context", {}),
                usage_conditions=entry.get("semantic_context", {}).get("when_to_use", []),
            )
        else:  # hybrid
            shape_id = f"DISCOVERED_SHAPE_{len(self.drawing.shapes)}"
            self.drawing.add_shape(shape_id, program, source=signature)
            rule_id = f"DISCOVERED_RULE_{len(self.grammar.rules)}"
            self.grammar.rules[rule_id] = GrammarRule(
                rule_id=rule_id,
                language="drawing",
                domain="drawing",
                pattern="discovered_hybrid",
                rpn_program=program,
                examples=[signature],
                description="Discovered hybrid program from ARC task",
                semantics=entry.get("semantic_context", {}),
                usage_conditions=entry.get("semantic_context", {}).get("when_to_use", []),
            )

    def commit_pending(self) -> None:
        """Commit staged entries to galaxies."""
        for entry in self._pending:
            self._commit_entry(entry)
        self._pending.clear()

    def prune_discovered(self, executor) -> Dict[str, int]:
        """
        Prune discovered programs that fail execution on a small grid.

        Args:
            executor: ARCRPNExecutor-like with execute(grid, program) -> grid
        """
        removed_rules = 0
        removed_shapes = 0
        test_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for rule_id in list(self.grammar.rules.keys()):
            if not rule_id.startswith("DISCOVERED_RULE_"):
                continue
            rule = self.grammar.rules[rule_id]
            try:
                executor.execute(test_grid, rule.rpn_program)
            except Exception:
                del self.grammar.rules[rule_id]
                removed_rules += 1

        for shape_id in list(self.drawing.shapes.keys()):
            if not shape_id.startswith("DISCOVERED_SHAPE_"):
                continue
            shape = self.drawing.shapes[shape_id]
            payload = getattr(shape, "payload", {}) if hasattr(shape, "payload") else shape
            program = payload.get("procedural_programs", {}).get("composition") or payload.get("rpn_program")
            if not program:
                del self.drawing.shapes[shape_id]
                removed_shapes += 1
                continue
            try:
                executor.execute(test_grid, program)
            except Exception:
                del self.drawing.shapes[shape_id]
                removed_shapes += 1

        return {"removed_rules": removed_rules, "removed_shapes": removed_shapes}

    def prune_low_quality(self) -> Dict[str, int]:
        """Remove low-quality duplicates and clean library/pending lists."""
        removed_programs = self.deduplicator.prune_low_quality(min_usage=2, min_score=0.45)
        valid = set(self.deduplicator.canonical_programs.keys())
        before_lib = len(self.library)
        before_pending = len(self._pending)
        self.library = [e for e in self.library if e.get("hash") in valid]
        self._pending = [e for e in self._pending if e.get("hash") in valid]
        return {
            "removed_programs": removed_programs,
            "removed_from_library": before_lib - len(self.library),
            "removed_from_pending": before_pending - len(self._pending),
            "unique_remaining": len(valid),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        state = {
            "library": self.library,
            "pending": self._pending,
            "total_entries": len(self.library),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[DualShadowCopy] Saved {len(self.library)} entries to {path}")
        dedupe_path = path.parent / "deduplication_index.json"
        self.deduplicator.save(dedupe_path)
        semantic_path = path.parent / "semantic_context.json"
        self.semantic_context.save(semantic_path)

    def load(self, path: Path) -> None:
        if not path.exists():
            print(f"[DualShadowCopy] No checkpoint at {path}, starting fresh")
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        self.library = state.get("library", [])
        self._pending = state.get("pending", [])
        print(f"[DualShadowCopy] Loaded {len(self.library)} shadow entries from {path}")
        dedupe_path = path.parent / "deduplication_index.json"
        self.deduplicator.load(dedupe_path)
        semantic_path = path.parent / "semantic_context.json"
        self.semantic_context.load(semantic_path)

    def summary(self) -> Dict[str, int]:
        return {
            "entries": len(self.library),
            "drawing_shapes": len(self.drawing.shapes),
            "grammar_rules": len(self.grammar.rules),
            "pending": len(self._pending),
        }

    # ------------------------------------------------------------------ #
    # Pattern/task confidence helpers
    # ------------------------------------------------------------------ #
    def get_pattern_success_rate(self, pattern_id: str) -> float | None:
        return self._pattern_confidence.get(pattern_id)

    def update_pattern_confidence(self, pattern_id: str, confidence: float) -> None:
        prev = self._pattern_confidence.get(pattern_id, 0.5)
        count = self._pattern_counts.get(pattern_id, 0)
        new_conf = (prev * count + confidence) / (count + 1)
        self._pattern_confidence[pattern_id] = new_conf
        self._pattern_counts[pattern_id] = count + 1

    def get_task_history(self, task_id: str) -> Dict[str, float] | None:
        return self._task_history.get(task_id)

    def update_task_history(self, task_id: str, success: bool) -> None:
        hist = self._task_history.get(task_id, {"success": 0, "total": 0})
        hist["total"] += 1
        if success:
            hist["success"] += 1
        hist["success_rate"] = hist["success"] / max(1, hist["total"])
        self._task_history[task_id] = hist


__all__ = ["DualShadowCopy"]
