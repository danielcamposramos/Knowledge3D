"""
DualShadowCopy: records successful discoveries across Drawing + Grammar galaxies.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, GrammarRule
from knowledge3d.training.arc_agi.content_deduplicator import ContentDeduplicator
from knowledge3d.training.arc_agi.quality_scorer import QualityScorer
from knowledge3d.training.arc_agi.semantic_context import SemanticContext
from knowledge3d.training.arc_agi.pattern_quality import compute_pattern_quality_opcode_aware, _infer_tier_from_tokens


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
        # Phase 5: exploration traces (kept separate from program library).
        self.explorations: List[Dict] = []

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
        semantic_context: Dict | None = None,
    ) -> None:
        """
        Record discovery with deduplication + quality filtering.

        - Compute quality score; skip if below threshold.
        - Deduplicate by content hash; only persist first occurrence.
        """
        quality_score = self.quality_scorer.score_quality(program, score)
        if quality_score < 0.45:
            return

        # Composition learning needs dedup keyed by (program + template + patterns),
        # otherwise identical RPN strings collapse and TRM can't learn mapping.
        dedupe_key = ""
        if program_type in {"composition"}:
            ctx = semantic_context or {}
            template_used = ctx.get("template_used")
            patterns = ctx.get("patterns_matched")
            struct = ctx.get("structure")
            if isinstance(template_used, str) and template_used and isinstance(patterns, list):
                pats = sorted({str(p) for p in patterns if p})
                struct_sig = ""
                if isinstance(struct, dict) and struct:
                    # Keep this stable and compact (no whitespace / ordering variance).
                    keys = [
                        "n_quantities",
                        "n_operations",
                        "aggregation",
                        "multi_step_indicator",
                        "has_rate",
                        "has_duration",
                    ]
                    struct_sig = ",".join(f"{k}={struct.get(k)}" for k in keys if k in struct)
                dedupe_key = f"composition:{template_used}:{','.join(pats)}:{struct_sig}"

        prog_hash, is_new = self.deduplicator.add_or_reference(
            program=program,
            program_type=program_type,
            score=quality_score,
            context=task_signature,
            dedupe_key=dedupe_key,
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
            "semantic_context": semantic_context or context or {},
        }
        # Cache opcode-aware metrics up front (outside hot path)
        entry["quality_score_opcode"] = compute_pattern_quality_opcode_aware(entry, [])
        entry["tier"] = _infer_tier_from_tokens(program.split())

        if self.staged:
            self.library.append(entry)
            self._pending.append(entry)
        else:
            self.library.append(entry)
            self._commit_entry(entry)
        # Maintain descending order by cached quality (keeps refiner hot path simple)
        self.library.sort(key=lambda e: e.get("quality_score_opcode", e.get("quality_score", 0.0)), reverse=True)

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
        elif program_type == "math":
            rule_id = f"DISCOVERED_MATH_RULE_{len(self.grammar.rules)}"
            self.grammar.rules[rule_id] = GrammarRule(
                rule_id=rule_id,
                language="math",
                domain=entry.get("domain", signature.get("problem_type", "math_general")),
                pattern=entry.get("pattern", "discovered_math"),
                rpn_program=program,
                examples=[signature],
                description=f"Discovered math rule: {signature.get('problem_type', 'unknown')}",
                semantics=entry.get("semantic_context", {}),
            )
        elif program_type == "reading":
            rule_id = f"DISCOVERED_READING_RULE_{len(self.grammar.rules)}"
            self.grammar.rules[rule_id] = GrammarRule(
                rule_id=rule_id,
                language="math",
                domain="math_reading",
                pattern=entry.get("pattern", "discovered_reading"),
                rpn_program=program,
                examples=[signature],
                description="Discovered reading program (Galaxy-based token sequence)",
                semantics=entry.get("semantic_context", {}),
            )
        elif program_type == "composition":
            rule_id = f"DISCOVERED_COMPOSITION_RULE_{len(self.grammar.rules)}"
            self.grammar.rules[rule_id] = GrammarRule(
                rule_id=rule_id,
                language="math",
                domain="math_composition",
                pattern=entry.get("pattern", "discovered_composition"),
                rpn_program=program,
                examples=[signature],
                description="Discovered composition strategy (template selection)",
                semantics=entry.get("semantic_context", {}),
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

    def query_by_patterns(self, patterns: frozenset[str]) -> Dict | None:
        """
        Retrieve best matching composition entry by pattern overlap.

        Returns the entry dict or None.
        """
        entry, _score = self.query_by_patterns_scored(patterns)
        return entry

    @staticmethod
    def _structure_matches(entry_struct: Dict[str, object] | None, query_struct: Dict[str, object] | None) -> bool:
        """
        Backwards-compatible strict structure match.

        The math benchmark reader now prefers fuzzy structure similarity via
        `_structure_similarity`, but we keep this helper because other code
        may rely on strict agreement semantics.
        """
        if query_struct is None:
            return True
        if not isinstance(entry_struct, dict):
            return False
        keys = [
            "n_quantities",
            "n_operations",
            "aggregation",
            "multi_step_indicator",
            "has_rate",
            "has_duration",
        ]
        for k in keys:
            if k in query_struct and entry_struct.get(k) != query_struct.get(k):
                return False
        return True

    @staticmethod
    def _structure_similarity(entry_struct: Dict[str, object] | None, query_struct: Dict[str, object] | None) -> float:
        """
        Fuzzy structure similarity for composition retrieval.

        Strict equality was too conservative: many correct retrieves share the
        same pattern set but differ slightly in counts (e.g. an extra extracted
        number) or flags (multi-step cues). We score similarity in [0, 1] and
        fold it into the retrieval score, instead of hard-rejecting entries.
        """
        if query_struct is None:
            return 1.0
        if not isinstance(entry_struct, dict):
            return 0.0

        def _get_int(d: Dict[str, object], key: str) -> int | None:
            v = d.get(key)
            if isinstance(v, bool) or v is None:
                return None
            try:
                return int(v)  # type: ignore[arg-type]
            except Exception:
                return None

        def _get_bool(d: Dict[str, object], key: str) -> bool | None:
            v = d.get(key)
            if v is None:
                return None
            return bool(v)

        def _num_sim(a: int | None, b: int | None) -> float:
            if a is None or b is None:
                return 0.65
            delta = abs(int(a) - int(b))
            if delta == 0:
                return 1.0
            if delta == 1:
                return 0.85
            if delta == 2:
                return 0.65
            if delta == 3:
                return 0.45
            return 0.25

        def _bool_sim(a: bool | None, b: bool | None) -> float:
            if a is None or b is None:
                return 0.65
            return 1.0 if bool(a) == bool(b) else 0.35

        def _str_sim(a: object, b: object) -> float:
            a_s = "" if a is None else str(a)
            b_s = "" if b is None else str(b)
            if not a_s or not b_s:
                return 0.75
            return 1.0 if a_s == b_s else 0.25

        parts: list[tuple[float, float]] = []
        # Counts matter most, but allow small drift.
        parts.append((2.0, _num_sim(_get_int(entry_struct, "n_quantities"), _get_int(query_struct, "n_quantities"))))
        parts.append((2.0, _num_sim(_get_int(entry_struct, "n_operations"), _get_int(query_struct, "n_operations"))))
        # Aggregation is informative but often implicit/missing.
        parts.append((1.0, _str_sim(entry_struct.get("aggregation"), query_struct.get("aggregation"))))
        # Flags help disambiguate multi-step vs single-step.
        parts.append((1.5, _bool_sim(_get_bool(entry_struct, "multi_step_indicator"), _get_bool(query_struct, "multi_step_indicator"))))
        parts.append((1.0, _bool_sim(_get_bool(entry_struct, "has_rate"), _get_bool(query_struct, "has_rate"))))
        parts.append((1.0, _bool_sim(_get_bool(entry_struct, "has_duration"), _get_bool(query_struct, "has_duration"))))
        # These fields are noisy, but help in edge cases.
        parts.append((0.5, _bool_sim(_get_bool(entry_struct, "has_labels"), _get_bool(query_struct, "has_labels"))))
        parts.append((0.5, _bool_sim(_get_bool(entry_struct, "has_goals"), _get_bool(query_struct, "has_goals"))))

        denom = sum(w for w, _ in parts) or 1.0
        num = sum(w * s for w, s in parts)
        return max(0.0, min(1.0, float(num / denom)))

    def query_by_patterns_scored(
        self,
        patterns: frozenset[str],
        *,
        structure: Dict[str, object] | None = None,
    ) -> tuple[Dict | None, float]:
        """
        Retrieve best matching composition entry by pattern overlap, with a score.

        Returns:
            (entry_or_none, score)
        """
        if not patterns:
            return None, 0.0
        best = None
        best_score = 0.0
        for entry in self.library:
            if entry.get("program_type") != "composition":
                continue
            ctx = entry.get("semantic_context", {}) or {}
            pats = ctx.get("patterns_matched")
            if not isinstance(pats, list) or not pats:
                continue
            struct_sim = self._structure_similarity(ctx.get("structure"), structure)
            if float(struct_sim) < 0.30:
                continue
            entry_set = {str(p) for p in pats if p}
            if not entry_set:
                continue
            inter = len(patterns.intersection(entry_set))
            if inter <= 0:
                continue
            union = len(patterns.union(entry_set))
            jaccard = inter / max(1, union)
            q = float(entry.get("quality_score_opcode", entry.get("quality_score", 0.0)) or 0.0)
            score = jaccard * float(struct_sim) * (0.25 + q)
            if score > best_score:
                best_score = score
                best = entry
        return best, float(best_score)

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
        # Refresh opcode-aware quality scores before pruning
        self.recompute_opcode_quality()
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

    def recompute_opcode_quality(self) -> None:
        """Compute opcode-aware quality scores for all patterns."""
        for entry in self.library:
            history = entry.get("execution_history", [])
            entry["quality_score_opcode"] = compute_pattern_quality_opcode_aware(entry, history)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        state = {
            "library": self.library,
            "pending": self._pending,
            "explorations": self.explorations,
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
        self.explorations = state.get("explorations", [])
        print(f"[DualShadowCopy] Loaded {len(self.library)} shadow entries from {path}")
        dedupe_path = path.parent / "deduplication_index.json"
        self.deduplicator.load(dedupe_path)
        semantic_path = path.parent / "semantic_context.json"
        self.semantic_context.load(semantic_path)
        # Re-cache opcode-aware quality/tier once at load (outside hot path)
        for entry in self.library:
            prog = entry.get("program", "") or ""
            entry["quality_score_opcode"] = compute_pattern_quality_opcode_aware(entry, entry.get("execution_history", []))
            entry["tier"] = _infer_tier_from_tokens(prog.split())
        # Keep library sorted by cached quality
        self.library.sort(key=lambda e: e.get("quality_score_opcode", e.get("quality_score", 0.0)), reverse=True)

    def summary(self) -> Dict[str, int]:
        return {
            "entries": len(self.library),
            "drawing_shapes": len(self.drawing.shapes),
            "grammar_rules": len(self.grammar.rules),
            "pending": len(self._pending),
        }

    def get_rule_success_rates(
        self,
        *,
        recent: int = 2000,
        default: float = 0.5,
        prior_weight: int = 2,
    ) -> Dict[str, float]:
        """
        Compute historical success rate per rule from exploration traces.

        Rates are derived from `explorations[*].patterns_matched` with correctness
        provided by the benchmark evaluator (`explorations[*].correct`).

        Args:
            recent: Only consider the last N exploration entries (0 = all).
            default: Default prior rate for unseen rules.
            prior_weight: Pseudo-count weight for the prior (Laplace-style smoothing).
        """
        items: List[Dict[str, Any]] = [e for e in self.explorations if isinstance(e, dict)]
        if recent and recent > 0:
            items = items[-int(recent) :]

        ok: Dict[str, int] = {}
        total: Dict[str, int] = {}
        for e in items:
            pats = e.get("patterns_matched")
            if not isinstance(pats, list) or not pats:
                continue
            correct = e.get("correct")
            if correct is None:
                continue
            is_ok = bool(correct)
            for rid in pats:
                if not rid:
                    continue
                rule_id = str(rid)
                total[rule_id] = int(total.get(rule_id, 0)) + 1
                if is_ok:
                    ok[rule_id] = int(ok.get(rule_id, 0)) + 1

        prior_w = max(0, int(prior_weight))
        prior_s = float(default) * float(prior_w)
        rates: Dict[str, float] = {}
        for rule_id, t in total.items():
            s = int(ok.get(rule_id, 0))
            denom = float(t + prior_w)
            rates[rule_id] = (float(s) + prior_s) / denom if denom > 0 else float(default)
        return rates

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

    # ------------------------------------------------------------------ #
    # Exploration logging (Phase 5)
    # ------------------------------------------------------------------ #
    def record_exploration(
        self,
        *,
        problem_text: str,
        concepts_explored: List[str],
        patterns_matched: List[str],
        templates_tried: List[str],
        template_used: str = "",
        success: bool,
        rpn_program: str = "",
        result: object | None = None,
        reason: str = "",
        tsinghua: Dict[str, object] | None = None,
    ) -> None:
        """
        Record an exploration attempt (separate from the RPN discovery library).

        This is intended for analysis + future policy learning and does not
        participate in deduplication, quality scoring, or galaxy commits.
        """
        problem_hash = int(hash(problem_text or "") % 100000)
        self.explorations.append(
            {
                "ts": float(time.time()),
                "problem_hash": problem_hash,
                "problem_text": (problem_text or "")[:240],
                "concepts": list(concepts_explored),
                "patterns_matched": list(patterns_matched),
                "templates_tried": list(templates_tried),
                "template_used": str(template_used or "")[:64],
                "success": bool(success),
                "rpn_program": str(rpn_program or "")[:240],
                "result": None if result is None else str(result)[:64],
                "reason": str(reason or "")[:120],
                "tsinghua": tsinghua or {},
                # Filled later by the benchmark runner (ground-truth aware).
                "expected_num": None,
                "got_num": None,
                "correct": None,
            }
        )
        # Bound growth to keep checkpoints small (training logs belong in session JSONL).
        max_items = 2000
        if len(self.explorations) > max_items:
            self.explorations = self.explorations[-max_items:]

    def annotate_exploration_eval(
        self,
        *,
        problem_text: str,
        expected_num: float | None,
        got_num: float | None,
        correct: bool | None,
    ) -> None:
        """
        Attach benchmark evaluation to the most recent exploration entry.

        This keeps exploration generation (reader) decoupled from ground-truth
        evaluation (runner) while still enabling error analysis.
        """
        problem_hash = int(hash(problem_text or "") % 100000)
        for entry in reversed(self.explorations):
            if entry.get("problem_hash") == problem_hash:
                entry["expected_num"] = expected_num
                entry["got_num"] = got_num
                entry["correct"] = correct
                return


__all__ = ["DualShadowCopy"]
