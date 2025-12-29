"""SleepTime consolidation utilities for ARC-AGI vocabulary."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, GrammarRule


class SleepTimeConsolidator:
    """Consolidates learned patterns after training runs."""

    def __init__(
        self,
        shadow_copy: DualShadowCopy,
        drawing_galaxy: DrawingGalaxy,
        grammar_galaxy: GrammarGalaxy,
        *,
        min_quality: float = 0.6,
        min_uses_for_canonical: int = 5,
        canonical_success_threshold: float = 0.7,
    ) -> None:
        self.shadow = shadow_copy
        self.drawing = drawing_galaxy
        self.grammar = grammar_galaxy
        self.min_quality = min_quality
        self.min_uses_canonical = min_uses_for_canonical
        self.canonical_threshold = canonical_success_threshold
        self._pruned_audit: List[Dict] = []

    def consolidate(self) -> Dict:
        stats: Dict[str, object] = {}
        stats["pruned_count"] = self._prune_low_quality()
        stats["rule_stats"] = self._analyze_grammar_rules()
        stats["shape_stats"] = self._analyze_drawing_shapes()
        canonical = self._promote_canonical_patterns(stats["rule_stats"])
        stats["canonical_promoted"] = len(canonical)
        stats["top_canonical_candidates"] = canonical
        stats["total_rules"] = len(self.grammar.rules)
        stats["total_shapes"] = len(self.drawing.shapes)
        if self._pruned_audit:
            stats["pruned_entries_audit"] = self._pruned_audit
        stats["rule_stats_detail"] = self._serialize_stat_block(stats["rule_stats"], kind="rule")
        stats["shape_stats_detail"] = self._serialize_stat_block(stats["shape_stats"], kind="shape")
        self._log_consolidation_report(stats)
        return stats

    # ------------------------------------------------------------------ #
    # Consolidation helpers
    # ------------------------------------------------------------------ #
    def _prune_low_quality(self) -> int:
        removed_hashes = set()
        kept_entries: List[Dict] = []
        pruned_entries: List[Dict] = []
        for entry in self.shadow.library:
            score = float(entry.get("quality_score", 0.0))
            if score < self.min_quality:
                prog_hash = entry.get("hash")
                if prog_hash:
                    removed_hashes.add(prog_hash)
                pruned_entries.append(
                    {
                        "hash": entry.get("hash", "unknown"),
                        "quality_score": score,
                        "program_type": entry.get("program_type", "unknown"),
                        "program": entry.get("program", "")[:200],
                    }
                )
                continue
            kept_entries.append(entry)

        pruned = len(self.shadow.library) - len(kept_entries)
        self._pruned_audit = pruned_entries
        print("\n[SLEEPTIME PRUNING AUDIT]")
        print(f"  Threshold: {self.min_quality:.2f}")
        print(f"  Total entries: {len(self.shadow.library)}")
        print(f"  Pruned: {pruned}")
        print(f"  Kept: {len(kept_entries)}")
        if pruned_entries:
            preview = pruned_entries[:5]
            print("  Sample pruned entries:")
            for item in preview:
                print(
                    f"    hash={item['hash'][:16]}... quality={item['quality_score']:.3f} type={item['program_type']}"
                )
        if not pruned:
            return 0

        self.shadow.library = kept_entries
        if getattr(self.shadow, "_pending", None):
            self.shadow._pending = [e for e in self.shadow._pending if e.get("hash") not in removed_hashes]

        for prog_hash in removed_hashes:
            self.shadow.deduplicator.canonical_programs.pop(prog_hash, None)
            self.shadow.deduplicator.usage_metadata.pop(prog_hash, None)

        return pruned

    def _analyze_grammar_rules(self) -> Dict[str, Dict]:
        rule_lookup = {rule_id.lower(): rule_id for rule_id in self.grammar.rules}
        stats: Dict[str, Dict] = defaultdict(lambda: {
            "uses": 0,
            "successes": 0,
            "total_quality": 0.0,
            "avg_quality": 0.0,
            "success_rate": 0.0,
            "tasks_solved": [],
        })

        for entry in self.shadow.library:
            if entry.get("program_type") != "transformation":
                continue
            program = entry.get("program", "")
            quality = float(entry.get("quality_score", 0.0))
            matched_rules = self._parse_tokens(program, rule_lookup)
            if not matched_rules:
                continue
            for rule_id in matched_rules:
                data = stats[rule_id]
                data["uses"] += 1
                data["total_quality"] += quality
                if quality >= self.canonical_threshold:
                    data["successes"] += 1
                signature = entry.get("task_signature") or {}
                if signature and len(data["tasks_solved"]) < 10:
                    formatted = self._format_task_signature(signature)
                    if formatted not in data["tasks_solved"]:
                        data["tasks_solved"].append(formatted)

        for rule_id, data in list(stats.items()):
            uses = data["uses"]
            if not uses:
                continue
            data["avg_quality"] = data["total_quality"] / uses
            data["success_rate"] = data["successes"] / uses

        return dict(stats)

    def _analyze_drawing_shapes(self) -> Dict[str, Dict]:
        shape_lookup = {shape_id.lower(): shape_id for shape_id in self.drawing.shapes}
        stats: Dict[str, Dict] = defaultdict(lambda: {
            "uses": 0,
            "successes": 0,
            "total_quality": 0.0,
            "avg_quality": 0.0,
            "success_rate": 0.0,
        })

        for entry in self.shadow.library:
            if entry.get("program_type") not in {"visual", "hybrid"}:
                continue
            program = entry.get("program", "")
            quality = float(entry.get("quality_score", 0.0))
            matched_shapes = self._parse_tokens(program, shape_lookup)
            if not matched_shapes:
                continue
            for shape_id in matched_shapes:
                data = stats[shape_id]
                data["uses"] += 1
                data["total_quality"] += quality
                if quality >= self.canonical_threshold:
                    data["successes"] += 1

        for shape_id, data in list(stats.items()):
            uses = data["uses"]
            if not uses:
                continue
            data["avg_quality"] = data["total_quality"] / uses
            data["success_rate"] = data["successes"] / uses

        return dict(stats)

    def _promote_canonical_patterns(self, rule_stats: Dict[str, Dict]) -> List[Tuple[str, float]]:
        promoted: List[Tuple[str, float]] = []
        for rule_id, data in rule_stats.items():
            uses = data.get("uses", 0)
            success_rate = data.get("success_rate", 0.0)
            if uses < self.min_uses_canonical:
                continue
            if success_rate < self.canonical_threshold:
                continue
            rule = self.grammar.rules.get(rule_id)
            if not rule or getattr(rule, "is_canonical", False):
                continue
            rule.is_canonical = True
            semantics = getattr(rule, "semantics", {}) or {}
            semantics["canonical"] = True
            rule.semantics = semantics
            promoted.append((rule_id, success_rate))
        promoted.sort(key=lambda item: item[1], reverse=True)
        return promoted

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def _log_consolidation_report(self, stats: Dict[str, object]) -> None:
        print("\n[SLEEPTIME CONSOLIDATION]")
        print(f"  Pruned: {stats['pruned_count']} low-quality entries (< {self.min_quality:.2f})")
        print(
            f"  Grammar rules: {stats['total_rules']} total, {stats['canonical_promoted']} promoted to canonical"
        )
        print(f"  Drawing shapes: {stats['total_shapes']} total")

        rule_stats: Dict[str, Dict] = stats.get("rule_stats", {})  # type: ignore[assignment]
        shape_stats: Dict[str, Dict] = stats.get("shape_stats", {})  # type: ignore[assignment]

        if rule_stats:
            print("\n  Top 10 Grammar Rules by Usage:")
            sorted_rules = sorted(rule_stats.items(), key=lambda item: item[1].get("uses", 0), reverse=True)[:10]
            for idx, (rule_id, data) in enumerate(sorted_rules, start=1):
                print(
                    f"    {idx}. {rule_id}: {data.get('uses', 0)} uses, "
                    f"success={data.get('success_rate', 0.0):.2f}, avg_quality={data.get('avg_quality', 0.0):.2f}"
                )
        else:
            print("\n  No grammar usage data available.")

        if shape_stats:
            print("\n  Top 10 Drawing Shapes by Usage:")
            sorted_shapes = sorted(shape_stats.items(), key=lambda item: item[1].get("uses", 0), reverse=True)[:10]
            for idx, (shape_id, data) in enumerate(sorted_shapes, start=1):
                print(
                    f"    {idx}. {shape_id}: {data.get('uses', 0)} uses, "
                    f"success={data.get('success_rate', 0.0):.2f}"
                )
        else:
            print("\n  No drawing usage data available.")

        promoted: List[Tuple[str, float]] = stats.get("top_canonical_candidates", [])  # type: ignore[assignment]
        if promoted:
            print("\n  Canonical Patterns Promoted:")
            for rule_id, success in promoted:
                print(f"    - {rule_id} (success={success:.2f})")
        else:
            print("\n  Canonical Patterns Promoted: none this cycle")

        if stats.get("pruned_entries_audit"):
            print("\n  Pruning audit available for detailed inspection.")

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _parse_tokens(self, program: str, lookup: Dict[str, str]) -> List[str]:
        tokens = program.split()
        matched: List[str] = []
        for token in tokens:
            normalized = token.lower()
            if normalized in lookup:
                matched.append(lookup[normalized])
        return matched

    def _serialize_stat_block(self, stats: Dict[str, Dict], *, kind: str) -> List[Dict]:
        serialized: List[Dict] = []
        for key, data in stats.items():
            serialized.append(
                {
                    f"{kind}_id": key,
                    "uses": data.get("uses", 0),
                    "success_rate": data.get("success_rate", 0.0),
                    "avg_quality": data.get("avg_quality", 0.0),
                    "tasks_solved": data.get("tasks_solved", []) if kind == "rule" else [],
                }
            )
        return serialized

    @staticmethod
    def _format_task_signature(signature: Dict) -> str:
        if "task_id" in signature:
            return str(signature["task_id"])
        if "source" in signature:
            return str(signature["source"])
        return str(signature)


__all__ = ["SleepTimeConsolidator"]
