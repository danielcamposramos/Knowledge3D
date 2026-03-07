"""Knowledgeverse-integrated Grammar Galaxy wrapper."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from knowledge3d.training.arc_agi.grammar_galaxy import (
    GrammarGalaxy as LegacyGrammarGalaxy,
)
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule, default_grammar_rules
from knowledge3d.training.math_benchmarks.calculus_grammar_rules import get_calculus_rules


_BENCHMARK_GSM8K_RULE_IDS = {
    "gsm_consume_from_total",
    "gsm_rate_application",
    "gsm_sequential_computation",
    "gsm_comparison_delta",
    "gsm_percent_of",
    "gsm_answer_final_stack",
}

_BENCHMARK_CALCULUS_RULE_IDS = {
    "apply_power_rule_natural",
    "apply_power_rule_leibniz",
    "apply_sum_rule_natural",
    "apply_product_rule_natural",
    "apply_fundamental_theorem_calculus_natural",
    "apply_fundamental_theorem_calculus_latex",
}


class GrammarGalaxy(LegacyGrammarGalaxy):
    """Wrap legacy GrammarGalaxy with Knowledgeverse-compatible interfaces."""

    def __init__(
        self,
        knowledgeverse: Any = None,
        rules: list[GrammarRule] | None = None,
    ):
        self.knowledgeverse = knowledgeverse
        self.name = "Grammar"
        self._extra_entries: list[dict[str, Any]] = []
        super().__init__(rules=rules or default_grammar_rules())
        self._bootstrap_arc_patterns()
        self._bootstrap_benchmark_patterns()

    @property
    def entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for rule in self.rules.values():
            entries.append(
                {
                    "type": "canonical_rule",
                    "rule_id": rule.rule_id,
                    "language": rule.language,
                    "pattern": rule.pattern,
                    "rpn_program": rule.rpn_program,
                    "domain": getattr(rule, "domain", "general"),
                    "symbol_refs": list(getattr(rule, "symbol_refs", []) or []),
                    "word_refs": list(getattr(rule, "word_refs", []) or []),
                    "description": getattr(rule, "description", None),
                    "semantics": dict(getattr(rule, "semantics", {}) or {}),
                    "usage_conditions": list(getattr(rule, "usage_conditions", []) or []),
                    "is_canonical": getattr(rule, "is_canonical", False),
                }
            )
        for rule_id, info in self._local_discoveries.items():
            entries.append(
                {
                    "type": "discovered_rule",
                    "rule_id": rule_id,
                    "rpn_program": info.get("rpn_program", ""),
                    "quality_score": float(info.get("quality_score", 0.0)),
                    "usage_count": int(info.get("usage_count", 0)),
                }
            )
        entries.extend(self._extra_entries)
        return entries

    def add_entry(self, entry: dict[str, Any], *, record_event: bool = True) -> None:
        """Add a generic manager entry into this specialized galaxy."""
        if "rule_id" in entry and "rpn_program" in entry:
            rule = GrammarRule(
                rule_id=str(entry["rule_id"]),
                language=str(entry.get("language", "en")),
                pattern=str(entry.get("pattern", "custom")),
                rpn_program=str(entry["rpn_program"]),
                domain=str(entry.get("domain", "text")),
                description=entry.get("description"),
                semantics=dict(entry.get("semantics", {})),
                usage_conditions=list(entry.get("usage_conditions", [])),
                is_canonical=bool(entry.get("is_canonical", False)),
            )
            self.add_rule(rule, persist=False)
        else:
            self._extra_entries.append(dict(entry))
            if record_event:
                self._log_discovery_event(
                    event_type="grammar_discovery",
                    payload={
                        "entry_type": str(entry.get("type", "generic")),
                        "entry_id": str(entry.get("id", entry.get("rule_id", ""))),
                        "specialist": "grammar",
                        "galaxy": "Grammar",
                        "confidence": 0.8,
                        "verification": "entry_registered",
                    },
                )

    def add_rule(self, rule: GrammarRule, persist: bool = False) -> bool:
        added = super().add_rule(rule=rule, persist=persist)
        if added:
            self._log_discovery_event(
                event_type="grammar_discovery",
                payload={
                    "rule_id": rule.rule_id,
                    "language": rule.language,
                    "pattern": rule.pattern,
                    "specialist": "grammar",
                    "galaxy": "Grammar",
                    "confidence": 0.85,
                    "verification": "rule_registered",
                },
            )
        return added

    def summary(self) -> dict[str, int]:
        return {
            "canonical_rules": len(self.rules),
            "discovered_rules": len(self._local_discoveries),
            "extra_entries": len(self._extra_entries),
            "total": len(self.rules) + len(self._local_discoveries) + len(self._extra_entries),
        }

    def list_benchmark_rules(self, family: str | None = None) -> list[GrammarRule]:
        """Return canonical benchmark-facing rules loaded into the Grammar Galaxy."""
        out: list[GrammarRule] = []
        for rule in self.rules.values():
            semantics = getattr(rule, "semantics", {}) or {}
            benchmark_family = str(semantics.get("benchmark_family", "")).strip()
            if not benchmark_family and not str(rule.rule_id).startswith("arc_"):
                continue
            if family is not None:
                if family == "ARC_AGI_2" and str(rule.rule_id).startswith("arc_"):
                    out.append(rule)
                    continue
                if benchmark_family != family:
                    continue
            out.append(rule)
        return out

    def get_high_confidence_rules(self, min_score: float = 0.70) -> list[dict[str, Any]]:
        """Compatibility surface used by legacy ARC refiners."""
        selected: list[dict[str, Any]] = []
        for rule in self.rules.values():
            score = float(getattr(rule, "quality_score", 1.0))
            if score >= min_score:
                selected.append(
                    {
                        "id": rule.rule_id,
                        "rpn_program": rule.rpn_program,
                        "quality_score": score,
                    }
                )
        for rule_id, info in self._local_discoveries.items():
            score = float(info.get("quality_score", 0.0))
            if score >= min_score:
                selected.append(
                    {
                        "id": rule_id,
                        "rpn_program": info.get("rpn_program", ""),
                        "quality_score": score,
                    }
                )
        return selected

    def _log_discovery_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.knowledgeverse is None:
            return
        try:
            self.knowledgeverse.log_event(event_type=event_type, event_data=payload)
        except Exception:
            # Discovery logging should not break core galaxy operations.
            return

    # ------------------------------------------------------------------ #
    # ARC galaxy-first pattern system
    # ------------------------------------------------------------------ #
    def _bootstrap_arc_patterns(self) -> None:
        """Populate canonical ARC transform rules as procedural knowledge."""
        arc_rules = [
            GrammarRule(
                rule_id="arc_identity",
                language="visual",
                pattern="identity",
                rpn_program="GRID CLONE",
                domain="visual",
                description="Identity transform",
                semantics={"pattern_type": "spatial_transform", "transform": {"op": "identity"}},
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_flip_h",
                language="visual",
                pattern="mirror_horizontal",
                rpn_program="GRID_W 1 SUB RANGE REVERSE_COLS APPLY",
                domain="visual",
                description="Mirror across vertical axis",
                semantics={
                    "pattern_type": "spatial_transform",
                    "transform": {"op": "flip_h"},
                    "composition": {"inverse": "arc_flip_h"},
                },
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_flip_v",
                language="visual",
                pattern="mirror_vertical",
                rpn_program="GRID_H 1 SUB RANGE REVERSE_ROWS APPLY",
                domain="visual",
                description="Mirror across horizontal axis",
                semantics={
                    "pattern_type": "spatial_transform",
                    "transform": {"op": "flip_v"},
                    "composition": {"inverse": "arc_flip_v"},
                },
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_rot90_cw",
                language="visual",
                pattern="rotate_clockwise_90",
                rpn_program="GRID_H GRID_W SWAP GRID_NEW ROT90_KERNEL APPLY",
                domain="visual",
                description="Rotate 90 degrees clockwise",
                semantics={
                    "pattern_type": "spatial_transform",
                    "transform": {"op": "rot90"},
                    "composition": {"4_applications": "arc_identity"},
                },
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_rot180",
                language="visual",
                pattern="rotate_180",
                rpn_program="GRID REV_ROWS REV_COLS",
                domain="visual",
                description="Rotate 180 degrees",
                semantics={"pattern_type": "spatial_transform", "transform": {"op": "rot180"}},
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_rot270_ccw",
                language="visual",
                pattern="rotate_counterclockwise_90",
                rpn_program="GRID TRANSPOSE REV_ROWS",
                domain="visual",
                description="Rotate 270 degrees clockwise (90 CCW)",
                semantics={"pattern_type": "spatial_transform", "transform": {"op": "rot270"}},
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_transpose",
                language="visual",
                pattern="transpose",
                rpn_program="GRID TRANSPOSE",
                domain="visual",
                description="Transpose matrix",
                semantics={"pattern_type": "spatial_transform", "transform": {"op": "transpose"}},
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_color_map",
                language="visual",
                pattern="color_substitution",
                rpn_program="COLOR_MAP_TABLE GRID_MAP_COLOR",
                domain="visual",
                description="Map colors consistently from train examples",
                semantics={
                    "pattern_type": "value_transform",
                    "transform": {"op": "color_map"},
                },
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="arc_rotate_then_color",
                language="visual",
                pattern="rotation_with_recolor",
                rpn_program="ROT90_CW COLOR_MAP_TABLE GRID_MAP_COLOR",
                domain="visual",
                description="Composite transform: rotate then recolor",
                semantics={
                    "pattern_type": "composite_transform",
                    "transform": {"op": "composed", "steps": [{"op": "rot90"}, {"op": "color_map"}]},
                    "components": ["arc_rot90_cw", "arc_color_map"],
                },
                is_canonical=True,
            ),
        ]
        for rule in arc_rules:
            self.add_rule(rule, persist=False)

    def _bootstrap_benchmark_patterns(self) -> None:
        """Load canonical benchmark-facing math and reasoning rules."""
        for rule in self._selected_gsm8k_rules():
            self.add_rule(rule, persist=False)
        for rule in self._selected_calculus_rules():
            self.add_rule(rule, persist=False)
        for rule in self._multiple_choice_benchmark_rules():
            self.add_rule(rule, persist=False)

    def _selected_gsm8k_rules(self) -> list[GrammarRule]:
        rules = [
            GrammarRule(
                rule_id="gsm_consume_from_total",
                language="natural",
                pattern=r"(left|remaining|remainder|after|spent|used|gave|lost|ate)",
                rpn_program="STACK total consumed SUB",
                domain="math",
                word_refs=["word_left", "word_remaining"],
                description="Subtract consumed quantities from a running total.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "consume_from_total",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "composition"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_rate_application",
                language="natural",
                pattern=r"(each|per|every|times)",
                rpn_program="STACK quantity rate MUL",
                domain="math",
                word_refs=["word_each", "word_every"],
                description="Apply a rate or per-unit multiplier to a quantity.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "rate_application",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "composition"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_sequential_computation",
                language="natural",
                pattern=r"(then|after|finally|next)",
                rpn_program="STACK step_1 step_2 step_3 EVAL_CHAIN",
                domain="math",
                word_refs=["word_then", "word_after"],
                description="Chain multiple benchmark steps while carrying the intermediate stack state.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "sequential_computation",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "composition"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_comparison_delta",
                language="natural",
                pattern=r"(more than|less than|fewer than|difference)",
                rpn_program="STACK larger smaller SUB",
                domain="math",
                description="Compute comparison deltas between two quantities.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "comparison_delta",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "comparison"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_percent_of",
                language="natural",
                pattern=r"(percent of|percentage|%)",
                rpn_program="STACK base percent MUL 100 DIV",
                domain="math",
                description="Apply a percentage to a base quantity.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "percentage_application",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "percentage"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_answer_final_stack",
                language="natural",
                pattern=r"(how many|how much|what is|what does)",
                rpn_program="STACK TOP EMIT",
                domain="math",
                description="Emit the final stack value as the benchmark answer.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "answer_final_stack",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "emit"],
                is_canonical=True,
            ),
        ]
        return [rule for rule in rules if rule.rule_id in _BENCHMARK_GSM8K_RULE_IDS]

    def _selected_calculus_rules(self) -> list[GrammarRule]:
        selected: list[GrammarRule] = []
        for rule in get_calculus_rules():
            if rule.rule_id not in _BENCHMARK_CALCULUS_RULE_IDS:
                continue
            semantics = dict(getattr(rule, "semantics", {}) or {})
            semantics.update(
                {
                    "benchmark_family": "MATH",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "symbolic_math",
                }
            )
            selected.append(
                GrammarRule(
                    rule_id=rule.rule_id,
                    language=rule.language,
                    pattern=rule.pattern,
                    rpn_program=rule.rpn_program,
                    domain=rule.domain,
                    symbol_refs=list(getattr(rule, "symbol_refs", []) or []),
                    word_refs=list(getattr(rule, "word_refs", []) or []),
                    examples=list(getattr(rule, "examples", []) or []),
                    description=rule.description or "Canonical symbolic math benchmark pattern",
                    semantics=semantics,
                    usage_conditions=["tablet_boundary", "benchmark_math", "symbolic_reasoning"],
                    is_canonical=True,
                )
            )
        return selected

    def _multiple_choice_benchmark_rules(self) -> list[GrammarRule]:
        return [
            GrammarRule(
                rule_id="benchmark_choice_score_and_emit",
                language="benchmark",
                pattern="multiple_choice_selection",
                rpn_program="PARSE_PROMPT PARSE_OPTIONS SCORE_OPTIONS ARGMAX EMIT_CHOICE",
                domain="benchmark_reasoning",
                description="Canonical multiple-choice benchmark flow for tablet-mediated evaluation.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "multiple_choice_letter",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "multiple_choice",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_choice_eliminate_then_emit",
                language="benchmark",
                pattern="multiple_choice_elimination",
                rpn_program="PARSE_PROMPT PARSE_OPTIONS ELIMINATE_CONTRADICTIONS SCORE_REMAINING ARGMAX EMIT_CHOICE",
                domain="benchmark_reasoning",
                description="Contrastive multiple-choice flow that prunes implausible answers before emission.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "multiple_choice_letter",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "contrastive_multiple_choice",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice"],
                is_canonical=True,
            ),
        ]

    def list_arc_rules(self) -> list[GrammarRule]:
        """Return canonical and learned ARC visual rules."""
        out: list[GrammarRule] = []
        for rule in self.rules.values():
            semantics = getattr(rule, "semantics", {}) or {}
            if rule.language == "visual" and "transform" in semantics:
                out.append(rule)
        return out

    def discover_arc_pattern(self, train_examples: list[dict[str, Any]]) -> GrammarRule:
        """
        Discover or retrieve the best ARC transform rule from examples.

        Returns an existing high-confidence rule when available. If no rule
        reaches confidence threshold, synthesizes a discovered compositional rule.
        """
        candidates = self._score_arc_rules(train_examples)
        if candidates:
            best_score, best_rule, _ = candidates[0]
            if best_score >= 0.80:
                return best_rule
        return self._synthesize_arc_rule(train_examples, candidates)

    def propose_arc_transform(self, train_examples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Propose ARC transform metadata for execution/ranking layers.

        Returns:
          {
            "rule_id": str,
            "confidence": float,
            "transform": dict,
          }
        """
        scored = self._score_arc_rules(train_examples)
        if scored:
            score, rule, transform = scored[0]
            return {
                "rule_id": rule.rule_id,
                "confidence": score,
                "transform": transform,
            }
        fallback = self.get_rule("arc_identity")
        return {
            "rule_id": fallback.rule_id,
            "confidence": 0.0,
            "transform": {"op": "identity"},
        }

    def _score_arc_rules(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[tuple[float, GrammarRule, dict[str, Any]]]:
        scored: list[tuple[float, GrammarRule, dict[str, Any]]] = []
        for rule in self.list_arc_rules():
            transform = self._rule_to_transform(rule, train_examples)
            if not transform:
                continue
            confidence = self._evaluate_transform_confidence(train_examples, transform)
            scored.append((confidence, rule, transform))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _rule_to_transform(
        self,
        rule: GrammarRule,
        train_examples: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        semantics = getattr(rule, "semantics", {}) or {}
        transform = dict(semantics.get("transform", {}))
        op = str(transform.get("op", ""))
        if not op:
            return None
        if op == "color_map":
            mapping = self._infer_color_mapping(train_examples)
            if not mapping:
                return None
            transform["mapping"] = mapping
            return transform
        if op == "composed":
            steps = list(transform.get("steps", []))
            resolved_steps: list[dict[str, Any]] = []
            prefix_steps: list[dict[str, Any]] = []
            for step in steps:
                step_op = str(step.get("op", ""))
                if step_op == "color_map":
                    mapping = self._infer_color_mapping_with_prefix(train_examples, prefix_steps)
                    if not mapping:
                        return None
                    resolved = {"op": "color_map", "mapping": mapping}
                    resolved_steps.append(resolved)
                    prefix_steps.append(resolved)
                elif step_op:
                    resolved = {"op": step_op}
                    resolved_steps.append(resolved)
                    prefix_steps.append(resolved)
            if not resolved_steps:
                return None
            transform["steps"] = resolved_steps
            return transform
        return transform

    def _synthesize_arc_rule(
        self,
        train_examples: list[dict[str, Any]],
        scored: list[tuple[float, GrammarRule, dict[str, Any]]],
    ) -> GrammarRule:
        """
        Create a discovered compositional rule when canonical confidence is low.
        """
        top = scored[:3]
        if top:
            # Compose top two candidate operations when possible.
            first = top[0][2]
            second = top[1][2] if len(top) > 1 else {"op": "identity"}
            composed = {"op": "composed", "steps": [first, second]}
            conf = self._evaluate_transform_confidence(train_examples, composed)
            if conf > top[0][0]:
                transform = composed
                confidence = conf
            else:
                transform = top[0][2]
                confidence = top[0][0]
        else:
            transform = {"op": "identity"}
            confidence = 0.0

        digest = hashlib.sha1(
            json.dumps(train_examples, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        rule_id = f"arc_discovered_{digest}"
        if rule_id in self.rules:
            return self.rules[rule_id]

        if transform.get("op") == "composed":
            rpn_program = " ".join(self._transform_step_to_rpn(step) for step in transform.get("steps", []))
        else:
            rpn_program = self._transform_step_to_rpn(transform)

        discovered = GrammarRule(
            rule_id=rule_id,
            language="visual",
            pattern="learned_transform",
            rpn_program=rpn_program,
            domain="visual",
            description="Discovered ARC transform pattern from train examples",
            semantics={
                "pattern_type": "discovered",
                "transform": transform,
                "confidence": confidence,
            },
            is_canonical=False,
        )
        self.add_rule(discovered, persist=False)
        return discovered

    def _transform_step_to_rpn(self, step: dict[str, Any]) -> str:
        op = str(step.get("op", "identity"))
        if op == "flip_h":
            return "GRID_W 1 SUB RANGE REVERSE_COLS APPLY"
        if op == "flip_v":
            return "GRID_H 1 SUB RANGE REVERSE_ROWS APPLY"
        if op == "rot90":
            return "GRID_H GRID_W SWAP GRID_NEW ROT90_KERNEL APPLY"
        if op == "rot180":
            return "GRID REV_ROWS REV_COLS"
        if op == "rot270":
            return "GRID TRANSPOSE REV_ROWS"
        if op == "transpose":
            return "GRID TRANSPOSE"
        if op == "color_map":
            return "COLOR_MAP_TABLE GRID_MAP_COLOR"
        return "GRID CLONE"

    def _evaluate_transform_confidence(
        self,
        train_examples: list[dict[str, Any]],
        transform: dict[str, Any],
    ) -> float:
        if not train_examples:
            return 0.0
        score = 0.0
        for example in train_examples:
            predicted = self._apply_transform(example.get("input", []), transform)
            expected = example.get("output", [])
            score += self._grid_match_score(predicted, expected)
        return score / max(1, len(train_examples))

    def _apply_transform(
        self,
        grid: list[list[int]] | list[list[Any]],
        transform: dict[str, Any],
    ) -> list[list[int]]:
        rows = [list(map(int, row)) for row in grid]
        op = str(transform.get("op", "identity"))
        if op == "composed":
            result = rows
            for step in transform.get("steps", []):
                result = self._apply_transform(result, dict(step))
            return result
        if op == "identity":
            return rows
        if op == "flip_h":
            return [list(reversed(row)) for row in rows]
        if op == "flip_v":
            return list(reversed(rows))
        if op == "rot90":
            return [list(col) for col in zip(*rows[::-1])]
        if op == "rot180":
            return [list(reversed(row)) for row in reversed(rows)]
        if op == "rot270":
            return [list(col) for col in zip(*rows)][::-1]
        if op == "transpose":
            return [list(col) for col in zip(*rows)]
        if op == "color_map":
            mapping = {int(k): int(v) for k, v in dict(transform.get("mapping", {})).items()}
            return [[mapping.get(val, val) for val in row] for row in rows]
        return rows

    def _infer_color_mapping(self, train_examples: list[dict[str, Any]]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for example in train_examples:
            inp = example.get("input", [])
            out = example.get("output", [])
            if len(inp) != len(out):
                return {}
            for in_row, out_row in zip(inp, out):
                if len(in_row) != len(out_row):
                    return {}
                for in_val, out_val in zip(in_row, out_row):
                    in_i = int(in_val)
                    out_i = int(out_val)
                    prev = mapping.get(in_i)
                    if prev is None:
                        mapping[in_i] = out_i
                    elif prev != out_i:
                        return {}
        return mapping

    def _infer_color_mapping_with_prefix(
        self,
        train_examples: list[dict[str, Any]],
        prefix_steps: list[dict[str, Any]],
    ) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for example in train_examples:
            inp = example.get("input", [])
            out = example.get("output", [])
            transformed = [list(map(int, row)) for row in inp]
            for step in prefix_steps:
                transformed = self._apply_transform(transformed, step)
            if len(transformed) != len(out):
                return {}
            for in_row, out_row in zip(transformed, out):
                if len(in_row) != len(out_row):
                    return {}
                for in_val, out_val in zip(in_row, out_row):
                    in_i = int(in_val)
                    out_i = int(out_val)
                    prev = mapping.get(in_i)
                    if prev is None:
                        mapping[in_i] = out_i
                    elif prev != out_i:
                        return {}
        return mapping

    def _grid_match_score(self, predicted: list[list[int]], expected: list[list[int]]) -> float:
        if not predicted or not expected:
            return 0.0
        if len(predicted) != len(expected):
            return 0.0
        total = 0
        matched = 0
        for pred_row, exp_row in zip(predicted, expected):
            if len(pred_row) != len(exp_row):
                return 0.0
            total += len(pred_row)
            matched += sum(1 for a, b in zip(pred_row, exp_row) if int(a) == int(b))
        return (matched / total) if total else 0.0


__all__ = ["GrammarGalaxy", "GrammarRule", "default_grammar_rules"]
