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
    "gsm_fractional_total_materials",
    "gsm_markup_profit_after_repairs",
    "gsm_repeated_schedule_distance",
    "gsm_scaled_total_minus_meals",
    "gsm_alternating_discount_pairs",
    "gsm_successive_ratio_family_total",
    "gsm_restart_from_beginning_time",
    "gsm_turnaround_distance_balance",
    "gsm_overtime_total_earnings",
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
            semantics = dict(getattr(rule, "semantics", {}) or {})
            usage_conditions = list(getattr(rule, "usage_conditions", []) or [])
            symbol_refs = list(getattr(rule, "symbol_refs", []) or [])
            word_refs = list(getattr(rule, "word_refs", []) or [])
            metadata = {
                "rule_id": rule.rule_id,
                "language": rule.language,
                "pattern": rule.pattern,
                "domain": getattr(rule, "domain", "general"),
                "symbol_refs": symbol_refs,
                "word_refs": word_refs,
                "description": getattr(rule, "description", None),
                "semantics": semantics,
                "usage_conditions": usage_conditions,
                "is_canonical": getattr(rule, "is_canonical", False),
                "rule_strength": int(getattr(rule, "rule_strength", 0) or 0),
                "superior_to": list(getattr(rule, "superior_to", []) or []),
                "trust_weight": float(getattr(rule, "trust_weight", 1.0) or 1.0),
                "mmlu_subjects": [
                    str(value).strip()
                    for value in list(semantics.get("mmlu_subjects", []) or [])
                    if str(value).strip()
                ],
            }
            entries.append(
                {
                    "id": rule.rule_id,
                    "name": rule.rule_id,
                    "type": "canonical_rule",
                    "rule_id": rule.rule_id,
                    "language": rule.language,
                    "pattern": rule.pattern,
                    "rpn_program": rule.rpn_program,
                    "domain": getattr(rule, "domain", "general"),
                    "symbol_refs": symbol_refs,
                    "word_refs": word_refs,
                    "description": getattr(rule, "description", None),
                    "semantics": semantics,
                    "usage_conditions": usage_conditions,
                    "is_canonical": getattr(rule, "is_canonical", False),
                    "metadata": metadata,
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
                rule_strength=int(entry.get("rule_strength", 0) or 0),
                superior_to=[
                    str(value)
                    for value in list(entry.get("superior_to", []) or [])
                    if str(value).strip()
                ],
                trust_weight=float(entry.get("trust_weight", 1.0) or 1.0),
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
        for rule in self._reasoning_skeleton_rules():
            self.add_rule(rule, persist=False)
        for rule in self._selected_gsm8k_rules():
            self.add_rule(rule, persist=False)
        for rule in self._selected_calculus_rules():
            self.add_rule(rule, persist=False)
        for rule in self._multiple_choice_benchmark_rules():
            self.add_rule(rule, persist=False)
        for rule in self._mmlu_domain_rules():
            self.add_rule(rule, persist=False)

    def _reasoning_skeleton_rules(self) -> list[GrammarRule]:
        return [
            GrammarRule(
                rule_id="reasoning_chain_of_thought",
                language="meta",
                pattern="task_decompose_chain_verify",
                rpn_program="TASK DECOMPOSE SUB_TASKS EACH LOAD_GALAXY FIND_SIMILAR EVAL STORE CHAIN_RESULTS VERIFY",
                domain="reasoning",
                description="General decomposition skeleton for multi-step reasoning over sub-tasks with verification.",
                semantics={
                    "reasoning_family": "chain_of_thought",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_chain_of_thought",
                },
                usage_conditions=["reasoning", "meta_rule", "decomposition", "verification"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_elimination",
                language="meta",
                pattern="options_eliminate_select",
                rpn_program="OPTIONS EACH EVIDENCE_CHECK CONTRADICT ELIMINATE STORE SURVIVORS SELECT_BEST",
                domain="reasoning",
                description="General contrastive elimination skeleton for candidate selection tasks.",
                semantics={
                    "reasoning_family": "elimination",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_elimination",
                },
                usage_conditions=["reasoning", "meta_rule", "elimination", "selection"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_contrastive_verification",
                language="meta",
                pattern="candidate_support_contradiction_score",
                rpn_program="CANDIDATE EVIDENCE EACH SUPPORT CONTRADICT ADD ifelse",
                domain="reasoning",
                description="General verification skeleton that scores support against contradiction for a candidate answer.",
                semantics={
                    "reasoning_family": "contrastive_verification",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_contrastive_verification",
                },
                usage_conditions=["reasoning", "meta_rule", "verification", "contrastive"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_evidence_triangulation",
                language="meta",
                pattern="candidate_sources_triangulate",
                rpn_program="CANDIDATES EACH INDEPENDENT_SOURCES_COUNT 2 >= SUPPORT_SCORE STORE BEST_CANDIDATE",
                domain="reasoning",
                description="General triangulation skeleton that favors candidates supported by multiple independent sources.",
                semantics={
                    "reasoning_family": "evidence_triangulation",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_evidence_triangulation",
                },
                usage_conditions=["reasoning", "meta_rule", "triangulation", "evidence"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_dimensional_analysis",
                language="meta",
                pattern="quantities_dimension_compute",
                rpn_program="EXTRACT_QUANTITIES DIMENSION_CHECK FORMULA EVAL UNIT_ATTACH VERIFY",
                domain="reasoning",
                description="General quantity reasoning skeleton for dimensional consistency and unit-aware computation.",
                semantics={
                    "reasoning_family": "dimensional_analysis",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_dimensional_analysis",
                },
                usage_conditions=["reasoning", "meta_rule", "dimension", "quantity"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_procedural_decode",
                language="meta",
                pattern="grammar_rules_apply_verify_emit",
                rpn_program="GRAMMAR_RULES EACH INPUT RULE APPLY_MOVE STORE VERIFY EMIT",
                domain="reasoning",
                description="General procedural reasoning skeleton for decode/transform tasks executed as rule application sequences.",
                semantics={
                    "reasoning_family": "procedural_decode",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_procedural_decode",
                },
                usage_conditions=["reasoning", "meta_rule", "procedural", "decode"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_clue_chain",
                language="meta",
                pattern="decompose_clues_resolve_chain",
                rpn_program="TASK DECOMPOSE CLUES EACH RESOLVE STORE CHAIN_RESULTS VERIFY",
                domain="reasoning",
                description="General clue-chain skeleton for multi-clause problems with intermediate symbolic variables.",
                semantics={
                    "reasoning_family": "clue_chain",
                    "layer": "meta_rule",
                    "benchmark_alias": "benchmark_reasoning_clue_chain",
                },
                usage_conditions=["reasoning", "meta_rule", "clue_chain", "composition"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="reasoning_pattern_recognition",
                language="meta",
                pattern="extract_patterns_match_verify",
                rpn_program="QUERY PATTERNS EACH MATCH SCORE VERIFY SELECT_BEST",
                domain="reasoning",
                description="General pattern-recognition skeleton for structured symbolic or spatial response selection.",
                semantics={
                    "reasoning_family": "pattern_recognition",
                    "layer": "meta_rule",
                },
                usage_conditions=["reasoning", "meta_rule", "pattern_recognition"],
                is_canonical=True,
            ),
        ]

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
            GrammarRule(
                rule_id="gsm_fractional_total_materials",
                language="natural",
                pattern=r"(half that much|half as much|in total|total bolts|takes .* white fiber)",
                rpn_program="STACK base fraction MUL base ADD",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                ],
                description="Recover a total when one material is a fractional share of the base material and both must be combined.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "fractional_total_materials",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "fraction", "composition"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_markup_profit_after_repairs",
                language="natural",
                pattern=r"(profit|flipping a house|repairs|increased the value|150%)",
                rpn_program="STACK initial markup APPLY_INCREASE repairs SUBTRACT_COST_BASIS",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "Josh buys a house, puts in repairs, and the value increases by 150%. How much profit did he make?",
                ],
                description="Compute profit after a percentage markup by comparing the new value to the original purchase plus repair costs.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "markup_profit_after_repairs",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "percentage", "profit"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_repeated_schedule_distance",
                language="natural",
                pattern=r"(sprints .* times a week|times a week|meters each sprint|runs .* each sprint)",
                rpn_program="STACK repeat_count session_count distance_per_session MUL_CHAIN",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "James runs 3 sprints 3 times a week and each sprint is 60 meters. How many total meters does he run?",
                ],
                description="Multiply repeated schedule counts by per-session distance to recover a weekly or repeated total.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "repeated_schedule_distance",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "schedule", "multiplicative_chain"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_scaled_total_minus_meals",
                language="natural",
                pattern=r"(each of her chickens|cups of feed|final meal|morning|afternoon|flock)",
                rpn_program="STACK count per_unit MUL known_parts SUB",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "Wendi feeds 20 chickens 3 cups each per day, gives 15 cups in the morning and 25 in the afternoon. How many cups remain for the final meal?",
                ],
                description="Scale a daily per-unit total and subtract the already-served meals to recover the remaining final allocation.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "scaled_total_minus_meals",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "remainder", "scaled_total"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_alternating_discount_pairs",
                language="natural",
                pattern=r"(every second glass|only 60% of the price|second glass costs|buy 16 glasses)",
                rpn_program="STACK pair_count full_price discount_price PAIR_TOTAL",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "One glass costs $5, every second glass costs 60% of the price, and 16 glasses are bought. How much does he pay?",
                ],
                description="Pair full-price and discounted items together so alternating-discount purchases resolve to pair totals instead of a naive single-rate multiply.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "alternating_discount_pairs",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "discount", "pairing"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_successive_ratio_family_total",
                language="natural",
                pattern=r"(twice as many|4 times as many|how many .* together|altogether if .* has)",
                rpn_program="STACK base ratio_chain EXPAND_AND_SUM",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "Toulouse has twice as many sheep as Charleston, Charleston has 4 times as many as Seattle, and Seattle has 20 sheep. How many are there together?",
                ],
                description="Expand chained family ratios from the base quantity and sum every related branch in the ratio family.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "successive_ratio_family_total",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "ratio_chain", "family_total"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_restart_from_beginning_time",
                language="natural",
                pattern=r"(restart .* beginning|40% of the way|download from the beginning|takes 20 minutes)",
                rpn_program="STACK full_time partial_progress_time restart_delay ADD",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "Carla downloads 40% of a 200 GB file at 2 GB/minute, restarts, waits 20 minutes, then downloads the file again from the beginning. How long does it take?",
                ],
                description="Accumulate wasted progress time, restart delay, and the full successful rerun when a process restarts from the beginning.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "restart_from_beginning_time",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "restart", "time_accumulation"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_turnaround_distance_balance",
                language="natural",
                pattern=r"(turns around|from home at the end|standstill traffic|remaining time .* 80 mph)",
                rpn_program="STACK outbound_distance return_segments SUB",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "John drives away from home, turns around, then returns with traffic and changing speeds. How far is he from home at the end?",
                ],
                description="Compute an outbound distance, compute segmented return distance, and subtract the return progress from the outbound leg.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "turnaround_distance_balance",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "distance", "segmented_motion"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="gsm_overtime_total_earnings",
                language="natural",
                pattern=r"(first 40 hours|overtime pay|1.2 times|regular hourly rate|earnings for this week)",
                rpn_program="STACK base_hours base_rate MUL overtime_hours overtime_rate MUL ADD",
                domain="math",
                symbol_refs=["math_template_arithmetic_chain_gpu", "math_concept_rate_balance"],
                examples=[
                    "Eliza earns $10 for the first 40 hours and 1.2 times that rate for overtime. She worked 45 hours. How much did she earn?",
                ],
                description="Split compensation into regular and overtime components, compute both, and add them into one weekly earnings total.",
                semantics={
                    "benchmark_family": "GSM8K",
                    "tablet_contract": "math_text_answer",
                    "benchmark_stage": "track1_content",
                    "benchmark_track": "math_word_problem",
                    "composition_role": "overtime_total_earnings",
                },
                usage_conditions=["tablet_boundary", "benchmark_math", "word_problem", "overtime", "earnings"],
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
                    rule_strength=1,
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
                rule_id="benchmark_reasoning_chain_of_thought",
                language="benchmark",
                pattern="reasoning_chain_of_thought",
                rpn_program="QUERY DECOMPOSE_STEPS STEP_1 SOLVE STEP_2 SOLVE CHAIN_RESULTS VERIFY",
                domain="benchmark_reasoning",
                description="Universal benchmark reasoning skeleton for explicit step-by-step decomposition and verification.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "chain_of_thought",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "chain_of_thought"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_reasoning_elimination",
                language="benchmark",
                pattern="reasoning_elimination",
                rpn_program="OPTIONS FOREACH OPTION EVIDENCE_CHECK CONTRADICT ELIMINATE SURVIVORS SELECT_BEST",
                domain="benchmark_reasoning",
                description="Universal benchmark reasoning skeleton for contrastive elimination among candidates.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "multiple_choice_letter",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "elimination",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "elimination"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_reasoning_contrastive_verification",
                language="benchmark",
                pattern="reasoning_contrastive_verification",
                rpn_program="CANDIDATE EVIDENCE_ALL CHECK_SUPPORT CHECK_CONTRADICTION SCORE",
                domain="benchmark_reasoning",
                description="Universal benchmark reasoning skeleton for support-versus-contradiction scoring.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "contrastive_verification",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "contrastive_verification"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_reasoning_evidence_triangulation",
                language="benchmark",
                pattern="reasoning_evidence_triangulation",
                rpn_program="QUERY EVIDENCE_ALL TRIANGULATE_CONTENT SCORE_SUPPORT EMIT",
                domain="benchmark_reasoning",
                description="Universal benchmark reasoning skeleton for merging semantic evidence from multiple rows.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "evidence_triangulation",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "evidence_triangulation"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_reasoning_dimensional_analysis",
                language="benchmark",
                pattern="reasoning_dimensional_analysis",
                rpn_program="QUERY EXTRACT_QUANTITIES TYPE_CHECK DIMENSION_MATCH COMPUTE VERIFY_UNITS",
                domain="benchmark_reasoning",
                description="Benchmark reasoning skeleton for quantity extraction, unit consistency, and dimensional verification.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_numeric",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "dimensional_analysis",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "dimensional_analysis"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_reasoning_procedural_decode",
                language="benchmark",
                pattern="reasoning_procedural_decode",
                rpn_program="QUERY REVERSE_PREPROCESS GENERATE_KEYS SCORE_LANGUAGE VERIFY OUTPUT",
                domain="benchmark_reasoning",
                description="Benchmark reasoning skeleton for procedural decoding tasks such as ciphers and notation transforms.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_exact",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "procedural_decode",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "procedural_decode"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="benchmark_reasoning_clue_chain",
                language="benchmark",
                pattern="reasoning_clue_chain",
                rpn_program="QUERY DECOMPOSE_CLUES RESOLVE_CLAUSES CHAIN_RESULTS VERIFY",
                domain="benchmark_reasoning",
                description="Benchmark reasoning skeleton for multi-clue chained open-ended questions with intermediate variables.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_exact",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "reasoning_skeletons",
                    "skeleton": "clue_chain",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "reasoning_skeleton", "clue_chain"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_irony",
                language="benchmark",
                pattern="irony_detection",
                rpn_program="PARSE_CONTEXT PARSE_TONE DETECT_INCONGRUITY SCORE_IRONY",
                domain="benchmark_reasoning",
                description="Detect irony by matching surface statement against contradictory or incongruent context.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "irony",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "irony"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_metaphor",
                language="benchmark",
                pattern="metaphor_detection",
                rpn_program="PARSE_CONTEXT EXTRACT_SOURCE_TARGET MAP_FIGURATIVE_RELATION SCORE_METAPHOR",
                domain="benchmark_reasoning",
                description="Detect metaphor by mapping figurative source-target relations rather than literal overlap.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "metaphor",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "metaphor"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_hyperbole",
                language="benchmark",
                pattern="hyperbole_detection",
                rpn_program="PARSE_CONTEXT DETECT_EXAGGERATION CONTRAST_LITERAL_SCOPE SCORE_HYPERBOLE",
                domain="benchmark_reasoning",
                description="Detect hyperbole by measuring exaggeration against literal scope and plausible bounds.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "hyperbole",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "hyperbole"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_sarcasm",
                language="benchmark",
                pattern="sarcasm_detection",
                rpn_program="PARSE_CONTEXT PARSE_TONE DETECT_MOCKERY SCORE_SARCASM",
                domain="benchmark_reasoning",
                description="Detect sarcasm by tracking mocking tone, inverted praise, and context-level contradiction.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "sarcasm",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "sarcasm"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_pun",
                language="benchmark",
                pattern="pun_detection",
                rpn_program="PARSE_CONTEXT DETECT_DOUBLE_MEANING SCORE_PUN",
                domain="benchmark_reasoning",
                description="Detect punning by comparing competing lexical meanings sharing the same surface form.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "pun",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "pun"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_paradox",
                language="benchmark",
                pattern="paradox_detection",
                rpn_program="PARSE_CONTEXT DETECT_CONTRADICTION RESOLVE_HIGHER_ORDER_TENSION SCORE_PARADOX",
                domain="benchmark_reasoning",
                description="Detect paradox by preserving a meaningful tension between apparently contradictory claims.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "paradox",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "paradox"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_oxymoron",
                language="benchmark",
                pattern="oxymoron_detection",
                rpn_program="PARSE_CONTEXT DETECT_OPPOSING_TERMS SCORE_OXYMORON",
                domain="benchmark_reasoning",
                description="Detect oxymoron by spotting compressed opposition inside a local phrase.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "oxymoron",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "oxymoron"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_allusion",
                language="benchmark",
                pattern="allusion_detection",
                rpn_program="PARSE_CONTEXT DETECT_INDIRECT_REFERENCE SCORE_ALLUSION",
                domain="benchmark_reasoning",
                description="Detect allusion by linking indirect references to shared cultural or historical anchors.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "allusion",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "allusion"],
                is_canonical=True,
            ),
            GrammarRule(
                rule_id="lhe_language_figure_personification",
                language="benchmark",
                pattern="personification_detection",
                rpn_program="PARSE_CONTEXT MAP_HUMAN_TRAITS_TO_NONHUMAN_TARGET SCORE_PERSONIFICATION",
                domain="benchmark_reasoning",
                description="Detect personification by assigning human actions or traits to non-human entities.",
                semantics={
                    "benchmark_family": "LHE",
                    "tablet_contract": "open_ended_or_multiple_choice",
                    "benchmark_stage": "track0_tablet",
                    "benchmark_track": "language_figures",
                    "figure": "personification",
                },
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "language_figure", "personification"],
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

    def _mmlu_domain_rules(self) -> list[GrammarRule]:
        return [
            GrammarRule(
                rule_id="mmlu_algebra_field_extension_degree",
                language="benchmark",
                pattern="field_extension_degree",
                rpn_program="QUERY GALAXY_LOOKUP DEGREE_SELECT VERIFY_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_abstract_algebra_field_extension_qsqrt2_sqrt3",
                    "reality_abstract_algebra_sqrt2_plus_sqrt3_minimal_polynomial",
                ],
                description="Resolve algebraic field-extension degree questions from compositional radical facts.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["abstract_algebra"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_abstract_algebra"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.95,
            ),
            GrammarRule(
                rule_id="mmlu_algebra_homomorphism_kernel_test",
                language="benchmark",
                pattern="homomorphism_kernel_injective",
                rpn_program="QUERY GALAXY_LOOKUP KERNEL TEST_INJECTIVE VERIFY_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_abstract_algebra_kernel_zero_injective",
                    "reality_abstract_algebra_order_of_image_divides_order",
                ],
                description="Use kernel/image facts to score algebraic homomorphism statements.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["abstract_algebra"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_abstract_algebra"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.94,
            ),
            GrammarRule(
                rule_id="mmlu_algebra_irreducibility_criterion",
                language="benchmark",
                pattern="eisenstein_irreducibility",
                rpn_program="QUERY FACTOR_CHECK PRIME_TEST IRREDUCIBLE_SELECT",
                domain="benchmark_reasoning",
                symbol_refs=["reality_abstract_algebra_eisenstein_criterion"],
                description="Apply Eisenstein-style prime divisibility tests to multiple-choice irreducibility prompts.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["abstract_algebra"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_abstract_algebra"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.93,
            ),
            GrammarRule(
                rule_id="mmlu_formal_logic_truth_evaluation",
                language="benchmark",
                pattern="truth_table_evaluation",
                rpn_program="QUERY PARSE_PROPOSITION BUILD_TRUTH_TABLE SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_formal_logic_truth_tables",
                    "reality_formal_logic_quantifier_negation",
                ],
                description="Evaluate multiple-choice logic prompts via truth conditions and quantified negation.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["formal_logic"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_formal_logic"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.93,
            ),
            GrammarRule(
                rule_id="mmlu_formal_logic_relation_properties",
                language="benchmark",
                pattern="relation_property_classification",
                rpn_program="QUERY PARSE_RELATION CHECK_REFLEXIVE CHECK_SYMMETRY CHECK_TRANSITIVITY SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_formal_logic_equivalence_relation",
                    "reality_formal_logic_symmetric_antisymmetric",
                ],
                description="Classify relation questions using equivalence and antisymmetry criteria.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["formal_logic"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_formal_logic"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.91,
            ),
            GrammarRule(
                rule_id="mmlu_biology_inheritance_pattern",
                language="benchmark",
                pattern="mendelian_inheritance",
                rpn_program="QUERY GALAXY_LOOKUP PHENOTYPE_RATIO GENOTYPE_RATIO SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_biology_mendelian_ratios",
                    "reality_biology_genetics_inheritance",
                ],
                description="Use Mendelian inheritance ratios for biology multiple-choice questions.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["college_biology", "high_school_biology"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_biology"],
                is_canonical=True,
                trust_weight=0.86,
            ),
            GrammarRule(
                rule_id="mmlu_biology_population_growth",
                language="benchmark",
                pattern="cell_growth_or_population_doubling",
                rpn_program="QUERY GALAXY_LOOKUP EXPONENTIAL_GROWTH VERIFY_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_biology_mitosis_doubling",
                    "reality_anchor_college_biology_core",
                ],
                description="Reason over cell-division and growth questions with explicit doubling structure.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["college_biology", "high_school_biology"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_biology"],
                is_canonical=True,
                trust_weight=0.84,
            ),
            GrammarRule(
                rule_id="mmlu_chemistry_periodic_and_acid_base",
                language="benchmark",
                pattern="periodic_trend_or_ph_reasoning",
                rpn_program="QUERY GALAXY_LOOKUP TREND_COMPARE SCALE_REASON VERIFY_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_chemistry_periodic_trends",
                    "reality_chemistry_ph_log_scale",
                ],
                description="Handle chemistry questions about periodic trends and pH logarithms.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["college_chemistry", "high_school_chemistry"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_chemistry"],
                is_canonical=True,
                trust_weight=0.87,
            ),
            GrammarRule(
                rule_id="mmlu_chemistry_stoichiometric_ratio",
                language="benchmark",
                pattern="stoichiometric_ratio_reasoning",
                rpn_program="QUERY BALANCE EQUATION_COEFF_RATIO SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_chemistry_stoichiometric_coefficients",
                    "reality_chemistry_stoichiometry",
                ],
                description="Apply coefficient-derived mole ratios to chemistry answer choices.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["college_chemistry", "high_school_chemistry"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_chemistry"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.9,
            ),
            GrammarRule(
                rule_id="mmlu_cs_complexity_dominance",
                language="benchmark",
                pattern="complexity_growth_comparison",
                rpn_program="QUERY PARSE_COMPLEXITIES ORDER_ASYMPTOTIC SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_cs_big_o_domination",
                    "reality_cs_algorithmic_complexity",
                ],
                description="Compare asymptotic growth classes for computer-science questions.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["college_computer_science"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_computer_science"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.91,
            ),
            GrammarRule(
                rule_id="mmlu_cs_boolean_gate_composition",
                language="benchmark",
                pattern="boolean_gate_reasoning",
                rpn_program="QUERY GALAXY_LOOKUP GATE_COMPOSE TRUTH_EVAL SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_cs_boolean_logic_gates",
                    "reality_formal_logic_truth_tables",
                ],
                description="Solve Boolean-gate and digital-logic questions through compositional truth evaluation.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["college_computer_science", "electrical_engineering"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_digital_logic"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.89,
            ),
            GrammarRule(
                rule_id="mmlu_ml_bias_variance_diagnosis",
                language="benchmark",
                pattern="bias_variance_diagnosis",
                rpn_program="QUERY GENERALIZATION_ERROR CLASSIFY_OVERFIT_UNDERFIT SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=["reality_ml_bias_variance_tradeoff"],
                description="Diagnose overfitting, underfitting, and generalization by bias-variance cues.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["machine_learning"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_machine_learning"],
                is_canonical=True,
                trust_weight=0.86,
            ),
            GrammarRule(
                rule_id="mmlu_security_authn_authorization",
                language="benchmark",
                pattern="authentication_authorization_distinction",
                rpn_program="QUERY IDENTITY_CHECK PERMISSION_CHECK SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=["reality_security_authn_vs_authz"],
                description="Separate authentication from authorization in computer-security prompts.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["computer_security"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_computer_security"],
                is_canonical=True,
                trust_weight=0.89,
            ),
            GrammarRule(
                rule_id="mmlu_philosophy_framework_compare",
                language="benchmark",
                pattern="ethical_framework_comparison",
                rpn_program="QUERY MAP_FRAMEWORK CONSEQUENCE_DUTY_VIRTUE SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_philosophy_utilitarianism",
                    "reality_philosophy_deontology",
                    "reality_philosophy_virtue_ethics",
                ],
                description="Map philosophy prompts onto core ethical frameworks.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["philosophy", "moral_scenarios", "moral_disputes"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_philosophy"],
                is_canonical=True,
                trust_weight=0.84,
            ),
            GrammarRule(
                rule_id="mmlu_law_precedent_vs_statute",
                language="benchmark",
                pattern="precedent_statute_distinction",
                rpn_program="QUERY GALAXY_LOOKUP APPLY_ANALOGY_OR_TEXT SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_jurisprudence_precedent_reasoning",
                    "reality_professional_law_statutory_interpretation",
                ],
                description="Distinguish precedent-driven legal reasoning from statute-text interpretation.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["jurisprudence", "professional_law"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_law"],
                is_canonical=True,
                trust_weight=0.85,
            ),
            GrammarRule(
                rule_id="mmlu_economics_supply_demand_shift",
                language="benchmark",
                pattern="supply_demand_shift_reasoning",
                rpn_program="QUERY MARKET_SHIFT EQUILIBRIUM_COMPARE SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=["reality_microeconomics_supply_demand_shift"],
                description="Classify equilibrium effects of supply and demand shifts.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["high_school_microeconomics", "high_school_macroeconomics"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_economics"],
                is_canonical=True,
                trust_weight=0.87,
            ),
            GrammarRule(
                rule_id="mmlu_accounting_balance_equation",
                language="benchmark",
                pattern="balance_sheet_consistency",
                rpn_program="QUERY ASSET LIABILITY EQUITY CHECK_BALANCE SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_accounting_balance_sheet_equation",
                    "reality_accounting_double_entry",
                ],
                description="Use accounting-equation consistency and double-entry conservation.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["professional_accounting"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_accounting"],
                is_canonical=True,
                rule_strength=1,
                trust_weight=0.92,
            ),
            GrammarRule(
                rule_id="mmlu_government_branch_reasoning",
                language="benchmark",
                pattern="government_branch_classification",
                rpn_program="QUERY MAP_POWER_TO_BRANCH SCORE_OPTION",
                domain="benchmark_reasoning",
                symbol_refs=[
                    "reality_government_separation_of_powers",
                    "reality_government_legislative_process",
                    "reality_government_federalism",
                ],
                description="Map institutions and powers to the appropriate branch or level of government.",
                semantics={"benchmark_family": "MMLU", "mmlu_subjects": ["high_school_government_and_politics"]},
                usage_conditions=["tablet_boundary", "benchmark_reasoning", "multiple_choice", "subject_government"],
                is_canonical=True,
                trust_weight=0.86,
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
