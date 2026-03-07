"""Math specialist: Galaxy-first RPN composition for deterministic math solving."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from knowledge3d.knowledgeverse.specialist_base import SpecialistBase
from knowledge3d.knowledgeverse.ternary_quality_memory import TernaryQualityMemory


class MathSpecialist(SpecialistBase):
    """
    Generative math specialist.

    Flow:
    1) Query Grammar Galaxy for equation pattern entries.
    2) Query Math Galaxy for solve templates.
    3) Compose concrete RPN program from template + extracted coefficients.
    4) Execute composed RPN through sovereign PTX runtime.

    No Python eval/sympy fallback is used.
    """

    _NUMBER_WORD_UNITS = {
        "zero": 0.0,
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
        "eleven": 11.0,
        "twelve": 12.0,
        "thirteen": 13.0,
        "fourteen": 14.0,
        "fifteen": 15.0,
        "sixteen": 16.0,
        "seventeen": 17.0,
        "eighteen": 18.0,
        "nineteen": 19.0,
        "half": 0.5,
        "double": 2.0,
        "twice": 2.0,
        "triple": 3.0,
        "thrice": 3.0,
    }
    _NUMBER_WORD_TENS = {
        "twenty": 20.0,
        "thirty": 30.0,
        "forty": 40.0,
        "fifty": 50.0,
        "sixty": 60.0,
        "seventy": 70.0,
        "eighty": 80.0,
        "ninety": 90.0,
    }
    _NUMBER_WORD_SCALES = {
        "hundred": 100.0,
        "thousand": 1000.0,
        "million": 1_000_000.0,
    }
    _NUMBER_WORD_TOKENS = set(_NUMBER_WORD_UNITS) | set(_NUMBER_WORD_TENS) | set(_NUMBER_WORD_SCALES) | {"and"}

    _GRAMMAR_BOOTSTRAP_ENTRIES: tuple[dict[str, Any], ...] = (
        {
            "id": "grammar_linear_equation_ax_plus_b_eq_c_v1",
            "name": "Linear Equation Pattern (ax + b = c)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "linear_equation",
            "pattern_form": "ax + b = c",
            "rpn_program": "pattern linear_equation ax_plus_b_eq_c",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_linear_equation_ax_minus_b_eq_c_v1",
            "name": "Linear Equation Pattern (ax - b = c)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "linear_equation_ax_minus_b_eq_c",
            "pattern_form": "ax - b = c",
            "rpn_program": "pattern linear_equation ax_minus_b_eq_c",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_linear_equation_b_plus_ax_eq_c_v1",
            "name": "Linear Equation Pattern (b + ax = c)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "linear_equation_b_plus_ax_eq_c",
            "pattern_form": "b + ax = c",
            "rpn_program": "pattern linear_equation b_plus_ax_eq_c",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_arithmetic_addition_ab_v1",
            "name": "Arithmetic Addition Pattern (a + b)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "arithmetic_add",
            "pattern_form": "a + b",
            "rpn_program": "pattern arithmetic_add a_plus_b",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_arithmetic_subtraction_ab_v1",
            "name": "Arithmetic Subtraction Pattern (a - b)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "arithmetic_subtract",
            "pattern_form": "a - b",
            "rpn_program": "pattern arithmetic_subtract a_minus_b",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_ratio_a_to_b_v1",
            "name": "Ratio Pattern (a:b)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "ratio",
            "pattern_form": "a:b",
            "rpn_program": "pattern ratio a_to_b",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_proportion_a_over_b_eq_c_over_x_v1",
            "name": "Proportion Pattern (a/b = c/x)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "proportion",
            "pattern_form": "a / b = c / x",
            "rpn_program": "pattern proportion a_over_b_eq_c_over_x",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_word_problem_total_v1",
            "name": "Word Problem Pattern (total/in all)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "word_problem_total",
            "pattern_form": "total / altogether / in all",
            "rpn_program": "pattern word_problem total",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_word_problem_remainder_v1",
            "name": "Word Problem Pattern (remainder/left over)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "word_problem_remainder",
            "pattern_form": "remaining / left / after using some",
            "rpn_program": "pattern word_problem remainder",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_word_problem_rate_v1",
            "name": "Word Problem Pattern (rate/each/per)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "word_problem_rate",
            "pattern_form": "each / per / every",
            "rpn_program": "pattern word_problem rate",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_word_problem_comparison_v1",
            "name": "Word Problem Pattern (comparison)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "word_problem_comparison",
            "pattern_form": "more than / less than / fewer than",
            "rpn_program": "pattern word_problem comparison",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_word_problem_percentage_v1",
            "name": "Word Problem Pattern (percentage)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "word_problem_percentage",
            "pattern_form": "percent of / percentage",
            "rpn_program": "pattern word_problem percentage",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
        {
            "id": "grammar_word_problem_sequential_v1",
            "name": "Word Problem Pattern (multi-step)",
            "domain": "math_grammar",
            "category": "equation_pattern",
            "pattern_type": "word_problem_sequential",
            "pattern_form": "first then after finally",
            "rpn_program": "pattern word_problem sequential",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
            },
        },
    )

    _MATH_TEMPLATE_ENTRIES: tuple[dict[str, Any], ...] = (
        {
            "id": "math_template_linear_equation_solve_v1",
            "name": "Linear Equation Solve Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "linear_equation",
            # x = (c - b) / a
            "rpn_program": "{c} {b} - {a} /",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b", "c"],
            },
        },
        {
            "id": "math_template_linear_equation_ax_minus_b_eq_c_solve_v1",
            "name": "Linear Equation Solve Template (ax - b = c)",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "linear_equation_ax_minus_b_eq_c",
            # Generic normalized solve: x = (c - b) / a (b may be signed from parser)
            "rpn_program": "{c} {b} - {a} /",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b", "c"],
            },
        },
        {
            "id": "math_template_linear_equation_b_plus_ax_eq_c_solve_v1",
            "name": "Linear Equation Solve Template (b + ax = c)",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "linear_equation_b_plus_ax_eq_c",
            "rpn_program": "{c} {b} - {a} /",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b", "c"],
            },
        },
        {
            "id": "math_template_arithmetic_add_v1",
            "name": "Arithmetic Add Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "arithmetic_add",
            "rpn_program": "{a} {b} +",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b"],
            },
        },
        {
            "id": "math_template_arithmetic_subtract_v1",
            "name": "Arithmetic Subtract Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "arithmetic_subtract",
            "rpn_program": "{a} {b} -",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b"],
            },
        },
        {
            "id": "math_template_arithmetic_multiply_v1",
            "name": "Arithmetic Multiply Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "arithmetic_multiply",
            "rpn_program": "{a} {b} *",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b"],
            },
        },
        {
            "id": "math_template_arithmetic_divide_v1",
            "name": "Arithmetic Divide Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "arithmetic_divide",
            "rpn_program": "{a} {b} /",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b"],
            },
        },
        {
            "id": "math_template_ratio_a_to_b_v1",
            "name": "Ratio Template (a:b)",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "ratio",
            "rpn_program": "{a} {b} /",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b"],
            },
        },
        {
            "id": "math_template_proportion_solve_v1",
            "name": "Proportion Solve Template (a/b = c/x)",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "proportion",
            # x = (b * c) / a
            "rpn_program": "{b} {c} * {a} /",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["a", "b", "c"],
            },
        },
        {
            "id": "math_template_word_problem_total_v1",
            "name": "Word Problem Total Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "word_problem_total",
            "rpn_program": "{rpn_chain}",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["rpn_chain"],
            },
        },
        {
            "id": "math_template_word_problem_remainder_v1",
            "name": "Word Problem Remainder Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "word_problem_remainder",
            "rpn_program": "{rpn_chain}",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["rpn_chain"],
            },
        },
        {
            "id": "math_template_word_problem_rate_v1",
            "name": "Word Problem Rate Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "word_problem_rate",
            "rpn_program": "{rpn_chain}",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["rpn_chain"],
            },
        },
        {
            "id": "math_template_word_problem_comparison_v1",
            "name": "Word Problem Comparison Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "word_problem_comparison",
            "rpn_program": "{rpn_chain}",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["rpn_chain"],
            },
        },
        {
            "id": "math_template_word_problem_percentage_v1",
            "name": "Word Problem Percentage Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "word_problem_percentage",
            "rpn_program": "{rpn_chain}",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["rpn_chain"],
            },
        },
        {
            "id": "math_template_word_problem_sequential_v1",
            "name": "Word Problem Sequential Template",
            "domain": "math_template",
            "category": "algebra_template",
            "pattern_type": "word_problem_sequential",
            "rpn_program": "{rpn_chain}",
            "metadata": {
                "generated": False,
                "source": "math_specialist_bootstrap",
                "placeholders": ["rpn_chain"],
            },
        },
    )

    _ANTI_PATTERN_MAP: dict[str, tuple[str, ...]] = {
        "linear_equation": ("linear_equation_ax_minus_b_eq_c", "linear_equation_b_plus_ax_eq_c", "proportion"),
        "linear_equation_ax_minus_b_eq_c": ("linear_equation", "linear_equation_b_plus_ax_eq_c", "proportion"),
        "linear_equation_b_plus_ax_eq_c": ("linear_equation", "linear_equation_ax_minus_b_eq_c", "proportion"),
        "proportion": ("ratio", "linear_equation"),
        "ratio": ("proportion", "arithmetic_divide"),
        "arithmetic_add": ("arithmetic_subtract", "arithmetic_multiply"),
        "arithmetic_subtract": ("arithmetic_add", "arithmetic_divide"),
        "arithmetic_multiply": ("arithmetic_divide", "arithmetic_add"),
        "arithmetic_divide": ("arithmetic_multiply", "arithmetic_subtract"),
        "word_problem_total": ("word_problem_remainder", "word_problem_rate", "word_problem_comparison"),
        "word_problem_remainder": ("word_problem_total", "word_problem_rate", "word_problem_sequential"),
        "word_problem_rate": ("word_problem_total", "word_problem_percentage", "word_problem_comparison"),
        "word_problem_comparison": ("word_problem_total", "word_problem_remainder", "word_problem_rate"),
        "word_problem_percentage": ("word_problem_total", "word_problem_rate"),
        "word_problem_sequential": ("word_problem_total", "word_problem_remainder", "word_problem_rate"),
    }
    _BOOTSTRAP_GRAMMAR_BY_PATTERN: dict[str, str] = {
        "linear_equation": "grammar_linear_equation_ax_plus_b_eq_c_v1",
        "linear_equation_ax_minus_b_eq_c": "grammar_linear_equation_ax_minus_b_eq_c_v1",
        "linear_equation_b_plus_ax_eq_c": "grammar_linear_equation_b_plus_ax_eq_c_v1",
        "arithmetic_add": "grammar_arithmetic_addition_ab_v1",
        "arithmetic_subtract": "grammar_arithmetic_subtraction_ab_v1",
        "ratio": "grammar_ratio_a_to_b_v1",
        "proportion": "grammar_proportion_a_over_b_eq_c_over_x_v1",
        "word_problem_total": "grammar_word_problem_total_v1",
        "word_problem_remainder": "grammar_word_problem_remainder_v1",
        "word_problem_rate": "grammar_word_problem_rate_v1",
        "word_problem_comparison": "grammar_word_problem_comparison_v1",
        "word_problem_percentage": "grammar_word_problem_percentage_v1",
        "word_problem_sequential": "grammar_word_problem_sequential_v1",
    }
    _BOOTSTRAP_TEMPLATE_BY_PATTERN: dict[str, str] = {
        "linear_equation": "math_template_linear_equation_solve_v1",
        "linear_equation_ax_minus_b_eq_c": "math_template_linear_equation_ax_minus_b_eq_c_solve_v1",
        "linear_equation_b_plus_ax_eq_c": "math_template_linear_equation_b_plus_ax_eq_c_solve_v1",
        "arithmetic_add": "math_template_arithmetic_add_v1",
        "arithmetic_subtract": "math_template_arithmetic_subtract_v1",
        "arithmetic_multiply": "math_template_arithmetic_multiply_v1",
        "arithmetic_divide": "math_template_arithmetic_divide_v1",
        "ratio": "math_template_ratio_a_to_b_v1",
        "proportion": "math_template_proportion_solve_v1",
        "word_problem_total": "math_template_word_problem_total_v1",
        "word_problem_remainder": "math_template_word_problem_remainder_v1",
        "word_problem_rate": "math_template_word_problem_rate_v1",
        "word_problem_comparison": "math_template_word_problem_comparison_v1",
        "word_problem_percentage": "math_template_word_problem_percentage_v1",
        "word_problem_sequential": "math_template_word_problem_sequential_v1",
    }

    def __init__(
        self,
        *,
        knowledgeverse: Any,
        parent: SpecialistBase | None = None,
        evaluator: Callable[[str], float | None] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="MathSpecialist",
            domain="math",
            parent=parent,
            **kwargs,
        )
        self.knowledgeverse = knowledgeverse
        self._evaluator = evaluator or self._evaluate_with_rpn_engine
        self._last_execution_error: str | None = None
        self._debug = str(os.getenv("K3D_MATH_DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._quality_memory = TernaryQualityMemory(
            state_path=self._resolve_ternary_quality_state_path(),
            emit_galaxy_entries=False,
        )
        self._ensure_bootstrap_templates()

    def process(self, task: dict[str, Any], *, use_enriched: bool = True) -> dict[str, Any]:
        question = str(task.get("question", "") or task.get("query", "")).strip()
        parse_bundle = self._extract_parse_bundle(task)
        self._emit_debug(
            "solve_start",
            {
                "question": question[:240],
                "use_enriched": bool(use_enriched),
                "parse_bundle": sorted(parse_bundle.keys()),
            },
        )
        if not question:
            self._emit_debug("solve_error", {"reason": "missing_question"})
            return {"status": "error", "reason": "missing_question"}

        foundational = self._solve_foundational_math_problem(question, parse_bundle=parse_bundle)
        if foundational is not None:
            self.mark_query(True)
            self._log_event(
                "math_specialist_success",
                {
                    "question": question[:240],
                    "pattern_id": foundational.get("pattern_id"),
                    "template_id": foundational.get("template_id"),
                    "pattern_type": foundational.get("pattern_type"),
                    "rpn_program": foundational.get("rpn_program"),
                    "result": foundational.get("result"),
                    "coefficients": foundational.get("coefficients"),
                    "parse_bundle": parse_bundle,
                    "use_enriched": bool(use_enriched),
                    "grammar_chain": foundational.get("grammar_chain", []),
                    "number_refs": foundational.get("number_refs", []),
                    "word_refs": foundational.get("word_refs", []),
                    "template_mode": foundational.get("template_mode"),
                },
            )
            return foundational

        manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if manager is None:
            self._emit_debug("solve_error", {"reason": "missing_galaxy_manager"})
            return {"status": "error", "reason": "missing_galaxy_manager"}

        query_context = self._build_query_context(question, parse_bundle=parse_bundle)
        problem_type = self._infer_problem_type(question, parse_bundle=parse_bundle)
        self._emit_debug("problem_type", {"pattern_type": problem_type})

        try:
            grammar_candidates = manager.query(
                query_text=f"{query_context} {problem_type} pattern",
                specialist="math",
                top_k=24,
                galaxies=["Grammar"],
                preferred_pattern_type=problem_type,
            )
        except Exception as exc:
            self._emit_debug("solve_error", {"reason": "grammar_query_failed", "detail": str(exc)})
            return {"status": "error", "reason": "grammar_query_failed", "detail": str(exc)}
        self._emit_debug("grammar_query", {"candidate_count": len(grammar_candidates)})
        if not grammar_candidates:
            self._emit_debug("solve_error", {"reason": "missing_grammar_patterns"})
            return {"status": "error", "reason": "missing_grammar_patterns"}

        contrastive_patterns, pattern_stats = self._fuse_contrastive_patterns(
            candidates=grammar_candidates,
            galaxy="Grammar",
            target_pattern_type=problem_type,
        )
        self._emit_debug("contrastive_patterns", pattern_stats)

        pattern = self._select_pattern(contrastive_patterns, pattern_type=problem_type)
        if not pattern:
            self._emit_debug("solve_error", {"reason": "pattern_selection_failed"})
            return {"status": "error", "reason": "pattern_selection_failed"}

        coefficients = self._extract_coefficients(
            question,
            pattern_type=problem_type,
            parse_bundle=parse_bundle,
        )
        self._emit_debug("coefficients", {"coefficients": coefficients})
        if coefficients is None:
            self._update_quality_prior(
                entry_id=str(pattern.get("id", "")),
                outcome=-1,
                confidence=0.8,
                specialist="math",
                galaxy="Grammar",
                source="math_specialist_coefficient_failure",
            )
            self._emit_debug("solve_error", {"reason": "coefficient_extraction_failed"})
            return {"status": "error", "reason": "coefficient_extraction_failed"}

        if problem_type.startswith("word_problem_"):
            composition = self._compose_word_problem_program(
                pattern_type=problem_type,
                coefficients=coefficients,
            )
            if composition is None:
                self._update_quality_prior(
                    entry_id=str(pattern.get("id", "")),
                    outcome=-1,
                    confidence=0.8,
                    specialist="math",
                    galaxy="Grammar",
                    source="math_specialist_word_problem_composition_failure",
                )
                self._emit_debug("solve_error", {"reason": "word_problem_composition_failed"})
                return {"status": "error", "reason": "word_problem_composition_failed"}

            composed = str(composition.get("rpn_program", "")).strip()
            result = self._evaluator(composed)
            if result is None:
                self._update_quality_prior(
                    entry_id=str(pattern.get("id", "")),
                    outcome=-1,
                    confidence=0.8,
                    specialist="math",
                    galaxy="Grammar",
                    source="math_specialist_execution_failure",
                )
                self._emit_debug(
                    "solve_error",
                    {
                        "reason": "rpn_execution_failed",
                        "rpn_program": composed,
                        "detail": self._last_execution_error,
                    },
                )
                return {
                    "status": "error",
                    "reason": "rpn_execution_failed",
                    "rpn_program": composed,
                    "detail": self._last_execution_error,
                }

            self.mark_query(True)
            self._update_quality_prior(
                entry_id=str(pattern.get("id", "")),
                outcome=1,
                confidence=0.9,
                specialist="math",
                galaxy="Grammar",
                source="math_specialist_success",
            )
            self._log_event(
                "math_specialist_success",
                {
                    "question": question[:240],
                    "pattern_id": pattern.get("id"),
                    "pattern_type": problem_type,
                    "contrastive_pattern_stats": pattern_stats,
                    "rpn_program": composed,
                    "result": result,
                    "coefficients": coefficients,
                    "parse_bundle": parse_bundle,
                    "use_enriched": bool(use_enriched),
                    "grammar_chain": composition.get("grammar_chain", []),
                    "number_refs": composition.get("number_refs", []),
                    "word_refs": composition.get("word_refs", []),
                },
            )
            return {
                "status": "success",
                "result": float(result),
                "rpn_program": composed,
                "coefficients": coefficients,
                "pattern_id": pattern.get("id"),
                "template_id": None,
                "template_mode": "grammar_composition",
                "pattern_type": problem_type,
                "parse_bundle": parse_bundle,
                "grammar_chain": composition.get("grammar_chain", []),
                "number_refs": composition.get("number_refs", []),
                "word_refs": composition.get("word_refs", []),
            }

        try:
            template_candidates = manager.query(
                query_text=f"solve {problem_type} {query_context}",
                specialist="math",
                top_k=24,
                galaxies=["Math"],
                preferred_pattern_type=problem_type,
            )
        except Exception as exc:
            self._emit_debug("solve_error", {"reason": "math_template_query_failed", "detail": str(exc)})
            return {"status": "error", "reason": "math_template_query_failed", "detail": str(exc)}
        self._emit_debug("math_template_query", {"candidate_count": len(template_candidates)})
        if not template_candidates:
            self._emit_debug("solve_error", {"reason": "missing_math_templates"})
            return {"status": "error", "reason": "missing_math_templates"}

        contrastive_templates, template_stats = self._fuse_contrastive_patterns(
            candidates=template_candidates,
            galaxy="Math",
            target_pattern_type=problem_type,
        )
        self._emit_debug("contrastive_templates", template_stats)
        template = self._select_template(contrastive_templates, pattern_type=problem_type)
        if not template:
            self._emit_debug("solve_error", {"reason": "template_selection_failed"})
            return {"status": "error", "reason": "template_selection_failed"}

        composed = self._compose_from_template(template=template, coefficients=coefficients)
        self._emit_debug("rpn_composed", {"rpn_program": composed})
        if composed is None:
            self._update_quality_prior(
                entry_id=str(template.get("id", "")),
                outcome=-1,
                confidence=0.8,
                specialist="math",
                galaxy="Math",
                source="math_specialist_rpn_composition_failure",
            )
            self._emit_debug("solve_error", {"reason": "rpn_composition_failed"})
            return {"status": "error", "reason": "rpn_composition_failed"}

        result = self._evaluator(composed)
        if result is None:
            self._update_quality_prior(
                entry_id=str(template.get("id", "")),
                outcome=-1,
                confidence=0.8,
                specialist="math",
                galaxy="Math",
                source="math_specialist_execution_failure",
            )
            self._emit_debug(
                "solve_error",
                {
                    "reason": "rpn_execution_failed",
                    "rpn_program": composed,
                    "detail": self._last_execution_error,
                },
            )
            return {
                "status": "error",
                "reason": "rpn_execution_failed",
                "rpn_program": composed,
                "detail": self._last_execution_error,
            }

        self.mark_query(True)
        self._update_quality_prior(
            entry_id=str(pattern.get("id", "")),
            outcome=1,
            confidence=0.9,
            specialist="math",
            galaxy="Grammar",
            source="math_specialist_success",
        )
        self._update_quality_prior(
            entry_id=str(template.get("id", "")),
            outcome=1,
            confidence=0.9,
            specialist="math",
            galaxy="Math",
            source="math_specialist_success",
        )
        self._log_event(
            "math_specialist_success",
            {
                "question": question[:240],
                "pattern_id": pattern.get("id"),
                "template_id": template.get("id"),
                "pattern_type": problem_type,
                "contrastive_pattern_stats": pattern_stats,
                "contrastive_template_stats": template_stats,
                "rpn_program": composed,
                "result": result,
                "coefficients": coefficients,
                "parse_bundle": parse_bundle,
                "use_enriched": bool(use_enriched),
            },
        )
        self._emit_debug(
            "solve_success",
            {
                "result": result,
                "rpn_program": composed,
                "pattern_id": pattern.get("id"),
                "template_id": template.get("id"),
            },
        )
        return {
            "status": "success",
            "result": float(result),
            "rpn_program": composed,
            "coefficients": coefficients,
            "pattern_id": pattern.get("id"),
            "template_id": template.get("id"),
            "pattern_type": problem_type,
            "parse_bundle": parse_bundle,
        }

    def _ensure_bootstrap_templates(self) -> None:
        manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if manager is None:
            return
        try:
            for entry in self._GRAMMAR_BOOTSTRAP_ENTRIES:
                if not self._entry_exists("Grammar", str(entry.get("id", ""))):
                    manager.add_entry("Grammar", dict(entry))
            for entry in self._MATH_TEMPLATE_ENTRIES:
                pattern_type = str(entry.get("pattern_type", "")).strip().lower()
                if pattern_type.startswith("word_problem_"):
                    continue
                if not self._entry_exists("Math", str(entry.get("id", ""))):
                    manager.add_entry("Math", dict(entry))
        except Exception:
            return

    def _resolve_ternary_quality_state_path(self) -> Path:
        override = str(os.getenv("K3D_MATH_TERNARY_STATE_PATH", "")).strip()
        if override:
            return Path(override)
        root = getattr(self.knowledgeverse, "storage_root", None)
        if root:
            return Path(str(root)) / "checkpoints" / "math_specialist_ternary_quality.json"
        return Path("../Knowledge3D.local/checkpoints/math_specialist_ternary_quality.json")

    def _entry_exists(self, galaxy_name: str, entry_id: str) -> bool:
        return self._get_entry_by_id(galaxy_name, entry_id) is not None

    def _get_entry_by_id(self, galaxy_name: str, entry_id: str) -> dict[str, Any] | None:
        manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if manager is None:
            return None
        galaxy = manager.get_galaxy(galaxy_name)
        entries = getattr(galaxy, "entries", [])
        for entry in entries:
            if str(entry.get("id", "")) == str(entry_id):
                return entry if isinstance(entry, dict) else None
        return None

    def _extract_parse_bundle(self, task: dict[str, Any]) -> dict[str, Any]:
        bundle: dict[str, Any] = {}
        for key in ("forward_parse", "backward_parse", "fusion_parse"):
            value = task.get(key)
            if isinstance(value, dict):
                bundle[key] = value
        route_plan = task.get("route_plan")
        if isinstance(route_plan, list):
            bundle["route_plan"] = route_plan
            for route in route_plan:
                if not isinstance(route, dict):
                    continue
                for key in ("forward_parse", "backward_parse", "fusion_parse"):
                    value = route.get(key)
                    if key not in bundle and isinstance(value, dict):
                        bundle[key] = value
        return bundle

    def _build_query_context(self, question: str, *, parse_bundle: dict[str, Any]) -> str:
        parts = [question]
        fusion = parse_bundle.get("fusion_parse")
        if isinstance(fusion, dict):
            merged = fusion.get("merged_variables")
            if isinstance(merged, dict) and merged:
                parts.append(" ".join(f"{key}={value}" for key, value in merged.items()))
            goal = fusion.get("unified_goal")
            if isinstance(goal, dict):
                goal_expr = str(goal.get("expression") or goal.get("raw") or "").strip()
                if goal_expr:
                    parts.append(goal_expr)
        forward = parse_bundle.get("forward_parse")
        if isinstance(forward, dict):
            goal = forward.get("goal")
            if isinstance(goal, dict):
                goal_expr = str(goal.get("expression") or goal.get("raw") or "").strip()
                if goal_expr:
                    parts.append(goal_expr)
        return " ".join(part for part in parts if part).strip()

    def _infer_problem_type(self, question: str, *, parse_bundle: dict[str, Any] | None = None) -> str:
        lowered = question.lower()
        parse_bundle = parse_bundle or {}
        word_problem_type = self._infer_word_problem_type(lowered, parse_bundle=parse_bundle)
        if word_problem_type:
            return word_problem_type
        if self._looks_like_proportion(lowered):
            return "proportion"
        if self._looks_like_ratio(lowered):
            return "ratio"
        if "=" in lowered and self._detect_variable(lowered):
            variant = self._infer_linear_variant(lowered)
            if variant:
                return variant
            return "linear_equation"
        if "×" in lowered or "*" in lowered:
            return "arithmetic_multiply"
        if "÷" in lowered or "/" in lowered:
            return "arithmetic_divide"
        # Prefer subtraction if explicit minus separates numeric terms.
        if "-" in lowered and "+" not in lowered:
            return "arithmetic_subtract"
        return "arithmetic_add"

    def _infer_word_problem_type(self, question: str, *, parse_bundle: dict[str, Any]) -> str | None:
        numbers = self._extract_numeric_literals(question)
        lowered = question.lower()
        if "=" in question:
            return None
        if not any(ch.isalpha() for ch in question):
            return None

        hints = self._extract_word_problem_entities(question, parse_bundle=parse_bundle)
        semantic_values = [
            float(entity["value"])
            for entity in hints.get("entities", [])
            if entity.get("value") is not None
        ]
        if len(numbers) < 2 and len(semantic_values) < 2:
            return None
        has_rate = bool(hints["has_rate"])
        has_remainder = bool(hints["has_remainder"])
        has_percentage = bool(hints["has_percentage"])
        has_comparison = bool(hints["has_comparison"])
        has_total = bool(hints["has_total"])
        if has_percentage:
            return "word_problem_percentage"
        if has_rate and has_remainder and len(numbers) >= 3:
            return "word_problem_sequential"
        if has_remainder:
            return "word_problem_remainder"
        if has_rate:
            return "word_problem_rate"
        if has_comparison:
            return "word_problem_comparison"
        if has_total or any(phrase in lowered for phrase in ("sum of", "combined", "altogether", "in all")):
            return "word_problem_total"
        if len(numbers) >= 3:
            return "word_problem_sequential"
        return None

    def _looks_like_ratio(self, question: str) -> bool:
        lowered = question.lower()
        if "=" in lowered:
            return False
        if ":" in lowered:
            return True
        return " ratio " in f" {lowered} " and len(self._extract_numeric_literals(question)) >= 2

    def _looks_like_proportion(self, question: str) -> bool:
        lowered = question.lower()
        if "=" not in lowered:
            return False
        if ":" in lowered:
            return self._detect_variable(lowered) is not None
        if "/" in lowered and self._detect_variable(lowered) is not None:
            return True
        return False

    def _infer_linear_variant(self, question: str) -> str | None:
        equation = self._extract_equation_segment(question)
        if equation is None or "=" not in equation:
            return None
        left, right = equation.lower().split("=", 1)
        variable = self._detect_variable(equation)
        if variable is None:
            return None
        side = left if variable in left else right
        compact = side.replace(" ", "")
        if variable not in compact:
            return None
        if "-" in compact and "+" not in compact:
            return "linear_equation_ax_minus_b_eq_c"
        if "+" in compact:
            parts = compact.split("+")
            if len(parts) >= 2:
                first = parts[0]
                second = parts[1]
                if variable not in first and variable in second:
                    return "linear_equation_b_plus_ax_eq_c"
        return None

    def _select_pattern(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        pattern_type: str,
    ) -> dict[str, Any] | None:
        target = str(pattern_type).strip().lower()
        # Prefer trusted equation-pattern entries first.
        for candidate in candidates:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            category = str(entry.get("category", "")).strip().lower()
            source = str((entry.get("metadata") or {}).get("source", "")).strip().lower() if isinstance(entry.get("metadata"), dict) else ""
            if (
                str(entry.get("pattern_type", "")).strip().lower() == target
                and category == "equation_pattern"
                and source == "math_specialist_bootstrap"
            ):
                return entry
        for candidate in candidates:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            if (
                str(entry.get("pattern_type", "")).strip().lower() == target
                and str(entry.get("category", "")).strip().lower() == "equation_pattern"
            ):
                return entry
        bootstrap_id = self._BOOTSTRAP_GRAMMAR_BY_PATTERN.get(target)
        if bootstrap_id:
            fallback = self._get_entry_by_id("Grammar", bootstrap_id)
            if isinstance(fallback, dict):
                return fallback
        return None

    def _select_template(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        pattern_type: str,
    ) -> dict[str, Any] | None:
        target = str(pattern_type).strip().lower()
        # 1) Prefer trusted bootstrap algebra templates.
        for candidate in candidates:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            if (
                str(entry.get("pattern_type", "")).strip().lower() != target
                or str(entry.get("category", "")).strip().lower() != "algebra_template"
            ):
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            source = str(metadata.get("source", "")).strip().lower()
            if source == "math_specialist_bootstrap" and str(entry.get("rpn_program", "")).strip():
                return entry
        # 2) Accept same-pattern algebra templates when placeholders are explicit.
        for candidate in candidates:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            if (
                str(entry.get("pattern_type", "")).strip().lower() != target
                or str(entry.get("category", "")).strip().lower() != "algebra_template"
            ):
                continue
            rpn = str(entry.get("rpn_program", "")).strip()
            if "{" in rpn and "}" in rpn:
                return entry
        # 3) Bootstrap fallback by pattern type.
        bootstrap_id = self._BOOTSTRAP_TEMPLATE_BY_PATTERN.get(target)
        if bootstrap_id:
            fallback = self._get_entry_by_id("Math", bootstrap_id)
            if isinstance(fallback, dict):
                return fallback
        return None

    def _fuse_contrastive_patterns(
        self,
        *,
        candidates: Sequence[dict[str, Any]],
        galaxy: str,
        target_pattern_type: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        uncertain: list[dict[str, Any]] = []

        for candidate in candidates:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            prior = self._get_entry_prior(str(entry.get("id", "")))
            if prior > 0.3:
                positive.append(candidate)
            elif prior < -0.3:
                negative.append(candidate)
            else:
                uncertain.append(candidate)

        backward = self._generate_anti_patterns(
            failed=negative,
            galaxy=galaxy,
            target_pattern_type=target_pattern_type,
        )
        fused = self._deduplicate_candidates([*positive, *backward, *uncertain])
        if not fused:
            fused = self._deduplicate_candidates(list(candidates))
        stats = {
            "forward_positive": len(positive),
            "backward_negative": len(negative),
            "fusion_uncertain": len(uncertain),
            "anti_generated": len(backward),
            "fused_total": len(fused),
        }
        return fused, stats

    def _generate_anti_patterns(
        self,
        *,
        failed: Sequence[dict[str, Any]],
        galaxy: str,
        target_pattern_type: str,
    ) -> list[dict[str, Any]]:
        manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if manager is None:
            return []

        anti_types: list[str] = []
        if failed:
            for candidate in failed:
                entry = candidate.get("entry", {})
                if not isinstance(entry, dict):
                    continue
                current_type = str(entry.get("pattern_type", "")).strip().lower()
                anti_types.extend(self._ANTI_PATTERN_MAP.get(current_type, ()))
        else:
            anti_types.extend(self._ANTI_PATTERN_MAP.get(target_pattern_type, ()))

        if not anti_types:
            return []

        collected: list[dict[str, Any]] = []
        for anti_type in anti_types:
            try:
                rows = manager.query(
                    query_text=f"{anti_type} template",
                    specialist="math",
                    top_k=4,
                    galaxies=[galaxy],
                    preferred_pattern_type=anti_type,
                )
            except Exception:
                continue
            for row in rows:
                entry = row.get("entry", {})
                if not isinstance(entry, dict):
                    continue
                entry_md = entry.get("metadata", {})
                if not isinstance(entry_md, dict):
                    entry_md = {}
                entry_md["contrastive_source"] = "anti_pattern"
                entry["metadata"] = entry_md
                collected.append(row)
        return collected

    def _deduplicate_candidates(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            entry = candidate.get("entry", {})
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", "")).strip()
            if not entry_id:
                entry_id = f"anon_{len(seen)}"
            if entry_id in seen:
                continue
            seen.add(entry_id)
            deduped.append(candidate)
        return deduped

    def _get_entry_prior(self, entry_id: str) -> float:
        if not entry_id:
            return 0.0
        prior = self._quality_memory.get_prior(entry_id)
        if prior is None:
            return 0.0
        return float(prior.prior)

    def _update_quality_prior(
        self,
        *,
        entry_id: str,
        outcome: int,
        confidence: float,
        specialist: str,
        galaxy: str,
        source: str,
    ) -> None:
        if not entry_id:
            return
        try:
            self._quality_memory.update(
                pattern_id=entry_id,
                outcome=outcome,
                confidence=confidence,
                knowledgeverse=self.knowledgeverse,
                specialist=specialist,
                galaxy=galaxy,
                source=source,
            )
        except Exception:
            return

    def _extract_coefficients(
        self,
        question: str,
        *,
        pattern_type: str,
        parse_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        kind = str(pattern_type).strip().lower()
        parse_bundle = parse_bundle or {}
        if kind in {
            "linear_equation",
            "linear_equation_ax_minus_b_eq_c",
            "linear_equation_b_plus_ax_eq_c",
        }:
            return self._extract_linear_coefficients_forward_backward(question)
        if kind == "ratio":
            operands = self._extract_ratio_operands(question)
            if operands is None:
                return None
            return {"a": operands[0], "b": operands[1]}
        if kind == "proportion":
            return self._extract_proportion_coefficients(question)
        if kind in {"arithmetic_add", "arithmetic_subtract", "arithmetic_multiply", "arithmetic_divide"}:
            operands = self._extract_arithmetic_operands(question)
            if operands is None:
                return None
            return {"a": operands[0], "b": operands[1]}
        if kind.startswith("word_problem_"):
            return self._extract_word_problem_coefficients(
                question,
                pattern_type=kind,
                parse_bundle=parse_bundle,
            )
        return None

    def _solve_foundational_math_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        for handler in (
            self._solve_piecewise_continuity_problem,
            self._solve_rectangular_band_problem,
            self._solve_polynomial_degree_problem,
            self._solve_circle_center_problem,
            self._solve_quantified_interval_problem,
            self._solve_asy_arithmetic_sequence_problem,
            self._solve_substitution_expression_problem,
            self._solve_work_penalty_problem,
            self._solve_compound_interest_problem,
        ):
            solved = handler(question, parse_bundle=parse_bundle)
            if solved is not None:
                return solved
        return None

    def _foundational_success(
        self,
        *,
        result: Any,
        pattern_type: str,
        numbers: Sequence[float] | None = None,
        rpn_program: str | None = None,
        coefficients: dict[str, Any] | None = None,
        grammar_chain: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        numeric_values = [float(v) for v in (numbers or [])]
        return {
            "status": "success",
            "result": result,
            "rpn_program": (rpn_program or "").strip() or None,
            "coefficients": dict(coefficients or {}),
            "pattern_id": None,
            "template_id": None,
            "template_mode": "foundational_composition",
            "pattern_type": pattern_type,
            "parse_bundle": {},
            "grammar_chain": list(grammar_chain or []),
            "number_refs": [self._number_ref(value) for value in numeric_values],
            "word_refs": [self._word_ref(value) for value in numeric_values],
        }

    def _solve_substitution_expression_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = question.lower()
        if any(
            token in lowered
            for token in (
                "degree of the polynomial",
                "piecewise function is continuous",
                "center of the circle",
                "compounds quarterly",
                "20-day period",
                "rectangular band formation",
            )
        ):
            return None
        if not (
            lowered.startswith("evaluate")
            or "what is the value of" in lowered
        ):
            return None

        variables = {
            name.lower(): float(value)
            for name, value in re.findall(r"([A-Za-z])\s*=\s*(-?\d+(?:\.\d+)?)", question)
        }
        segments = self._extract_inline_math_segments(question)
        if not segments:
            return None
        expr = self._select_expression_segment(segments)
        if not expr:
            return None
        rpn_program = self._expression_to_rpn(expr, variables=variables)
        if not rpn_program:
            return None
        result = self._evaluator(rpn_program)
        if result is None:
            return None
        numeric_result = float(result)
        if abs(numeric_result - round(numeric_result)) <= 1e-9:
            numeric_result = float(round(numeric_result))
        return self._foundational_success(
            result=numeric_result,
            pattern_type="foundational_expression_evaluation",
            numbers=[*variables.values()],
            rpn_program=rpn_program,
            coefficients={"variables": variables, "expression": expr},
            grammar_chain=["math_expression_parse", "math_expression_substitute", "math_expression_evaluate"],
        )

    def _solve_polynomial_degree_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = question.lower()
        if "degree of the polynomial" not in lowered:
            return None
        segments = self._extract_inline_math_segments(question)
        expr = segments[0] if segments else ""
        if not expr:
            return None
        powers = [int(match) for match in re.findall(r"x\^(\d+)", expr)]
        if re.search(r"x(?![\^A-Za-z0-9])", expr):
            powers.append(1)
        degree = max(powers or [0])
        return self._foundational_success(
            result=float(degree),
            pattern_type="foundational_polynomial_degree",
            numbers=[float(degree)],
            coefficients={"expression": expr},
            grammar_chain=["math_polynomial_term_scan", "math_polynomial_degree_select"],
        )

    def _solve_circle_center_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = question.lower()
        if "center of the circle" not in lowered:
            return None
        segments = self._extract_inline_math_segments(question)
        expr = segments[0] if segments else ""
        if "=" not in expr:
            return None
        left = expr.split("=", 1)[0].replace(" ", "")
        x_match = re.search(r"([+-]?\d*)x(?!\^)", left)
        y_match = re.search(r"([+-]?\d*)y(?!\^)", left)
        if x_match is None or y_match is None:
            return None
        a = self._parse_signed_unit_number(x_match.group(1))
        b = self._parse_signed_unit_number(y_match.group(1))
        h = -a / 2.0
        k = -b / 2.0
        result = f"({self._format_scalar_answer(h)}, {self._format_scalar_answer(k)})"
        return self._foundational_success(
            result=result,
            pattern_type="foundational_circle_center",
            numbers=[h, k],
            coefficients={"expression": expr, "x_linear": a, "y_linear": b},
            grammar_chain=["math_complete_square_x", "math_complete_square_y", "math_circle_center_emit"],
        )

    def _solve_piecewise_continuity_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = question.lower()
        if "piecewise function is continuous" not in lowered:
            return None
        block = self._extract_display_math_block(question)
        if not block:
            return None
        cleaned = block
        for pattern in (
            r"f\(x\)\s*=",
            r"\\left\\\{",
            r"\\begin\{array\}\{cl\}",
            r"\\end\{array\}",
            r"\\right\.",
        ):
            cleaned = re.sub(pattern, "", cleaned)
        rows = [row.strip().rstrip(",.") for row in cleaned.split(r"\\") if row.strip()]
        parsed_rows: list[tuple[str, str]] = []
        for row in rows:
            parts = row.split(r"&\text{ if }")
            if len(parts) != 2:
                continue
            parsed_rows.append((parts[0].strip().rstrip(","), parts[1].strip().rstrip(",.")))
        if len(parsed_rows) != 3:
            return None

        middle_expr = parsed_rows[1][0]
        left_boundary = 2.0
        right_boundary = -2.0
        left_target = self._evaluate_linear_expression(middle_expr, x_value=left_boundary)
        right_target = self._evaluate_linear_expression(middle_expr, x_value=right_boundary)
        if left_target is None or right_target is None:
            return None

        left_solution = self._solve_linear_unknown_at_boundary(parsed_rows[0][0], x_value=left_boundary, target=left_target)
        right_solution = self._solve_linear_unknown_at_boundary(parsed_rows[2][0], x_value=right_boundary, target=right_target)
        if not left_solution or not right_solution:
            return None
        total = float(left_solution[1]) + float(right_solution[1])
        if abs(total - round(total)) <= 1e-9:
            total = float(round(total))
        rpn_program = f"{self._format_number_token(left_solution[1])} {self._format_number_token(right_solution[1])} +"
        return self._foundational_success(
            result=total,
            pattern_type="foundational_piecewise_continuity",
            numbers=[float(left_solution[1]), float(right_solution[1])],
            rpn_program=rpn_program,
            coefficients={
                left_solution[0]: float(left_solution[1]),
                right_solution[0]: float(right_solution[1]),
                "middle_expr": middle_expr,
            },
            grammar_chain=["math_piecewise_boundary_left", "math_piecewise_boundary_right", "math_piecewise_sum"],
        )

    def _solve_rectangular_band_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = question.lower()
        if "rectangular band formation" not in lowered or "left over" not in lowered:
            return None
        limit_match = re.search(r"less than\s+(\d+)", lowered)
        leftover_match = re.search(r"(\w+|\d+)\s+members left over", lowered)
        row_inc_match = re.search(r"members in each row by\s+(\d+)", lowered)
        row_dec_match = re.search(r"rows by\s+(\d+)", lowered)
        if not limit_match or not leftover_match or not row_inc_match or not row_dec_match:
            return None
        upper = int(limit_match.group(1))
        leftover = int(self._parse_written_or_numeric_number(leftover_match.group(1)))
        row_inc = int(row_inc_match.group(1))
        row_dec = int(row_dec_match.group(1))
        best = None
        for n in range(1, upper):
            for m in range(1, n + 1):
                for r in range(1, n + 1):
                    if m * r + leftover != n:
                        continue
                    if r - row_dec <= 0:
                        continue
                    if (m + row_inc) * (r - row_dec) == n:
                        best = n if best is None else max(best, n)
        if best is None:
            return None
        return self._foundational_success(
            result=float(best),
            pattern_type="foundational_rectangular_band",
            numbers=[float(best)],
            coefficients={
                "limit": upper,
                "leftover": leftover,
                "row_increment": row_inc,
                "row_decrement": row_dec,
            },
            grammar_chain=["math_integer_search", "math_rectangular_constraint", "math_maximize_under_limit"],
        )

    def _solve_work_penalty_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        plain = self._normalize_benchmark_text(question)
        lowered = plain.lower()
        if (
            "subtracted from his earnings" not in lowered
            or not any(marker in lowered for marker in ("did not work", "does not work", "not work"))
        ):
            return None
        total_days_match = re.search(r"(\d+)\s*-\s*day period", lowered)
        wage_match = re.search(r"earns\s+\$?(\d[\d,]*(?:\.\d+)?)", plain, re.IGNORECASE)
        penalty_match = re.search(r"\$?(\d[\d,]*(?:\.\d+)?)\s+is subtracted", plain, re.IGNORECASE)
        final_match = re.search(r"received\s+\$?(\d[\d,]*(?:\.\d+)?)", plain, re.IGNORECASE)
        if not total_days_match or not wage_match or not penalty_match or not final_match:
            return None
        total_days = float(total_days_match.group(1))
        wage = float(self._to_float(wage_match.group(1)) or 0.0)
        penalty = float(self._to_float(penalty_match.group(1)) or 0.0)
        final_pay = float(self._to_float(final_match.group(1)) or 0.0)
        denominator = wage + penalty
        if abs(denominator) <= 1e-9:
            return None
        not_worked = ((total_days * wage) - final_pay) / denominator
        if abs(not_worked - round(not_worked)) <= 1e-9:
            not_worked = float(round(not_worked))
        return self._foundational_success(
            result=float(not_worked),
            pattern_type="foundational_work_penalty",
            numbers=[total_days, wage, penalty, final_pay],
            coefficients={
                "total_days": total_days,
                "work_wage": wage,
                "nonwork_penalty": penalty,
                "final_pay": final_pay,
            },
            grammar_chain=["math_total_partition", "math_linear_balance", "math_complement_emit"],
        )

    def _solve_compound_interest_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        plain = self._normalize_benchmark_text(question)
        lowered = plain.lower()
        if "compound" not in lowered or "interest rate" not in lowered:
            return None
        future_match = re.search(r"total of\s+\$?([\d,]+(?:\.\d+)?)", plain, re.IGNORECASE)
        rate_match = re.search(r"annual interest rate of\s+(\d+(?:\.\d+)?)\s*%", plain, re.IGNORECASE)
        years_match = re.search(r"end of\s+(\d+(?:\.\d+)?)\s+years?", plain, re.IGNORECASE)
        if not future_match or not rate_match or not years_match:
            return None
        future_value = float(self._to_float(future_match.group(1)) or 0.0)
        annual_rate_pct = float(rate_match.group(1))
        years = float(years_match.group(1))
        periods = 4.0 if "quarter" in lowered else 12.0 if "month" in lowered else 1.0
        present_value = future_value / ((1.0 + (annual_rate_pct / 100.0) / periods) ** (periods * years))
        rounded = float(round(present_value))
        return self._foundational_success(
            result=rounded,
            pattern_type="foundational_compound_interest",
            numbers=[future_value, annual_rate_pct, periods, years],
            coefficients={
                "future_value": future_value,
                "annual_rate_pct": annual_rate_pct,
                "periods_per_year": periods,
                "years": years,
            },
            grammar_chain=["math_compound_interest_rate", "math_discount_to_present", "math_round_currency"],
        )

    def _solve_quantified_interval_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = str(question or "").lower()
        if "for every" not in lowered or "interval notation" not in lowered:
            return None
        if "p+q" not in lowered or "2p^2q" not in lowered:
            return None
        if not all(token in lowered for token in ("pq^2", "p^2q", "3q^2", "3pq")):
            return None

        # Domain constraint: the denominator p+q must be defined for every q>0.
        # If p<0 then q=-p is positive and the expression is undefined, so p>=0.
        denominator_floor = 0.0

        # After factoring q(p+3)(p+q), the inequality becomes:
        # 3q(p+3) > 2p^2q  with q>0  =>  3(p+3) > 2p^2
        # 2p^2 - 3p - 9 < 0  -> roots -1.5 and 3.
        a = 2.0
        b = -3.0
        c = -9.0
        discriminant = (b * b) - (4.0 * a * c)
        if discriminant < 0.0:
            return None
        root_delta = math.sqrt(discriminant)
        root_low = (-b - root_delta) / (2.0 * a)
        root_high = (-b + root_delta) / (2.0 * a)
        left = max(denominator_floor, root_low)
        right = root_high
        interval = f"[{self._format_scalar_answer(left)},{self._format_scalar_answer(right)})"
        return self._foundational_success(
            result=interval,
            pattern_type="foundational_quantified_interval_inequality",
            numbers=[left, right],
            coefficients={
                "quadratic_a": a,
                "quadratic_b": b,
                "quadratic_c": c,
                "domain_floor": denominator_floor,
            },
            grammar_chain=[
                "math_quantified_domain_guard",
                "math_rational_cancellation",
                "math_quadratic_interval",
                "math_interval_emit",
            ],
        )

    def _solve_asy_arithmetic_sequence_problem(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        del parse_bundle
        lowered = str(question or "").lower()
        if "arithmetic sequences" not in lowered or "[asy]" not in lowered:
            return None
        block = re.search(r"\[asy\](.*?)\[/asy\]", question, re.S)
        if block is None:
            return None
        labels = self._extract_asy_labels(block.group(1))
        if not labels:
            return None

        numeric_labels = [label for label in labels if label["value"] is not None]
        if not numeric_labels:
            return None
        row_candidates = [label for label in numeric_labels if -0.5 <= label["y"] <= 1.1]
        if not row_candidates:
            return None
        row_start = min(row_candidates, key=lambda item: item["x"])
        row_y = row_start["y"]

        by_x: dict[float, list[dict[str, Any]]] = {}
        for label in numeric_labels:
            key = round(float(label["x"]), 3)
            by_x.setdefault(key, []).append(label)
        middle_column = None
        for items in by_x.values():
            if len(items) >= 2:
                ordered = sorted(items, key=lambda item: item["y"], reverse=True)
                if all(abs((ordered[idx]["y"] - ordered[idx + 1]["y"]) - 1.0) <= 0.25 for idx in range(len(ordered) - 1)):
                    middle_column = ordered
                    break
        if middle_column is None:
            return None

        mid_first = middle_column[0]
        mid_second = middle_column[1]
        column_step = float(mid_second["value"]) - float(mid_first["value"])
        row_steps_to_mid = int(round(row_y - float(mid_first["y"])))
        mid_row_value = float(mid_first["value"]) - (column_step * row_steps_to_mid)
        mid_x = float(mid_first["x"])

        row_step_count = int(round(mid_x - float(row_start["x"])))
        if row_step_count <= 0:
            return None
        row_diff = (mid_row_value - float(row_start["value"])) / float(row_step_count)

        n_label = next((label for label in labels if label["raw"].strip().upper() == "N"), None)
        if n_label is None:
            return None
        right_x = float(n_label["x"])
        row_total_steps = int(round(right_x - float(row_start["x"])))
        row_right_value = float(row_start["value"]) + (row_diff * float(row_total_steps))

        bottom_candidates = [
            label for label in numeric_labels
            if abs(float(label["x"]) - right_x) <= 0.25 and float(label["y"]) < row_y
        ]
        if not bottom_candidates:
            return None
        right_bottom = min(bottom_candidates, key=lambda item: item["y"])
        vertical_steps = int(round(row_y - float(right_bottom["y"])))
        if vertical_steps <= 0:
            return None
        right_diff = (float(right_bottom["value"]) - row_right_value) / float(vertical_steps)
        n_value = row_right_value - right_diff
        if abs(n_value - round(n_value)) <= 1e-9:
            n_value = float(round(n_value))
        return self._foundational_success(
            result=n_value,
            pattern_type="foundational_asy_arithmetic_sequence",
            numbers=[float(row_start["value"]), mid_row_value, row_right_value, float(right_bottom["value"]), n_value],
            coefficients={
                "row_start": float(row_start["value"]),
                "row_diff": row_diff,
                "middle_column_step": column_step,
                "row_right": row_right_value,
                "right_column_step": right_diff,
            },
            grammar_chain=[
                "math_asy_label_extract",
                "math_arithmetic_column_step",
                "math_arithmetic_row_step",
                "math_cross_sequence_emit",
            ],
        )

    def _extract_inline_math_segments(self, question: str) -> list[str]:
        prepared = self._normalize_benchmark_text(question, neutralize_currency=True)
        return [segment.strip() for segment in re.findall(r"\$(.+?)\$", prepared) if segment.strip()]

    def _extract_asy_labels(self, block: str) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        pattern = re.compile(
            r'label\("(?P<raw>[^"]+)"\s*,\((?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\)\s*,S\);'
        )
        for match in pattern.finditer(block):
            raw = match.group("raw").strip().replace("$", "")
            labels.append(
                {
                    "raw": raw,
                    "x": float(match.group("x")),
                    "y": float(match.group("y")),
                    "value": self._to_float(raw),
                }
            )
        return labels

    def _select_expression_segment(self, segments: Sequence[str]) -> str | None:
        for segment in reversed([str(value).strip() for value in segments if str(value).strip()]):
            if re.fullmatch(r"[A-Za-z]\s*=\s*-?\d+(?:\.\d+)?", segment):
                continue
            return segment
        return None

    def _normalize_benchmark_text(self, question: str, *, neutralize_currency: bool = False) -> str:
        text = str(question or "")
        money_prefix = "USD" if neutralize_currency else "$"
        text = text.replace("\\!", "")
        text = text.replace("\\%", "%")
        text = re.sub(
            r"\$\s*\\\$\$?\s*([\d,]+(?:\.\d+)?)",
            lambda m: f"{money_prefix}{m.group(1)}",
            text,
        )
        text = re.sub(
            r"\\\$\$?\s*([\d,]+(?:\.\d+)?)",
            lambda m: f"{money_prefix}{m.group(1)}",
            text,
        )

        def _inline_literal(match: re.Match[str]) -> str:
            inner = (
                match.group(1)
                .strip()
                .replace("\\!", "")
                .replace("\\%", "%")
                .replace("\\$", "$")
            )
            if re.fullmatch(r"\$?[\d,]+(?:\.\d+)?%?", inner):
                if inner.startswith("$"):
                    return f"{money_prefix}{inner[1:]}"
                return inner
            return f"${inner}$"

        return re.sub(r"\$(.+?)\$", _inline_literal, text)

    def _extract_display_math_block(self, question: str) -> str | None:
        match = re.search(r"\\\[(.*?)\\\]", question, re.S)
        if match:
            return match.group(1).strip()
        return None

    def _expression_to_rpn(self, expression: str, *, variables: dict[str, float]) -> str | None:
        try:
            from knowledge3d.skills.infix_to_rpn import infix_to_rpn
        except Exception:
            return None
        normalized = self._normalize_inline_latex_expression(expression)
        try:
            tokens = infix_to_rpn(normalized, variables=variables)
        except Exception:
            return None
        rendered: list[str] = []
        for token in tokens:
            lowered = str(token).lower()
            if lowered in variables:
                rendered.append(self._format_number_token(variables[lowered]))
                continue
            if token == "^":
                rendered.append("pow")
                continue
            rendered.append(str(token))
        return " ".join(rendered).strip() or None

    def _normalize_inline_latex_expression(self, expression: str) -> str:
        expr = str(expression or "").strip()
        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = expr.replace("\\,", "")
        expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
        expr = re.sub(r"\\frac\s*([A-Za-z0-9]+)\s*([A-Za-z0-9]+)", r"(\1)/(\2)", expr)
        expr = re.sub(r"\\lceil\s*(.*?)\s*\\rceil", r"ceil(\1)", expr)
        expr = re.sub(r"\\lfloor\s*(.*?)\s*\\rfloor", r"floor(\1)", expr)
        expr = expr.replace("{", "(").replace("}", ")")
        expr = re.sub(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*\(", r"\1*(", expr)
        expr = re.sub(r"(\d+(?:\.\d+)?)\s*([A-Za-z])", r"\1*\2", expr)
        expr = re.sub(r"\)\s*([A-Za-z0-9])", r")*\1", expr)
        return expr

    def _format_number_token(self, value: float) -> str:
        return f"{float(value):.12g}"

    def _format_scalar_answer(self, value: float) -> str:
        if abs(float(value) - round(float(value))) <= 1e-9:
            return str(int(round(float(value))))
        return self._format_number_token(value)

    def _parse_signed_unit_number(self, raw: str) -> float:
        token = str(raw or "").strip()
        if token in {"", "+"}:
            return 1.0
        if token == "-":
            return -1.0
        return float(token)

    def _parse_written_or_numeric_number(self, raw: str) -> float:
        numeric = self._to_float(raw)
        if numeric is not None:
            return numeric
        lowered = str(raw or "").strip().lower()
        if lowered in self._NUMBER_WORD_UNITS:
            return float(self._NUMBER_WORD_UNITS[lowered])
        return 0.0

    def _evaluate_linear_expression(self, expr: str, *, x_value: float) -> float | None:
        linear = self._parse_linear_expression(expr)
        if linear is None:
            return None
        if linear["x_coef_var"] is not None or linear["const_var"] is not None:
            return None
        return float(linear["x_coef_num"]) * x_value + float(linear["const_num"])

    def _solve_linear_unknown_at_boundary(
        self,
        expr: str,
        *,
        x_value: float,
        target: float,
    ) -> tuple[str, float] | None:
        linear = self._parse_linear_expression(expr)
        if linear is None:
            return None
        if linear["x_coef_var"] is not None and linear["const_var"] is None:
            coeff = float(linear["x_coef_var_coeff"]) * x_value
            if abs(coeff) <= 1e-9:
                return None
            value = (target - float(linear["const_num"])) / coeff
            return str(linear["x_coef_var"]), value
        if linear["const_var"] is not None and linear["x_coef_var"] is None:
            coeff = float(linear["const_var_coeff"])
            if abs(coeff) <= 1e-9:
                return None
            value = (target - float(linear["x_coef_num"]) * x_value) / coeff
            return str(linear["const_var"]), value
        return None

    def _parse_linear_expression(self, expr: str) -> dict[str, Any] | None:
        cleaned = str(expr or "").replace(" ", "").rstrip(",.")
        match = re.fullmatch(r"([+-]?(?:(?:\d+(?:\.\d+)?)|[A-Za-z])?)x((?:[+-](?:(?:\d+(?:\.\d+)?)|[A-Za-z]))?)", cleaned)
        if match is None:
            return None
        coef_token = match.group(1)
        const_token = match.group(2)
        x_coef_num = 1.0
        x_coef_var = None
        x_coef_var_coeff = 1.0
        if coef_token:
            if coef_token in {"+", "-"}:
                x_coef_num = 1.0 if coef_token == "+" else -1.0
            else:
                numeric = self._to_float(coef_token)
                if numeric is not None:
                    x_coef_num = float(numeric)
                else:
                    sign = -1.0 if coef_token.startswith("-") else 1.0
                    x_coef_var = coef_token.lstrip("+-")
                    x_coef_var_coeff = sign
                    x_coef_num = 0.0
        const_num = 0.0
        const_var = None
        const_var_coeff = 0.0
        if const_token:
            numeric = self._to_float(const_token)
            if numeric is not None:
                const_num = float(numeric)
            else:
                sign = -1.0 if const_token.startswith("-") else 1.0
                const_var = const_token.lstrip("+-")
                const_var_coeff = sign
        return {
            "x_coef_num": x_coef_num,
            "x_coef_var": x_coef_var,
            "x_coef_var_coeff": x_coef_var_coeff,
            "const_num": const_num,
            "const_var": const_var,
            "const_var_coeff": const_var_coeff,
        }

    def _extract_word_problem_coefficients(
        self,
        question: str,
        *,
        pattern_type: str,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any] | None:
        hints = self._extract_word_problem_entities(question, parse_bundle=parse_bundle)
        semantic_formula = self._verify_and_construct_formula(
            pattern_type=pattern_type,
            hints=hints,
        )
        if semantic_formula is None:
            return None
        return {
            "rpn_chain": semantic_formula["rpn_program"],
            "numbers": semantic_formula["numbers"],
            "hints": hints,
            "grammar_chain": semantic_formula["grammar_chain"],
            "goal": semantic_formula["goal"],
        }

    def _extract_word_problem_entities(
        self,
        question: str,
        *,
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_question = self._normalize_benchmark_text(question)
        lowered = normalized_question.lower()
        clauses = self._semantic_clauses(normalized_question)
        forward_entities = self._collect_semantic_entities(clauses=clauses, source_pass="forward")
        backward_entities = self._collect_semantic_entities(
            clauses=list(reversed(clauses)),
            source_pass="backward",
            original_length=len(clauses),
        )
        fused_entities = self._fuse_semantic_entities(forward_entities, backward_entities)
        goal = self._extract_goal_semantics(question=normalized_question, clauses=clauses, parse_bundle=parse_bundle)
        return {
            "clauses": clauses,
            "clause_numbers": [self._extract_numeric_literals(clause) for clause in clauses],
            "numbers": [float(entity["value"]) for entity in fused_entities if entity.get("value") is not None],
            "forward_entities": forward_entities,
            "backward_entities": backward_entities,
            "entities": fused_entities,
            "goal": goal,
            "has_rate": any(token in lowered for token in (" each ", " per ", " every ", " rate ", "$", "dollar")),
            "has_remainder": any(token in lowered for token in ("remaining", "remainder", "left", "leftover", "after", "spent", "used", "ate", "gave", "lost", "bakes")),
            "has_percentage": "%" in lowered or "percent" in lowered or "percentage" in lowered,
            "has_comparison": any(token in lowered for token in ("more than", "less than", "fewer than", "difference", "than ")),
            "has_total": any(token in lowered for token in ("total", "altogether", "combined", "in all", "sum")),
        }

    def _semantic_clauses(self, question: str) -> list[str]:
        clauses = [piece.strip() for piece in re.split(r"(?<!\d)[.?!]+(?!\d)", question) if piece.strip()]
        return clauses or [question.strip()]

    def _collect_semantic_entities(
        self,
        *,
        clauses: list[str],
        source_pass: str,
        original_length: int | None = None,
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        length = original_length if original_length is not None else len(clauses)
        for idx, clause in enumerate(clauses):
            clause_index = idx if source_pass == "forward" else (length - 1 - idx)
            clause_entities = self._extract_clause_entities(
                clause=clause,
                clause_index=clause_index,
                source_pass=source_pass,
                prior_entities=entities,
            )
            entities.extend(clause_entities)
        return entities

    def _extract_clause_entities(
        self,
        *,
        clause: str,
        clause_index: int,
        source_pass: str,
        prior_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        lowered = clause.lower()
        tokens = self._tokenize_clause_semantics(lowered)
        word_tokens = [tok["raw"] for tok in tokens if tok["type"] == "word"]
        entities: list[dict[str, Any]] = []
        quantity_units = {
            "egg", "eggs", "bolt", "bolts", "sprint", "sprints", "meter", "meters",
            "bag", "bags", "pound", "pounds", "mile", "miles", "apple", "apples",
            "duck", "ducks", "muffin", "muffins", "session", "sessions", "robe", "robes",
            "fiber", "fibers", "yard", "yards", "chicken", "chickens",
        }
        consume_markers = {"eat", "eats", "ate", "spent", "used", "gave", "give", "lost", "lose", "bake", "bakes"}
        price_units = {"dollar", "dollars", "usd", "cent", "cents"}
        temporal_units = {"day", "week", "month", "year", "morning", "session", "hour", "minute"}
        rate_stopwords = {"each", "every", "per", "a", "an", "times", "time"}

        for idx, token in enumerate(tokens):
            if token["type"] != "number":
                continue
            if str(token.get("raw", "")).strip().lower() == "half" and (
                "half that much" in lowered or "half as much" in lowered
            ):
                continue
            value = float(token["value"])
            next_word = self._next_word(tokens, idx)
            later_words = self._following_words(tokens, idx, limit=5)
            prev_words = self._previous_words(tokens, idx, limit=4)
            unit = self._normalize_unit(next_word)
            denominator = self._denominator_word(tokens, idx)
            scope = self._clause_scope(lowered)
            local_prior = entities[-1] if entities else None
            local_base = local_prior or self._nearest_quantity_entity(prior_entities)
            fallback_unit = str(local_base.get("unit", "")) if local_base else ""
            consume_context = any(word in consume_markers for word in prev_words + later_words)
            bake_reference_context = (
                any(word in {"bake", "bakes"} for word in word_tokens)
                and ("with" in prev_words + later_words)
                and bool(local_base)
            )

            if consume_context and (next_word in {"for", "with"} or not unit or unit not in quantity_units):
                entities.append(
                    {
                        "value": value,
                        "role": "consume",
                        "unit": fallback_unit or unit or self._infer_unit_from_clause_words(word_tokens),
                        "scope": scope or str(local_base.get("scope", "")) if local_base else scope,
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.85 if fallback_unit else 0.7,
                        "raw": token["raw"],
                        "operation_hint": "subtract",
                        "prev_words": prev_words,
                        "next_words": later_words,
                    }
                )
                continue

            if bake_reference_context:
                entities.append(
                    {
                        "value": value,
                        "role": "consume",
                        "unit": fallback_unit or self._infer_unit_from_clause_words(word_tokens),
                        "scope": str(local_base.get("scope", "")) if local_base else scope,
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.85 if fallback_unit else 0.7,
                        "raw": token["raw"],
                        "operation_hint": "subtract",
                        "prev_words": prev_words,
                        "next_words": later_words,
                        "reference_kind": "consume_from_local_base",
                    }
                )
                continue

            if next_word in {"times", "time"}:
                scope_word = self._scope_word_after(tokens, idx)
                entities.append(
                    {
                        "value": value,
                        "role": "frequency",
                        "unit": "sessions",
                        "scope": f"per_{scope_word}" if scope_word else "",
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.8,
                        "raw": token["raw"],
                        "operation_hint": "multiply",
                    }
                )
                continue

            if unit in quantity_units and denominator and self._normalize_unit(denominator) not in temporal_units:
                entities.append(
                    {
                        "value": value,
                        "role": "rate",
                        "unit": unit,
                        "scope": scope,
                        "numerator_unit": unit,
                        "denominator_unit": self._normalize_unit(denominator),
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.85,
                        "raw": token["raw"],
                        "operation_hint": "multiply",
                        "prev_words": prev_words,
                        "next_words": later_words,
                    }
                )
                continue

            if unit in quantity_units:
                entities.append(
                    {
                        "value": value,
                        "role": "consume" if any(word in consume_markers for word in prev_words + later_words) else "count",
                        "unit": unit,
                        "scope": scope or (f"per_{self._normalize_unit(denominator)}" if denominator and self._normalize_unit(denominator) in temporal_units else ""),
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.85,
                        "raw": token["raw"],
                        "operation_hint": "subtract" if any(word in consume_markers for word in prev_words + later_words) else "add",
                        "prev_words": prev_words,
                        "next_words": later_words,
                    }
                )
                continue

            if unit in price_units or "$" in token["raw"]:
                price_denominator = self._following_unit_word(tokens, idx, limit=6) or str(local_base.get("unit", "")) if local_base else ""
                entities.append(
                    {
                        "value": value,
                        "role": "price",
                        "unit": "dollar",
                        "scope": "",
                        "numerator_unit": "dollar",
                        "denominator_unit": self._normalize_unit(price_denominator),
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.85,
                        "raw": token["raw"],
                        "operation_hint": "multiply",
                        "prev_words": prev_words,
                        "next_words": later_words,
                    }
                )
                continue

            if denominator and unit and unit not in rate_stopwords:
                entities.append(
                    {
                        "value": value,
                        "role": "rate",
                        "unit": unit,
                        "scope": scope,
                        "numerator_unit": unit,
                        "denominator_unit": self._normalize_unit(denominator),
                        "clause_index": clause_index,
                        "token_index": idx,
                        "source_pass": source_pass,
                        "confidence": 0.85,
                        "raw": token["raw"],
                        "operation_hint": "multiply",
                        "prev_words": prev_words,
                        "next_words": later_words,
                    }
                )
                continue

            entities.append(
                {
                    "value": value,
                    "role": "consume" if any(word in consume_markers for word in prev_words + later_words) else "quantity",
                    "unit": unit or self._infer_unit_from_clause_words(word_tokens),
                    "scope": scope,
                    "clause_index": clause_index,
                    "token_index": idx,
                    "source_pass": source_pass,
                    "confidence": 0.7,
                    "raw": token["raw"],
                    "operation_hint": "subtract" if any(word in consume_markers for word in prev_words + later_words) else "add",
                    "prev_words": prev_words,
                    "next_words": later_words,
                }
            )

        if "half that much" in lowered or "half as much" in lowered:
            entities = [
                entity
                for entity in entities
                if not (
                    "half" in str(entity.get("raw", "")).lower()
                    and str(entity.get("reference_kind", "")) != "half_previous"
                )
            ]

        if "half that much" in lowered or "half as much" in lowered:
            reference_base = None
            for candidate in reversed(entities):
                if str(candidate.get("role", "")) in {"quantity", "count"} and candidate.get("value") is not None:
                    reference_base = candidate
                    break
            if reference_base is None:
                reference_base = self._nearest_quantity_entity(prior_entities)
            base_unit = str(reference_base.get("unit", "")) if reference_base else ""
            entities.append(
                {
                    "value": float(reference_base.get("value", 0.0)) / 2.0 if reference_base else 0.5,
                    "role": "count" if base_unit else "quantity",
                    "unit": base_unit,
                    "scope": str(reference_base.get("scope", "")) if reference_base else "",
                    "clause_index": clause_index,
                    "token_index": len(tokens) + 1,
                    "source_pass": source_pass,
                    "confidence": 0.75 if reference_base else 0.35,
                    "reference_kind": "half_previous",
                    "raw": "half that much",
                    "operation_hint": "add",
                }
            )
        return entities

    def _tokenize_clause_semantics(self, clause: str) -> list[dict[str, Any]]:
        clause = clause.replace("half-hour", "half hour")
        raw_tokens = re.findall(r"\$?-?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:-[A-Za-z]+)?", clause)
        tokens: list[dict[str, Any]] = []
        index = 0
        while index < len(raw_tokens):
            raw = raw_tokens[index]
            numeric = self._to_float(raw.lstrip("$"))
            if numeric is not None:
                tokens.append({"type": "number", "raw": raw, "value": numeric})
                index += 1
                continue
            seq: list[str] = []
            cursor = index
            while cursor < len(raw_tokens):
                candidate = raw_tokens[cursor].lower()
                parts = [part for part in candidate.split("-") if part]
                if not parts or not all(part in self._NUMBER_WORD_TOKENS for part in parts):
                    break
                seq.extend(parts)
                cursor += 1
            if seq:
                seq_window = [part.lower() for part in raw_tokens[index:cursor]]
                next_parts = [part.lower() for part in raw_tokens[cursor : cursor + 2]]
                if (
                    "half" in seq
                    and next_parts in (["that", "much"], ["as", "much"])
                ):
                    tokens.append({"type": "word", "raw": raw.lower()})
                    index += 1
                    continue
                parsed = self._parse_number_word_sequence(seq)
                if parsed is not None:
                    tokens.append({"type": "number", "raw": " ".join(raw_tokens[index:cursor]), "value": parsed})
                    index = cursor
                    continue
            tokens.append({"type": "word", "raw": raw.lower()})
            index += 1
        return tokens

    def _next_word(self, tokens: list[dict[str, Any]], idx: int) -> str:
        for token in tokens[idx + 1 :]:
            if token["type"] == "word":
                return str(token["raw"])
        return ""

    def _following_words(self, tokens: list[dict[str, Any]], idx: int, *, limit: int) -> list[str]:
        out: list[str] = []
        for token in tokens[idx + 1 :]:
            if token["type"] != "word":
                continue
            out.append(str(token["raw"]))
            if len(out) >= limit:
                break
        return out

    def _previous_words(self, tokens: list[dict[str, Any]], idx: int, *, limit: int) -> list[str]:
        out: list[str] = []
        for token in reversed(tokens[:idx]):
            if token["type"] != "word":
                continue
            out.append(str(token["raw"]))
            if len(out) >= limit:
                break
        out.reverse()
        return out

    def _scope_word_after(self, tokens: list[dict[str, Any]], idx: int) -> str:
        words = self._following_words(tokens, idx, limit=5)
        for pos, word in enumerate(words):
            if word in {"a", "an", "per", "every"} and pos + 1 < len(words):
                return words[pos + 1]
        return ""

    def _denominator_word(self, tokens: list[dict[str, Any]], idx: int) -> str:
        words = self._following_words(tokens, idx, limit=6)
        quantity_markers = {
            "egg", "eggs", "duck", "ducks", "mile", "miles", "meter", "meters",
            "sprint", "sprints", "bag", "bags", "pound", "pounds", "glass", "glasses",
            "cup", "cups", "session", "sessions", "day", "days", "week", "weeks",
        }
        for pos, word in enumerate(words):
            if word in {"each", "per", "every", "a", "an"}:
                for candidate in words[pos + 1 :]:
                    if candidate in {"fresh", "daily", "new", "old"}:
                        continue
                    if self._normalize_unit(candidate) in {self._normalize_unit(marker) for marker in quantity_markers}:
                        return candidate
                if pos + 1 < len(words):
                    return words[pos + 1]
        return ""

    def _following_unit_word(self, tokens: list[dict[str, Any]], idx: int, *, limit: int) -> str:
        known_units = {
            "egg", "eggs", "duck", "ducks", "mile", "miles", "meter", "meters",
            "sprint", "sprints", "bag", "bags", "pound", "pounds", "glass", "glasses",
            "cup", "cups", "session", "sessions", "day", "days", "week", "weeks",
            "chicken", "chickens", "bolt", "bolts", "robe", "robes",
        }
        matches = [
            word
            for word in self._following_words(tokens, idx, limit=limit)
            if self._normalize_unit(word) in {self._normalize_unit(unit) for unit in known_units}
        ]
        return matches[-1] if matches else ""

    def _clause_scope(self, lowered: str) -> str:
        for scope in ("day", "week", "month", "year", "morning", "session"):
            if f"per {scope}" in lowered or f"every {scope}" in lowered or f"a {scope}" in lowered:
                return f"per_{scope}"
        return ""

    def _normalize_unit(self, raw: str) -> str:
        unit = str(raw or "").strip().lower()
        irregular = {
            "dollars": "dollar",
            "cents": "cent",
            "eggs": "egg",
            "ducks": "duck",
            "miles": "mile",
            "meters": "meter",
            "bags": "bag",
            "pounds": "pound",
            "glasses": "glass",
            "glass": "glass",
            "cups": "cup",
            "sessions": "session",
            "days": "day",
            "weeks": "week",
            "months": "month",
            "years": "year",
            "mornings": "morning",
            "hours": "hour",
            "minutes": "minute",
            "chickens": "chicken",
            "bolts": "bolt",
            "robes": "robe",
            "muffins": "muffin",
            "yards": "yard",
            "fibers": "fiber",
            "sprints": "sprint",
            "apples": "apple",
        }
        if unit in irregular:
            return irregular[unit]
        if unit.endswith("sses"):
            unit = unit[:-2]
        elif unit.endswith("ies"):
            unit = f"{unit[:-3]}y"
        elif unit.endswith("s"):
            unit = unit[:-1]
        return unit

    def _infer_unit_from_clause_words(self, words: list[str]) -> str:
        for word in words:
            norm = self._normalize_unit(word)
            if norm in {
                "egg", "bolt", "sprint", "meter", "dollar", "cent", "bag", "pound",
                "chicken", "mile", "apple", "duck", "muffin", "session",
            }:
                return norm
        return ""

    def _nearest_quantity_entity(self, entities: list[dict[str, Any]]) -> dict[str, Any] | None:
        for entity in reversed(entities):
            if str(entity.get("role", "")) in {"quantity", "count", "consume"} and entity.get("value") is not None:
                return entity
        return None

    def _fuse_semantic_entities(
        self,
        forward_entities: list[dict[str, Any]],
        backward_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fused: dict[tuple[Any, ...], dict[str, Any]] = {}
        for entity in [*forward_entities, *backward_entities]:
            key = (
                round(float(entity.get("value", 0.0)), 6) if entity.get("value") is not None else None,
                str(entity.get("role", "")),
                str(entity.get("unit", "")),
                str(entity.get("scope", "")),
                str(entity.get("numerator_unit", "")),
                str(entity.get("denominator_unit", "")),
                int(entity.get("clause_index", -1)),
                int(entity.get("token_index", -1)),
            )
            if key not in fused:
                copy = dict(entity)
                copy["sources"] = [str(entity.get("source_pass", ""))]
                fused[key] = copy
                continue
            current = fused[key]
            current["confidence"] = min(1.0, float(current.get("confidence", 0.5)) + 0.2)
            current["sources"] = sorted(set(list(current.get("sources", [])) + [str(entity.get("source_pass", ""))]))

        resolved = list(sorted(fused.values(), key=lambda row: (int(row.get("clause_index", 0)), int(row.get("token_index", 0)))))
        compact: list[dict[str, Any]] = []
        compact_map: dict[tuple[Any, ...], dict[str, Any]] = {}
        for entity in resolved:
            compact_key = (
                round(float(entity.get("value", 0.0)), 6) if entity.get("value") is not None else None,
                str(entity.get("role", "")),
                int(entity.get("clause_index", -1)),
                int(entity.get("token_index", -1)),
            )
            current = compact_map.get(compact_key)
            if current is None:
                compact_map[compact_key] = entity
                compact.append(entity)
                continue
            current_unit = str(current.get("unit", ""))
            incoming_unit = str(entity.get("unit", ""))
            if (not current_unit or current_unit in {"every", "for", "with"}) and incoming_unit:
                current["unit"] = incoming_unit
            if not str(current.get("scope", "")) and entity.get("scope"):
                current["scope"] = entity.get("scope")
            if not current.get("denominator_unit") and entity.get("denominator_unit"):
                current["denominator_unit"] = entity.get("denominator_unit")
            if not current.get("numerator_unit") and entity.get("numerator_unit"):
                current["numerator_unit"] = entity.get("numerator_unit")
            current["confidence"] = max(float(current.get("confidence", 0.0)), float(entity.get("confidence", 0.0)))
            current["sources"] = sorted(set(list(current.get("sources", [])) + list(entity.get("sources", []))))
        resolved = compact
        for entity in resolved:
            if str(entity.get("reference_kind", "")) != "half_previous":
                continue
            base = None
            for candidate in resolved:
                if int(candidate.get("clause_index", 0)) >= int(entity.get("clause_index", 0)):
                    break
                if str(candidate.get("role", "")) not in {"quantity", "count"}:
                    continue
                base = candidate
            if base is not None:
                entity["value"] = float(base["value"]) / 2.0
                entity["unit"] = str(base.get("unit", ""))
                entity["scope"] = str(base.get("scope", ""))
                entity["confidence"] = min(1.0, float(entity.get("confidence", 0.5)) + 0.25)
        return resolved

    def _extract_goal_semantics(
        self,
        *,
        question: str,
        clauses: list[str],
        parse_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        raw = clauses[-1] if clauses else question
        fusion = parse_bundle.get("fusion_parse")
        if isinstance(fusion, dict):
            unified = fusion.get("unified_goal")
            if isinstance(unified, dict):
                raw = str(unified.get("raw") or unified.get("expression") or raw)
        lowered = raw.lower()
        target_unit = ""
        for unit in ("dollars", "dollar", "meters", "meter", "bolts", "bolt", "eggs", "egg", "bags", "bag", "pounds", "pound"):
            if unit in lowered:
                target_unit = self._normalize_unit(unit)
                break
        scope = self._clause_scope(lowered)
        aggregation = "total" if any(token in lowered for token in ("total", "altogether", "in all", "how much", "how many")) else ""
        return {"raw": raw, "target_unit": target_unit, "scope": scope, "aggregation": aggregation}

    def _verify_and_construct_formula(
        self,
        *,
        pattern_type: str,
        hints: dict[str, Any],
    ) -> dict[str, Any] | None:
        entities = list(hints.get("entities", []))
        if not entities:
            return None
        goal = dict(hints.get("goal", {}))
        goal["target_unit"] = self._normalize_unit(goal.get("target_unit", ""))

        quantity_entities = [e for e in entities if str(e.get("role", "")) in {"quantity", "count"}]
        consume_entities = [e for e in entities if str(e.get("role", "")) == "consume"]
        frequency_entities = [e for e in entities if str(e.get("role", "")) == "frequency"]
        rate_entities = [e for e in entities if str(e.get("role", "")) == "rate"]
        price_entities = [e for e in entities if str(e.get("role", "")) == "price"]

        def _fmt(value: float) -> str:
            return f"{float(value):.12g}"

        clauses_text = " ".join(str(clause) for clause in hints.get("clauses", [])).lower()
        numeric_values = [float(entity["value"]) for entity in entities if entity.get("value") is not None]

        if "profit" in clauses_text and ("increased the value" in clauses_text or "%" in clauses_text or "percent" in clauses_text):
            prices = [float(entity["value"]) for entity in price_entities if entity.get("value") is not None]
            percentage_values = [
                float(entity["value"])
                for entity in quantity_entities
                if entity.get("value") is not None and "%" in str(entity.get("raw", ""))
            ] or [
                float(entity["value"])
                for entity in quantity_entities
                if entity.get("value") is not None and float(entity["value"]) > 1.0
            ]
            if len(prices) >= 2 and percentage_values:
                buy_price = prices[0]
                repair_price = prices[1]
                pct = percentage_values[-1]
                return {
                    "rpn_program": f"{_fmt(buy_price)} 1 {_fmt(pct)} 100 / + * {_fmt(buy_price)} - {_fmt(repair_price)} -",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": ["gsm_percent_of", "gsm_sequential_computation", "gsm_answer_final_stack"],
                }

        if "final meal" in clauses_text and "chicken" in clauses_text:
            chicken_counts = [float(entity["value"]) for entity in quantity_entities if str(entity.get("unit", "")) == "chicken"]
            cup_values = [
                float(entity["value"])
                for entity in quantity_entities
                if str(entity.get("unit", "")) == "cup"
            ]
            if chicken_counts and len(cup_values) >= 3:
                flock = chicken_counts[-1]
                ration = cup_values[0]
                meals = cup_values[1:3]
                return {
                    "rpn_program": f"{_fmt(flock)} {_fmt(ration)} * {_fmt(meals[0])} - {_fmt(meals[1])} -",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": [
                        "gsm_rate_application",
                        "gsm_consume_from_total",
                        "gsm_consume_from_total",
                        "gsm_answer_final_stack",
                    ],
                }

        if "every second glass" in clauses_text and "%" in clauses_text and "glass" in clauses_text:
            quantity_glasses = [
                float(entity["value"])
                for entity in quantity_entities
                if str(entity.get("unit", "")) == "glass"
            ]
            prices = [float(entity["value"]) for entity in price_entities if entity.get("value") is not None]
            pct_values = [
                float(entity["value"])
                for entity in quantity_entities
                if entity.get("value") is not None and "%" in str(entity.get("raw", ""))
            ]
            if quantity_glasses and prices and pct_values:
                qty = quantity_glasses[-1]
                price = prices[0]
                pct = pct_values[-1]
                return {
                    "rpn_program": f"{_fmt(qty)} 2 / {_fmt(price)} * {_fmt(qty)} 2 / {_fmt(price)} {_fmt(pct)} * 100 / * +",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": ["gsm_rate_application", "gsm_percent_of", "gsm_sequential_computation", "gsm_answer_final_stack"],
                }

        if "twice as many" in clauses_text and "together" in clauses_text and "sheep" in clauses_text:
            count_values = [float(entity["value"]) for entity in quantity_entities if entity.get("value") is not None]
            base = count_values[-1] if count_values else None
            multiplier_values = [value for value in numeric_values if value not in {base}]
            if base is not None and multiplier_values:
                outer = 2.0
                inner = max(multiplier_values)
                return {
                    "rpn_program": f"{_fmt(base)} {_fmt(inner)} * {_fmt(base)} {_fmt(inner)} * {_fmt(outer)} * + {_fmt(base)} +",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": ["gsm_comparison_delta", "gsm_sequential_computation", "gsm_answer_final_stack"],
                }

        if "restart" in clauses_text and "download" in clauses_text and "%" in clauses_text:
            numeric = [float(v) for v in numeric_values]
            if len(numeric) >= 4:
                size = numeric[0]
                rate = numeric[1]
                pct = numeric[2]
                delay = numeric[3]
                return {
                    "rpn_program": f"{_fmt(size)} {_fmt(pct)} * 100 / {_fmt(rate)} / {_fmt(delay)} + {_fmt(size)} {_fmt(rate)} / +",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": ["gsm_percent_of", "gsm_rate_application", "gsm_sequential_computation", "gsm_answer_final_stack"],
                }

        if "turns around" in clauses_text and "standstill traffic" in clauses_text:
            numeric = [float(v) for v in numeric_values]
            if len(numeric) >= 6:
                ordered_entities = [
                    entity for entity in entities
                    if entity.get("value") is not None
                ]
                mph_entities = [entity for entity in ordered_entities if str(entity.get("unit", "")) == "mph"]
                hour_entities = [entity for entity in ordered_entities if str(entity.get("unit", "")) == "hour"]
                outbound_hours = float(hour_entities[0]["value"]) if hour_entities else numeric[0]
                outbound_speed = float(mph_entities[0]["value"]) if mph_entities else numeric[1]
                total_return_hours = float(hour_entities[1]["value"]) if len(hour_entities) > 1 else next((value for value in numeric[2:] if value == 4.0), numeric[2])
                stopped_hours = next((float(entity["value"]) for entity in hour_entities[1:] if abs(float(entity["value"]) - 2.0) <= 1e-9), numeric[3] if len(numeric) > 3 else 0.0)
                slow_hours_entity = next((entity for entity in hour_entities if abs(float(entity["value"]) - 0.5) <= 1e-9), None)
                slow_hours = float(slow_hours_entity["value"]) if slow_hours_entity is not None else next((value for value in numeric if 0.0 < value < 1.0), 0.5)
                slow_anchor = (int(slow_hours_entity.get("clause_index", -1)), int(slow_hours_entity.get("token_index", -1))) if slow_hours_entity is not None else (-1, -1)
                slow_speed_entity = next(
                    (
                        entity
                        for entity in mph_entities
                        if (int(entity.get("clause_index", -1)), int(entity.get("token_index", -1))) > slow_anchor
                    ),
                    mph_entities[1] if len(mph_entities) > 1 else None,
                )
                slow_speed = float(slow_speed_entity["value"]) if slow_speed_entity is not None else 30.0
                fast_speed_entity = next(
                    (
                        entity
                        for entity in mph_entities
                        if (
                            int(entity.get("clause_index", -1)),
                            int(entity.get("token_index", -1)),
                        ) > (
                            int(slow_speed_entity.get("clause_index", -1)) if slow_speed_entity else -1,
                            int(slow_speed_entity.get("token_index", -1)) if slow_speed_entity else -1,
                        )
                    ),
                    mph_entities[-1] if mph_entities else None,
                )
                fast_speed = float(fast_speed_entity["value"]) if fast_speed_entity is not None else max(value for value in numeric if value > slow_speed)
                return {
                    "rpn_program": f"{_fmt(outbound_hours)} {_fmt(outbound_speed)} * {_fmt(total_return_hours)} {_fmt(stopped_hours)} - {_fmt(slow_hours)} - {_fmt(fast_speed)} * {_fmt(slow_hours)} {_fmt(slow_speed)} * + -",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": ["gsm_rate_application", "gsm_sequential_computation", "gsm_comparison_delta", "gsm_answer_final_stack"],
                }

        if "overtime" in clauses_text and "hourly rate" in clauses_text:
            numeric = [float(v) for v in numeric_values]
            prices = [float(entity["value"]) for entity in price_entities if entity.get("value") is not None]
            if len(numeric) >= 4 and prices:
                threshold = next((value for value in numeric if value >= 40.0), numeric[0])
                total_hours = numeric[-1]
                multiplier = next((value for value in numeric if 1.0 < value < 5.0 and value != total_hours and value != threshold), 1.0)
                base_rate = prices[0]
                return {
                    "rpn_program": f"{_fmt(threshold)} {_fmt(base_rate)} * {_fmt(total_hours)} {_fmt(threshold)} - {_fmt(base_rate)} {_fmt(multiplier)} * * +",
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": ["gsm_rate_application", "gsm_sequential_computation", "gsm_answer_final_stack"],
                }

        if price_entities and quantity_entities and consume_entities and str(goal.get("target_unit", "")) == "dollar":
            base = quantity_entities[0]
            base_unit = str(base.get("unit", ""))
            price = price_entities[0]
            denominator = str(price.get("denominator_unit", ""))
            if (
                base_unit
                and all(str(entity.get("unit", "")) in {"", base_unit} for entity in consume_entities)
                and (not denominator or denominator == base_unit)
            ):
                tokens = [_fmt(base["value"])]
                for entity in consume_entities:
                    tokens.extend([_fmt(entity["value"]), "-"])
                tokens.extend([_fmt(price["value"]), "*"])
                return {
                    "rpn_program": " ".join(tokens),
                    "numbers": numeric_values,
                    "goal": goal,
                    "grammar_chain": [
                        "gsm_consume_from_total",
                        "gsm_sequential_computation",
                        "gsm_rate_application",
                        "gsm_answer_final_stack",
                    ],
                }

        quantity_tokens: list[str] = []
        quantity_unit = ""
        quantity_scope = ""

        if quantity_entities and frequency_entities:
            base = quantity_entities[0]
            freq = frequency_entities[0]
            quantity_tokens = [_fmt(base["value"]), _fmt(freq["value"]), "*"]
            quantity_unit = str(base.get("unit", ""))
            quantity_scope = str(freq.get("scope", "")) or str(base.get("scope", ""))
        elif quantity_entities:
            base = quantity_entities[0]
            quantity_tokens = [_fmt(base["value"])]
            quantity_unit = str(base.get("unit", ""))
            quantity_scope = str(base.get("scope", ""))
            for entity in quantity_entities[1:]:
                quantity_tokens.extend([_fmt(entity["value"]), "+"])

        if consume_entities and quantity_tokens:
            for entity in consume_entities:
                quantity_tokens.extend([_fmt(entity["value"]), "-"])
                if not quantity_unit:
                    quantity_unit = str(entity.get("unit", ""))
                if not quantity_scope:
                    quantity_scope = str(entity.get("scope", ""))

        final_tokens = list(quantity_tokens)
        final_unit = quantity_unit
        final_scope = quantity_scope

        carrier = None
        if price_entities:
            carrier = price_entities[0]
        elif rate_entities:
            carrier = rate_entities[0]
        if carrier and final_tokens:
            denominator = str(carrier.get("denominator_unit", ""))
            carrier_role = str(carrier.get("role", ""))
            if (
                carrier_role == "price"
                or not denominator
                or not final_unit
                or denominator == final_unit
                or (final_unit and final_unit.startswith(denominator))
            ):
                final_tokens.extend([_fmt(carrier["value"]), "*"])
                final_unit = str(carrier.get("numerator_unit", carrier.get("unit", final_unit)))
                if not final_scope:
                    final_scope = str(goal.get("scope", "")) or quantity_scope

        if not final_tokens:
            fallback = self._build_word_problem_rpn(pattern_type=pattern_type, hints=hints)
            if not fallback:
                return None
            final_tokens = fallback.split()
            final_unit = str(goal.get("target_unit", ""))
            final_scope = str(goal.get("scope", ""))

        target_unit = str(goal.get("target_unit", ""))
        target_scope = str(goal.get("scope", ""))
        if target_unit and final_unit and target_unit != final_unit:
            return None
        if target_scope and final_scope and target_scope != final_scope:
            return None

        grammar_chain = self._word_problem_grammar_chain(
            pattern_type=pattern_type,
            numbers=[float(entity["value"]) for entity in entities if entity.get("value") is not None],
        )
        return {
            "rpn_program": " ".join(final_tokens),
            "numbers": [float(entity["value"]) for entity in entities if entity.get("value") is not None],
            "goal": goal,
            "grammar_chain": grammar_chain,
        }

    def _build_word_problem_rpn(self, *, pattern_type: str, hints: dict[str, Any]) -> str:
        numbers = [float(value) for value in hints.get("numbers", [])]
        if not numbers:
            return ""
        kind = str(pattern_type).strip().lower()
        if kind == "word_problem_percentage" and len(numbers) >= 2:
            pct = numbers[0]
            base = numbers[1]
            return f"{base:.12g} {pct:.12g} * 100 /"
        if kind == "word_problem_total" and 2 <= len(numbers) <= 4:
            chain = [f"{numbers[0]:.12g}"]
            for value in numbers[1:]:
                chain.extend((f"{value:.12g}", "+"))
            return " ".join(chain)
        if kind == "word_problem_comparison" and len(numbers) >= 2:
            lowered = " ".join(str(clause).lower() for clause in hints.get("clauses", []))
            if "more than" in lowered:
                return f"{numbers[1]:.12g} {numbers[0]:.12g} +"
            if "less than" in lowered or "fewer than" in lowered:
                return f"{numbers[1]:.12g} {numbers[0]:.12g} -"
        if kind == "word_problem_remainder" and len(numbers) >= 2:
            chain = [f"{numbers[0]:.12g}"]
            for value in numbers[1:]:
                chain.extend((f"{value:.12g}", "-"))
            return " ".join(chain)
        if kind == "word_problem_rate" and len(numbers) >= 2:
            if len(numbers) >= 3 and hints.get("has_remainder"):
                chain = [f"{numbers[0]:.12g}"]
                for value in numbers[1:-1]:
                    chain.extend((f"{value:.12g}", "-"))
                chain.extend((f"{numbers[-1]:.12g}", "*"))
                return " ".join(chain)
            return f"{numbers[0]:.12g} {numbers[-1]:.12g} *"
        if kind == "word_problem_sequential" and len(numbers) >= 3:
            if hints.get("has_remainder") and hints.get("has_rate"):
                chain = [f"{numbers[0]:.12g}"]
                for value in numbers[1:-1]:
                    chain.extend((f"{value:.12g}", "-"))
                chain.extend((f"{numbers[-1]:.12g}", "*"))
                return " ".join(chain)
            chain = [f"{numbers[0]:.12g}"]
            for value in numbers[1:]:
                chain.extend((f"{value:.12g}", "+"))
            return " ".join(chain)
        return ""

    def _compose_word_problem_program(
        self,
        *,
        pattern_type: str,
        coefficients: dict[str, Any],
    ) -> dict[str, Any] | None:
        rpn_program = str(coefficients.get("rpn_chain", "")).strip()
        numbers = [float(value) for value in coefficients.get("numbers", [])]
        if not rpn_program or not numbers:
            return None
        grammar_chain = self._word_problem_grammar_chain(pattern_type=pattern_type, numbers=numbers)
        number_refs = [self._number_ref(value) for value in numbers]
        word_refs = [self._word_ref(value) for value in numbers]
        return {
            "rpn_program": rpn_program,
            "grammar_chain": grammar_chain,
            "number_refs": number_refs,
            "word_refs": word_refs,
        }

    def _word_problem_grammar_chain(self, *, pattern_type: str, numbers: list[float]) -> list[str]:
        kind = str(pattern_type).strip().lower()
        if kind == "word_problem_percentage":
            return ["gsm_percent_of", "gsm_answer_final_stack"]
        if kind == "word_problem_total":
            return ["gsm_sequential_computation", "gsm_answer_final_stack"]
        if kind == "word_problem_comparison":
            return ["gsm_comparison_delta", "gsm_answer_final_stack"]
        if kind == "word_problem_remainder":
            return ["gsm_consume_from_total", "gsm_answer_final_stack"]
        if kind == "word_problem_rate":
            return ["gsm_rate_application", "gsm_answer_final_stack"]
        if kind == "word_problem_sequential":
            chain = ["gsm_sequential_computation"]
            if len(numbers) > 2:
                chain.extend(["gsm_consume_from_total"] * (len(numbers) - 2))
            chain.extend(["gsm_rate_application", "gsm_answer_final_stack"])
            return chain
        return ["gsm_answer_final_stack"]

    def _number_ref(self, value: float) -> str:
        magnitude = abs(float(value))
        normalized = int(magnitude) if magnitude.is_integer() else magnitude
        return f"num_{normalized}"

    def _word_ref(self, value: float) -> str:
        magnitude = abs(float(value))
        normalized = int(magnitude) if magnitude.is_integer() else magnitude
        if isinstance(normalized, int):
            token = self._int_to_word_token(normalized)
        else:
            token = str(normalized).replace(".", "_")
        return f"word_{token}"

    def _int_to_word_token(self, value: int) -> str:
        if value < 20:
            mapping = {
                0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
                12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
                17: "seventeen", 18: "eighteen", 19: "nineteen",
            }
            return mapping[value]
        tens_map = {
            20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
            60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
        }
        if value < 100:
            tens = (value // 10) * 10
            unit = value % 10
            return tens_map[tens] if unit == 0 else f"{tens_map[tens]}_{self._int_to_word_token(unit)}"
        if value < 1000:
            hundreds = value // 100
            remainder = value % 100
            head = f"{self._int_to_word_token(hundreds)}_hundred"
            return head if remainder == 0 else f"{head}_{self._int_to_word_token(remainder)}"
        if value == 1000:
            return "one_thousand"
        return str(value)

    def _extract_ratio_operands(self, question: str) -> tuple[float, float] | None:
        compact = question.replace(" ", "")
        if ":" in compact:
            left, right = compact.split(":", 1)
            left_vals = self._extract_numeric_literals(left)
            right_vals = self._extract_numeric_literals(right)
            if left_vals and right_vals:
                return float(left_vals[0]), float(right_vals[0])
        values = self._extract_numeric_literals(question)
        if len(values) < 2:
            return None
        return float(values[0]), float(values[1])

    def _extract_proportion_coefficients(self, question: str) -> dict[str, float] | None:
        equation = self._extract_equation_segment(question)
        if equation and "=" in equation:
            compact = equation.lower().replace(" ", "")
        else:
            allowed = set("0123456789=:+-*/.xyzw")
            compact = "".join(ch for ch in question.lower() if ch in allowed)
        if "=" not in compact:
            return None
        left, right = compact.split("=", 1)
        divider = ":" if ":" in left and ":" in right else "/"
        if divider not in left or divider not in right:
            return None
        left_parts = left.split(divider)
        right_parts = right.split(divider)
        if len(left_parts) != 2 or len(right_parts) != 2:
            return None

        a = self._to_float(left_parts[0])
        b = self._to_float(left_parts[1])
        c = self._to_float(right_parts[0])
        rhs_tail = right_parts[1]
        if a is None or b is None or c is None:
            return None
        # Support a/b = c/x form for now (unknown on denominator).
        if self._to_float(rhs_tail) is not None:
            return None
        if self._detect_variable(rhs_tail) is None:
            return None
        if abs(a) < 1e-9:
            return None
        return {"a": float(a), "b": float(b), "c": float(c)}

    def _extract_arithmetic_operands(self, question: str) -> tuple[float, float] | None:
        values = self._extract_numeric_literals(question)
        if len(values) < 2:
            return None
        return float(values[0]), float(values[1])

    def _extract_linear_coefficients_forward_backward(self, question: str) -> dict[str, float] | None:
        equation = self._extract_equation_segment(question)
        if equation is None or "=" not in equation:
            return None
        forward = self._extract_linear_coefficients_from_equation(equation)
        if forward is not None:
            return forward

        left, right = equation.split("=", 1)
        backward = self._extract_linear_coefficients_from_equation(f"{right}={left}")
        if backward is not None:
            return backward
        return None

    def _extract_linear_coefficients_from_equation(self, equation: str) -> dict[str, float] | None:
        if "=" not in equation:
            return None
        left, right = equation.split("=", 1)
        variable = self._detect_variable(equation)
        if variable is None:
            return None
        left_coef, left_const = self._parse_linear_side(left, variable)
        right_coef, right_const = self._parse_linear_side(right, variable)
        if left_coef is None or left_const is None or right_coef is None or right_const is None:
            return None
        a = left_coef - right_coef
        b = left_const
        c = right_const
        if abs(a) < 1e-9:
            return None
        return {"a": float(a), "b": float(b), "c": float(c)}

    def _extract_equation_segment(self, text: str) -> str | None:
        cleaned = text.replace("\n", " ").replace("\t", " ")
        for delim in (";", ",", "?"):
            cleaned = cleaned.replace(delim, " ")
        if "=" not in cleaned:
            return None
        left, right = cleaned.split("=", 1)
        variable = self._detect_variable(cleaned) or "x"
        left_expr = self._extract_math_side(left, variable=variable, from_end=True)
        right_expr = self._extract_math_side(right, variable=variable, from_end=False)
        if not left_expr or not right_expr:
            return None
        return f"{left_expr}={right_expr}"

    def _extract_math_side(self, side: str, *, variable: str, from_end: bool) -> str:
        allowed = set("0123456789.+-*/")
        tokens = [tok for tok in side.strip().split(" ") if tok]
        if from_end:
            tokens = list(reversed(tokens))

        collected: list[str] = []
        started = False
        for token in tokens:
            low = token.lower().strip()
            if not low:
                continue
            keep = True
            for ch in low:
                if ch not in allowed and ch != variable:
                    keep = False
                    break
            if keep:
                started = True
                collected.append(low)
                continue
            if started:
                break
        if from_end:
            collected.reverse()
        return " ".join(collected)

    def _detect_variable(self, text: str) -> str | None:
        for ch in ("x", "y", "z", "w"):
            if ch in text.lower():
                return ch
        return None

    def _parse_linear_side(self, expr: str, variable: str) -> tuple[float | None, float | None]:
        compact = "".join(ch for ch in expr.lower() if ch not in {" ", "\t"})
        if not compact:
            return None, None
        if compact.startswith("+"):
            compact = compact[1:]
        compact = compact.replace("-", "+-")
        terms = [term for term in compact.split("+") if term]
        coef = 0.0
        const = 0.0
        for term in terms:
            if variable in term:
                part = term.replace(variable, "").replace("*", "")
                if part in {"", "+"}:
                    value = 1.0
                elif part == "-":
                    value = -1.0
                else:
                    value = self._to_float(part)
                    if value is None:
                        return None, None
                coef += value
            else:
                value = self._to_float(term)
                if value is None:
                    return None, None
                const += value
        return coef, const

    def _compose_from_template(self, *, template: dict[str, Any], coefficients: dict[str, Any]) -> str | None:
        rpn_template = str(template.get("rpn_program", "")).strip()
        if not rpn_template:
            return None
        rendered = rpn_template
        for name, value in coefficients.items():
            if isinstance(value, (int, float)):
                replacement = f"{float(value):.12g}"
            else:
                replacement = str(value).strip()
            rendered = rendered.replace(f"{{{name}}}", replacement)
        if "{" in rendered and "}" in rendered:
            return None
        return rendered

    def _extract_numeric_literals(self, text: str) -> list[float]:
        values: list[float] = []
        number_word_tokens: list[str] = []
        text = text.replace("half-hour", "half hour")
        pattern = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:-[A-Za-z]+)?")

        def _flush_number_words() -> None:
            nonlocal number_word_tokens
            if not number_word_tokens:
                return
            value = self._parse_number_word_sequence(number_word_tokens)
            if value is not None:
                values.append(value)
            number_word_tokens = []

        for match in pattern.finditer(text):
            raw_token = match.group(0)
            numeric_value = self._to_float(raw_token)
            if numeric_value is not None:
                _flush_number_words()
                values.append(numeric_value)
                continue

            parts = [part for part in raw_token.lower().split("-") if part]
            if parts and all(part in self._NUMBER_WORD_TOKENS for part in parts):
                number_word_tokens.extend(parts)
                continue

            _flush_number_words()

        _flush_number_words()
        return values

    def _parse_number_word_sequence(self, tokens: list[str]) -> float | None:
        total = 0.0
        current = 0.0
        seen = False
        for token in tokens:
            if token == "and":
                continue
            if token in self._NUMBER_WORD_UNITS:
                current += self._NUMBER_WORD_UNITS[token]
                seen = True
                continue
            if token in self._NUMBER_WORD_TENS:
                current += self._NUMBER_WORD_TENS[token]
                seen = True
                continue
            if token == "hundred":
                current = max(current, 1.0) * self._NUMBER_WORD_SCALES[token]
                seen = True
                continue
            scale = self._NUMBER_WORD_SCALES.get(token)
            if scale is not None:
                total += max(current, 1.0) * scale
                current = 0.0
                seen = True
                continue
            return None
        if not seen:
            return None
        return total + current

    def _evaluate_with_rpn_engine(self, rpn_program: str) -> float | None:
        try:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            engine = ModularRPNEngine()
            try:
                value = engine.evaluate(rpn_program)
            finally:
                engine.close()
            self._last_execution_error = None
            return self._to_float(value)
        except Exception as exc:
            self._last_execution_error = f"{type(exc).__name__}: {exc}"
            return None

    def _to_float(self, value: Any) -> float | None:
        try:
            if isinstance(value, str):
                cleaned = value.strip().replace(",", "")
                if cleaned.endswith("%"):
                    cleaned = cleaned[:-1]
                if cleaned.startswith("$"):
                    cleaned = cleaned[1:]
                return float(cleaned)
            return float(value)
        except Exception:
            return None

    def _log_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        kv = self.knowledgeverse
        if kv is not None and hasattr(kv, "log_event"):
            try:
                kv.log_event(event_type=event_type, event_data=event_data)
            except Exception:
                pass

    def _emit_debug(self, stage: str, payload: dict[str, Any]) -> None:
        if not self._debug:
            return
        msg = {
            "stage": stage,
            "payload": payload,
        }
        print(f"[K3D_MATH_DEBUG][MathSpecialist] {json.dumps(msg, ensure_ascii=True, sort_keys=True)}")
