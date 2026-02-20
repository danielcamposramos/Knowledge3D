"""Math specialist: Galaxy-first RPN composition for deterministic math solving."""

from __future__ import annotations

import json
import os
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
    }
    _BOOTSTRAP_GRAMMAR_BY_PATTERN: dict[str, str] = {
        "linear_equation": "grammar_linear_equation_ax_plus_b_eq_c_v1",
        "linear_equation_ax_minus_b_eq_c": "grammar_linear_equation_ax_minus_b_eq_c_v1",
        "linear_equation_b_plus_ax_eq_c": "grammar_linear_equation_b_plus_ax_eq_c_v1",
        "arithmetic_add": "grammar_arithmetic_addition_ab_v1",
        "arithmetic_subtract": "grammar_arithmetic_subtraction_ab_v1",
        "ratio": "grammar_ratio_a_to_b_v1",
        "proportion": "grammar_proportion_a_over_b_eq_c_over_x_v1",
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
        self._emit_debug(
            "solve_start",
            {"question": question[:240], "use_enriched": bool(use_enriched)},
        )
        if not question:
            self._emit_debug("solve_error", {"reason": "missing_question"})
            return {"status": "error", "reason": "missing_question"}

        manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if manager is None:
            self._emit_debug("solve_error", {"reason": "missing_galaxy_manager"})
            return {"status": "error", "reason": "missing_galaxy_manager"}

        problem_type = self._infer_problem_type(question)
        self._emit_debug("problem_type", {"pattern_type": problem_type})

        try:
            grammar_candidates = manager.query(
                query_text=f"{question} {problem_type} pattern",
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

        coefficients = self._extract_coefficients(question, pattern_type=problem_type)
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

        try:
            template_candidates = manager.query(
                query_text=f"solve {problem_type}",
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

    def _infer_problem_type(self, question: str) -> str:
        lowered = question.lower()
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

    def _extract_coefficients(self, question: str, *, pattern_type: str) -> dict[str, float] | None:
        kind = str(pattern_type).strip().lower()
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
        return None

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

    def _compose_from_template(self, *, template: dict[str, Any], coefficients: dict[str, float]) -> str | None:
        rpn_template = str(template.get("rpn_program", "")).strip()
        if not rpn_template:
            return None
        rendered = rpn_template
        for name, value in coefficients.items():
            rendered = rendered.replace(f"{{{name}}}", f"{value:.12g}")
        if "{" in rendered and "}" in rendered:
            return None
        return rendered

    def _extract_numeric_literals(self, text: str) -> list[float]:
        values: list[float] = []
        token: list[str] = []
        for char in text:
            if char.isdigit() or char in {".", "-"}:
                token.append(char)
                continue
            if token:
                value = self._to_float("".join(token))
                if value is not None:
                    values.append(value)
                token = []
        if token:
            value = self._to_float("".join(token))
            if value is not None:
                values.append(value)
        return values

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
