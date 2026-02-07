"""Knowledgeverse TRM navigator with resilient composition helpers."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Sequence

from .galaxy_manager import GalaxyManager
from .navigator_specialist import NavigatorSpecialist
from .resilience import SelfHealingWrapper
from .specialist_router import SpecialistRouter


class TRMNavigator:
    """Deterministic navigator surface used by benchmark/integration flows."""

    def __init__(self, knowledgeverse: Any | None = None, galaxy_manager: GalaxyManager | None = None):
        self.knowledgeverse = knowledgeverse
        self.galaxy_manager = galaxy_manager or getattr(knowledgeverse, "galaxy_manager", None) or GalaxyManager()
        self.specialist_router = SpecialistRouter()
        self.navigator_specialist = NavigatorSpecialist(
            knowledgeverse=knowledgeverse,
            router=self.specialist_router,
        )
        self._trace: list[str] = []

    @SelfHealingWrapper.circuit_breaker(failure_threshold=5, timeout=60.0)
    def navigate_and_compose(
        self,
        query: str,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        if specialist == "auto":
            composed = self.navigator_specialist.navigate_and_compose(
                trm_navigator=self,
                query=query,
                use_enriched=use_enriched,
                specialist=specialist,
                domain_hint=domain_hint,
            )
            return composed

        route = self.route(query=query, specialist=specialist, domain_hint=domain_hint)
        results = self.query(
            query=query,
            galaxy_names=route["galaxy_names"],
            top_k=20,
            specialist=route["specialist"],
            domain_hint=route["domain"],
        )
        composed = self.compose(
            query=query,
            patterns=results,
            specialist=route["specialist"],
            use_enriched=use_enriched,
        )
        if isinstance(composed, dict):
            composed["route"] = route
        return composed

    def route(
        self,
        query: str,
        *,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        resolved = self.specialist_router.route(
            query=query,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        self._trace.append(
            "route specialist="
            f"{resolved['specialist']} domain={resolved['domain']} reason={resolved['reason']}"
        )
        return resolved

    def query(
        self,
        query: str,
        galaxy_names: Sequence[str] | None = None,
        top_k: int = 10,
        specialist: str = "any",
        domain_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        route = self.route(
            query=query,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        resolved_specialist = str(route["specialist"])
        names = [str(name) for name in route["galaxy_names"]] if route["galaxy_names"] else None
        self._trace.append(f"query specialist={resolved_specialist} top_k={top_k}")

        if not names:
            return self.galaxy_manager.query(
                query_text=query,
                specialist=resolved_specialist,
                top_k=top_k,
            )

        tokens = {tok for tok in re.split(r"[^A-Za-z0-9_]+", query.lower()) if tok}
        scored: list[tuple[int, dict[str, Any], str]] = []
        for name in names:
            galaxy = self.galaxy_manager.get_galaxy(name)
            for entry in galaxy.entries:
                haystack = str(entry).lower()
                score = sum(1 for tok in tokens if tok in haystack)
                if score <= 0 and tokens:
                    continue
                scored.append((max(score, 1), entry, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"galaxy": name, "score": score, "entry": entry}
            for score, entry, name in scored[: max(1, int(top_k))]
        ]

    def compose(
        self,
        query: str | None = None,
        patterns: Sequence[dict[str, Any]] | None = None,
        specialist: str = "math",
        task_examples: Sequence[dict[str, Any]] | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        self._trace.append(f"compose specialist={specialist} enriched={use_enriched}")
        if task_examples:
            transform = self._infer_arc_transform(task_examples, prefer_enriched=use_enriched)
            return {
                "program_type": "arc_transform",
                "transform": transform,
                "specialist": specialist,
                "patterns_used": len(patterns or []),
            }

        return {
            "program_type": "math_expression",
            "expression": query or "",
            "specialist": specialist,
            "patterns_used": len(patterns or []),
            "use_enriched": bool(use_enriched),
        }

    def execute(self, program: dict[str, Any], input_data: Any | None = None) -> Any:
        program_type = str(program.get("program_type", "unknown"))
        self._trace.append(f"execute type={program_type}")

        if program_type == "arc_transform":
            if input_data is None:
                raise ValueError("ARC execution requires input_data grid")
            transform = program.get("transform", {"op": "identity"})
            return self._apply_arc_transform(input_data, transform)

        if program_type == "math_expression":
            expression = str(program.get("expression", ""))
            use_enriched = bool(program.get("use_enriched", True))
            return self._solve_math(expression, use_enriched=use_enriched)

        raise ValueError(f"Unsupported program type: {program_type}")

    def select_answer(self, reasoning: Any, options: Sequence[str]) -> str:
        self._trace.append("select_answer")
        if not options:
            return ""
        numeric_reasoning = self._to_float(reasoning)
        if numeric_reasoning is not None:
            for option in options:
                val = self._to_float(option)
                if val is None:
                    continue
                if abs(val - numeric_reasoning) <= 1e-6:
                    return str(option)
        reason = str(reasoning).strip().lower()
        for option in options:
            normalized = str(option).strip().lower()
            if normalized == reason or reason in normalized:
                return str(option)
        return str(options[0])

    def get_reasoning_trace(self) -> list[str]:
        return list(self._trace)

    def clear_trace(self) -> None:
        self._trace.clear()

    def learn_from_feedback(self, *, query: str, specialist: str, success: bool) -> None:
        """Update persistent routing weights from observed outcomes."""
        self.navigator_specialist.learn_routing_topology(
            query=query,
            specialist=specialist,
            success=success,
        )

    def consolidate_weights_from_events(self, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """
        Consolidate routing weights from buffered Shadow Copy events.

        Returns a lightweight summary for SleepTime reporting.
        """
        updated = 0
        specialists: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "")).lower()
            data = event.get("data", {})
            if not isinstance(data, dict):
                data = {}
            specialist = str(event.get("specialist") or data.get("specialist") or "grammar")
            query = str(data.get("query") or data.get("prompt") or event_type or specialist)
            success = ("success" in event_type) or (
                "fail" not in event_type and float(event.get("confidence", 0.0)) >= 0.65
            )
            self.learn_from_feedback(query=query, specialist=specialist, success=success)
            updated += 1
            specialists.add(specialist)
        self.navigator_specialist.save_state()
        return {
            "updated_count": updated,
            "updated_specialists": sorted(specialists),
            "weights_path": str(self.navigator_specialist.weight_store.path),
        }

    def save_weights(self) -> None:
        self.navigator_specialist.save_state()

    def _infer_arc_transform(
        self,
        task_examples: Sequence[dict[str, Any]],
        *,
        prefer_enriched: bool,
    ) -> dict[str, Any]:
        # Empty-mind baseline intentionally limits adaptation.
        if not prefer_enriched:
            return {"op": "identity"}

        # Galaxy-first ARC transform proposal from Grammar Galaxy.
        grammar_best_transform: dict[str, Any] | None = None
        grammar_best_score = -1.0
        grammar_rule_id: str | None = None
        try:
            grammar = self.galaxy_manager.get_galaxy("Grammar")
            proposer = getattr(grammar, "propose_arc_transform", None)
            if callable(proposer):
                proposal = proposer(list(task_examples))
                proposed = proposal.get("transform")
                confidence = float(proposal.get("confidence", 0.0))
                if isinstance(proposed, dict):
                    grammar_best_transform = proposed
                    grammar_best_score = confidence
                    grammar_rule_id = str(proposal.get("rule_id", ""))
        except Exception:
            grammar_best_transform = None
            grammar_best_score = -1.0
            grammar_rule_id = None

        op_candidates = (
            "identity",
            "flip_h",
            "flip_v",
            "rot90",
            "rot180",
            "rot270",
            "transpose",
        )
        best_transform: dict[str, Any] = {"op": "identity"}
        best_score = -1.0
        for op in op_candidates:
            score = 0.0
            for example in task_examples:
                predicted = self._apply_arc_transform(example["input"], {"op": op})
                score += self._grid_match_score(predicted, example["output"])
            avg_score = score / max(1, len(task_examples))
            if avg_score > best_score:
                best_score = avg_score
                best_transform = {"op": op}

        mapping = self._infer_color_mapping(task_examples)
        if mapping:
            score = 0.0
            for example in task_examples:
                predicted = self._apply_arc_transform(
                    example["input"],
                    {"op": "color_map", "mapping": mapping},
                )
                score += self._grid_match_score(predicted, example["output"])
            avg_score = score / max(1, len(task_examples))
            if avg_score > best_score:
                best_score = avg_score
                best_transform = {"op": "color_map", "mapping": mapping}

        if best_score >= 0.45:
            if grammar_best_transform is not None and grammar_best_score >= best_score:
                self._trace.append(
                    f"arc_transform source=grammar rule={grammar_rule_id or 'unknown'} confidence={grammar_best_score:.3f}"
                )
                return grammar_best_transform
            return best_transform

        if grammar_best_transform is not None and grammar_best_score >= 0.35:
            self._trace.append(
                f"arc_transform source=grammar rule={grammar_rule_id or 'unknown'} confidence={grammar_best_score:.3f}"
            )
            return grammar_best_transform

        return {"op": "identity"}

    def _infer_color_mapping(self, task_examples: Sequence[dict[str, Any]]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for example in task_examples:
            inp = example["input"]
            out = example["output"]
            if len(inp) != len(out):
                return {}
            for in_row, out_row in zip(inp, out):
                if len(in_row) != len(out_row):
                    return {}
                for in_val, out_val in zip(in_row, out_row):
                    in_int = int(in_val)
                    out_int = int(out_val)
                    prev = mapping.get(in_int)
                    if prev is None:
                        mapping[in_int] = out_int
                    elif prev != out_int:
                        return {}
        return mapping

    def _apply_arc_transform(self, grid: Sequence[Sequence[int]], transform: dict[str, Any]) -> list[list[int]]:
        op = str(transform.get("op", "identity"))
        rows = [list(map(int, row)) for row in grid]
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

    def _grid_match_score(
        self,
        predicted: Sequence[Sequence[int]],
        expected: Sequence[Sequence[int]],
    ) -> float:
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

    def _solve_math(self, text: str, *, use_enriched: bool) -> float | None:
        # Empty-mind baseline supports only direct arithmetic.
        if not use_enriched:
            expr = self._extract_arithmetic_expr(text)
            return self._safe_eval(expr) if expr else None

        derivative = self._solve_derivative_prompt(text)
        if derivative is not None:
            return derivative

        expr = self._extract_arithmetic_expr(text)
        if expr:
            val = self._safe_eval(expr)
            if val is not None:
                return val
        return None

    def _solve_derivative_prompt(self, text: str) -> float | None:
        lowered = text.lower()
        x_value = self._extract_eval_x(text)
        if "sin(x)" in lowered and x_value is not None:
            return math.cos(x_value)
        if "cos(x)" in lowered and x_value is not None:
            return -math.sin(x_value)
        if "e^x" in lowered and x_value is not None:
            return math.exp(x_value)

        quotient_match = re.search(
            r"f\(x\)\s*=\s*\(([-+]?\d+)x([+-]\d+)\)\s*/\s*\(([-+]?\d+)x([+-]\d+)\)",
            text.replace(" ", ""),
        )
        if quotient_match and x_value is not None:
            a, b, c, d = [float(part) for part in quotient_match.groups()]
            numerator = (a * c * x_value + a * d) - (a * c * x_value + b * c)
            denominator = (c * x_value + d) ** 2
            if denominator == 0:
                return None
            return numerator / denominator

        poly_match = re.search(r"derivative of ([^@]+?) at x\s*=\s*([-+]?\d*\.?\d+)", lowered)
        if poly_match:
            expr_raw = poly_match.group(1)
            eval_x = float(poly_match.group(2))
            return self._differentiate_polynomial(expr_raw, eval_x)

        generic = re.search(r"f\(x\)\s*=\s*([^,]+?)\s+at x\s*=\s*([-+]?\d*\.?\d+)", lowered)
        if generic:
            expr_raw = generic.group(1)
            eval_x = float(generic.group(2))
            return self._differentiate_polynomial(expr_raw, eval_x)

        return None

    def _differentiate_polynomial(self, expr_raw: str, x_value: float) -> float | None:
        expr = expr_raw.replace(" ", "")
        if "/" in expr and "x" in expr:
            return None
        normalized = expr.replace("-", "+-")
        terms = [term for term in normalized.split("+") if term]
        result = 0.0
        matched_any = False
        for term in terms:
            if "x^" in term:
                coef_part, pow_part = term.split("x^", 1)
                coef = self._parse_coef(coef_part)
                power = self._to_float(pow_part)
                if power is None:
                    continue
                matched_any = True
                result += coef * power * (x_value ** (power - 1))
                continue
            if term.endswith("x"):
                coef = self._parse_coef(term[:-1])
                matched_any = True
                result += coef
                continue
        if not matched_any:
            return None
        return result

    def _parse_coef(self, raw: str) -> float:
        if raw in ("", "+"):
            return 1.0
        if raw == "-":
            return -1.0
        val = self._to_float(raw)
        return float(val) if val is not None else 0.0

    def _extract_eval_x(self, text: str) -> float | None:
        match = re.search(r"at x\s*=\s*([-+]?\d*\.?\d+)", text.lower())
        if not match:
            return None
        return float(match.group(1))

    def _extract_arithmetic_expr(self, text: str) -> str | None:
        # First quoted/extracted expressions.
        match = re.search(r"([\-+*/()0-9\.\s]{3,})", text)
        if match:
            expr = match.group(1).strip()
            if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/"):
                return expr
        # Last fallback: collect tokens.
        tokens = re.findall(r"[-+]?\d*\.?\d+|[()+\-*/]", text)
        if len(tokens) >= 3 and any(tok in "+-*/" for tok in tokens):
            return " ".join(tokens)
        return None

    def _safe_eval(self, expr: str) -> float | None:
        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError:
            return None
        if not self._is_safe_math_ast(node):
            return None
        try:
            value = eval(compile(node, "<math>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None
        return self._to_float(value)

    def _is_safe_math_ast(self, node: ast.AST) -> bool:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
            ast.Mod,
            ast.FloorDiv,
        )
        for child in ast.walk(node):
            if not isinstance(child, allowed_nodes):
                return False
            if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float)):
                return False
        return True

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None
