"""
Algebra solver that chained regex preprocessing (deprecated).

Deprecated in favor of MathSymbolGalaxy + SovereignComposer sovereign parsing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from knowledge3d.training.math_benchmarks.problem_classifier import ProblemClassifier, ProblemClassification
from knowledge3d.training.math_benchmarks.strategy_selector import StrategySelector, SolvingStrategy

DEPRECATION_MSG = "AlgebraSolver is deprecated. Use SovereignComposer + MathSymbolGalaxy for RPN composition."


class AlgebraSolver:
    def __init__(self, rpn_engine=None):
        raise RuntimeError(DEPRECATION_MSG)

    @property
    def rpn_engine(self):
        if self._rpn_engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
            self._rpn_engine = ModularRPNEngine()
        return self._rpn_engine

    def solve(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
        self._execution_trace = []
        classification = self.classifier.classify(problem_text)
        self._trace("classify", {"classification": classification})

        strategy = self.strategy_selector.select(classification)
        if strategy is None:
            return None, {"error": "no_strategy", "classification": classification}
        self._trace("strategy", {"strategy": strategy.strategy_name})

        variables = self._extract_variables(problem_text, classification, strategy)
        self._trace("variables", {"variables": variables})

        results = []
        for i, rpn_template in enumerate(strategy.rpn_chains):
            if not rpn_template.strip():
                continue
            rpn_program = self._substitute_variables(rpn_template, variables)
            self._trace(f"rpn_step_{i}", {"template": rpn_template, "program": rpn_program})
            tokens = self._parse_rpn(rpn_program)
            if not tokens:
                continue
            try:
                expr = " ".join(str(t) for t in tokens)
                result = self.rpn_engine.evaluate(expr)
                results.append(result)
                self._trace(f"result_{i}", {"result": result})
                if "STORE_" in rpn_template:
                    var_match = re.search(r"STORE_(\w+)", rpn_template)
                    if var_match:
                        variables[var_match.group(1).lower()] = result
            except Exception as e:
                self._trace(f"error_{i}", {"error": str(e)})

        # Direct expression fallback if strategy produced nothing
        if (not results) or results[-1] is None:
            direct_result = self._try_direct_expression_eval(problem_text)
            if direct_result is not None:
                results.append(direct_result)
                self._trace("direct_eval", {"result": direct_result})

        final_answer = results[-1] if results else None
        metadata = {
            "classification": {
                "type": classification.problem_type,
                "subtype": classification.subtype,
                "confidence": classification.confidence,
            },
            "strategy": strategy.strategy_name,
            "variables": variables,
            "trace": self._execution_trace,
        }
        return final_answer, metadata

    def _extract_variables(self, problem_text: str, classification: ProblemClassification, strategy: SolvingStrategy) -> Dict[str, float]:
        variables = dict(classification.coefficients)
        assignments = re.findall(r"\b([a-z])\s*=\s*(-?\d+\.?\d*)", problem_text.lower())
        for var, val in assignments:
            variables[var] = float(val)

        n_match = re.search(r"(?:first|sum of)\s+(\d+)", problem_text.lower())
        if n_match and "n" not in variables:
            variables["n"] = float(n_match.group(1))

        choose_match = re.search(r"(?:choose|select|pick)\s+(\d+)\s+(?:from|out of)\s+(\d+)", problem_text.lower())
        if choose_match:
            variables["k"] = float(choose_match.group(1))
            variables["n"] = float(choose_match.group(2))

        quad_match = re.search(r"x[²\^2]\s*([+\-])\s*(\d+)\s*x\s*([+\-])\s*(\d+)", problem_text)
        if quad_match:
            variables.setdefault("a", 1.0)
            sign_b = 1 if quad_match.group(1) == "+" else -1
            variables["b"] = sign_b * float(quad_match.group(2))
            sign_c = 1 if quad_match.group(3) == "+" else -1
            variables["c"] = sign_c * float(quad_match.group(4))

        # Fill required vars from numbered coefficients as fallback
        numbered = [v for k, v in variables.items() if k.startswith("n")]
        idx = 0
        for req in strategy.required_vars:
            if req not in variables and idx < len(numbered):
                variables[req] = numbered[idx]
                idx += 1

        return variables

    def _substitute_variables(self, template: str, variables: Dict[str, float]) -> str:
        result = template
        for var, val in variables.items():
            result = result.replace(f"{{{var}}}", str(val))
        return result

    def _parse_rpn(self, program: str) -> List[Any]:
        tokens: List[Any] = []
        for token in program.split():
            if token.startswith("STORE_") or token.startswith("RECALL_"):
                continue
            try:
                tokens.append(float(token))
            except ValueError:
                tokens.append(token.lower())
        return tokens

    def _trace(self, step: str, data: Dict[str, Any]) -> None:
        self._execution_trace.append({"step": step, **data})

    def solve_batch(self, problems: List[str]) -> List[Tuple[Any, Dict[str, Any]]]:
        return [self.solve(p) for p in problems]

    def _normalize_latex(self, text: str) -> str:
        result = text
        result = re.sub(r"\$\$?", "", result)
        result = re.sub(r"\\\[|\\\]", "", result)
        result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
        while r"\frac" in result:
            result = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1/\2)", result)
        result = re.sub(r"\\binom\{(\d+)\}\{(\d+)\}", r"binom(\1,\2)", result)
        result = re.sub(r"\^{(\d+)}", r"^\1", result)
        result = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", result)
        result = re.sub(r"\\[a-zA-Z]+", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _try_direct_expression_eval(self, problem_text: str) -> Optional[float]:
        expr_pattern = r"([\d\.\+\-\*\/\^\!\(\)\s]+)"
        text = self._normalize_latex(problem_text)
        matches = re.findall(expr_pattern, text)
        for match in reversed(matches):
            tokens = self._expr_to_rpn(match.strip())
            if tokens:
                try:
                    expr = " ".join(str(t) for t in tokens)
                    result = self.rpn_engine.evaluate(expr)
                    if result is not None:
                        return result
                except Exception:
                    continue
        return None

    def _expr_to_rpn(self, expr: str) -> List[Any]:
        binom_match = re.search(r"binom\((\d+),(\d+)\)", expr)
        if binom_match:
            n, k = float(binom_match.group(1)), float(binom_match.group(2))
            return [n, k, "binomial"]

        token_pattern = r"(\d+\.?\d*|\+|\-|\*|\/|\^|\!|\(|\))"
        raw_tokens = re.findall(token_pattern, expr)
        output: List[Any] = []
        op_stack: List[str] = []
        precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3, "!": 4}
        right_assoc = {"^"}

        for tok in raw_tokens:
            if re.match(r"\d", tok):
                output.append(float(tok))
            elif tok == "!":
                output.append("factorial")
            elif tok == "(":
                op_stack.append(tok)
            elif tok == ")":
                while op_stack and op_stack[-1] != "(":
                    output.append(self._op_to_rpn(op_stack.pop()))
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
            elif tok in precedence:
                while (
                    op_stack
                    and op_stack[-1] != "("
                    and op_stack[-1] in precedence
                    and (
                        precedence[op_stack[-1]] > precedence[tok]
                        or (precedence[op_stack[-1]] == precedence[tok] and tok not in right_assoc)
                    )
                ):
                    output.append(self._op_to_rpn(op_stack.pop()))
                op_stack.append(tok)

        while op_stack:
            output.append(self._op_to_rpn(op_stack.pop()))
        return output

    def _op_to_rpn(self, op: str) -> str:
        mapping = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "pow", "!": "factorial"}
        return mapping.get(op, op)
