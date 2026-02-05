"""
Recursive Solver for Calculus Problems.

Implements the "Divide and Conquer" strategy:
1. Parse expression to AST (using SymPy as oracle).
2. Recursively traverse AST.
3. Apply atomic grammar rules (Power, Product, Quotient, Chain) at each node.
4. Compose numeric results.

This avoids the need for hard-coded regexes for complex nested expressions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, Union, List

try:
    import sympy
    from sympy import Add, Mul, Pow, Symbol, Integer, Float, Number
    from sympy.core.function import AppliedUndef
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except ImportError:
    sympy = None

from knowledge3d.training.math_benchmarks.latex_normalizer import normalize_latex_to_natural
from knowledge3d.training.math_benchmarks.router_embedder import embed_text
from knowledge3d.cranium.sovereign_trm import PAD_ID, BOS_ID, RULE_OFFSET


class RecursiveSolver:
    """
    Solves numeric derivative problems by decomposing expressions.
    """

    def __init__(
        self,
        verbose: bool = False,
        policy_model: Optional[Any] = None,
        policy_registry: Optional[List[str]] = None,
    ):
        self.verbose = verbose
        self._policy_model = policy_model
        self._policy_registry = list(policy_registry or [])
        self._policy_sequence: List[str] = []
        self._policy_index = 0
        self._last_trace_lines: List[str] = []
        self._last_steps: List[Dict[str, Any]] = []
        self._last_expression: Optional[str] = None
        self._last_point: Optional[float] = None
        self._neural_steps = 0
        self._heuristic_steps = 0
        self._policy_mismatches = 0
        if not sympy:
            print("Warning: SymPy not found. RecursiveSolver disabled.")

    def solve(self, problem_text: str) -> Optional[float]:
        """
        Main entry point. Parses text, extracts f(x) and point, solves.
        """
        if not sympy:
            return None
        normalized_text = normalize_latex_to_natural(problem_text)
        self._init_policy_sequence(normalized_text)
        self._neural_steps = 0
        self._heuristic_steps = 0
        self._policy_mismatches = 0

        # 1. Extract evaluation point "at x=..." or "f'(...)"
        point, var_char = self._extract_point(normalized_text)
        if point is None:
            return None
        
        # 2. Extract function definition
        func_expr = self._extract_function(normalized_text, var_char)
        if not func_expr:
            return None

        if self.verbose:
            print(f"[RecursiveSolver] Function: {func_expr}, Point: {var_char}={point}")

        try:
            # 3. Recursive Solve
            var_sym = Symbol(var_char)
            result, trace = self._differentiate_eval(func_expr, var_sym, point)
            steps = self._trace_lines_to_steps(trace)
            self._last_trace_lines = list(trace)
            self._last_steps = list(steps)
            self._last_expression = str(func_expr)
            self._last_point = float(point)

            if self._policy_sequence and self._heuristic_steps > 0:
                print("[Neural] Autonomy fallback detected")
                print(f"[Neural] Problem: {problem_text}")
                print(f"[Neural] Policy sequence: {self._policy_sequence}")
                print(f"[Neural] Mismatches: {self._policy_mismatches}")
            
            if self.verbose:
                print("--- TRACE START ---")
                for line in trace:
                    print(line)
                print("--- TRACE END ---")
                
            return float(result)
        except Exception as e:
            if self.verbose:
                print(f"[RecursiveSolver] Error: {e}")
            return None

    def get_last_trace(self) -> Dict[str, Any]:
        mode = "heuristic"
        if self._neural_steps > 0:
            if self._heuristic_steps == 0:
                mode = "neural"
            else:
                mode = "mixed"
        
        # Resonance Diagnostic: Report non-neural thoughts
        if mode != "neural":
            print(f"[RecursiveSolver] Drift Detected ({mode}): {self._last_expression}")
            print(f"  Neural/Heuristic Steps: {self._neural_steps}/{self._heuristic_steps}")
            print(f"  Crystal Emanations: {self._policy_sequence}")

        return {
            "trace_lines": list(self._last_trace_lines),
            "step_sequence": list(self._last_steps),
            "expression": self._last_expression,
            "point": self._last_point,
            "policy_mode": mode,
            "policy_steps": self._neural_steps,
            "policy_mismatches": self._policy_mismatches,
        }

    def _extract_point(self, text: str) -> Tuple[Optional[float], str]:
        # Try "at x=2"
        m = re.search(r"at\s+([a-zA-Z])\s*=\s*([-+]?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            return float(m.group(2)), m.group(1)
        
        # Try "f'(2)"
        m = re.search(r"[a-zA-Z]'\(\s*([-+]?\d+\.?\d*)\s*\)", text)
        if m:
            # Assume variable is 'x' if not specified, or infer from context later
            return float(m.group(1)), 'x'
            
        # Try "evaluation of ... at 2" (simple fallback)
        m = re.search(r"at\s*([-+]?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            return float(m.group(1)), 'x'

        return None, 'x'

    def _init_policy_sequence(self, text: str) -> None:
        self._policy_sequence = []
        self._policy_index = 0
        if self._policy_model is None or not self._policy_registry:
            return
        try:
            import torch  # type: ignore
        except Exception:
            return
        dim = getattr(self._policy_model.encoder, "in_features", 256)
        embedding = embed_text(text, dim=dim)
        device = next(self._policy_model.parameters()).device
        emb = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0)
        max_len = 64
        tokens = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
        tokens[0, 0] = BOS_ID
        for step in range(1, max_len):
            logits = self._policy_model(emb, tokens[:, :step])
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id == PAD_ID:
                break
            tokens[0, step] = next_id
        predicted_ids: List[int] = []
        for tok in tokens[0].tolist():
            if tok >= RULE_OFFSET:
                predicted_ids.append(int(tok) - RULE_OFFSET)
        self._policy_sequence = [
            self._policy_registry[i] if 0 <= i < len(self._policy_registry) else f"unknown_{i}"
            for i in predicted_ids
        ]

    def _peek_policy_rule(self) -> Optional[str]:
        if self._policy_index < len(self._policy_sequence):
            return self._policy_sequence[self._policy_index]
        return None

    def _consume_policy_rule(self) -> Optional[str]:
        rule = self._peek_policy_rule()
        if rule is None:
            return None
        self._policy_index += 1
        print(f"[Neural] Selected rule: {rule}")
        return rule

    def _extract_function(self, text: str, var_char: str) -> Optional[Any]:
        """
        Extracts the mathematical expression f(x) from the text.
        Returns a SymPy expression object.
        """
        # Heuristics to find the math part.
        
        # Case 1: Explicit "f(x) = "
        # Note: SymPy parsing is fragile with natural language. We need to isolate the math.
        
        # Try to find the equation part
        candidates = []
        
        # "f(x) = x^2 + 2x" (allow any leading words)
        m = re.search(r"[a-zA-Z]\([a-zA-Z]\)\s*=\s*([^,;.]+)", text)
        if m:
            candidates.append(m.group(1).strip(" ."))
        # "where f(x) = x^2 + 2x"
        m = re.search(r"where\s+[a-zA-Z]\([a-zA-Z]\)\s*=\s*(.*)", text, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip(" ."))

        # "derivative of x^2 + 2x"
        m = re.search(r"derivative of\s+(.*?)(?:\s+at\s+|$)", text, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip(" ."))
            
        # "d/dx [ ... ]"
        m = re.search(r"\\frac\{d\}\{d[a-zA-Z]\}\s*\[(.*?)(\s*)\]", text)
        if m:
            candidates.append(m.group(1))

        if not candidates:
            # Fallback for "Given f(x) = ... , find ..." or simple "f'(2) where f(x)=..."
            if "=" in text:
                parts = text.split("=")
                # Take the part after the equals sign
                rhs = parts[1]
                # Stop at comma or "find" or "evaluate"
                for stop in [",", "find", "evaluate", "calculate", "at"]:
                    if stop in rhs.lower():
                        # Split by delimiter, take first part
                        rhs = re.split(f"{stop}", rhs, flags=re.IGNORECASE)[0]
                candidates.append(rhs.strip(" ."))

        for cand in candidates:
            try:
                clean = cand.strip()
                clean = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", clean)
                clean = re.sub(r"\\cdot", "*", clean)
                clean = clean.replace("{(", "(").replace(")}", ")")
                # Handle implicit multiplication carefully if possible, but standard transform covers common cases
                expr = parse_expr(
                    clean,
                    local_dict={"e": sympy.E, "pi": sympy.pi},
                    transformations=standard_transformations
                    + (implicit_multiplication_application, convert_xor),
                )
                return expr
            except Exception:
                continue
                
        return None

    def _trace_lines_to_steps(self, trace_lines: List[str]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        rule_map = {
            "sum rule": "sum_rule",
            "product/const rule": "product_rule",
            "quotient rule": "quotient_rule",
            "power rule": "power_rule",
            "power (const base)": "power_const_base",
            "sin chain rule": "sin_rule",
            "cos chain rule": "cos_rule",
            "exp chain rule": "exp_rule",
        }
        for line in trace_lines:
            if "[Decompose]" in line:
                # Match optional <status> tag: [Decompose] <honest> Sum Rule: ...
                match = re.search(r"\[Decompose\](?:\s+<([^>]+)>)?\s+([^:]+):\s*(.+)", line)
                if not match:
                    continue
                status = match.group(1) or "heuristic"
                label = match.group(2).strip()
                expr = match.group(3).strip()
                key = label.lower()
                steps.append(
                    {
                        "kind": "decompose",
                        "rule": rule_map.get(key, key.replace(" ", "_")),
                        "status": status,
                        "label": label,
                        "expr": expr,
                    }
                )
            elif "[Base]" in line:
                match = re.search(r"\[Base\](?:\s+<([^>]+)>)?\s+(.+?)\s+->", line)
                if not match:
                    continue
                status = match.group(1) or "heuristic"
                steps.append(
                    {
                        "kind": "base",
                        "status": status,
                        "label": match.group(1).strip(),
                    }
                )
        return steps

    def _evaluate_func(self, expr: Any, var: Any, val: float) -> float:
        """Evaluates f(x) at x=val."""
        return float(expr.subs(var, val))

    def _differentiate_eval(self, expr: Any, var: Any, val: float, depth: int = 0) -> Tuple[float, List[str]]:
        """
        Recursively computes f'(val) using differentiation rules.
        Returns (result, trace_log).
        """
        trace = []
        indent = "  " * depth

        forced_rule: Optional[str] = None
        step_status = "heuristic"
        policy_rule = self._peek_policy_rule()
        if policy_rule:
            pr_lower = policy_rule.lower()
            pr_norm = re.sub(r"[^a-z0-9]+", "_", pr_lower).strip("_")
            if pr_norm.startswith("constant") and (isinstance(expr, (int, float)) or expr.is_Number):
                self._consume_policy_rule()
                self._neural_steps += 1
                forced_rule = "constant"
                step_status = "honest"
            elif pr_norm.startswith("variable") and isinstance(expr, Symbol):
                self._consume_policy_rule()
                self._neural_steps += 1
                forced_rule = "variable"
                step_status = "honest"
            elif pr_norm == "sum_rule" and expr.is_Add:
                forced_rule = "sum_rule"
                self._consume_policy_rule()
                self._neural_steps += 1
                step_status = "honest"
            elif pr_norm == "quotient_rule" and expr.is_Mul:
                num, den = expr.as_numer_denom()
                if den is not None and den != 1:
                    forced_rule = "quotient_rule"
                    self._consume_policy_rule()
                    self._neural_steps += 1
                    step_status = "honest"
            elif pr_norm == "product_rule" and expr.is_Mul:
                forced_rule = "product_rule"
                self._consume_policy_rule()
                self._neural_steps += 1
                step_status = "honest"
            elif pr_norm == "power_rule" and expr.is_Pow:
                forced_rule = "power_rule"
                self._consume_policy_rule()
                self._neural_steps += 1
                step_status = "honest"
            elif pr_norm == "power_const_base" and expr.is_Pow:
                base, exp = expr.args
                if exp.has(var) and not base.has(var):
                    forced_rule = "power_const_base"
                    self._consume_policy_rule()
                    self._neural_steps += 1
                    step_status = "honest"
            elif pr_norm == "sin_rule" and expr.func == sympy.sin:
                forced_rule = "sin_rule"
                self._consume_policy_rule()
                self._neural_steps += 1
                step_status = "honest"
            elif pr_norm == "cos_rule" and expr.func == sympy.cos:
                forced_rule = "cos_rule"
                self._consume_policy_rule()
                self._neural_steps += 1
                step_status = "honest"
            elif pr_norm == "exp_rule" and expr.func == sympy.exp:
                forced_rule = "exp_rule"
                self._consume_policy_rule()
                self._neural_steps += 1
                step_status = "honest"
            else:
                self._policy_mismatches += 1
                step_status = "hallucination"
                if self.verbose:
                    print(f"[Neural] Mismatch: predicted {policy_rule} for expr {expr}")
        
        if forced_rule is None:
            self._heuristic_steps += 1
        
        # Base Cases
        if isinstance(expr, (int, float)) or expr.is_Number:
            trace.append(f"{indent}[Base] <{step_status}> Constant {expr} -> derivative is 0")
            return 0.0, trace
        
        if isinstance(expr, Symbol):
            if expr == var:
                trace.append(f"{indent}[Base] <{step_status}> Variable {expr} -> derivative is 1")
                return 1.0, trace
            trace.append(f"{indent}[Base] <{step_status}> Other symbol {expr} -> derivative is 0")
            return 0.0, trace # Treat other symbols as constants for now

        # Recursive Cases
        
        # Sum Rule: (u + v)' = u' + v'
        if expr.is_Add and (forced_rule is None or forced_rule == "sum_rule"):
            trace.append(f"{indent}[Decompose] <{step_status}> Sum Rule: {expr}")
            total = 0.0
            for arg in expr.args:
                res, sub_trace = self._differentiate_eval(arg, var, val, depth + 1)
                trace.extend(sub_trace)
                total += res
            trace.append(f"{indent}[Result] Sum -> {total}")
            return total, trace

        # Quotient Rule: (u/v)' = (u'v - uv') / v^2
        # SymPy represents division as Mul(u, Pow(v, -1)) usually, but let's check explicit structure if possible.
        # Often better to handle as product rule with negative power if standard Mul.
        # But let's check for explicit fraction structure via as_numer_denom
        if expr.is_Mul and (forced_rule is None or forced_rule == "quotient_rule"):
            num, den = expr.as_numer_denom()
            if den is not None and den != 1:
                # It is a fraction
                trace.append(f"{indent}[Decompose] <{step_status}> Quotient Rule: {num} / {den}")
                
                num_prime, num_trace = self._differentiate_eval(num, var, val, depth + 1)
                den_prime, den_trace = self._differentiate_eval(den, var, val, depth + 1)
                trace.extend(num_trace)
                trace.extend(den_trace)
                
                num_val = self._evaluate_func(num, var, val)
                den_val = self._evaluate_func(den, var, val)
                
                if den_val == 0:
                    trace.append(f"{indent}[Error] Division by zero in quotient rule")
                    return 0.0, trace

                result = (num_prime * den_val - num_val * den_prime) / (den_val ** 2)
                trace.append(f"{indent}[Result] Quotient -> {result}")
                return result, trace

        # Constant Multiple / Product Rule: (uv)' = u'v + uv'
        if expr.is_Mul and (forced_rule is None or forced_rule == "product_rule"):
            trace.append(f"{indent}[Decompose] <{step_status}> Product/Const Rule: {expr}")
            # SymPy flattens multiplications: a*b*c. Treat as a * (b*c...)
            args = expr.args
            u = args[0]
            v = Mul(*args[1:]) # Rest of the terms
            
            trace.append(f"{indent}  u = {u}")
            trace.append(f"{indent}  v = {v}")
            
            u_prime, u_trace = self._differentiate_eval(u, var, val, depth + 1)
            v_prime, v_trace = self._differentiate_eval(v, var, val, depth + 1)
            
            trace.extend(u_trace)
            trace.extend(v_trace)
            
            u_val = self._evaluate_func(u, var, val)
            v_val = self._evaluate_func(v, var, val)
            
            # u'v + uv'
            result = u_prime * v_val + u_val * v_prime
            trace.append(f"{indent}[Result] Product -> {u_prime}*{v_val} + {u_val}*{v_prime} = {result}")
            return result, trace

        # Power Rule / Chain Rule: (u^n)' = n * u^(n-1) * u'
        if expr.is_Pow and (forced_rule is None or forced_rule in ("power_rule", "power_const_base")):
            base, exp = expr.args
            
            if exp.has(var):
                # Handle constant base: d/dx (a^g(x)) = ln(a) * a^g(x) * g'(x)
                if not base.has(var):
                    base_val = self._evaluate_func(base, var, val)
                    if base_val <= 0:
                        trace.append(f"{indent}[Warning] Non-positive base in power: {expr}")
                        return 0.0, trace
                    trace.append(f"{indent}[Decompose] <{step_status}> Power (const base): {expr}")
                    exp_val = self._evaluate_func(exp, var, val)
                    exp_prime, exp_trace = self._differentiate_eval(exp, var, val, depth + 1)
                    trace.extend(exp_trace)
                    result = float(sympy.log(base_val)) * (base_val ** exp_val) * exp_prime
                    trace.append(
                        f"{indent}[Result] Power const -> ln({base_val}) * {base_val}^{exp_val} * {exp_prime} = {result}"
                    )
                    return result, trace
                # f(x)^g(x) general case is Phase 2 scope.
                trace.append(f"{indent}[Warning] Variable exponent not supported in Phase 1: {expr}")
                return 0.0, trace
            
            trace.append(f"{indent}[Decompose] <{step_status}> Power Rule: {expr}")
            n = float(exp)
            
            base_val = self._evaluate_func(base, var, val)
            
            base_prime, base_trace = self._differentiate_eval(base, var, val, depth + 1)
            trace.extend(base_trace)
            
            # n * base^(n-1) * base_prime
            result = n * (base_val ** (n - 1)) * base_prime
            trace.append(f"{indent}[Result] Power -> {n} * {base_val}^({n-1}) * {base_prime} = {result}")
            return result, trace

        # Standard Functions (Chain Rule applied: f(g(x))' = f'(g(x)) * g'(x))
        # Sin
        if expr.func == sympy.sin and (forced_rule is None or forced_rule == "sin_rule"):
            trace.append(f"{indent}[Decompose] <{step_status}> Sin Chain Rule: {expr}")
            arg = expr.args[0]
            
            arg_val = self._evaluate_func(arg, var, val)
            
            arg_prime, arg_trace = self._differentiate_eval(arg, var, val, depth + 1)
            trace.extend(arg_trace)
            
            result = float(sympy.cos(arg_val)) * arg_prime
            trace.append(f"{indent}[Result] Sin -> cos({arg_val}) * {arg_prime} = {result}")
            return result, trace
            
        # Cos
        if expr.func == sympy.cos and (forced_rule is None or forced_rule == "cos_rule"):
            trace.append(f"{indent}[Decompose] <{step_status}> Cos Chain Rule: {expr}")
            arg = expr.args[0]
            
            arg_val = self._evaluate_func(arg, var, val)
            
            arg_prime, arg_trace = self._differentiate_eval(arg, var, val, depth + 1)
            trace.extend(arg_trace)
            
            result = -float(sympy.sin(arg_val)) * arg_prime
            trace.append(f"{indent}[Result] Cos -> -sin({arg_val}) * {arg_prime} = {result}")
            return result, trace
            
        # Exp
        if expr.func == sympy.exp and (forced_rule is None or forced_rule == "exp_rule"):
            trace.append(f"{indent}[Decompose] <{step_status}> Exp Chain Rule: {expr}")
            arg = expr.args[0]
            
            arg_val = self._evaluate_func(arg, var, val)
            
            arg_prime, arg_trace = self._differentiate_eval(arg, var, val, depth + 1)
            trace.extend(arg_trace)
            
            result = float(sympy.exp(arg_val)) * arg_prime
            trace.append(f"{indent}[Result] Exp -> exp({arg_val}) * {arg_prime} = {result}")
            return result, trace

        # Fallback for unknown functions
        trace.append(f"{indent}[Warning] Unknown function {expr.func}")
        return 0.0, trace
