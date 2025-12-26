"""
RPN Validator - lightweight checks before GPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

# Constants supported by `ModularRPNEngine.CONSTANTS`.
#
# NOTE: Keep this small and math-focused. This validator is used as a cheap
# pre-flight check before GPU execution (and during ingestion filtering).
VALID_CONSTANTS: Set[str] = {
    "pi",
    "π",
    "tau",
    "phi",
    "φ",
    "e",
}

# Opcodes supported by ModularRPNEngine (math + stack).
VALID_OPCODES: Set[str] = {
    # Arithmetic
    "+",
    "-",
    "*",
    "/",
    "^",
    "pow",
    "neg",
    "sqrt",
    "abs",
    "max",
    "min",
    "mod",
    "%",
    # Trigonometry
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "atan2",
    "arcsin",
    "arccos",
    "arctan",
    # Logarithmic / exponential
    "log",
    "ln",
    "log2",
    "log10",
    "exp",
    "gamma",
    "beta",
    # Stack ops
    "dup",
    "swap",
    "drop",
    "over",
    "rot",
    "clear",
    # Rounding / discrete
    "floor",
    "ceil",
    "round",
    "gcd",
    "factorial",
    "!",
    "binomial",
    "binom",
    # Comparisons
    "eq",
    "neq",
    "lt",
    "gt",
    "le",
    "ge",
    "gte",
    # Ternary / specialized
    "tern_add",
    "tern_mul",
    "tern_sub",
    "ifelse",
}


def is_valid_rpn(program: str) -> bool:
    """
    Check if an RPN program is structurally valid.
    - Non-empty
    - Only numbers, variables (single letters), or valid opcodes
    - No parentheses (should be parsed out)
    """
    if not program or not program.strip():
        return False

    tokens = program.strip().split()
    if not tokens:
        return False

    has_number = False
    has_opcode = False

    for token in tokens:
        lower = token.lower()

        # Constants (pi, e, ...)
        if token in VALID_CONSTANTS or lower in VALID_CONSTANTS:
            has_number = True
            continue

        # Number
        try:
            float(token)
            has_number = True
            continue
        except ValueError:
            pass

        # Opcode
        if lower in VALID_OPCODES:
            has_opcode = True
            continue

        # Parentheses are invalid in RPN
        if token in "()[]{}":
            return False

        # Allow single-letter variables (x, y, n)
        if len(token) == 1 and token.isalpha():
            continue

        # Unknown token → invalid
        return False

    return has_number or has_opcode


def estimate_stack_balance(program: str) -> int:
    """
    Estimate final stack size (should be 1 for a well-formed program).
    """
    tokens = program.strip().split()
    stack_size = 0

    for token in tokens:
        lower = token.lower()

        # Numbers push 1
        try:
            float(token)
            stack_size += 1
            continue
        except ValueError:
            pass

        if token in VALID_CONSTANTS or lower in VALID_CONSTANTS:
            stack_size += 1
            continue

        # Binary ops: pop 2, push 1 (net -1)
        if lower in {
            "+",
            "-",
            "*",
            "/",
            "^",
            "pow",
            "mod",
            "%",
            "max",
            "min",
            "atan2",
            "eq",
            "neq",
            "lt",
            "gt",
            "le",
            "ge",
            "gte",
            "gcd",
            "binomial",
            "binom",
            "beta",
        }:
            stack_size -= 1
        # Unary ops: pop 1, push 1 (net 0)
        elif lower in {
            "sqrt",
            "abs",
            "sin",
            "cos",
            "tan",
            "log",
            "ln",
            "exp",
            "log2",
            "log10",
            "asin",
            "acos",
            "atan",
            "arcsin",
            "arccos",
            "arctan",
            "sinh",
            "cosh",
            "tanh",
            "floor",
            "ceil",
            "round",
            "gamma",
            "factorial",
            "!",
        }:
            pass
        # Stack ops
        elif lower == "dup":
            stack_size += 1
        elif lower in {"swap", "over", "rot"}:
            pass
        elif lower == "drop":
            stack_size -= 1
        elif lower == "ifelse":
            # cond then else -> value
            stack_size -= 2

    return stack_size


@dataclass(frozen=True)
class StackShapeResult:
    ok: bool
    reason: str | None = None
    final_stack: int | None = None


_BINARY_OPS: Set[str] = {
    "+",
    "-",
    "*",
    "/",
    "^",
    "pow",
    "mod",
    "%",
    "max",
    "min",
    "atan2",
    "eq",
    "neq",
    "lt",
    "gt",
    "le",
    "ge",
    "gte",
    "gcd",
    "binomial",
    "binom",
    "beta",
}

_UNARY_OPS: Set[str] = {
    "neg",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "log2",
    "log10",
    "exp",
    "asin",
    "acos",
    "atan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "round",
    "gamma",
    "factorial",
    "!",
}


def validate_stack_shape(program: str) -> StackShapeResult:
    """
    Validate that an RPN program is executable as a single-value numeric chain.

    This is intentionally a conservative check: it only reasons about stack
    *shape* (underflow + final size), not semantic correctness.
    """
    if not is_valid_rpn(program):
        return StackShapeResult(ok=False, reason="invalid_token")

    tokens = (program or "").strip().split()
    stack = 0
    for token in tokens:
        lower = token.lower()
        if token in VALID_CONSTANTS or lower in VALID_CONSTANTS:
            stack += 1
            continue
        try:
            float(token)
            stack += 1
            continue
        except ValueError:
            pass
        if len(token) == 1 and token.isalpha():
            stack += 1
            continue

        if lower == "clear":
            stack = 0
            continue
        if lower == "dup":
            if stack < 1:
                return StackShapeResult(ok=False, reason="stack_underflow_dup")
            stack += 1
            continue
        if lower == "drop":
            if stack < 1:
                return StackShapeResult(ok=False, reason="stack_underflow_drop")
            stack -= 1
            continue
        if lower == "swap":
            if stack < 2:
                return StackShapeResult(ok=False, reason="stack_underflow_swap")
            continue
        if lower == "over":
            if stack < 2:
                return StackShapeResult(ok=False, reason="stack_underflow_over")
            stack += 1
            continue
        if lower == "rot":
            if stack < 3:
                return StackShapeResult(ok=False, reason="stack_underflow_rot")
            continue
        if lower == "ifelse":
            if stack < 3:
                return StackShapeResult(ok=False, reason="stack_underflow_ifelse")
            stack -= 2
            continue

        if lower in _UNARY_OPS:
            if stack < 1:
                return StackShapeResult(ok=False, reason="stack_underflow_unary")
            continue
        if lower in _BINARY_OPS:
            if stack < 2:
                return StackShapeResult(ok=False, reason="stack_underflow_binary")
            stack -= 1
            continue

        # Unknown token type: `is_valid_rpn` should have rejected it already.
        return StackShapeResult(ok=False, reason="invalid_token")

    if stack != 1:
        return StackShapeResult(ok=False, reason="final_stack_not_single", final_stack=stack)
    return StackShapeResult(ok=True, reason=None, final_stack=1)


__all__ = ["is_valid_rpn", "estimate_stack_balance", "validate_stack_shape", "VALID_OPCODES", "VALID_CONSTANTS"]
