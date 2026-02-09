"""Deterministic symbolic RPN task generation and evaluation helpers."""

from __future__ import annotations

import math
import random
from typing import Any


BINARY_OPS = ("ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MAX", "MIN")
UNARY_OPS = ("ABS", "NEG")


def evaluate_rpn_program(program: str) -> float | int:
    """Evaluate a small deterministic RPN program."""
    stack: list[float] = []
    for token in program.split():
        if token.upper() in BINARY_OPS:
            if len(stack) < 2:
                return 0
            b = stack.pop()
            a = stack.pop()
            op = token.upper()
            if op == "ADD":
                stack.append(a + b)
            elif op == "SUB":
                stack.append(a - b)
            elif op == "MUL":
                stack.append(a * b)
            elif op == "DIV":
                if abs(b) < 1e-9:
                    return 0
                stack.append(a / b)
            elif op == "MOD":
                if abs(b) < 1e-9:
                    return 0
                stack.append(a % b)
            elif op == "POW":
                if abs(a) > 100 or abs(b) > 8:
                    return 0
                stack.append(a**b)
            elif op == "MAX":
                stack.append(max(a, b))
            elif op == "MIN":
                stack.append(min(a, b))
        elif token.upper() in UNARY_OPS:
            if not stack:
                return 0
            a = stack.pop()
            op = token.upper()
            if op == "ABS":
                stack.append(abs(a))
            elif op == "NEG":
                stack.append(-a)
        elif token.upper() == "SQRT":
            if not stack:
                return 0
            a = stack.pop()
            if a < 0:
                return 0
            stack.append(math.sqrt(a))
        else:
            try:
                stack.append(float(token))
            except Exception:
                return 0

    if len(stack) != 1:
        return 0
    value = stack[0]
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return value


def _generate_single_program(rng: random.Random) -> str:
    nums = [rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 5)]
    op1 = rng.choice(BINARY_OPS)
    op2 = rng.choice(BINARY_OPS)
    program = f"{nums[0]} {nums[1]} {op1} {nums[2]} {op2}"
    if rng.random() < 0.2:
        program += f" {rng.choice(UNARY_OPS)}"
    return program


def generate_rpn_tasks(count: int, seed: int = 1341) -> list[dict[str, Any]]:
    """Generate deterministic symbolic RPN tasks."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    idx = 0
    while idx < max(0, int(count)):
        program = _generate_single_program(rng)
        expected = evaluate_rpn_program(program)
        # Skip malformed/noisy programs from defensive evaluator.
        if expected == 0 and program.count("0") == 0:
            continue
        tasks.append(
            {
                "id": f"rpn_{idx:04d}",
                "category": "symbolic_rpn",
                "rpn_program": program,
                "expected": expected,
                "query": f"evaluate rpn {program}",
            }
        )
        idx += 1
    return tasks

