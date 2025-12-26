"""
RPN Parser implementing classic infix↔postfix algorithms.

Algorithm A1 (infix → postfix) and A2 (postfix → infix) mirror the RPN
papers included in the knowledge base. This is lightweight and sovereign
(no numpy, no external deps).
"""

from __future__ import annotations

from typing import List


class RPNParser:
    """Infix/Postfix transformer using stack-based algorithms."""

    def __init__(self) -> None:
        # Basic precedence table; extend as needed.
        self.precedence = {
            "+": 1,
            "-": 1,
            "*": 2,
            "/": 2,
            "^": 3,
        }
        # Functions treated as highest-precedence, right-associative.
        self.functions = {
            "sin",
            "cos",
            "tan",
            "log",
            "ln",
            "exp",
            "sqrt",
            "arcsin",
            "arccos",
            "arctan",
            "abs",
            "floor",
            "ceil",
        }

    def infix_to_rpn(self, expr: str) -> str:
        """Algorithm A1: Infix to Postfix transformation."""
        output: List[str] = []
        stack: List[str] = []

        for token in self._tokenize(expr):
            if self._is_operand(token):
                output.append(token)
            elif token in self.functions:
                stack.append(token)
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if stack:
                    stack.pop()  # discard '('
                # If a function preceded the parens, emit it now
                if stack and stack[-1] in self.functions:
                    output.append(stack.pop())
            elif token in self.precedence:
                while (
                    stack
                    and stack[-1] != "("
                    and stack[-1] not in self.functions
                    and self.precedence.get(stack[-1], 0) >= self.precedence[token]
                ):
                    output.append(stack.pop())
                stack.append(token)

        while stack:
            output.append(stack.pop())

        return " ".join(output)

    def postfix_to_infix(self, expr: str) -> str:
        """Algorithm A2: Postfix to Infix transformation."""
        stack: List[str] = []
        for token in expr.split():
            if self._is_operand(token):
                stack.append(token)
            elif token in self.precedence:
                if len(stack) < 2:
                    continue
                b = stack.pop()
                a = stack.pop()
                stack.append(f"({a} {token} {b})")
        return stack[-1] if stack else ""

    def token_grasp(self, token: str) -> int:
        """
        Minimal grasp estimation: number of operands consumed by an operator.
        Used to approximate Left Grasp Bound (LGB) ideas from RPN literature.
        """
        if token in {"+", "-", "*", "/", "^"}:
            return 2
        if token in self.functions:
            return 1
        return 1

    def _tokenize(self, expr: str) -> List[str]:
        tokens: List[str] = []
        buf = ""
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch.isalnum() or ch == ".":
                buf += ch
            else:
                if buf:
                    lowered = buf.lower()
                    tokens.append(lowered if lowered in self.functions else buf)
                    buf = ""
                if ch.strip():
                    tokens.append(ch)
            i += 1
        if buf:
            lowered = buf.lower()
            tokens.append(lowered if lowered in self.functions else buf)
        return tokens

    def _is_operand(self, token: str) -> bool:
        if token in {"(", ")"}:
            return False
        try:
            float(token)
            return True
        except ValueError:
            return token not in self.precedence and token not in self.functions


__all__ = ["RPNParser"]
