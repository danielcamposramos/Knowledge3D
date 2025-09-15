from __future__ import annotations

from typing import List, Tuple
import re


class RPNCalculator:
    def __init__(self, stack_size: int = 100, precision: int = 15):
        self.stack_size = int(stack_size)
        self.precision = int(precision)
        self.stack: List[float] = []
        self.history: List[Tuple[str, float]] = []

    def parse_expression(self, expression: str) -> List[str]:
        """Parse expression into RPN tokens using regex (numbers and operators)."""
        expr = (expression or '').strip()
        # Allow decimals and scientific notation, ops: + - * / ^
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[+\-*/^]", expr)
        return tokens

    def evaluate(self, expression: str) -> float:
        tokens = self.parse_expression(expression)
        self.clear_stack()  # Auto-clean stack per spec
        for tok in tokens:
            if tok in {'+','-','*','/','^'}:
                if len(self.stack) < 2:
                    raise ValueError('Insufficient operands')
                b = self.stack.pop()
                a = self.stack.pop()
                if tok == '+':
                    res = a + b
                elif tok == '-':
                    res = a - b
                elif tok == '*':
                    res = a * b
                elif tok == '/':
                    if b == 0:
                        raise ValueError('Division by zero')
                    res = a / b
                else:  # '^'
                    res = a ** b
                # Push rounded result
                self.stack.append(round(res, self.precision))
            else:
                # number
                if len(self.stack) >= self.stack_size:
                    raise ValueError('Stack overflow')
                self.stack.append(float(tok))
        if len(self.stack) != 1:
            raise ValueError('Invalid RPN expression')
        result = self.stack[0]
        self.history.append((expression, result))
        return result

    def get_stack(self) -> List[float]:
        return list(self.stack)

    def clear_stack(self) -> None:
        self.stack.clear()

    def get_history(self) -> List[Tuple[str, float]]:
        return list(self.history)

