from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Union


Token = Union[float, int, str]


@dataclass
class RPN:
    max_depth: int = 64
    keep_top: int = 5

    def __post_init__(self) -> None:
        self.keep_top = max(5, self.keep_top)
        self._stack: List[float] = []

    def reset(self) -> None:
        self._stack.clear()

    def push(self, x: float) -> None:
        self._stack.append(float(x))
        self._auto_clean()

    def pop(self) -> float:
        if not self._stack:
            raise RuntimeError("RPN underflow")
        return self._stack.pop()

    def peek(self, n: int = 0) -> float:
        idx = len(self._stack) - 1 - n
        if idx < 0:
            raise RuntimeError("RPN underflow")
        return self._stack[idx]

    def _auto_clean(self) -> None:
        if len(self._stack) > self.max_depth:
            keep = max(1, self.keep_top)
            del self._stack[: len(self._stack) - keep]

    def eval(self, tokens: Iterable[Token]) -> float:
        for t in tokens:
            if isinstance(t, (int, float)):
                self.push(float(t))
                continue
            if t == '+':
                self.push(self.pop() + self.pop())
            elif t == '-':
                b, a = self.pop(), self.pop()
                self.push(a - b)
            elif t == '*':
                self.push(self.pop() * self.pop())
            elif t == '/':
                b, a = self.pop(), self.pop()
                self.push(a / b if b != 0 else 0.0)
            elif t == 'sqrt':
                import math
                self.push(math.sqrt(self.pop()))
            elif t == 'abs':
                self.push(abs(self.pop()))
            elif t == 'dup':
                self.push(self.peek())
            elif t == 'drop':
                self.pop()
            elif t == 'swap':
                a, b = self.pop(), self.pop()
                self.push(a)
                self.push(b)
            else:
                raise ValueError(f"unknown op: {t}")
        return self.peek(0)

    def dot(self, a: Iterable[float], b: Iterable[float]) -> float:
        acc = 0.0
        for x, y in zip(a, b):
            acc = self.eval([acc, x, y, '*', '+'])
        return acc

    def norm(self, a: Iterable[float]) -> float:
        acc = 0.0
        for x in a:
            acc = self.eval([acc, x, x, '*', '+'])
        import math
        return math.sqrt(acc)

    def cosine(self, a: Iterable[float], b: Iterable[float]) -> float:
        d = self.dot(a, b)
        na = self.norm(a)
        nb = self.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return self.eval([d, na, '/', nb, '/'])

