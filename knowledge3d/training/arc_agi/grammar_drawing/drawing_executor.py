"""Stub drawing executor connecting grammar to procedural drawing specialist."""

from __future__ import annotations

from typing import Dict

import numpy as np


class DrawingGrammarExecutor:
    """Execute drawing grammar RPN programs (placeholder orchestration)."""

    def __init__(self):
        # Wiring to procedural drawing specialist can be added when needed.
        self._placeholder = True

    def execute_drawing_rpn(self, program: str, context: Dict | None = None) -> np.ndarray:
        """
        Execute drawing RPN program to generate visual output.

        Currently returns an empty canvas placeholder; hook to procedural_drawing_specialist later.
        """
        _ = program
        _ = context
        return np.zeros((1, 1), dtype=np.float32)


__all__ = ["DrawingGrammarExecutor"]
