from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore

from knowledge3d.cranium.phase10.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.phase10.text_to_3d_generator import TextTo3DGenerator


class PTXOps:
    """Utility wrapper exposing GPU PTX helpers to higher-level components."""

    def __init__(self) -> None:
        self._rpn_engine = ModularRPNEngine()
        self._shape_generator = TextTo3DGenerator()

    # ------------------------------------------------------------------
    def evaluate_rpn(self, expression: str) -> float:
        """Evaluate an RPN expression on the GPU and return the first scalar result."""
        result = self._rpn_engine.evaluate(expression)
        return float(result)

    def generate_shape(self, prompt: str, vertex_count: int = 32, shape_hint: Optional[int] = None) -> str:
        """Generate a GLB path for a prompt-driven shape using the PTX geometry kernel."""
        # The TextTo3DGenerator internally hashes the prompt to determine shape semantics.
        shape_path = self._shape_generator.generate_3d_from_text(prompt)
        return shape_path

    # ------------------------------------------------------------------
    @staticmethod
    def format_numeric(value: float, precision: int = 6) -> str:
        arr = np.array([value], dtype=np.float64)
        return np.array2string(arr, precision=precision, suppress_small=True)[1:-1]


# Shared singleton to avoid re-initialising GPU contexts repeatedly.
PTX_OPS = PTXOps()
