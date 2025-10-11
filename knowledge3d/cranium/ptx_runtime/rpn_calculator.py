from __future__ import annotations

from typing import Dict, List, Optional

from .modular_rpn_engine import ModularRPNEngine


class RPNCalculator:
    """Backward-compatible wrapper around the modular GPU RPN engine."""

    _ENGINE: ModularRPNEngine | None = None

    def __init__(self) -> None:
        if RPNCalculator._ENGINE is None:
            RPNCalculator._ENGINE = ModularRPNEngine()
        self._engine = RPNCalculator._ENGINE

    def evaluate(self, expression: str, instance_id: int = 0, variables: Optional[Dict[str, float]] = None) -> float:
        # Substitute variables if provided (Python-side)
        if variables:
            for var_name, var_value in variables.items():
                expression = expression.replace(var_name, str(var_value))

        result = self._engine.evaluate(expression, instance_id=instance_id)
        return float(result)

    def evaluate_vector(
        self,
        expression: str,
        instance_id: int = 0,
        variables: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        # Substitute variables if provided (Python-side)
        if variables:
            for var_name, var_value in variables.items():
                expression = expression.replace(var_name, str(var_value))

        result = self._engine.evaluate(expression, instance_id=instance_id)
        return [float(result)]

    def reset(self) -> None:
        self._engine.reset()


__all__ = ["RPNCalculator"]
