"""Drawing grammar rules (primitives, curves, transforms, compositions)."""

from __future__ import annotations

from typing import List

from .primitives import PRIMITIVE_RULES
from .curves import CURVE_RULES
from .transforms import TRANSFORM_RULES
from .compositions import COMPOSITION_RULES


def get_drawing_rules() -> List:
    rules = []
    rules.extend(PRIMITIVE_RULES)
    rules.extend(CURVE_RULES)
    rules.extend(TRANSFORM_RULES)
    rules.extend(COMPOSITION_RULES)
    return rules


__all__ = ["get_drawing_rules"]
