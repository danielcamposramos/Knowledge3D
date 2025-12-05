"""Opcode-aware quality scoring for Shadow Copy patterns."""
from __future__ import annotations

from typing import Dict, List


TIER1_HINTS = {"add", "sub", "mul", "div", "sqrt", "abs", "gt", "lt", "eq"}
TIER2_HINTS = {"reduce", "matvec", "norm", "dot", "mean", "median", "variance"}
TIER3_HINTS = {"trm", "swiglu", "symbolic", "gradient", "matmul", "svd", "qr", "cholesky"}


def _infer_tier_from_tokens(tokens: List[str]) -> int:
    """Heuristic tier inference from opcode-like tokens."""
    lower = [t.lower() for t in tokens]
    if any(any(h in tok for h in TIER3_HINTS) for tok in lower):
        return 3
    if any(any(h in tok for h in TIER2_HINTS) for tok in lower):
        return 2
    if any(any(h in tok for h in TIER1_HINTS) for tok in lower):
        return 1
    return 1


def compute_pattern_quality_opcode_aware(pattern_entry: Dict, execution_history: List[bool]) -> float:
    """
    Enhanced quality score using opcode/complexity heuristics.

    Components:
    - success_rate (0.6 weight)
    - complexity_penalty (0.2 weight)
    - tier_alignment (0.2 weight)
    """
    success_rate = sum(execution_history) / max(1, len(execution_history)) if execution_history else 1.0
    program = pattern_entry.get("program", "")
    tokens = [t for t in program.split() if t]
    tier = _infer_tier_from_tokens(tokens)

    tier1_count = sum(1 for tok in tokens if any(h in tok.lower() for h in TIER1_HINTS))
    tier2_count = sum(1 for tok in tokens if any(h in tok.lower() for h in TIER2_HINTS))
    tier3_count = sum(1 for tok in tokens if any(h in tok.lower() for h in TIER3_HINTS))
    total_ops = max(1, tier1_count + tier2_count + tier3_count)
    complexity_penalty = (tier1_count * 0.1 + tier2_count * 0.3 + tier3_count * 0.6) / total_ops

    expected_tier = pattern_entry.get("expected_tier", tier)
    tier_mismatch = abs(expected_tier - tier) / 2.0  # normalized 0-1

    quality = (
        0.6 * success_rate +
        0.2 * (1.0 - complexity_penalty) +
        0.2 * (1.0 - tier_mismatch)
    )
    return max(0.0, min(1.0, quality))


__all__ = ["compute_pattern_quality_opcode_aware", "_infer_tier_from_tokens"]
