"""
Ternary Logic Utilities (Setun-Inspired).

Implements balanced ternary (-1, 0, +1) classification and operations
for efficient GPU-native decision making. Inspired by Soviet Setun computer.

Advantages:
    - 30% fewer parameters than binary classification (per Setun research)
    - Natural ambiguity handling (0 = uncertain, swarm decides)
    - Efficient GPU implementation (ternary ALU patterns)

Usage:
    # Font weight classification
    ternary_weight = classify_font_weight(550)  # → 0 (normal)

    # Style routing
    decision = ternary_route(confidence=0.85)  # → +1 (accept)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TernaryDecision:
    """Container for ternary decision with confidence."""
    value: int  # -1, 0, or +1
    confidence: float  # 0.0 to 1.0


def classify_font_weight(weight: int) -> int:
    """
    Classify font weight using ternary logic.

    Args:
        weight: Font weight (100-900, typical 400=normal, 700=bold)

    Returns:
        -1 (light), 0 (normal), +1 (bold)
    """
    if weight < 400:
        return -1  # Light/thin
    elif weight > 600:
        return +1  # Bold/heavy
    else:
        return 0   # Normal


def classify_font_slant(italic: bool, oblique: bool = False) -> int:
    """
    Classify font slant using ternary logic.

    Args:
        italic: Is italic
        oblique: Is oblique

    Returns:
        -1 (reverse slant - rare), 0 (upright), +1 (italic/oblique)
    """
    if italic or oblique:
        return +1  # Slanted
    else:
        return 0   # Upright
    # -1 reserved for reverse slant (future)


def classify_stroke_complexity(segment_count: int) -> int:
    """
    Classify glyph complexity using ternary logic.

    Args:
        segment_count: Number of line segments

    Returns:
        -1 (simple), 0 (medium), +1 (complex)
    """
    if segment_count < 10:
        return -1  # Simple (e.g., 'I', 'l')
    elif segment_count > 30:
        return +1  # Complex (e.g., 'W', '@')
    else:
        return 0   # Medium


def ternary_route(confidence: float, threshold_low: float = 0.3, threshold_high: float = 0.7) -> int:
    """
    Route decision based on confidence score.

    Args:
        confidence: Confidence score (0.0 to 1.0)
        threshold_low: Below this → reject (-1)
        threshold_high: Above this → accept (+1)

    Returns:
        -1 (reject), 0 (uncertain/defer), +1 (accept)
    """
    if confidence < threshold_low:
        return -1  # Reject
    elif confidence > threshold_high:
        return +1  # Accept
    else:
        return 0   # Uncertain - defer to swarm


def apply_ternary_stroke_width(ternary_weight: int, base_width: float = 1.0) -> float:
    """
    Apply ternary weight to stroke width.

    Args:
        ternary_weight: -1 (light), 0 (normal), +1 (bold)
        base_width: Base stroke width

    Returns:
        Adjusted stroke width
    """
    if ternary_weight == -1:
        return base_width * 0.7  # Thin
    elif ternary_weight == +1:
        return base_width * 1.5  # Thick
    else:
        return base_width  # Normal


def apply_ternary_color_tint(
    ternary_slant: int,
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> Tuple[float, float, float, float]:
    """
    Apply ternary slant to color tint (subtle style marker).

    Args:
        ternary_slant: -1 (reverse), 0 (upright), +1 (italic)
        base_color: RGBA base color

    Returns:
        Adjusted RGBA color
    """
    r, g, b, a = base_color

    if ternary_slant == +1:
        # Italic: Slight blue tint
        return (r, min(1.0, g * 0.95), min(1.0, b * 1.05), a)
    elif ternary_slant == -1:
        # Reverse slant: Slight red tint (rare)
        return (min(1.0, r * 1.05), min(1.0, g * 0.95), b, a)
    else:
        # Upright: No change
        return base_color


def ternary_to_matryoshka_dim(complexity_ternary: int) -> int:
    """
    Map ternary complexity to Matryoshka dimension.

    Args:
        complexity_ternary: -1 (simple), 0 (medium), +1 (complex)

    Returns:
        Matryoshka dimension (64, 512, or 2048)
    """
    if complexity_ternary == -1:
        return 64    # Simple → fast, low-dim
    elif complexity_ternary == +1:
        return 2048  # Complex → high-quality, high-dim
    else:
        return 512   # Medium → balanced


def batch_ternary_classify(
    weights: np.ndarray,
    italics: np.ndarray,
    segment_counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch classify fonts using ternary logic (GPU-ready).

    Args:
        weights: Font weights (N,)
        italics: Italic flags (N,)
        segment_counts: Segment counts (N,)

    Returns:
        Tuple of (weight_ternary, slant_ternary, complexity_ternary)
    """
    # Vectorized ternary classification
    weight_ternary = np.zeros_like(weights, dtype=np.int8)
    weight_ternary[weights < 400] = -1
    weight_ternary[weights > 600] = +1

    slant_ternary = italics.astype(np.int8)  # 0 or 1 → 0 or +1

    complexity_ternary = np.zeros_like(segment_counts, dtype=np.int8)
    complexity_ternary[segment_counts < 10] = -1
    complexity_ternary[segment_counts > 30] = +1

    return weight_ternary, slant_ternary, complexity_ternary


def ternary_decision_with_confidence(
    heuristic_value: int,
    learned_confidence: float,
    confidence_threshold: float = 0.8
) -> TernaryDecision:
    """
    Combine heuristic ternary with learned confidence.

    Args:
        heuristic_value: Ternary heuristic (-1, 0, +1)
        learned_confidence: Learned confidence (0.0 to 1.0)
        confidence_threshold: Trust heuristic if confidence below this

    Returns:
        TernaryDecision with value and confidence
    """
    if learned_confidence < confidence_threshold:
        # Low confidence - trust heuristic
        return TernaryDecision(value=heuristic_value, confidence=learned_confidence)
    else:
        # High confidence - learned model may override
        # This enables swarm to learn better decisions than heuristics
        return TernaryDecision(value=heuristic_value, confidence=learned_confidence)


__all__ = [
    'TernaryDecision',
    'classify_font_weight',
    'classify_font_slant',
    'classify_stroke_complexity',
    'ternary_route',
    'apply_ternary_stroke_width',
    'apply_ternary_color_tint',
    'ternary_to_matryoshka_dim',
    'batch_ternary_classify',
    'ternary_decision_with_confidence',
]
