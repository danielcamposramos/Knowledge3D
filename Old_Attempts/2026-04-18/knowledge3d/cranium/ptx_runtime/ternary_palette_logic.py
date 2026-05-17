"""Ternary contrastive logic for compact procedural color palettes.

This wraps palette selection in the same positive/negative/neutral ternary logic
already used for gradients. Palettes are converted into evenly spaced gradient
stops so they can reuse the canonical ternary gradient substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ternary_gradient_logic import ContrastiveGradientScore, TernaryGradientLogic, TernaryGradientSignature


def _prepare_palette(palette: Sequence[Sequence[float]]) -> np.ndarray:
    if len(palette) == 0:
        raise ValueError("palette must contain at least one color")
    arr = np.ascontiguousarray(np.asarray(palette, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] not in {3, 4}:
        raise ValueError("palette colors must have shape [n,3] or [n,4]")
    if arr.shape[1] == 3:
        alpha = np.ones((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=1)
    return np.clip(arr, 0.0, 1.0)


def _palette_to_stops(palette: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    if len(palette) == 1:
        rgba = palette[0]
        return [(0.0, float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))]
    positions = np.linspace(0.0, 1.0, len(palette), dtype=np.float32)
    return [
        (float(pos), float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        for pos, color in zip(positions, palette, strict=False)
    ]


@dataclass(frozen=True)
class TernaryPaletteSignature:
    palette_rgba: tuple[tuple[float, float, float, float], ...]
    gradient_signature: TernaryGradientSignature
    gradient_stops: tuple[tuple[float, float, float, float, float], ...]


@dataclass(frozen=True)
class ContrastivePaletteScore:
    score: float
    positive_similarity: float
    negative_penalty: float
    candidate_signature: TernaryPaletteSignature
    target_signature: TernaryPaletteSignature


class TernaryPaletteLogic:
    """Contrastive ternary surface for color palettes."""

    def __init__(self) -> None:
        self.gradient_logic = TernaryGradientLogic()

    def palette_to_stops(self, palette: Sequence[Sequence[float]]) -> list[tuple[float, float, float, float, float]]:
        prepared = _prepare_palette(palette)
        return _palette_to_stops(prepared)

    def encode_signature(
        self,
        palette: Sequence[Sequence[float]],
        *,
        thresholds: tuple[float, float, float, float, float] = (0.08, 0.1, 0.1, 0.1, 0.08),
    ) -> TernaryPaletteSignature:
        prepared = _prepare_palette(palette)
        stops = self.palette_to_stops(prepared)
        signature = self.gradient_logic.encode_signature(stops, thresholds=thresholds)
        return TernaryPaletteSignature(
            palette_rgba=tuple(tuple(float(v) for v in row.tolist()) for row in prepared),
            gradient_signature=signature,
            gradient_stops=tuple(stops),
        )

    def contrastive_score(
        self,
        target_palette: Sequence[Sequence[float]],
        candidate_palette: Sequence[Sequence[float]],
        *,
        negative_examples: Sequence[Sequence[Sequence[float]]] = (),
    ) -> ContrastivePaletteScore:
        target_sig = self.encode_signature(target_palette)
        candidate_sig = self.encode_signature(
            candidate_palette,
            thresholds=target_sig.gradient_signature.thresholds,
        )
        score = self.gradient_logic.contrastive_score(
            target_sig.gradient_stops,
            candidate_sig.gradient_stops,
            negative_examples=[self.palette_to_stops(palette) for palette in negative_examples],
        )
        return ContrastivePaletteScore(
            score=float(score.score),
            positive_similarity=float(score.positive_similarity),
            negative_penalty=float(score.negative_penalty),
            candidate_signature=candidate_sig,
            target_signature=target_sig,
        )


__all__ = [
    "ContrastivePaletteScore",
    "TernaryPaletteLogic",
    "TernaryPaletteSignature",
]
