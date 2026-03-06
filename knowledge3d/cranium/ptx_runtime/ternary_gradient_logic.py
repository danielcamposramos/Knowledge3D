"""Ternary contrastive logic for procedural gradients.

This module applies the PM-KR ternary contrastive paradigm to color gradients:
- encode gradient stop deltas as ternary trends
- compare candidates against positives and negatives
- compose gradients from cascaded ternary layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps


def _prepare_stops(stops: Sequence[Sequence[float]]) -> np.ndarray:
    if not stops:
        raise ValueError("at least one gradient stop is required")
    arr = np.ascontiguousarray(np.asarray(stops, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("stops must have shape [n,5] as [pos,r,g,b,a]")
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1:] = np.clip(arr[:, 1:], 0.0, 1.0)
    return arr


@dataclass(frozen=True)
class TernaryGradientSignature:
    base_stop: tuple[float, float, float, float, float]
    delta_trits: tuple[tuple[int, int, int, int, int], ...]
    thresholds: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ContrastiveGradientScore:
    score: float
    positive_similarity: float
    negative_penalty: float
    candidate_signature: TernaryGradientSignature
    target_signature: TernaryGradientSignature


class TernaryGradientLogic:
    """Contrastive ternary surface for procedural gradients."""

    def __init__(self) -> None:
        self.ops = TernaryCodecOps()
        self._signature_cache: dict[
            tuple[tuple[tuple[float, float, float, float, float], ...], tuple[float, float, float, float, float]],
            TernaryGradientSignature,
        ] = {}

    def encode_signature(
        self,
        stops: Sequence[Sequence[float]],
        *,
        thresholds: tuple[float, float, float, float, float] = (0.08, 0.1, 0.1, 0.1, 0.08),
    ) -> TernaryGradientSignature:
        arr = _prepare_stops(stops)
        threshold_key = tuple(float(v) for v in thresholds)
        stops_key = tuple(tuple(float(v) for v in row.tolist()) for row in arr)
        cache_key = (stops_key, threshold_key)
        cached = self._signature_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(arr) == 1:
            signature = TernaryGradientSignature(
                base_stop=tuple(float(v) for v in arr[0].tolist()),
                delta_trits=(),
                thresholds=threshold_key,
            )
            self._signature_cache[cache_key] = signature
            return signature

        deltas = arr[1:] - arr[:-1]
        trit_columns: list[np.ndarray] = []
        for col in range(deltas.shape[1]):
            trits = self.ops.quantize_numpy(deltas[:, col], threshold=float(thresholds[col]))
            trit_columns.append(trits)

        trit_matrix = np.stack(trit_columns, axis=1).astype(np.int32, copy=False)
        signature = TernaryGradientSignature(
            base_stop=tuple(float(v) for v in arr[0].tolist()),
            delta_trits=tuple(tuple(int(v) for v in row) for row in trit_matrix.tolist()),
            thresholds=threshold_key,
        )
        self._signature_cache[cache_key] = signature
        return signature

    def contrastive_score(
        self,
        target_stops: Sequence[Sequence[float]],
        candidate_stops: Sequence[Sequence[float]],
        *,
        negative_examples: Sequence[Sequence[Sequence[float]]] = (),
    ) -> ContrastiveGradientScore:
        target = self.encode_signature(target_stops)
        candidate = self.encode_signature(candidate_stops, thresholds=target.thresholds)

        positive_similarity = self._signature_similarity(target, candidate)
        negative_penalty = 0.0
        for negative in negative_examples:
            negative_sig = self.encode_signature(negative, thresholds=target.thresholds)
            negative_penalty = max(negative_penalty, self._signature_similarity(candidate, negative_sig))

        score = positive_similarity - 0.8 * negative_penalty
        return ContrastiveGradientScore(
            score=float(score),
            positive_similarity=float(positive_similarity),
            negative_penalty=float(negative_penalty),
            candidate_signature=candidate,
            target_signature=target,
        )

    def compose_stops_from_cascade(
        self,
        *,
        base_stop: Sequence[float],
        position_layers: Sequence[Sequence[int]],
        color_layers: Sequence[Sequence[Sequence[int]]],
        base_spacing: float | None = None,
        position_step: float = 0.2,
        color_step: float = 0.18,
        alpha_step: float = 0.12,
    ) -> list[tuple[float, float, float, float, float]]:
        base = np.asarray(base_stop, dtype=np.float32)
        if base.shape != (5,):
            raise ValueError("base_stop must be length 5 [pos,r,g,b,a]")
        if len(position_layers) != len(color_layers):
            raise ValueError("position_layers and color_layers must have same number of layers")
        if not position_layers:
            return [tuple(float(v) for v in np.clip(base, 0.0, 1.0).tolist())]

        segment_count = len(position_layers[0])
        if segment_count < 1:
            return [tuple(float(v) for v in np.clip(base, 0.0, 1.0).tolist())]
        for layer in position_layers:
            if len(layer) != segment_count:
                raise ValueError("all position layers must share segment count")
        for layer in color_layers:
            if len(layer) != segment_count:
                raise ValueError("all color layers must share segment count")
            for step in layer:
                if len(step) != 4:
                    raise ValueError("color layer steps must be length 4 RGBA trits")

        spacing = float(base_spacing) if base_spacing is not None else (1.0 / float(segment_count))
        current_pos = float(np.clip(base[0], 0.0, 1.0))
        current_rgba = np.clip(base[1:].astype(np.float32, copy=True), 0.0, 1.0)
        stops: list[tuple[float, float, float, float, float]] = [
            (current_pos, float(current_rgba[0]), float(current_rgba[1]), float(current_rgba[2]), float(current_rgba[3]))
        ]

        for seg_idx in range(segment_count):
            pos_delta = spacing
            rgba_delta = np.zeros(4, dtype=np.float32)
            for layer_idx, pos_layer in enumerate(position_layers):
                scale = 1.0 / float(2**layer_idx)
                pos_delta += float(pos_layer[seg_idx]) * float(position_step) * scale
                color_step_vec = np.asarray(color_layers[layer_idx][seg_idx], dtype=np.float32)
                rgba_scale = np.array([color_step, color_step, color_step, alpha_step], dtype=np.float32) * scale
                rgba_delta += color_step_vec * rgba_scale

            current_pos = min(1.0, max(current_pos + max(0.02, pos_delta), current_pos))
            current_rgba = np.clip(current_rgba + rgba_delta, 0.0, 1.0)
            stops.append(
                (current_pos, float(current_rgba[0]), float(current_rgba[1]), float(current_rgba[2]), float(current_rgba[3]))
            )

        if stops[-1][0] < 1.0:
            last = stops[-1]
            stops[-1] = (1.0, last[1], last[2], last[3], last[4])
        return stops

    def _signature_similarity(
        self,
        a: TernaryGradientSignature,
        b: TernaryGradientSignature,
    ) -> float:
        rows = min(len(a.delta_trits), len(b.delta_trits))
        if rows == 0:
            return 1.0

        score = 0.0
        count = 0
        for row_idx in range(rows):
            row_a = a.delta_trits[row_idx]
            row_b = b.delta_trits[row_idx]
            for trit_a, trit_b in zip(row_a, row_b, strict=False):
                count += 1
                if trit_a == trit_b == 0:
                    score += 0.5
                elif trit_a == trit_b:
                    score += 1.0
                elif trit_a == -trit_b:
                    score -= 1.0
                else:
                    score -= 0.25

        if len(a.delta_trits) != len(b.delta_trits):
            score -= 0.25 * abs(len(a.delta_trits) - len(b.delta_trits))
            count += abs(len(a.delta_trits) - len(b.delta_trits))

        return float(score / max(count, 1))


__all__ = [
    "ContrastiveGradientScore",
    "TernaryGradientLogic",
    "TernaryGradientSignature",
]
