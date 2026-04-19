"""
GPU-native ternary diagnostics helper.

Provides two utilities backed by sovereign bridges:
 - TritOverlayGenerator: renders packed ternary fields into RGBA overlays
 - TritInspectorBridge: summarizes ternary fields for selected nodes

All methods are GPU-first and avoid CPU math except for final summaries.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple
import numpy as np

from knowledge3d.cranium.bridges.sovereign_bridges import (
    TritOverlayGenerator,
    TritInspectorBridge,
)


class TritInspector:
    """High-level ternary diagnostics for Galaxy fields."""

    def __init__(
        self,
        field_stride: int,
        overlay: Optional[TritOverlayGenerator] = None,
        inspector: Optional[TritInspectorBridge] = None,
    ) -> None:
        self.field_stride = int(field_stride)
        self.overlay = overlay or TritOverlayGenerator()
        self.inspector = inspector or TritInspectorBridge()

    def generate_overlay(
        self,
        trits_packed: np.ndarray,
        grid_shape: Tuple[int, int, int],
        field_type: int = 0,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Render RGBA8 overlay for a chosen ternary field."""
        return self.overlay.generate(
            trits_packed=trits_packed,
            grid_shape=grid_shape,
            field_stride=self.field_stride,
            field_type=field_type,
            threshold=threshold,
        )

    def inspect_node_trits(
        self,
        trits_packed: np.ndarray,
        node_index: int,
    ) -> dict:
        """Inspect a single node's ternary field."""
        summaries = self.inspector.inspect(
            trits_packed=trits_packed,
            node_indices=np.array([node_index], dtype=np.int32),
            field_stride=self.field_stride,
        )
        s = summaries[0]
        return {
            "count": int(s["count"]),
            "sum": int(s["sum"]),
            "mean": float(s["mean"]),
            "var": float(s["var"]),
            "bottleneck": bool(s["bottlenecks"]),
        }

    def trace_path_trits(
        self,
        trits_packed: np.ndarray,
        path_indices: Sequence[int],
    ) -> dict:
        """Aggregate ternary summaries across a path."""
        nodes = np.array(list(path_indices), dtype=np.int32)
        summaries = self.inspector.inspect(
            trits_packed=trits_packed,
            node_indices=nodes,
            field_stride=self.field_stride,
        )
        # Simple reductions on CPU after GPU summaries
        counts = summaries["count"].sum()
        sums = summaries["sum"].sum()
        means = summaries["mean"]
        bottlenecks = int(summaries["bottlenecks"].sum())
        return {
            "path_length": int(nodes.size),
            "mean_of_means": float(means.mean() if means.size else 0.0),
            "sum": int(sums),
            "count": int(counts),
            "bottlenecks": bottlenecks,
        }
