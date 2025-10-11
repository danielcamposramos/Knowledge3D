"""Dual-client synchronization helper."""

from __future__ import annotations

from typing import Tuple

import cupy as cp


class DualClientSync:
    """Maintain shared state for human and AI clients.

    This implementation keeps two buffers and exposes a lightweight API to
    stage updates and materialise synchronized results. It is intentionally
    simple but provides a concrete foundation that can be expanded with more
    elaborate atomic operations later on.
    """

    def __init__(self) -> None:
        self.human_buffer = None
        self.ai_buffer = None

    def stage(self, human_view: cp.ndarray, ai_view: cp.ndarray) -> None:
        self.human_buffer = cp.asarray(human_view, dtype=cp.float32)
        self.ai_buffer = cp.asarray(ai_view, dtype=cp.float32)

    def sync(self) -> Tuple[cp.ndarray, cp.ndarray]:
        if self.human_buffer is None or self.ai_buffer is None:
            raise RuntimeError("Buffers have not been staged")
        average = (self.human_buffer + self.ai_buffer) * 0.5
        self.human_buffer = average.copy()
        self.ai_buffer = average.copy()
        return self.human_buffer, self.ai_buffer


__all__ = ["DualClientSync"]
