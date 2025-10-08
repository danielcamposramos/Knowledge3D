from __future__ import annotations

"""
Enhanced sleep integration helpers.

The Step7.2 blueprint proposed richer metrics flowing from the multi-modal
confidence system into the sleep pipeline.  This module offers a thin Python
layer that packages those metrics into ``ConsolidationTicket`` instances so the
rest of the code base can be incrementally upgraded without touching the PTX
runtime.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from knowledge3d.cranium.actions.action_types import ActionBuffer, ActionType
from knowledge3d.cranium.ptx_runtime import SleepTimeCompute


@dataclass
class AlphaMetric:
    timestamp: float
    action: str
    alpha: float
    confidence: float
    curiosity: float


class EnhancedSleepIntegrator:
    """
    Collect α metrics and feed them into SleepTimeCompute.
    """

    def __init__(self, sleep_compute: Optional[SleepTimeCompute] = None) -> None:
        self.sleep_compute = sleep_compute
        self.history: List[AlphaMetric] = []

    def bind_compute(self, sleep_compute: SleepTimeCompute) -> None:
        self.sleep_compute = sleep_compute

    def record_from_buffer(self, buffer: ActionBuffer, *, alpha: float, confidence: float, curiosity: float) -> None:
        metric = AlphaMetric(
            timestamp=time.time(),
            action=buffer.get_action_type().name,
            alpha=float(alpha),
            confidence=float(confidence),
            curiosity=float(curiosity),
        )
        self.history.append(metric)

    def create_ticket(self, buffer: ActionBuffer, *, alpha: float, confidence: float, curiosity: float) -> Optional[SleepTimeCompute]:
        """
        Convenience helper used by the action router to trigger the sleep pipeline.
        """
        if self.sleep_compute is None:
            return None
        self.record_from_buffer(buffer, alpha=alpha, confidence=confidence, curiosity=curiosity)
        return self.sleep_compute

    def summarize(self) -> Dict[str, float]:
        if not self.history:
            return {"mean_alpha": 0.0, "count": 0}
        alphas = [m.alpha for m in self.history]
        return {
            "mean_alpha": float(sum(alphas) / len(alphas)),
            "count": float(len(alphas)),
        }

    def recommendations(self, threshold_low: float = 0.5, threshold_high: float = 2.0) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        slices: Dict[str, List[AlphaMetric]] = {}
        for metric in self.history:
            slices.setdefault(metric.action, []).append(metric)
        for action, metrics in slices.items():
            mean_alpha = sum(m.alpha for m in metrics) / len(metrics)
            efficiency = (
                sum(m.confidence for m in metrics) / len(metrics)
            ) / (mean_alpha + 1e-3)
            if efficiency < threshold_low:
                stats[action] = {
                    "recommendation": -0.2,
                    "mean_alpha": mean_alpha,
                    "efficiency": efficiency,
                }
            elif efficiency > threshold_high:
                stats[action] = {
                    "recommendation": +0.2,
                    "mean_alpha": mean_alpha,
                    "efficiency": efficiency,
                }
            else:
                stats[action] = {
                    "recommendation": 0.0,
                    "mean_alpha": mean_alpha,
                    "efficiency": efficiency,
                }
        return stats
