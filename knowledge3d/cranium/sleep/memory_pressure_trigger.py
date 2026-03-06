"""VRAM-pressure trigger for sleep-time consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge3d.cranium.sovereign import loader


@dataclass(frozen=True)
class MemoryPressureSnapshot:
    used_bytes: int
    total_bytes: int
    usage_ratio: float
    threshold_ratio: float
    reserve_bytes: int
    free_bytes: int
    should_consolidate: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "used_bytes": int(self.used_bytes),
            "total_bytes": int(self.total_bytes),
            "usage_ratio": float(self.usage_ratio),
            "threshold_ratio": float(self.threshold_ratio),
            "reserve_bytes": int(self.reserve_bytes),
            "free_bytes": int(self.free_bytes),
            "should_consolidate": bool(self.should_consolidate),
            "reason": str(self.reason),
        }


class MemoryPressureTrigger:
    """Query sovereign VRAM usage and decide whether sleep-time should fire."""

    def __init__(
        self,
        *,
        threshold_ratio: float = 0.82,
        reserve_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.threshold_ratio = float(max(0.0, min(1.0, threshold_ratio)))
        self.reserve_bytes = int(max(0, reserve_bytes))

    def snapshot(self) -> MemoryPressureSnapshot:
        used_bytes, total_bytes = loader.get_vram_usage()
        total = max(int(total_bytes), 1)
        used = max(0, int(used_bytes))
        free = max(0, total - used)
        ratio = float(used) / float(total)

        threshold_hit = ratio >= self.threshold_ratio
        reserve_hit = free <= self.reserve_bytes
        should_consolidate = bool(threshold_hit or reserve_hit)

        if threshold_hit and reserve_hit:
            reason = "threshold_and_reserve"
        elif threshold_hit:
            reason = "threshold_ratio"
        elif reserve_hit:
            reason = "reserve_bytes"
        else:
            reason = "ok"

        return MemoryPressureSnapshot(
            used_bytes=used,
            total_bytes=total,
            usage_ratio=ratio,
            threshold_ratio=self.threshold_ratio,
            reserve_bytes=self.reserve_bytes,
            free_bytes=free,
            should_consolidate=should_consolidate,
            reason=reason,
        )


__all__ = ["MemoryPressureSnapshot", "MemoryPressureTrigger"]
