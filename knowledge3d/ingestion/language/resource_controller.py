"""
Resource-aware ingestion controller for Step 15.

Provides linear (text → audio → visual) ingestion with VRAM monitoring and
spill-to-House behaviour when the 12 GB RTX 3060 budget is exceeded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard, OOMSpillManager
from knowledge3d.cranium.sovereign.loader import get_vram_usage


IngestionHandler = Callable[[Sequence[Any]], Sequence[Any]]


@dataclass
class ResourceSafeIngestionController:
    """
    Linear ingestion orchestrator with VRAM budget enforcement.

    Parameters
    ----------
    vram_budget_gb:
        Soft VRAM limit (defaults to 8 GB leaving headroom on a 12 GB card).
    spill_dir:
        Directory where overflow batches are stored when spilling to House.
    logger:
        Callable used for logging (defaults to built-in `print`).
    """

    vram_budget_gb: float = 8.0
    spill_dir: Path | None = None
    logger: Callable[[str], None] = print

    latency_guard: LatencyGuard = field(init=False)
    spill_manager: OOMSpillManager = field(init=False)
    spill_path: Path = field(init=False)
    results_buffer: Dict[str, List[Any]] = field(init=False, default_factory=lambda: {"text": [], "audio": [], "visual": []})

    def __post_init__(self) -> None:
        self.latency_guard = LatencyGuard(threshold_us=95.0)
        self.spill_manager = OOMSpillManager()
        default_spill = Path("../Knowledge3D.local/house_spill").resolve()
        self.spill_path = (self.spill_dir or default_spill)
        self.spill_path.mkdir(parents=True, exist_ok=True)
        self.vram_budget_bytes = int(self.vram_budget_gb * 1e9)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def batch_ingest_linear(
        self,
        *,
        text_items: Sequence[Any] | None,
        text_handler: IngestionHandler | None,
        audio_items: Sequence[Any] | None = None,
        audio_handler: IngestionHandler | None = None,
        visual_items: Sequence[Any] | None = None,
        visual_handler: IngestionHandler | None = None,
        batch_size: int = 128,
    ) -> Dict[str, List[Any]]:
        """
        Sequentially ingest text → audio → visual batches.

        Each handler receives a slice of `batch_size` items and returns the
        processed results (any serializable structure).
        """
        self.logger("Starting resource-safe ingestion (linear sequence).")
        schedule = [
            ("text", text_items, text_handler),
            ("audio", audio_items, audio_handler),
            ("visual", visual_items, visual_handler),
        ]

        for modality, items, handler in schedule:
            if not items or handler is None:
                continue
            self._process_modality(modality, items, handler, batch_size)

        return self.results_buffer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_modality(
        self,
        modality: str,
        items: Sequence[Any],
        handler: IngestionHandler,
        batch_size: int,
    ) -> None:
        self.logger(f"Ingesting {modality} modality ({len(items)} items)…")
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            batch_index = start // batch_size

            self._maybe_spill(modality, len(batch))

            self.latency_guard.start()
            processed = handler(batch)
            elapsed_ns, breached = self.latency_guard.stop()

            self.results_buffer.setdefault(modality, []).extend(processed)
            self.logger(
                f"[{modality}] batch {batch_index} size={len(batch)} "
                f"latency={elapsed_ns / 1_000:.2f}µs{' BREACH' if breached else ''}"
            )

    def _maybe_spill(self, modality: str, batch_count: int) -> None:
        used, total = self._safe_vram_query()
        estimate = self._estimate_batch_vram(batch_count, modality)
        if used + estimate <= self.vram_budget_bytes:
            return

        self.logger(
            f"VRAM budget exceeded for {modality}: "
            f"used={self._format_bytes(used)}, estimate={self._format_bytes(estimate)} "
            f"(budget={self._format_bytes(self.vram_budget_bytes)}). Spilling to House."
        )
        self._spill_to_house(modality)

    def _spill_to_house(self, modality: str) -> None:
        payload = self.results_buffer.get(modality, [])
        if not payload:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        spill_file = self.spill_path / f"{modality}-spill-{timestamp}.json"
        spill_metadata = {
            "modality": modality,
            "item_count": len(payload),
            "timestamp": timestamp,
        }
        with spill_file.open("w", encoding="utf-8") as handle:
            json.dump(spill_metadata, handle, indent=2)

        # Reset buffer for modality after spill
        self.results_buffer[modality] = []

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value <= 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = min(int(math.log(value, 1024)), len(units) - 1)
        scaled = value / (1024 ** idx)
        return f"{scaled:.2f} {units[idx]}"

    def _estimate_batch_vram(self, count: int, modality: str) -> int:
        if modality == "text":
            return int(count * 2_048)  # ~2 KB per sentence
        if modality == "audio":
            return int(count * 90_000)  # ~90 KB per clip
        if modality == "visual":
            return int(count * 200_000)  # ~200 KB per glyph/frame
        return int(count * 1_024)

    def _safe_vram_query(self) -> tuple[int, int]:
        try:
            return get_vram_usage()
        except RuntimeError:
            # Driver does not expose cuMemGetInfo; fall back to zeros.
            return 0, int(self.vram_budget_gb * 1e9)


__all__ = ["ResourceSafeIngestionController"]
