"""
Sleep Scheduler - Phase D

Detects idle time and auto-triggers RPN + glyph consolidation.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class SleepScheduler:
    """Detect idle time and trigger consolidation automatically."""

    def __init__(
        self,
        rpn_engine,
        idle_threshold: float = 300.0,  # 5 minutes
        log_path: str = "/K3D/Knowledge3D.local/logs/sleep_scheduler.jsonl",
    ):
        self.rpn_engine = rpn_engine
        self.idle_threshold = idle_threshold
        self.log_path = Path(log_path)
        self.last_activity = time.time()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.consolidation_count = 0

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def mark_activity(self):
        """Call this whenever ingestion happens to reset idle timer."""
        self.last_activity = time.time()

    def start(self):
        """Start background monitoring thread."""
        if self.running:
            print("[SLEEP] Scheduler already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[SLEEP] Scheduler started - idle threshold: {self.idle_threshold:.1f}s")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print("[SLEEP] Scheduler stopped")

    def _monitor_loop(self):
        """Background thread: check for idle time every 30 seconds."""
        while self.running:
            time.sleep(30)  # Check every 30 seconds

            idle_time = time.time() - self.last_activity
            if idle_time > self.idle_threshold:
                print(
                    f"[SLEEP] Idle for {idle_time:.1f}s - starting consolidation #{self.consolidation_count + 1}..."
                )
                self._run_consolidation()
                self.last_activity = time.time()  # Reset after consolidation

    def _run_consolidation(self):
        """Run both RPN and glyph consolidation."""
        try:
            start_time = time.time()
            results = {}

            # Phase 1: RPN Cluster Refinement
            print("[SLEEP] Phase 1: RPN cluster refinement...")
            from knowledge3d.cranium.sleep_time_consolidator import (
                SleepTimeConsolidator,
            )

            rpn_consolidator = SleepTimeConsolidator(self.rpn_engine)
            rpn_result = rpn_consolidator.consolidate()
            results["rpn"] = rpn_result
            print(
                "[SLEEP] Phase 1 complete: "
                f"{rpn_result.get('cluster_refinement', {}).get('clusters', 0)} clusters"
            )

            # Phase 2: Glyph Consolidation
            print("[SLEEP] Phase 2: Glyph consolidation...")
            from knowledge3d.cranium.sleep.glyph_consolidator import (
                GlyphConsolidator,
            )

            glyph_consolidator = GlyphConsolidator()
            glyph_result = glyph_consolidator.consolidate(
                similarity_threshold=0.98,
                min_retention_ratio=0.6,
            )
            results["glyph"] = glyph_result.to_dict()
            print(
                "[SLEEP] Phase 2 complete: "
                f"{glyph_result.glyphs_before} → {glyph_result.glyphs_after} glyphs "
                f"({glyph_result.reduction_pct:.1f}% reduction)"
            )

            # Log metrics
            elapsed = time.time() - start_time
            self.consolidation_count += 1

            self._save_metrics(results, elapsed)
            print(
                f"[SLEEP] Consolidation #{self.consolidation_count} complete in {elapsed:.1f}s"
            )

        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[SLEEP] ERROR: Consolidation failed - {exc}")
            import traceback

            traceback.print_exc()

    def _save_metrics(self, results: dict, elapsed: float):
        """Save consolidation metrics to JSONL log."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "consolidation_number": self.consolidation_count,
            "elapsed_seconds": elapsed,
            "rpn_consolidation": results.get("rpn", {}),
            "glyph_consolidation": results.get("glyph", {}),
        }

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        print(f"[SLEEP] Metrics saved: {self.log_path}")
