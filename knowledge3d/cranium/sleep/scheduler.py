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
        """Run sleep-time consolidation via sovereign PTX kernels.

        Dispatches through :class:`SleepConsolidationDriver` which loads
        ``sleep_time_micro.ptx``, ``sleep_glyph_consolidator.ptx``, and
        ``sleep_cluster_refiner.ptx`` via the sovereign loader. No numpy,
        no fallbacks — per ``feedback_no_fallbacks_ever_including_sleeptime.md``.

        Working buffers are pulled from ``self.rpn_engine`` when the engine
        exposes ``sleep_ring_buffer()`` / ``sleep_glyph_embeddings()``; when
        they are unavailable (e.g. benchmark runs that never populate them)
        the tick is a no-op and the elapsed time is still recorded.
        """
        from knowledge3d.cranium.sleep.ptx_driver import SleepConsolidationDriver

        driver = getattr(self, "_driver", None)
        if driver is None:
            driver = SleepConsolidationDriver()
            self._driver = driver

        ring_buffer = None
        glyph_embeddings = None
        engine = self.rpn_engine
        if engine is not None:
            ring_fn = getattr(engine, "sleep_ring_buffer", None)
            if callable(ring_fn):
                try:
                    ring_buffer = ring_fn()
                except Exception:
                    ring_buffer = None
            glyph_fn = getattr(engine, "sleep_glyph_embeddings", None)
            if callable(glyph_fn):
                try:
                    glyph_embeddings = glyph_fn()
                except Exception:
                    glyph_embeddings = None

        t0 = time.time()
        results = driver.consolidate(
            ring_buffer=ring_buffer,
            glyph_embeddings=glyph_embeddings,
        )
        self.consolidation_count += 1
        self._save_metrics(results, time.time() - t0)

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
