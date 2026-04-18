from __future__ import annotations

import sys
import threading
import time
from typing import Any


class TickDriver:
    def __init__(self, kv: Any, *, max_hz: float = 50.0, idle_backoff_ms: int = 20) -> None:
        self._kv = kv
        self._interval_s = 0.0 if max_hz <= 0.0 else 1.0 / float(max_hz)
        self._idle_backoff_s = max(0.0, float(idle_backoff_ms) / 1000.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks_total = 0
        self._ticks_since_start = 0
        self._idle_ticks = 0
        self._active_ticks = 0
        self._error_ticks = 0
        self._last_tick_wall_ms = 0.0
        self._last_error = ""

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._ticks_since_start = 0
        self._thread = threading.Thread(target=self._run, name="k3d-tick-driver", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stats(self) -> dict[str, Any]:
        running = self.is_running()
        return {
            "active": running,
            "running": running,
            "ticks_total": int(self._ticks_total),
            "ticks_since_start": int(self._ticks_since_start),
            "last_tick_wall_ms": float(self._last_tick_wall_ms),
            "idle_ticks": int(self._idle_ticks),
            "active_ticks": int(self._active_ticks),
            "error_ticks": int(self._error_ticks),
            "last_error": str(self._last_error),
        }

    def _run(self) -> None:
        logged_error = False
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                processed = int(self._kv.run_ticks(1))
                self._last_error = ""
                logged_error = False
            except Exception as exc:
                processed = 0
                self._error_ticks += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                if not logged_error:
                    print(f"[TickDriver] {self._last_error}", file=sys.stderr, flush=True)
                    logged_error = True
            elapsed_s = max(0.0, time.monotonic() - started)
            self._last_tick_wall_ms = elapsed_s * 1000.0
            self._ticks_total += 1
            self._ticks_since_start += 1
            if processed > 0:
                self._active_ticks += 1
                sleep_s = max(0.0, self._interval_s - elapsed_s)
            else:
                self._idle_ticks += 1
                sleep_s = self._idle_backoff_s
            if self._stop.wait(timeout=sleep_s):
                break
