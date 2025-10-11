"""
Latency guard bridge for GPU kernels.

Uses the PTX kernel `gre_sub100micro_gate.ptx` to record `%globaltimer`
timestamps before and after an operation. If the elapsed time exceeds the
configured threshold (default 100 microseconds), a sentinel flag is written.

The bridge exposes a convenient context manager as well as imperative APIs so
other bridges can wrap their hot kernels without relying on CPU timers.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class LatencyGuard:
    """
    GPU latency guard built on top of `gre_sub100micro_gate.ptx`.

    Parameters
    ----------
    threshold_us:
        Maximum allowed latency in microseconds (default: 100).
    """

    def __init__(self, threshold_us: float = 100.0) -> None:
        self.threshold_us = float(threshold_us)
        self.threshold_ns = int(self.threshold_us * 1_000.0)

        ptx_path = Path(__file__).resolve().parent.parent / "kernels" / "gre_sub100micro_gate.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"PTX kernel not found: {ptx_path}")

        self._module = cp.RawModule(path=str(ptx_path))
        self._kernel = self._module.get_function("gre_sub100micro_gate")

        # Device buffers reused across invocations.
        self._timestamps = cp.zeros(2, dtype=cp.uint64)
        self._flag = cp.zeros(1, dtype=cp.uint32)

        self.last_elapsed_ns: int = 0
        self.last_flag: int = 0

    def _launch(self, mode: int, stream: Optional[cp.cuda.Stream]) -> None:
        """Internal helper to launch the PTX kernel."""
        if mode not in (0, 1):
            raise ValueError(f"Unsupported guard mode: {mode}")

        stream = stream if stream is not None else cp.cuda.Stream.null

        self._kernel(
            (1,),
            (32,),
            (
                self._timestamps.data.ptr,
                self._flag.data.ptr,
                self.threshold_ns,
                mode,
            ),
            stream=stream,
        )

    def start(self, stream: Optional[cp.cuda.Stream] = None) -> None:
        """Record the start timestamp on the GPU."""
        self._launch(mode=0, stream=stream)

    def stop(self, stream: Optional[cp.cuda.Stream] = None) -> Tuple[int, bool]:
        """
        Record the stop timestamp, compute elapsed nanoseconds, and return the result.

        Returns
        -------
        elapsed_ns:
            Time between start/stop in nanoseconds.
        breached:
            True if the latency exceeded the configured threshold.
        """
        self._launch(mode=1, stream=stream)
        (stream or cp.cuda.Stream.null).synchronize()

        timestamps = self._timestamps.get()
        flag = int(self._flag.get()[0])

        elapsed_ns = int(timestamps[1] - timestamps[0])
        self.last_elapsed_ns = elapsed_ns
        self.last_flag = flag

        breached = flag == 0xDEADBEEF
        return elapsed_ns, breached

    def check(self, func: Callable[[], None], stream: Optional[cp.cuda.Stream] = None) -> Tuple[int, bool]:
        """
        Run `func` between guard start/stop calls and return the measurement.
        """
        self.start(stream=stream)
        try:
            func()
        finally:
            elapsed_ns, breached = self.stop(stream=stream)
        return elapsed_ns, breached

    @contextlib.contextmanager
    def measure(self, stream: Optional[cp.cuda.Stream] = None) -> Iterator["LatencyGuard"]:
        """
        Context manager for easier integration.

        Example
        -------
        >>> guard = LatencyGuard()
        >>> with guard.measure():
        ...     launch_kernel()
        >>> guard.last_elapsed_ns
        42000
        """
        self.start(stream=stream)
        try:
            yield self
        finally:
            self.stop(stream=stream)


__all__ = ["LatencyGuard"]
