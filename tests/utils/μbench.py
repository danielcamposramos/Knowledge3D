"""
GPU-sovereign micro-benchmark utility for Knowledge3D testing.

Zero-dependency benchmarking that reports nanosecond-precision timing with
automatic baseline correction. Works in CPU-only environments.
"""
import time
import functools


class μBench:
    """
    GPU-sovereign micro-benchmark; reports nanoseconds.

    Automatically calibrates to subtract empty-loop overhead, ensuring
    honest microsecond budget tracking even in CPU-mocked tests.

    Usage:
        μ = μBench("test_name")
        stats = μ(my_function, arg1, arg2)
        print(f"p50: {stats['p50']}µs, p95: {stats['p95']}µs")
    """

    def __init__(self, prefix=""):
        self.prefix = prefix
        self._calibrate()

    def _calibrate(self):
        """Measure empty-loop cost to subtract from measurements."""
        t0 = time.perf_counter_ns()
        for _ in range(1000):
            pass
        t1 = time.perf_counter_ns()
        self.loop_ns = (t1 - t0) / 1000

    def __call__(self, fn, *a, **kw):
        """
        Run function 1000 times and return latency statistics.

        Args:
            fn: Function to benchmark
            *a, **kw: Arguments to pass to function

        Returns:
            dict with keys 'p50', 'p95', 'p99' (all in microseconds)
        """
        @functools.wraps(fn)
        def _wrapped():
            t0 = time.perf_counter_ns()
            fn(*a, **kw)
            t1 = time.perf_counter_ns()
            return (t1 - t0) - self.loop_ns

        # Run 1000 iterations, return p50/p95/p99 in µs
        samples = sorted(_wrapped() for _ in range(1000))
        p50 = samples[500] / 1e3
        p95 = samples[950] / 1e3
        p99 = samples[990] / 1e3
        return dict(p50=p50, p95=p95, p99=p99)


__all__ = ['μBench']
