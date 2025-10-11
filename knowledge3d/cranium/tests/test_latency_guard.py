import cupy as cp
import pytest

from knowledge3d.cranium.bridges.guard import LatencyGuard


_BUSY_WAIT_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__ void busy_wait(unsigned long long wait_cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < wait_cycles) {
        // prevent compiler optimizations
        asm volatile("");
    }
}
""",
    "busy_wait",
)


@pytest.mark.cuda
def test_latency_guard_within_threshold():
    guard = LatencyGuard(threshold_us=200.0)

    elapsed_ns, breached = guard.check(lambda: None)

    assert not breached
    assert elapsed_ns >= 0
    assert elapsed_ns < int(500_000)  # sanity bound: <0.5 ms


@pytest.mark.cuda
def test_latency_guard_detects_breach():
    guard = LatencyGuard(threshold_us=50.0)

    def _slow_gpu_op() -> None:
        # Wait for roughly ~150 µs (depends on GPU clock). Overshoot threshold.
        wait_cycles = cp.uint64(2_000_000)
        _BUSY_WAIT_KERNEL((1,), (1,), (wait_cycles,))

    elapsed_ns, breached = guard.check(_slow_gpu_op)

    assert breached, "Guard should flag when exceeding the threshold"
    assert elapsed_ns > int(50_000)  # > 50 µs


@pytest.mark.cuda
def test_latency_guard_context_manager():
    guard = LatencyGuard(threshold_us=150.0)

    with guard.measure():
        wait_cycles = cp.uint64(500_000)
        _BUSY_WAIT_KERNEL((1,), (1,), (wait_cycles,))

    assert guard.last_elapsed_ns > 0
    assert guard.last_flag in (0, 0xDEADBEEF)
