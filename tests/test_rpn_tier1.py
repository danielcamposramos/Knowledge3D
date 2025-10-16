"""Unit tests for the Tier‑1 (lightweight) RPN engine."""
from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine


@pytest.fixture(scope="module")
def tier1_engine():
    return LightweightRPNEngine()


class TestTier1RPN:
    def test_arithmetic_ops(self, tier1_engine: LightweightRPNEngine):
        op_codes = np.array([0, 0, 10], dtype=np.uint16)  # 2 + 3
        scalars = np.array([2.0, 3.0, 0.0], dtype=np.float32)
        vectors = np.zeros((3, 3), dtype=np.float32)

        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 5.0) < 1e-5

        # (10 - 3) * 2 = 14
        op_codes = np.array([0, 0, 11, 0, 12], dtype=np.uint16)
        scalars = np.array([10.0, 3.0, 2.0, 0.0, 0.0], dtype=np.float32)
        vectors = np.zeros((5, 3), dtype=np.float32)
        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 14.0) < 1e-5

    def test_math_ops(self, tier1_engine: LightweightRPNEngine):
        op_codes = np.array([0, 20], dtype=np.uint16)  # sqrt(16)
        scalars = np.array([16.0, 0.0], dtype=np.float32)
        vectors = np.zeros((2, 3), dtype=np.float32)

        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 4.0) < 1e-5

        op_codes = np.array([0, 24], dtype=np.uint16)  # sin(pi/2)
        scalars = np.array([np.pi / 2, 0.0], dtype=np.float32)
        vectors = np.zeros((2, 3), dtype=np.float32)
        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 1.0) < 1e-5

    def test_comparison_ops(self, tier1_engine: LightweightRPNEngine):
        op_codes = np.array([0, 0, 46], dtype=np.uint16)  # max(3, 7)
        scalars = np.array([3.0, 7.0, 0.0], dtype=np.float32)
        vectors = np.zeros((3, 3), dtype=np.float32)

        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 7.0) < 1e-5

        op_codes = np.array([0, 0, 44], dtype=np.uint16)  # eq(5, 5)
        scalars = np.array([5.0, 5.0, 0.0], dtype=np.float32)
        vectors = np.zeros((3, 3), dtype=np.float32)
        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 1.0) < 1e-5

    def test_stack_ops(self, tier1_engine: LightweightRPNEngine):
        # dup and mul: (5 dup *) → 25
        op_codes = np.array([0, 50, 12], dtype=np.uint16)
        scalars = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        vectors = np.zeros((3, 3), dtype=np.float32)
        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 25.0) < 1e-5

        # swap: (3 4 swap -) → 1 (4 - 3)
        op_codes = np.array([0, 0, 51, 11], dtype=np.uint16)
        scalars = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
        vectors = np.zeros((4, 3), dtype=np.float32)
        result = tier1_engine.execute_single(0, op_codes, scalars, vectors)
        assert abs(result - 1.0) < 1e-5

    def test_unsupported_op(self, tier1_engine: LightweightRPNEngine):
        op_codes = np.array([0, 0, 60], dtype=np.uint16)  # dot (Tier‑2)
        scalars = np.zeros(3, dtype=np.float32)
        vectors = np.zeros((3, 3), dtype=np.float32)

        with pytest.raises(ValueError):
            tier1_engine.execute_single(0, op_codes, scalars, vectors)

    def test_latency_hint(self, tier1_engine: LightweightRPNEngine):
        if not tier1_engine.gpu_enabled:
            pytest.skip("Tier‑1 GPU kernel unavailable on this host")

        import time

        op_codes = np.array([0, 0, 10], dtype=np.uint16)
        scalars = np.array([2.0, 3.0, 0.0], dtype=np.float32)
        vectors = np.zeros((3, 3), dtype=np.float32)

        # Warm-up
        for _ in range(16):
            tier1_engine.execute_single(0, op_codes, scalars, vectors)

        iterations = 512
        start = time.perf_counter()
        for _ in range(iterations):
            tier1_engine.execute_single(0, op_codes, scalars, vectors)
        elapsed = (time.perf_counter() - start) * 1e6 / iterations

        assert elapsed < 1.0, f"Tier‑1 latency {elapsed:.2f}µs exceeds 1µs target"
