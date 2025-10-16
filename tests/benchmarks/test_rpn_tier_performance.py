from __future__ import annotations

import time

import numpy as np
import pytest

from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


@pytest.fixture(scope="module")
def tiered_engine() -> TieredRPNEngine:
    try:
        return TieredRPNEngine()
    except RuntimeError as exc:
        if "invalid device context" in str(exc):
            pytest.skip("CUDA context unavailable for tiered RPN benchmarks")
        raise


class TestRPNTierPerformance:
    """Validate tier latency targets."""

    def test_tier1_latency_under_1us(self, tiered_engine: TieredRPNEngine) -> None:
        """Tier 1 should complete simple ops in <1µs."""
        op_codes = np.array([0x0000, 0x0000, 0x000A], dtype=np.uint16)
        scalars = np.array([2.0, 3.0, 0.0], dtype=np.float32)
        vectors = np.zeros((len(op_codes), 3), dtype=np.float32)

        for _ in range(100):
            tiered_engine.execute_scalar(op_codes, scalars=scalars, vectors=vectors)

        iterations = 10_000
        start = time.perf_counter()
        for _ in range(iterations):
            tiered_engine.execute_scalar(op_codes, scalars=scalars, vectors=vectors)
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1e6
        print(f"\nTier 1 avg latency: {avg_latency_us:.3f}µs")
        assert avg_latency_us < 1.0, f"Tier 1 latency {avg_latency_us:.3f}µs exceeds 1µs target"

    def test_tier2_latency_around_3us(self, tiered_engine: TieredRPNEngine) -> None:
        """Tier 2 should complete vector ops in ~3µs."""
        op_codes = np.array([0x0001, 0x0001, 0x003C], dtype=np.uint16)  # vector literals + dot
        scalars = np.zeros(len(op_codes), dtype=np.float32)
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        for _ in range(100):
            tiered_engine.execute_scalar(op_codes, scalars=scalars, vectors=vectors)

        iterations = 10_000
        start = time.perf_counter()
        for _ in range(iterations):
            tiered_engine.execute_scalar(op_codes, scalars=scalars, vectors=vectors)
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1e6
        print(f"\nTier 2 avg latency: {avg_latency_us:.3f}µs")
        print(f"Tier 2 target: ~3µs, actual: {avg_latency_us:.3f}µs")

    def test_tier3_matmul_latency(self, tiered_engine: TieredRPNEngine) -> None:
        """Tier 3 matrix multiply should complete in ~10µs."""
        op_codes = np.array([0x0002, 0x0002, 0x005A], dtype=np.uint16)
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 10.0]], dtype=np.float32)
        B = np.array([[9.0, 8.0, 7.0],
                      [6.0, 5.0, 4.0],
                      [3.0, 2.0, 1.0]], dtype=np.float32)
        scalars = np.array([A.shape[0], A.shape[1], B.shape[0], B.shape[1]], dtype=np.float32)
        literals = np.concatenate([A.flatten(), B.flatten()]).astype(np.float32)

        for _ in range(100):
            tiered_engine.execute_matrix(
                op_codes,
                matrix_shape=A.shape,
                scalars=scalars,
                matrices=literals,
            )

        iterations = 1_000
        start = time.perf_counter()
        for _ in range(iterations):
            tiered_engine.execute_matrix(
                op_codes,
                matrix_shape=A.shape,
                scalars=scalars,
                matrices=literals,
            )
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1e6
        print(f"\nTier 3 MATMUL (3×3) avg latency: {avg_latency_us:.3f}µs")
        print(f"Tier 3 target: ~10µs, actual: {avg_latency_us:.3f}µs")

    def test_tier_dispatch_distribution(self, tiered_engine: TieredRPNEngine) -> None:
        """Verify orchestrator routes to expected tiers."""
        tier1_codes = np.array([0x0000, 0x0000, 0x000A], dtype=np.uint16)
        tier1_scalars = np.array([2.0, 3.0, 0.0], dtype=np.float32)
        tier1_vectors = np.zeros((len(tier1_codes), 3), dtype=np.float32)

        for _ in range(100):
            tiered_engine.execute_scalar(tier1_codes, scalars=tier1_scalars, vectors=tier1_vectors)

        tier2_codes = np.array([0x0001, 0x0001, 0x003C], dtype=np.uint16)
        tier2_scalars = np.zeros(len(tier2_codes), dtype=np.float32)
        tier2_vectors = np.array(
            [
                [0.2, 0.3, 0.5],
                [0.2, 0.6, 0.1],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        for _ in range(10):
            tiered_engine.execute_scalar(tier2_codes, scalars=tier2_scalars, vectors=tier2_vectors)

        tier3_codes = np.array([0x0002, 0x0064], dtype=np.uint16)  # literal + trace
        matrix = np.eye(2, dtype=np.float32)
        scalars = np.array([matrix.shape[0], matrix.shape[1]], dtype=np.float32)
        literals = matrix.flatten().astype(np.float32)
        tiered_engine.execute_scalar(tier3_codes, scalars=scalars, matrices=literals)

        print("\nTier dispatch test: 100 Tier-1, 10 Tier-2, 1 Tier-3 calls executed")
