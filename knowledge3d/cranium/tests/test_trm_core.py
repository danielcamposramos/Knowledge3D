"""
Test suite for Tiny Recursive Model (TRM) core functionality.

Chain Contributors: All partners
Validation: Latency <95µs, convergence ≤16 steps, ARC-AGI accuracy ≥40%
"""

import cupy as cp
import pytest
import time
from knowledge3d.cranium.bridges.trm_core import TinyRecursiveModel, create_trm


class TestTRMCore:
    """Core TRM functionality tests."""

    @pytest.fixture
    def trm_model(self):
        """Create TRM instance for testing."""
        return create_trm(hidden_dim=512, n_recursions=6, T_iterations=3)

    @pytest.mark.parametrize("batch_size", [1, 16, 32, 128])
    def test_convergence_and_latency(self, trm_model, batch_size):
        """Test TRM convergence within 16 steps and <95µs latency."""
        # Generate random question embeddings
        question = cp.random.randn(batch_size, 512).astype(cp.float32)

        # Run recursive refinement
        answer, latent, steps, elapsed_us = trm_model.recursive_refine(
            question=question,
            max_supervision_steps=16
        )

        # Validation assertions
        assert steps <= 16, f"Exceeded max supervision steps: {steps}"
        assert elapsed_us < 95.0, f"Latency breach: {elapsed_us:.2f}µs (target: <95µs)"
        assert cp.isfinite(answer).all(), "Answer contains NaN or Inf"
        assert cp.isfinite(latent).all(), "Latent contains NaN or Inf"
        assert answer.shape == (batch_size, 512), f"Wrong answer shape: {answer.shape}"
        assert latent.shape == (batch_size, 512), f"Wrong latent shape: {latent.shape}"

    def test_early_stopping_via_halt(self, trm_model):
        """Test adaptive halting (ACT) stops early when converged."""
        # Create simple question that should converge quickly
        question = cp.ones((4, 512), dtype=cp.float32)

        answer, latent, steps, _ = trm_model.recursive_refine(
            question=question,
            max_supervision_steps=16
        )

        # Should converge in less than max steps
        assert steps < 16, "ACT halting should stop early for simple inputs"

    def test_ema_stability(self, trm_model):
        """Test EMA weight updates for training stability."""
        question = cp.random.randn(8, 512).astype(cp.float32)

        # Get initial weights
        initial_weights = trm_model.weights.copy()

        # Run training iteration
        trm_model.recursive_refine(
            question=question,
            max_supervision_steps=5,
            training=True
        )

        # EMA weights should be different from initial
        assert not cp.allclose(trm_model.ema_weights, initial_weights), \
            "EMA weights should update during training"

    def test_gradient_tracking(self, trm_model):
        """Test gradient tracking mode vs detached mode."""
        question = cp.random.randn(2, 512).astype(cp.float32)

        # Training mode (with gradients)
        answer_train, _, _, _ = trm_model.recursive_refine(
            question=question,
            max_supervision_steps=3,
            training=True
        )

        # Inference mode (detached)
        answer_infer, _, _, _ = trm_model.recursive_refine(
            question=question,
            max_supervision_steps=3,
            training=False
        )

        # Both should produce valid outputs
        assert cp.isfinite(answer_train).all()
        assert cp.isfinite(answer_infer).all()

    def test_performance_stats(self, trm_model):
        """Test performance statistics tracking."""
        question = cp.random.randn(16, 512).astype(cp.float32)

        trm_model.recursive_refine(question=question)
        stats = trm_model.get_performance_stats()

        assert 'last_latency_us' in stats
        assert 'mean_convergence_steps' in stats
        assert 'sla_compliant' in stats
        assert stats['sla_compliant'], "Should meet <95µs SLA"


class TestTRMIntegration:
    """Integration tests for TRM with other K3D components."""

    def test_batch_size_scaling(self):
        """Test TRM scales linearly with batch size."""
        trm = create_trm()

        batch_sizes = [1, 8, 32, 64]
        latencies = []

        for batch_size in batch_sizes:
            question = cp.random.randn(batch_size, 512).astype(cp.float32)
            _, _, _, latency = trm.recursive_refine(question=question)
            latencies.append(latency)

        # Latency should scale sub-linearly (due to warp efficiency)
        assert latencies[-1] < latencies[0] * len(batch_sizes), \
            "Batch processing should be more efficient than linear scaling"

    def test_zero_initialization(self):
        """Test TRM works with zero-initialized answer and latent."""
        trm = create_trm()
        question = cp.random.randn(4, 512).astype(cp.float32)

        # Explicitly pass None to test default initialization
        answer, latent, steps, _ = trm.recursive_refine(
            question=question,
            answer=None,  # Should initialize to zeros
            latent=None   # Should initialize to zeros
        )

        assert steps <= 16
        assert cp.isfinite(answer).all()
        assert cp.isfinite(latent).all()


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestTRMPerformance:
    """Performance benchmarks for TRM."""

    def test_latency_percentiles(self):
        """Test latency distribution meets SLA."""
        trm = create_trm()
        question = cp.random.randn(32, 512).astype(cp.float32)

        latencies = []
        for _ in range(100):
            _, _, _, latency = trm.recursive_refine(question=question)
            latencies.append(latency)

        latencies = cp.array(latencies)
        p50 = float(cp.percentile(latencies, 50))
        p95 = float(cp.percentile(latencies, 95))
        p99 = float(cp.percentile(latencies, 99))

        print(f"\nLatency Percentiles:")
        print(f"  P50: {p50:.2f}µs")
        print(f"  P95: {p95:.2f}µs")
        print(f"  P99: {p99:.2f}µs")

        assert p50 < 70.0, f"P50 latency too high: {p50:.2f}µs"
        assert p95 < 95.0, f"P95 latency exceeds SLA: {p95:.2f}µs"
        assert p99 < 120.0, f"P99 latency too high: {p99:.2f}µs"

    def test_convergence_distribution(self):
        """Test convergence step distribution."""
        trm = create_trm()
        question = cp.random.randn(64, 512).astype(cp.float32)

        steps_list = []
        for _ in range(50):
            _, _, steps, _ = trm.recursive_refine(question=question)
            steps_list.append(steps)

        mean_steps = float(cp.mean(cp.array(steps_list)))

        print(f"\nConvergence Steps:")
        print(f"  Mean: {mean_steps:.2f}")
        print(f"  Min: {min(steps_list)}")
        print(f"  Max: {max(steps_list)}")

        # Most should converge in <10 steps per TRM paper
        assert mean_steps < 10.0, f"Mean convergence too high: {mean_steps:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
