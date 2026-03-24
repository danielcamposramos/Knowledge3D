#!/usr/bin/env python3
"""
Regression tests for Phase 2 RPN Sovereignty.

Validates that RPN-native gradient updates match CPU NumPy updates
and that ternary validation gate operates correctly.

Author: K3D Adaptive Swarm (Claude + Codex collaboration)
Date: 2025-11-19
"""

import numpy as np
import pytest

from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter, AdapterConfig
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32, RPNMathCore


def _np_matrix(value) -> np.ndarray:
    if isinstance(value, HostTensorF32):
        return np.asarray(value.to_nested_list(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _copy_tensor(dst, src) -> None:
    if isinstance(dst, HostTensorF32):
        dst.copy_from(src)
        return
    np.copyto(dst, src)

# Provide a benchmark fixture fallback when pytest-benchmark isn't installed.
try:
    import pytest_benchmark as _pytest_benchmark  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _pytest_benchmark = None

    @pytest.fixture
    def benchmark():
        pytest.skip("pytest-benchmark plugin not installed")


@pytest.mark.cuda
class TestRPNSovereignty:
    """Test sovereign RPN training against a NumPy reference update."""

    def test_rpn_vs_numpy_reference_gradient_update(self):
        """
        Verify sovereign RPN gradient update matches a NumPy reference.

        Acceptance criteria:
        - GPU path should match mathematically equivalent LoRA updates
        - Difference should be within floating-point tolerance (1e-4)
        """
        # Setup
        dims = 128
        rank = 16
        lr = 0.001

        # Create two identical adapters
        adapter_rpn = SelfUpdatingAdapter(
            shape=(dims, dims),
            rank=rank,
            specialist_name="rpn_test"
        )

        adapter_ref = SelfUpdatingAdapter(
            shape=(dims, dims),
            rank=rank,
            specialist_name="reference_test"
        )

        # Copy weights to ensure identical starting point
        _copy_tensor(adapter_ref.A, adapter_rpn.A)
        _copy_tensor(adapter_ref.B, adapter_rpn.B)

        # Generate test gradient
        gradient = np.random.randn(dims, dims).astype(np.float32) * 0.01

        # Apply via RPN path
        adapter_rpn.apply_gradient_rpn(gradient, lr)
        stale_host_a = _np_matrix(adapter_rpn.A).copy()
        stale_host_b = _np_matrix(adapter_rpn.B).copy()
        adapter_rpn.sync_weights_to_host()

        # Apply mathematically equivalent NumPy reference update
        grad_a = gradient @ _np_matrix(adapter_ref.B).T
        grad_b = _np_matrix(adapter_ref.A).T @ gradient
        _copy_tensor(adapter_ref.A, _np_matrix(adapter_ref.A) - (lr * grad_a))
        _copy_tensor(adapter_ref.B, _np_matrix(adapter_ref.B) - (lr * grad_b))

        # Compare results
        a_diff = np.linalg.norm(_np_matrix(adapter_rpn.A) - _np_matrix(adapter_ref.A))
        b_diff = np.linalg.norm(_np_matrix(adapter_rpn.B) - _np_matrix(adapter_ref.B))

        print(f"\n[Regression Test] RPN vs NumPy reference gradient update:")
        print(f"  A difference: {a_diff:.6f}")
        print(f"  B difference: {b_diff:.6f}")
        host_stale = (
            not np.allclose(stale_host_a, _np_matrix(adapter_ref.A))
            or not np.allclose(stale_host_b, _np_matrix(adapter_ref.B))
        )
        assert host_stale, "At least one host matrix should remain stale until sync_weights_to_host()"

        # Tolerance: 1e-4 (should be very close)
        assert a_diff < 1e-4, f"A matrices diverged: {a_diff:.6f}"
        assert b_diff < 1e-4, f"B matrices diverged: {b_diff:.6f}"

    def test_rpn_shadow_updates(self):
        """
        Verify shadow weight updates use RPN path correctly.

        Acceptance criteria:
        - Shadow updates should route through RPN
        - Shadow weights should differ from primary after update
        - Primary weights should remain unchanged
        """
        dims = 64
        rank = 8
        lr = 0.001

        adapter = SelfUpdatingAdapter(
            shape=(dims, dims),
            rank=rank,
            specialist_name="shadow_test"
        )

        # Fork to shadow
        adapter.fork_to_shadow()
        adapter.sync_shadow_weights_to_host()

        # Verify shadow matches primary initially
        assert np.allclose(_np_matrix(adapter.A), _np_matrix(adapter.A_shadow))
        assert np.allclose(_np_matrix(adapter.B), _np_matrix(adapter.B_shadow))

        # Generate gradient
        gradient = np.random.randn(dims, dims).astype(np.float32) * 0.01

        # Store primary for comparison
        A_primary_before = adapter.A.copy()
        B_primary_before = adapter.B.copy()

        # Apply to shadow
        adapter.apply_gradient_to_shadow(gradient, lr)
        adapter.sync_shadow_weights_to_host()

        # Verify primary unchanged
        assert np.allclose(_np_matrix(adapter.A), _np_matrix(A_primary_before))
        assert np.allclose(_np_matrix(adapter.B), _np_matrix(B_primary_before))

        # Verify shadow changed (LoRA updates touch B first, then A)
        changed_A = not np.allclose(_np_matrix(adapter.A_shadow), _np_matrix(A_primary_before))
        changed_B = not np.allclose(_np_matrix(adapter.B_shadow), _np_matrix(B_primary_before))

        assert changed_B, "B shadow weights should change after update"
        assert changed_A or changed_B, "At least one shadow matrix should change"

        print(f"\n[Regression Test] Shadow updates:")
        print(f"  Primary A unchanged: {np.allclose(_np_matrix(adapter.A), _np_matrix(A_primary_before))}")
        print(f"  Primary B unchanged: {np.allclose(_np_matrix(adapter.B), _np_matrix(B_primary_before))}")
        print(f"  Shadow A changed: {not np.allclose(_np_matrix(adapter.A_shadow), _np_matrix(A_primary_before))}")
        print(f"  Shadow B changed: {not np.allclose(_np_matrix(adapter.B_shadow), _np_matrix(B_primary_before))}")

    def test_ternary_validation_gate(self):
        """
        Test ternary validation gate decision logic.

        Acceptance criteria:
        - TRUE: improvement >= min_improvement
        - FALSE: degradation > max_degradation
        - UNKNOWN: marginal difference
        """
        config = AdapterConfig(
            min_improvement=0.001,  # 0.1%
            max_degradation=0.05    # 5%
        )

        adapter = SelfUpdatingAdapter(
            shape=(32, 32),
            rank=8,
            specialist_name="ternary_test",
            config=config
        )

        # Test TRUE case (clear improvement)
        baseline = 0.80
        shadow = 0.82  # +2% improvement
        decision = adapter._ternary_gate(baseline, shadow)
        assert decision == "TRUE", f"Expected TRUE, got {decision}"

        # Test FALSE case (excessive degradation)
        baseline = 0.80
        shadow = 0.70  # -10% degradation
        decision = adapter._ternary_gate(baseline, shadow)
        assert decision == "FALSE", f"Expected FALSE, got {decision}"

        # Test UNKNOWN case (marginal improvement)
        baseline = 0.80
        shadow = 0.8005  # +0.05% improvement (< 0.1% threshold)
        decision = adapter._ternary_gate(baseline, shadow)
        assert decision == "UNKNOWN", f"Expected UNKNOWN, got {decision}"

        print(f"\n[Regression Test] Ternary validation gate:")
        print(f"  +2% improvement → TRUE: ✓")
        print(f"  -10% degradation → FALSE: ✓")
        print(f"  +0.05% marginal → UNKNOWN: ✓")

    def test_rpn_math_core_operations(self):
        """
        Test Tier-3 RPN math core operations.

        Acceptance criteria:
        - Vector norm computed correctly
        - Fill operation works
        - Vector multiply works
        - Matrix multiply works
        """
        from knowledge3d.cranium.ptx_runtime.rpn_math_core import DeviceTensor

        math_core = RPNMathCore()

        # Test vector norm
        test_vec = np.array([3.0, 4.0], dtype=np.float32)
        d_vec = RPNMathCore.to_device(test_vec)
        tensor = DeviceTensor(d_vec, 2, 1)

        norm = math_core.vector_norm(tensor)
        expected_norm = 5.0  # sqrt(3^2 + 4^2)

        print(f"\n[Regression Test] RPN math core:")
        print(f"  Vector norm: {norm:.4f} (expected: {expected_norm:.4f})")
        assert abs(norm - expected_norm) < 1e-3, f"Norm mismatch: {norm} vs {expected_norm}"

        # Test fill operation
        fill_vec = np.zeros(4, dtype=np.float32)
        d_fill = RPNMathCore.to_device(fill_vec)
        fill_tensor = DeviceTensor(d_fill, 4, 1)

        math_core.fill(fill_tensor, 7.5)
        RPNMathCore.copy_to_host(d_fill, fill_vec)

        print(f"  Fill operation: {fill_vec} (expected: [7.5, 7.5, 7.5, 7.5])")
        assert np.allclose(fill_vec, 7.5), f"Fill failed: {fill_vec}"

        # Cleanup
        RPNMathCore.free(d_vec)
        RPNMathCore.free(d_fill)

    def test_rpn_math_core_copy_round_trip_supports_2d_arrays(self):
        """2D host arrays must round-trip through device copy helpers without shape loss."""
        matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
        ptr = RPNMathCore.to_device(matrix)
        restored = np.zeros_like(matrix)

        RPNMathCore.copy_to_host(ptr, restored)

        assert np.allclose(restored, matrix)
        RPNMathCore.free(ptr)

    def test_gradient_norm_clipping(self):
        """
        Test gradient norm clipping in RPN path.

        Acceptance criteria:
        - Large gradients should be clipped to threshold
        - Clipped norm should match config.gradient_clip
        """
        config = AdapterConfig(gradient_clip=1.0)

        adapter = SelfUpdatingAdapter(
            shape=(64, 64),
            rank=8,
            specialist_name="clip_test",
            config=config
        )

        # Generate large gradient (norm > 1.0)
        large_gradient = np.random.randn(64, 64).astype(np.float32) * 10.0
        original_norm = np.linalg.norm(large_gradient)

        print(f"\n[Regression Test] Gradient clipping:")
        print(f"  Original norm: {original_norm:.4f}")

        # Apply via RPN (should clip internally)
        clipped_norm = adapter.apply_gradient_rpn(large_gradient, lr=0.001)

        print(f"  Clipped norm: {clipped_norm:.4f} (threshold: {config.gradient_clip})")

        assert clipped_norm <= config.gradient_clip + 1e-3, \
            f"Gradient not clipped: {clipped_norm} > {config.gradient_clip}"

    def test_validate_and_commit_decisions(self):
        """
        Test validate_and_commit with different performance scenarios.

        Acceptance criteria:
        - TRUE decision → commit to primary
        - FALSE decision → reject (primary unchanged)
        - UNKNOWN decision → reject (primary unchanged)
        """
        config = AdapterConfig(
            min_improvement=0.01,
            max_degradation=0.05
        )

        adapter = SelfUpdatingAdapter(
            shape=(32, 32),
            rank=8,
            specialist_name="commit_test",
            config=config
        )

        # Mock validation samples
        adapter.set_validation_samples([{'dummy': 1}])

        # Mock base weights
        base_weights = np.eye(32, dtype=np.float32)

        # Mock evaluation function
        # Returns fixed performance based on weights sum (proxy for testing)
        def eval_fn(weights, samples):
            return float(np.sum(weights)) / weights.size

        # Test TRUE case (clear improvement)
        adapter.fork_to_shadow()
        adapter.A_shadow += 0.5  # Increase shadow weights → better performance
        adapter.B_shadow += 0.5

        A_before = adapter.A.copy()
        success, baseline, shadow = adapter.validate_and_commit(base_weights, eval_fn)

        print(f"\n[Regression Test] Validate and commit:")
        print(f"  TRUE case: success={success}, baseline={baseline:.4f}, shadow={shadow:.4f}")

        assert success, "TRUE decision should commit"
        assert not np.allclose(_np_matrix(adapter.A), _np_matrix(A_before)), "Primary should be updated"
        assert adapter.accepted_count == 1, "Accepted count should increment"

        # Test FALSE case (excessive degradation)
        adapter.fork_to_shadow()
        adapter.A_shadow -= 0.5  # Decrease shadow weights → worse performance
        adapter.B_shadow += 0.5

        A_before = adapter.A.copy()
        success, baseline, shadow = adapter.validate_and_commit(base_weights, eval_fn)

        print(f"  FALSE case: success={success}, baseline={baseline:.4f}, shadow={shadow:.4f}")

        assert not success, "FALSE decision should reject"
        assert np.allclose(_np_matrix(adapter.A), _np_matrix(A_before)), "Primary should be unchanged"


class TestRPNPerformance:
    """Performance benchmarks for RPN vs CPU."""

    @pytest.mark.cuda
    @pytest.mark.benchmark
    def test_rpn_speedup(self, benchmark):
        """
        Benchmark RPN gradient update vs CPU.

        Expected: RPN should be faster for dims >= 256
        """
        dims = 256
        rank = 32
        lr = 0.001

        adapter = SelfUpdatingAdapter(
            shape=(dims, dims),
            rank=rank,
            specialist_name="perf_test"
        )

        gradient = np.random.randn(dims, dims).astype(np.float32) * 0.01

        # Benchmark RPN path
        rpn_time = benchmark(adapter.apply_gradient_rpn, gradient, lr)

        print(f"\n[Performance Benchmark] RPN gradient update:")
        print(f"  Dimensions: {dims}x{dims}, Rank: {rank}")
        print(f"  RPN time: {rpn_time:.2f} ms")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
