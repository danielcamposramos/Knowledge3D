"""
Tests for RPN-Powered Golden Ratio (φ) Fractal Calculations

Validates fractal tree constraints using modular RPN kernel.
"""

import numpy as np
import pytest


# Golden ratio constant for validation
PHI = 1.618033988749895


def _require_gpu():
    """Skip test if GPU not available."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_garden_fractal_module_loads():
    """Test that garden fractal module loads successfully."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_golden_angle_rpn

    # Should not raise
    assert compute_golden_angle_rpn is not None


@pytest.mark.cuda
def test_golden_angle_calculation():
    """Test golden angle θ = 2π/φ calculation."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_golden_angle_rpn

    angle = compute_golden_angle_rpn()

    # Expected: 2π/φ ≈ 2.39996 radians ≈ 137.5 degrees
    expected = 2 * np.pi / PHI
    assert np.abs(angle - expected) < 0.001, \
        f"Expected {expected:.5f}, got {angle:.5f}"


@pytest.mark.cuda
def test_max_depth_from_honesty():
    """Test max depth calculation from honesty score."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_max_depth_rpn

    # High honesty → deeper tree
    depth_high = compute_max_depth_rpn(honesty=1.0)

    # Low honesty → shallow tree
    depth_low = compute_max_depth_rpn(honesty=0.3)

    # Validate
    assert isinstance(depth_high, int)
    assert isinstance(depth_low, int)
    assert depth_high > depth_low, \
        f"High honesty depth ({depth_high}) should exceed low ({depth_low})"

    # Expected: φ × 1.0 × 10 ≈ 16
    assert 15 <= depth_high <= 17, f"Expected ~16, got {depth_high}"


@pytest.mark.cuda
def test_thickness_tapering():
    """Test branch thickness tapering with depth."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_thickness_rpn

    base = 1.0

    # Compute thickness at different depths
    t0 = compute_thickness_rpn(base, depth=0)
    t1 = compute_thickness_rpn(base, depth=1)
    t2 = compute_thickness_rpn(base, depth=2)
    t5 = compute_thickness_rpn(base, depth=5)

    # Validate tapering (thickness decreases with depth)
    assert t0 > t1 > t2 > t5, \
        f"Thickness should decrease: {t0:.3f} > {t1:.3f} > {t2:.3f} > {t5:.3f}"

    # Depth 0 should be base thickness
    assert np.abs(t0 - base) < 0.01, f"Expected {base}, got {t0}"

    # Depth 1 should be base/φ
    expected_t1 = base / PHI
    assert np.abs(t1 - expected_t1) < 0.01, \
        f"Expected {expected_t1:.3f}, got {t1:.3f}"


@pytest.mark.cuda
def test_fractal_constraints_batch():
    """Test batch fractal constraint computation."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_fractal_constraints_batch_rpn

    # Batch of honesty scores
    honesty_scores = [0.3, 0.5, 0.7, 0.9, 1.0]

    results = compute_fractal_constraints_batch_rpn(honesty_scores, base_thickness=1.0)

    # Validate structure
    assert 'golden_angle' in results
    assert 'max_depths' in results
    assert 'thickness_curves' in results

    # Golden angle is constant
    expected_angle = 2 * np.pi / PHI
    assert np.abs(results['golden_angle'] - expected_angle) < 0.001

    # Max depths should increase with honesty
    depths = results['max_depths']
    assert len(depths) == 5
    assert all(depths[i] <= depths[i+1] for i in range(4)), \
        f"Depths should increase with honesty: {depths}"

    # Thickness curves should exist for each tree
    curves = results['thickness_curves']
    assert len(curves) == 5
    for i, curve in enumerate(curves):
        assert len(curve) == depths[i] + 1, \
            f"Curve {i} should have depth+1 entries"


@pytest.mark.cuda
def test_thickness_curve_validation():
    """Test thickness curve follows φ taper."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_fractal_constraints_batch_rpn

    honesty_scores = [0.8]
    results = compute_fractal_constraints_batch_rpn(honesty_scores, base_thickness=1.0)

    curve = results['thickness_curves'][0]

    # Validate φ ratio between consecutive depths
    for d in range(len(curve) - 1):
        ratio = curve[d] / curve[d + 1]
        # Ratio should be φ
        assert np.abs(ratio - PHI) < 0.01, \
            f"Thickness ratio at depth {d} should be φ, got {ratio:.3f}"


@pytest.mark.cuda
def test_branching_density():
    """Test branching density follows φ^depth."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_branching_density_rpn

    # Depth 0 → 1 branch (trunk)
    b0 = compute_branching_density_rpn(0)
    assert b0 == 1, f"Depth 0 should have 1 branch, got {b0}"

    # Depth 1 → ~2 branches (φ^1 ≈ 1.6 → 2)
    b1 = compute_branching_density_rpn(1)
    assert b1 == 2, f"Depth 1 should have ~2 branches, got {b1}"

    # Depth 2 → ~3 branches (φ^2 ≈ 2.6 → 3)
    b2 = compute_branching_density_rpn(2)
    assert b2 == 3, f"Depth 2 should have ~3 branches, got {b2}"

    # Depth 5 → ~11 branches (φ^5 ≈ 11.1)
    b5 = compute_branching_density_rpn(5)
    assert 10 <= b5 <= 12, f"Depth 5 should have ~11 branches, got {b5}"


@pytest.mark.cuda
def test_golden_spiral_position():
    """Test golden spiral position calculation."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_golden_spiral_position_rpn

    # At θ=0, should be at (radius_base, 0)
    x0, y0 = compute_golden_spiral_position_rpn(theta=0.0, radius_base=1.0)
    assert np.abs(x0 - 1.0) < 0.01, f"x at θ=0 should be 1.0, got {x0}"
    assert np.abs(y0) < 0.01, f"y at θ=0 should be 0.0, got {y0}"

    # At θ=π/2, should be at (0, r) where r = base × φ^(π/2 / 2π)
    x90, y90 = compute_golden_spiral_position_rpn(theta=np.pi/2, radius_base=1.0)
    expected_r = 1.0 * (PHI ** (0.25))  # π/2 / 2π = 1/4
    assert np.abs(x90) < 0.01, f"x at θ=π/2 should be ~0, got {x90}"
    assert np.abs(y90 - expected_r) < 0.1, \
        f"y at θ=π/2 should be ~{expected_r:.3f}, got {y90:.3f}"


@pytest.mark.cuda
def test_batch_performance():
    """Test batch fractal constraint performance."""
    _require_gpu()

    from knowledge3d.tools.garden_fractal_rpn import compute_fractal_constraints_batch_rpn
    import time

    # Many honesty scores
    np.random.seed(42)
    honesty_scores = np.random.rand(20).tolist()

    # Time batch execution
    start = time.time()
    results = compute_fractal_constraints_batch_rpn(honesty_scores, base_thickness=1.0)
    elapsed = time.time() - start

    # Should complete quickly (target: <0.2s for 20 trees)
    assert elapsed < 1.0, f"Batch too slow: {elapsed:.3f}s for 20 trees"
    assert len(results['max_depths']) == 20

    print(f"\n✓ RPN fractal batch: {len(honesty_scores)} trees in {elapsed*1000:.1f}ms")
    print(f"  Average: {elapsed*1000/len(honesty_scores):.2f}ms per tree")
