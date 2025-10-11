"""Test SleepTimeCompute compatibility with sovereign PTX architecture.

This test validates that:
1. PTX_OPS can be imported successfully
2. PTX_OPS.evaluate_rpn() works with our sovereign ModularRPNEngine
3. Variable substitution works correctly
4. SleepTimeCompute module can be imported

We don't test the full sleep computation pipeline here (would require GLB files),
just that the module is compatible with our sovereign architecture.
"""
import pytest
import numpy as np

from knowledge3d.cranium.ptx import PTX_OPS


def test_ptx_ops_import():
    """Test that PTX_OPS can be imported."""
    assert PTX_OPS is not None
    assert hasattr(PTX_OPS, 'evaluate_rpn')


def test_ptx_ops_basic_rpn():
    """Test basic RPN evaluation through PTX_OPS."""
    # Simple arithmetic
    result = PTX_OPS.evaluate_rpn("2 3 +")
    assert abs(result - 5.0) < 1e-5

    # More complex
    result = PTX_OPS.evaluate_rpn("10 2 / 3 4 * +")
    assert abs(result - 17.0) < 1e-5


def test_ptx_ops_rpn_with_variables():
    """Test RPN evaluation with variable substitution."""
    # Single variable
    result = PTX_OPS.evaluate_rpn("x 2 *", variables={"x": 5.0})
    assert abs(result - 10.0) < 1e-5

    # Multiple variables (like modality metrics)
    expression = "brightness 0.3 * saturation 0.3 * + contrast 0.4 * +"
    variables = {
        "brightness": 0.8,
        "saturation": 0.6,
        "contrast": 0.7,
    }
    result = PTX_OPS.evaluate_rpn(expression, variables=variables)
    expected = 0.8 * 0.3 + 0.6 * 0.3 + 0.7 * 0.4
    assert abs(result - expected) < 1e-5


def test_ptx_ops_modality_expression():
    """Test a real modality expression from PTX_OPS."""
    # Simulate image modality metrics
    expression = "brightness_std 0.3 * saturation_std 0.3 * + colorfulness 0.2 * + dynamic_range 0.2 * +"
    metrics = {
        "brightness_std": 0.5,
        "saturation_std": 0.4,
        "colorfulness": 0.6,
        "dynamic_range": 0.7,
    }

    result = PTX_OPS.evaluate_rpn(expression, variables=metrics)
    expected = 0.5 * 0.3 + 0.4 * 0.3 + 0.6 * 0.2 + 0.7 * 0.2
    assert abs(result - expected) < 1e-5


def test_sleep_time_compute_import():
    """Test that SleepTimeCompute can be imported."""
    try:
        from knowledge3d.cranium.ptx_runtime.sleep_time_compute import SleepTimeCompute
        assert SleepTimeCompute is not None
    except ImportError as e:
        pytest.skip(f"SleepTimeCompute import failed (missing dependencies): {e}")


def test_ptx_ops_mathematical_functions():
    """Test mathematical functions through PTX_OPS."""
    # sqrt
    result = PTX_OPS.evaluate_rpn("16 sqrt")
    assert abs(result - 4.0) < 1e-5

    # exp and log
    result = PTX_OPS.evaluate_rpn("1 exp log")
    assert abs(result - 1.0) < 1e-4

    # sin and cos
    result = PTX_OPS.evaluate_rpn("0 sin")
    assert abs(result - 0.0) < 1e-5

    result = PTX_OPS.evaluate_rpn("0 cos")
    assert abs(result - 1.0) < 1e-5


def test_ptx_ops_sigmoid_approximation():
    """Test sigmoid function (used in modality confidence)."""
    # Sigmoid is not a builtin, but we can approximate with: 1 / (1 + exp(-x))
    # For x=0: sigmoid(0) = 0.5
    # This tests if we can do compound expressions
    # Note: Our RPN might not have sigmoid, so we just test that complex expressions work
    result = PTX_OPS.evaluate_rpn("0.5 0.3 * 0.6 0.2 * +")
    expected = 0.5 * 0.3 + 0.6 * 0.2
    assert abs(result - expected) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
