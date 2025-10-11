"""Test RPNCalculator wrapper with sovereign PTX backend.

The RPNCalculator is a backward-compatible wrapper around ModularRPNEngine.
This test validates that it correctly delegates to our sovereign PTX architecture.
"""
import numpy as np
import pytest

from knowledge3d.cranium.ptx_runtime.rpn_calculator import RPNCalculator


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    calc = RPNCalculator()

    # Addition
    result = calc.evaluate("2 3 +")
    assert abs(result - 5.0) < 1e-5

    # Subtraction
    result = calc.evaluate("10 3 -")
    assert abs(result - 7.0) < 1e-5

    # Multiplication
    result = calc.evaluate("4 5 *")
    assert abs(result - 20.0) < 1e-5

    # Division
    result = calc.evaluate("15 3 /")
    assert abs(result - 5.0) < 1e-5


def test_complex_expression():
    """Test complex RPN expression."""
    calc = RPNCalculator()

    # (2 + 3) * 4 = 20
    result = calc.evaluate("2 3 + 4 *")
    assert abs(result - 20.0) < 1e-5

    # 10 / 2 + 3 * 4 = 5 + 12 = 17
    result = calc.evaluate("10 2 / 3 4 * +")
    assert abs(result - 17.0) < 1e-5


def test_trigonometric_functions():
    """Test trigonometric operations."""
    calc = RPNCalculator()

    # sin(0) = 0
    result = calc.evaluate("0 sin")
    assert abs(result - 0.0) < 1e-5

    # cos(0) = 1
    result = calc.evaluate("0 cos")
    assert abs(result - 1.0) < 1e-5

    # sin(pi/2) ≈ 1
    result = calc.evaluate("3.14159265 2 / sin")
    assert abs(result - 1.0) < 1e-3


def test_vector_operations():
    """Test vector result operations."""
    calc = RPNCalculator()

    # Simple scalar that returns vector
    result = calc.evaluate_vector("5 3 +")
    assert isinstance(result, list)
    assert abs(result[0] - 8.0) < 1e-5


def test_variables():
    """Test variable substitution."""
    calc = RPNCalculator()

    # x + y where x=10, y=20
    result = calc.evaluate("x y +", variables={"x": 10.0, "y": 20.0})
    assert abs(result - 30.0) < 1e-5

    # x * 2 + y where x=5, y=3
    result = calc.evaluate("x 2 * y +", variables={"x": 5.0, "y": 3.0})
    assert abs(result - 13.0) < 1e-5


def test_reset():
    """Test calculator reset."""
    calc = RPNCalculator()

    # Evaluate something
    result = calc.evaluate("10 20 +")
    assert abs(result - 30.0) < 1e-5

    # Reset should work without errors
    calc.reset()

    # Should still work after reset
    result = calc.evaluate("5 5 *")
    assert abs(result - 25.0) < 1e-5


def test_singleton_engine():
    """Test that multiple calculators share the same engine."""
    calc1 = RPNCalculator()
    calc2 = RPNCalculator()

    # Both should share the same engine instance
    assert calc1._engine is calc2._engine

    # Both should work correctly
    result1 = calc1.evaluate("7 3 +")
    result2 = calc2.evaluate("7 3 +")
    assert abs(result1 - result2) < 1e-10


def test_instance_ids():
    """Test that instance IDs work correctly."""
    calc = RPNCalculator()

    # Different instance IDs should work independently
    result1 = calc.evaluate("10 5 +", instance_id=0)
    result2 = calc.evaluate("20 3 +", instance_id=1)

    assert abs(result1 - 15.0) < 1e-5
    assert abs(result2 - 23.0) < 1e-5


def test_stack_operations():
    """Test stack manipulation operations."""
    calc = RPNCalculator()

    # Test dup: 5 dup + = 5 + 5 = 10
    result = calc.evaluate("5 dup +")
    assert abs(result - 10.0) < 1e-5

    # Test dup with multiplication: 7 dup * = 7 * 7 = 49
    result = calc.evaluate("7 dup *")
    assert abs(result - 49.0) < 1e-5


def test_mathematical_functions():
    """Test mathematical functions."""
    calc = RPNCalculator()

    # sqrt(16) = 4
    result = calc.evaluate("16 sqrt")
    assert abs(result - 4.0) < 1e-5

    # neg: -(-5) = 5 (double negation)
    result = calc.evaluate("-5 neg")
    assert abs(result - 5.0) < 1e-5

    # exp(0) = 1
    result = calc.evaluate("0 exp")
    assert abs(result - 1.0) < 1e-5

    # log and exp are inverses: log(exp(2)) ≈ 2
    result = calc.evaluate("2 exp log")
    assert abs(result - 2.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
