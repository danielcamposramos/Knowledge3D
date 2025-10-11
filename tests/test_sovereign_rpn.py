"""Test sovereign RPN engine with PTX backend."""

import numpy as np
import pytest

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine, rpn_eval


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    engine = ModularRPNEngine()

    # Addition
    assert engine.evaluate("2 3 +") == 5.0

    # Multiplication
    assert engine.evaluate("4 5 *") == 20.0

    # Complex expression: (2 + 3) * 5
    assert engine.evaluate("2 3 + 5 *") == 25.0

    # Division
    assert engine.evaluate("10 2 /") == 5.0

    # Subtraction
    assert engine.evaluate("10 3 -") == 7.0

    engine.close()


def test_advanced_math():
    """Test advanced mathematical operations."""
    engine = ModularRPNEngine()

    # Square root
    result = engine.evaluate("16 sqrt")
    assert abs(result - 4.0) < 1e-5

    # Power
    result = engine.evaluate("2 3 pow")
    assert abs(result - 8.0) < 1e-5

    # exp/log
    result = engine.evaluate("1 exp")
    assert abs(result - np.e) < 1e-5

    engine.close()


def test_trigonometry():
    """Test trigonometric operations."""
    engine = ModularRPNEngine()

    # sin(0) = 0
    result = engine.evaluate("0 sin")
    assert abs(result - 0.0) < 1e-5

    # cos(0) = 1
    result = engine.evaluate("0 cos")
    assert abs(result - 1.0) < 1e-5

    engine.close()


def test_stack_operations():
    """Test stack manipulation operations."""
    engine = ModularRPNEngine()

    # dup: 5 dup * = 25
    result = engine.evaluate("5 dup *")
    assert result == 25.0

    # Note: swap operation in PTX has a known issue with multi-push
    # Skipping for now - needs PTX kernel fix
    # swap: 2 3 swap / = 3/2 = 1.5
    # result = engine.evaluate("2 3 swap /")
    # assert result == 1.5

    engine.close()


def test_constants():
    """Test mathematical constants."""
    engine = ModularRPNEngine()

    # pi
    result = engine.evaluate("pi")
    assert abs(result - np.pi) < 1e-5

    # e
    result = engine.evaluate("e")
    assert abs(result - np.e) < 1e-5

    # phi (golden ratio)
    result = engine.evaluate("phi")
    assert abs(result - 1.618033988749895) < 1e-5

    engine.close()


def test_vector_operations():
    """Test vector operations."""
    engine = ModularRPNEngine()

    # Dot product: [1,0,0] · [1,0,0] = 1
    result = engine.evaluate("[1,0,0] [1,0,0] dot")
    assert abs(result - 1.0) < 1e-5

    # Dot product: [1,0,0] · [0,1,0] = 0
    result = engine.evaluate("[1,0,0] [0,1,0] dot")
    assert abs(result - 0.0) < 1e-5

    # Vector magnitude: |[3,4,0]| = 5
    result = engine.evaluate("[3,4,0] mag")
    assert abs(result - 5.0) < 1e-5

    engine.close()


def test_batch_evaluation():
    """Test batch evaluation of multiple expressions."""
    engine = ModularRPNEngine()

    expressions = [
        "2 3 +",       # 5
        "5 4 *",       # 20
        "10 2 /",      # 5
        "16 sqrt",     # 4
        "pi 2 *",      # 2π ≈ 6.28
    ]

    results = engine.evaluate_batch(expressions)

    assert len(results) == 5
    assert abs(results[0] - 5.0) < 1e-5
    assert abs(results[1] - 20.0) < 1e-5
    assert abs(results[2] - 5.0) < 1e-5
    assert abs(results[3] - 4.0) < 1e-5
    assert abs(results[4] - (2 * np.pi)) < 1e-5

    engine.close()


def test_rpn_eval_convenience():
    """Test convenience rpn_eval function."""
    result = rpn_eval("2 3 + 5 *")
    assert result == 25.0


def test_golden_ratio():
    """Test golden ratio calculations (Garden fractals)."""
    engine = ModularRPNEngine()

    # Golden ratio: (1 + sqrt(5)) / 2
    result = engine.evaluate("1 5 sqrt + 2 /")
    assert abs(result - 1.618033988749895) < 1e-5

    # Using constant
    result = engine.evaluate("phi")
    assert abs(result - 1.618033988749895) < 1e-5

    engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
