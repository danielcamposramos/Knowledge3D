from __future__ import annotations


def test_rpn_validator_accepts_constants_and_extended_ops() -> None:
    from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn, validate_stack_shape

    assert is_valid_rpn("pi 2 *")
    assert validate_stack_shape("pi 2 *").ok

    assert is_valid_rpn("5 factorial")
    assert validate_stack_shape("5 factorial").ok

    assert is_valid_rpn("10 6 gcd")
    assert validate_stack_shape("10 6 gcd").ok

    assert is_valid_rpn("5 2 binomial")
    assert validate_stack_shape("5 2 binomial").ok


def test_rpn_validator_rejects_invalid_stack_shape() -> None:
    from knowledge3d.training.math_benchmarks.rpn_validator import validate_stack_shape

    assert not validate_stack_shape("3 4").ok
    assert not validate_stack_shape("10 2013 3 9 *").ok
    assert not validate_stack_shape("2 *").ok


def test_rpn_validator_accepts_single_value_programs() -> None:
    from knowledge3d.training.math_benchmarks.rpn_validator import validate_stack_shape

    assert validate_stack_shape("3 4 +").ok
    assert validate_stack_shape("3 2 pow").ok
    assert validate_stack_shape("9 sqrt").ok
