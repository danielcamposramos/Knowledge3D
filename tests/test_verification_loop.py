import pytest

pytest.importorskip("torch")

from scripts.verification_loop import VerificationLoop


def test_verification_loop_correct_rule():
    verifier = VerificationLoop()
    labels = verifier.verify_rule_sequence("Find f'(0) where f(x) = sin(x)", ["sin_rule"])
    assert labels == [1]


def test_verification_loop_wrong_rule():
    verifier = VerificationLoop()
    labels = verifier.verify_rule_sequence("Find f'(0) where f(x) = sin(x)", ["cos_rule"])
    assert labels == [0]
