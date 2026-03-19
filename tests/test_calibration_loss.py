import pytest

torch = pytest.importorskip("torch")

from knowledge3d.training.math_benchmarks.calibration_loss import (
    binary_calibration_loss,
    expected_calibration_error,
)


def test_binary_calibration_loss_zero():
    confidences = torch.tensor([0.0, 1.0, 0.5])
    correctness = torch.tensor([0.0, 1.0, 0.5])
    loss = binary_calibration_loss(confidences, correctness)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_expected_calibration_error_zero():
    confidences = torch.tensor([0.2, 0.8])
    correctness = torch.tensor([0.0, 1.0])
    ece = expected_calibration_error(confidences, correctness, num_bins=1)
    assert torch.isclose(ece, torch.tensor(0.0))
