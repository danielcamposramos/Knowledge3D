"""
Calibration losses for confidence training (Phase 5.1).
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("Calibration loss helpers require torch. Install torch to use training losses.")


def binary_calibration_loss(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between predicted confidence and correctness labels."""
    _require_torch()
    if mask is not None:
        confidences = confidences[mask]
        correctness = correctness[mask]
    if confidences.numel() == 0:
        return torch.tensor(0.0, device=confidences.device)
    return F.mse_loss(confidences, correctness.float())


def expected_calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    *,
    num_bins: int = 10,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expected Calibration Error (ECE) over confidences."""
    _require_torch()
    if mask is not None:
        confidences = confidences[mask]
        correctness = correctness[mask]
    if confidences.numel() == 0:
        return torch.tensor(0.0, device=confidences.device)

    confidences = confidences.view(-1)
    correctness = correctness.view(-1).float()
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    ece = torch.tensor(0.0, device=confidences.device)
    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        if in_bin.sum() == 0:
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_correct = correctness[in_bin].mean()
        weight = in_bin.float().mean()
        ece += weight * torch.abs(avg_confidence - avg_correct)
    return ece


def compute_training_loss(
    rule_logits: torch.Tensor,
    target_rules: torch.Tensor,
    confidence_preds: Optional[torch.Tensor],
    confidence_labels: Optional[torch.Tensor],
    *,
    rule_weight: float = 1.0,
    confidence_weight: float = 0.3,
    pad_id: int = 0,
    use_ece: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined rule + calibration loss."""
    _require_torch()
    rule_loss = F.cross_entropy(
        rule_logits.view(-1, rule_logits.shape[-1]),
        target_rules.view(-1),
        ignore_index=pad_id,
    )

    cal_loss = torch.tensor(0.0, device=rule_logits.device)
    if confidence_preds is not None and confidence_labels is not None:
        preds = confidence_preds.squeeze(-1)
        labels = confidence_labels
        mask = (target_rules != pad_id) & (labels >= 0)
        if use_ece:
            cal_loss = expected_calibration_error(preds, labels, mask=mask)
        else:
            cal_loss = binary_calibration_loss(preds, labels, mask=mask)

    total = rule_weight * rule_loss + confidence_weight * cal_loss
    return total, rule_loss, cal_loss


__all__ = [
    "binary_calibration_loss",
    "expected_calibration_error",
    "compute_training_loss",
]
