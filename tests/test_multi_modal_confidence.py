import numpy as np

from knowledge3d.cranium.actions.action_types import ActionType
from knowledge3d.cranium.actions.multi_modal_confidence_propagation import (
    DEFAULT_ALPHA_RANGES,
    MultiModalConfidencePropagator,
)


def test_alpha_within_bounds():
    propagator = MultiModalConfidencePropagator()
    action_type = ActionType.DIALOGUE
    adjusted, alpha = propagator.propagate_single(
        action_type,
        base_confidence=0.6,
        curiosity_score=0.5,
        input_confidence=0.8,
        context_embedding=np.zeros(4),
    )
    alpha_range = DEFAULT_ALPHA_RANGES[action_type.name]
    assert alpha_range.minimum <= alpha <= alpha_range.maximum
    assert adjusted <= 0.8


def test_alpha_optimizer_updates():
    propagator = MultiModalConfidencePropagator()
    action_type = ActionType.NAV_MOVE
    start_alpha = propagator.calculate_adaptive_alpha(
        action_type, 0.5, np.zeros(4)
    )
    propagator.update_rl(action_type, reward=1.0)
    updated_alpha = propagator.calculate_adaptive_alpha(
        action_type, 0.5, np.zeros(4)
    )
    assert updated_alpha >= start_alpha
