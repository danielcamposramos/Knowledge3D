from __future__ import annotations

import pytest

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.knowledgeverse.navigator_specialist import (
    HALTING_WEIGHT_PRIOR_UNIFORM,
    MEANING_CLASSES,
    NavigatorSpecialist,
)


def test_navigator_specialist_cold_start_is_uniform() -> None:
    swarm = AdaptiveSwarmTRM(SwarmConfig(base_dims=64, min_dims=64))
    class _KV:
        adaptive_swarm = swarm

    navigator = NavigatorSpecialist(knowledgeverse=_KV())
    meaning_dist, halting = navigator.emit([0.0] * 64, [0.0] * len(MEANING_CLASSES))
    assert meaning_dist == pytest.approx([1.0 / len(MEANING_CLASSES)] * len(MEANING_CLASSES))
    assert halting == pytest.approx(list(HALTING_WEIGHT_PRIOR_UNIFORM))
