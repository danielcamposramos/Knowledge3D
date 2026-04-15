from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import (
    REASONING_SLOT_RESOLUTION,
    REASONING_SLOT_SUBSUME,
    REASONING_SLOT_UNIFY,
    atlas_words,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


@pytest.mark.parametrize(
    ("slot", "atlas", "expected_nonzero", "expected_handle", "expected_belief"),
    [
        (REASONING_SLOT_UNIFY, atlas_words(42, 42, halt_after=1), True, None, 32768),
        (REASONING_SLOT_UNIFY, atlas_words(42, 7, halt_after=1), False, 0, 0),
        (REASONING_SLOT_RESOLUTION, atlas_words(5, 0xFFFFFFFB, halt_after=1), True, 0xFFFFFFFF, 32768),
        (REASONING_SLOT_SUBSUME, atlas_words(42, 42, halt_after=1), True, 1, 32768),
    ],
)
def test_resolution_family_dispatch(slot: int, atlas: bytes, expected_nonzero: bool, expected_handle: int | None, expected_belief: int) -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << slot,
                "galaxy_atlas": atlas,
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        if expected_nonzero:
            assert lane0["result_handle"] != 0
        if expected_handle is not None:
            assert lane0["result_handle"] == expected_handle
        assert lane0["belief_q15"] == expected_belief
    finally:
        bridge.cleanup()


def test_subsume_dispatch_rejects_nonmatching_polarity() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_SUBSUME,
                "galaxy_atlas": atlas_words(42, 0xFFFFFFD6, halt_after=1),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] == 0
        assert lane0["belief_q15"] == 0
    finally:
        bridge.cleanup()

