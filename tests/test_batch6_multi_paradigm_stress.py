from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import (
    REASONING_SLOT_ALPCHAIN,
    REASONING_SLOT_BIDUCE,
    REASONING_SLOT_CBR,
    REASONING_SLOT_CTX_SWITCH,
    REASONING_SLOT_DPLL,
    REASONING_SLOT_EBELIEF,
    REASONING_SLOT_RESOLUTION,
    REASONING_SLOT_RETE,
    REASONING_SLOT_SUBSUME,
    REASONING_SLOT_SUPERPOS,
    REASONING_SLOT_TABLEAUX,
    REASONING_SLOT_UNIFY,
    atlas_words,
    pack_case,
    reference_assign,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


ALL_SLOTS = [
    REASONING_SLOT_CBR,
    REASONING_SLOT_SUPERPOS,
    REASONING_SLOT_BIDUCE,
    REASONING_SLOT_EBELIEF,
    REASONING_SLOT_RETE,
    REASONING_SLOT_TABLEAUX,
    REASONING_SLOT_RESOLUTION,
    REASONING_SLOT_ALPCHAIN,
    REASONING_SLOT_DPLL,
    REASONING_SLOT_CTX_SWITCH,
    REASONING_SLOT_SUBSUME,
    REASONING_SLOT_UNIFY,
]


def test_multi_paradigm_stress_popcount_12_completes_without_hang() -> None:
    mask = 0
    for slot in ALL_SLOTS:
        mask |= 1 << slot
    assert mask.bit_count() == 12

    atlas = atlas_words(
        pack_case(33, 120, 7, 1),
        pack_case(3, 122, 7, 1),
        pack_case(7, 124, 7, 1),
        0x1234ABCD,
        halt_after=1,
        context_id=7,
        ethical_trit=0,
    )

    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        result = bridge.tick(
            {
                "paradigm_mask": mask,
                "galaxy_atlas": atlas,
                "n_cand_frustum": 32,
                "n_floor": 12,
                "n_hard_max": 32,
            },
            timeout_s=5.0,
        )
        assert result["halting_flag"] == bridge.FLAG_COMPLETE
        assert result["n_active"] >= 12

        histogram: dict[int, int] = {}
        for lane_index in range(result["n_active"]):
            assigned = reference_assign(mask, lane_index)
            histogram[assigned] = histogram.get(assigned, 0) + 1
            lane = bridge.read_lane_output(lane_index)
            assert lane["halt_flag"] == 1
            assert 0 <= lane["belief_q15"] <= 32768

        for slot in ALL_SLOTS:
            assert histogram.get(slot, 0) >= 1
    finally:
        bridge.cleanup()

