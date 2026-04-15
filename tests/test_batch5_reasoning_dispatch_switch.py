from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import (
    REASONING_SLOT_BIDUCE,
    REASONING_SLOT_CBR,
    REASONING_SLOT_EBELIEF,
    REASONING_SLOT_RETE,
    REASONING_SLOT_SUPERPOS,
    atlas_words,
    pack_alpha,
    pack_case,
    pack_fact,
    pack_opinion,
    pack_rule,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


@pytest.mark.parametrize(
    ("slot", "atlas"),
    [
        (
            REASONING_SLOT_CBR,
            atlas_words(
                pack_case(33, 120, 7, 1),
                pack_case(3, 40, 7, 1),
                pack_case(7, 122, 7, 1),
                halt_after=1,
                context_id=7,
                ethical_trit=0,
            ),
        ),
        (
            REASONING_SLOT_SUPERPOS,
            atlas_words(pack_rule(1, 2), 1, halt_after=1, context_id=0, ethical_trit=0),
        ),
        (
            REASONING_SLOT_BIDUCE,
            atlas_words(0b0011, 0b0111, halt_after=1, context_id=0, ethical_trit=0),
        ),
        (
            REASONING_SLOT_EBELIEF,
            atlas_words(pack_opinion(32, 16, 64), 24, 0, halt_after=1, context_id=0, ethical_trit=0),
        ),
        (
            REASONING_SLOT_RETE,
            atlas_words(pack_fact(0b0011, 7, 2, 1), pack_alpha(0b0001, 0, 2, 1, 1), halt_after=1, context_id=7, ethical_trit=0),
        ),
    ],
)
def test_reasoning_dispatch_switch_hits_each_wired_paradigm(slot: int, atlas: bytes) -> None:
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
        assert lane0["result_handle"] != 0
    finally:
        bridge.cleanup()
