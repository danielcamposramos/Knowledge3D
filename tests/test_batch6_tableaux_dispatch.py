from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import REASONING_SLOT_TABLEAUX, atlas_words, pack_branch


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_tableaux_dispatch_closes_on_known_clash() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_TABLEAUX,
                "galaxy_atlas": atlas_words(pack_branch(3, 0b0011), 0b0100, 5, 0xFFFFFFFB, halt_after=1),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] != 0
        assert lane0["belief_q15"] == 32768
    finally:
        bridge.cleanup()


def test_tableaux_dispatch_saturates_without_clash() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_TABLEAUX,
                "galaxy_atlas": atlas_words(pack_branch(4, 0b0011), 0b0100, 5, 7, halt_after=1),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] == pack_branch(4, 0b0111)
        assert lane0["belief_q15"] == 16384
    finally:
        bridge.cleanup()

