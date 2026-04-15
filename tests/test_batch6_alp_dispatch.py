from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import REASONING_SLOT_ALPCHAIN, atlas_words, pack_horn_rule


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_alp_dispatch_resolves_and_halts_with_assumption_pool() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_ALPCHAIN,
                "galaxy_atlas": atlas_words((1 << 3) | (1 << 0), pack_horn_rule(3, (1 << 1) | (1 << 2)), (1 << 0) | (1 << 1), halt_after=1),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] == (1 << 2)
        assert lane0["belief_q15"] == 32768
    finally:
        bridge.cleanup()


def test_alp_dispatch_blocks_on_integrity_constraint() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_ALPCHAIN,
                "galaxy_atlas": atlas_words((1 << 3) | (1 << 0), pack_horn_rule(3, (1 << 2), (1 << 2)), 0, halt_after=1),
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


def test_alp_dispatch_can_run_for_multiple_internal_ticks_before_halting() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        result = bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_ALPCHAIN,
                "galaxy_atlas": atlas_words((1 << 3) | (1 << 0), pack_horn_rule(3, (1 << 1) | (1 << 2)), (1 << 0) | (1 << 1), halt_after=2),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert result["halting_flag"] == bridge.FLAG_COMPLETE
        assert lane0["halt_flag"] == 1
        assert lane0["payload2"] == 2
        assert lane0["result_handle"] == (1 << 2)
    finally:
        bridge.cleanup()

