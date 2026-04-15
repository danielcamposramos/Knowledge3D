from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import REASONING_SLOT_DPLL, atlas_words, pack_clause, pack_trail


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_dpll_dispatch_halts_with_model_for_trivial_sat() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_DPLL,
                "galaxy_atlas": atlas_words(pack_clause(1 << 0, 0), pack_trail(1 << 0, 0), 0, 0xABCDEF01, halt_after=1),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] == 0xABCDEF01
        assert lane0["belief_q15"] == 32768
    finally:
        bridge.cleanup()


def test_dpll_dispatch_halts_unsat_on_conflict() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_DPLL,
                "galaxy_atlas": atlas_words(pack_clause(0, 1 << 0), pack_trail(1 << 0, 0), 0, 0xABCDEF01, halt_after=1),
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


def test_dpll_dispatch_can_close_via_learnt_clause() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_DPLL,
                "galaxy_atlas": atlas_words(pack_clause(1 << 0, 0), pack_trail(1 << 1, 0), pack_clause(0, 1 << 1), 0x1234ABCD, halt_after=1),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] == 0x1234ABCD
        assert lane0["belief_q15"] == 32768
        assert lane0["payload0"] != 0
    finally:
        bridge.cleanup()

