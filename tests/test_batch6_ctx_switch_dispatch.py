from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import (
    REASONING_SLOT_CBR,
    REASONING_SLOT_CTX_SWITCH,
    atlas_words,
    pack_case,
    pack_ctx_view,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_ctx_switch_lane_emits_context_view_handle() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_CTX_SWITCH,
                "galaxy_atlas": atlas_words(0, halt_after=1, context_id=42, ethical_trit=0),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert lane0["halt_flag"] == 1
        assert lane0["result_handle"] == pack_ctx_view(42, 1, 0)
    finally:
        bridge.cleanup()


def test_ctx_switch_plus_cbr_honors_context_filtered_retrieval() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": (1 << REASONING_SLOT_CTX_SWITCH) | (1 << REASONING_SLOT_CBR),
                "galaxy_atlas": atlas_words(
                    pack_case(9, 121, 42, 1),
                    pack_case(5, 100, 0, 1),
                    pack_case(7, 122, 42, 1),
                    halt_after=1,
                    context_id=42,
                    ethical_trit=0,
                ),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        lane1 = bridge.read_lane_output(1)
        ctx_lane = lane0 if lane0["result_handle"] == pack_ctx_view(42, 1, 0) else lane1
        cbr_lane = lane1 if ctx_lane is lane0 else lane0
        assert ctx_lane["halt_flag"] == 1
        assert cbr_lane["halt_flag"] == 1
        assert cbr_lane["result_handle"] != 0
        assert ((cbr_lane["result_handle"] >> 14) & 0x3F) == 42
    finally:
        bridge.cleanup()


def test_cbr_blocks_forbidden_ethics_under_context_filter() -> None:
    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        bridge.tick(
            {
                "paradigm_mask": (1 << REASONING_SLOT_CTX_SWITCH) | (1 << REASONING_SLOT_CBR),
                "galaxy_atlas": atlas_words(
                    pack_case(9, 121, 42, 1),
                    pack_case(5, 122, 42, 1),
                    pack_case(7, 123, 42, 1),
                    halt_after=1,
                    context_id=42,
                    ethical_trit=-1,
                ),
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        lane1 = bridge.read_lane_output(1)
        cbr_lane = lane0 if lane0["result_handle"] != pack_ctx_view(42, 1, -1) else lane1
        assert cbr_lane["halt_flag"] == 1
        assert cbr_lane["result_handle"] == 0
    finally:
        bridge.cleanup()

