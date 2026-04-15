from __future__ import annotations

import os
from pathlib import Path

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import REASONING_SLOT_CBR, atlas_words, pack_case


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_end_to_end_reasoning_smoke_runs_through_persistent_swarm() -> None:
    bridge = NChainSwarmBridge()
    expected = pack_case(7, 122, 7, 1)
    atlas = atlas_words(
        pack_case(33, 120, 7, 1),
        pack_case(3, 40, 7, 1),
        expected,
        halt_after=1,
        context_id=7,
        ethical_trit=0,
    )
    try:
        bridge.launch()
        result = bridge.tick(
            {
                "paradigm_mask": 1 << REASONING_SLOT_CBR,
                "galaxy_atlas": atlas,
                "n_cand_frustum": 8,
                "n_floor": 4,
                "n_hard_max": 16,
            },
            timeout_s=5.0,
        )
        lane0 = bridge.read_lane_output(0)
        assert result["halting_flag"] == bridge.FLAG_COMPLETE
        assert result["halt_epoch"] == 1
        assert lane0["result_handle"] == expected
        assert lane0["belief_q15"] >= 16384
        assert lane0["payload2"] == 1
    finally:
        bridge.cleanup()

    bridge_source = (Path.cwd() / "knowledge3d/cranium/bridges/n_chain_swarm_bridge.py").read_text(encoding="utf-8")
    assert "numpy" not in bridge_source
    assert "cupy" not in bridge_source
    assert "scipy" not in bridge_source
    assert "sympy" not in bridge_source
    assert "\nimport re\n" not in bridge_source
