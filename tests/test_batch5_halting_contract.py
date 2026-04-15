from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge
from tests._batch5_helpers import REASONING_SLOT_CBR, atlas_words, pack_case


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


@pytest.mark.parametrize("halt_after", [1, 5, 16])
def test_persistent_swarm_uses_reasoning_lane_halt_flag(halt_after: int) -> None:
    bridge = NChainSwarmBridge()
    atlas = atlas_words(
        pack_case(33, 120, 7, 1),
        pack_case(3, 40, 7, 1),
        pack_case(7, 122, 7, 1),
        halt_after=halt_after,
        context_id=7,
        ethical_trit=0,
    )
    try:
        bridge.launch()
        halt_epoch_before = bridge.halt_epoch
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
        assert result["halt_epoch"] == halt_epoch_before + 1
        assert lane0["halt_flag"] == 1
        assert lane0["payload2"] == halt_after
    finally:
        bridge.cleanup()
