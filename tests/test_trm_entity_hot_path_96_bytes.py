from __future__ import annotations

import ctypes
from pathlib import Path

from knowledge3d.cranium.bridges.sovereign_bridges import _EntityHotPathStruct as _SovereignEntityHotPathStruct
from knowledge3d.cranium.bridges.trm_step_fused_bridge import _EntityHotPathStruct as _BridgeEntityHotPathStruct


def test_entity_hot_path_is_96_bytes_on_both_bridge_surfaces() -> None:
    assert ctypes.sizeof(_BridgeEntityHotPathStruct) == 96
    assert ctypes.sizeof(_SovereignEntityHotPathStruct) == 96
    assert _BridgeEntityHotPathStruct.gaze_yaw.offset == 64
    assert _BridgeEntityHotPathStruct.attention_entity_id.offset == 76
    assert _BridgeEntityHotPathStruct.motor_output.offset == 80
    assert _BridgeEntityHotPathStruct.current_goal_star.offset == 92

    source = Path("knowledge3d/cranium/kernels/entity_hot_path.h").read_text(encoding="utf-8")
    assert "sizeof(EntityHotPath) == 96" in source
