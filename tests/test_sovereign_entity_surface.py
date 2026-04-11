from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine as BridgeModularRPNEngine
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_BH_BLACKBOARD_READ,
    OP_BH_BLACKBOARD_WRITE,
    OP_BH_BT_TICK,
    OP_BH_PERCEIVE,
    OP_BH_SEEK,
    OP_BH_SLEEP_CHECK,
)
from knowledge3d.cranium.sovereign_entity_bootstrap import build_entity_hot_path_array, build_entity_stars
from knowledge3d.ingestion import ingest_entity_bootstrap
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager


def test_entity_star_is_meaning_centric() -> None:
    stars = build_entity_stars()
    assert all(star.meaning_class in {"entity", "blackboard"} for star in stars)
    trm = next(star for star in stars if star.star_id == "entity:trm:primary")
    assert trm.behavior_rpn is not None
    assert trm.visual_rpn is not None
    assert trm.house_position[1] > 0.0
    assert trm.galaxy_ref == "House"


def test_entity_bootstrap_ingestion() -> None:
    class _FakeManager:
        def __init__(self) -> None:
            self.rows: list[tuple[str, object, str]] = []

        def store_meaning_star(self, galaxy_name: str, star, *, category: str = "meaning_star", metadata=None):
            self.rows.append((galaxy_name, star, category))
            return "inserted"

    manager = _FakeManager()
    count = ingest_entity_bootstrap(manager)
    assert count == 4
    reality_entities = [(galaxy, star, category) for galaxy, star, category in manager.rows if galaxy == "Reality"]
    assert len(reality_entities) == 4
    trm = next(star for _, star, _ in reality_entities if star.star_id == "entity:trm:primary")
    entry = trm.to_galaxy_entry(galaxy_name="Reality")
    assert entry["id"] == "entity:trm:primary"
    assert entry["metadata"]["meaning_star"]["behavior_rpn"] is not None


def test_build_entity_hot_path_array_scans_galaxy_entries(tmp_path: Path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    ingest_entity_bootstrap(manager)
    hot_paths = build_entity_hot_path_array(manager)
    assert len(hot_paths) == 1
    trm = hot_paths[0]
    assert trm["star_id"] == "entity:trm:primary"
    assert trm["physics_body_id"] == 0
    assert trm["perception_flags"] == 0x1


def test_entity_behavior_tokens_are_registered() -> None:
    assert ModularRPNEngine.OPCODES["bh_perceive"] == OP_BH_PERCEIVE
    assert ModularRPNEngine.OPCODES["bh_seek"] == OP_BH_SEEK
    assert ModularRPNEngine.OPCODES["bh_sleep_check"] == OP_BH_SLEEP_CHECK


def test_entity_behavior_source_and_fused_step_slot_are_present() -> None:
    kernel_source = Path("knowledge3d/cranium/kernels/modular_rpn_kernel.cu").read_text(encoding="utf-8")
    assert "case 0x180" in kernel_source
    assert "case 0x181" in kernel_source
    assert "case 0x189" in kernel_source

    fused_source = Path("knowledge3d/cranium/ptx/trm_step_fused.cu").read_text(encoding="utf-8")
    assert "BEHAVIOR_PHASE" in fused_source
    assert "entity_hot_path_ptr" in fused_source
    assert "state_machine_ptr" in fused_source
    assert "ring_buffer_ptr" in fused_source
    assert "tick" in fused_source


def test_bind_entity_soa_and_sleep_smoke() -> None:
    try:
        engine = BridgeModularRPNEngine()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialise ModularRPNEngine: {exc}")

    behavior_addrs = engine.bind_entity_behavior_programs(
        {"entity:trm:primary": b"BH_PERCEIVE 50.0 BH_SLEEP_CHECK BH_BT_TICK"}
    )
    assert behavior_addrs["entity:trm:primary"] > 0
    bind_info = engine.bind_entity_soa(
        [
            {
                "star_id": "entity:trm:primary",
                "star_table_idx": 0,
                "physics_body_id": 0,
                "house_x": 0.0,
                "house_y": 1.75,
                "house_z": 0.0,
                "sleep_state": 0,
                "faction": 0,
                "ai_tier": 0,
                "perception_flags": 0x1,
                "perception_radius": 50.0,
                "last_player_dist": 35.0,
                "awareness": 0.0,
                "blackboard_star_id": 0,
                "meta_rule_addr": 0,
                "cranial_origin": [0.0, 1.6, 0.0],
            }
        ]
    )
    assert bind_info["entity_count"] == 1
    result = engine.execute_single(
        instance_id=0,
        op_codes=[OP_BH_SLEEP_CHECK],
        scalars=[],
        vectors=[],
    )
    assert result == pytest.approx(0.0, abs=1.0e-5)


def test_blackboard_roundtrip_smoke() -> None:
    try:
        engine = BridgeModularRPNEngine()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialise ModularRPNEngine: {exc}")

    engine.bind_entity_soa(
        [
            {
                "star_id": "entity:trm:primary",
                "star_table_idx": 0,
                "physics_body_id": 0,
                "behavior_rpn_addr": 0,
                "house_x": 0.0,
                "house_y": 1.75,
                "house_z": 0.0,
                "sleep_state": 0,
                "faction": 0,
                "ai_tier": 0,
                "perception_flags": 0x1,
                "perception_radius": 30.0,
                "last_player_dist": 999.0,
                "awareness": 0.0,
                "blackboard_star_id": 0,
                "meta_rule_addr": 0,
                "cranial_origin": [0.0, 1.6, 0.0],
            }
        ]
    )
    result = engine.execute_single(
        instance_id=0,
        op_codes=[0x00, OP_BH_BLACKBOARD_WRITE, OP_BH_BLACKBOARD_READ],
        scalars=[42.0],
        vectors=[],
    )
    assert result == pytest.approx(42.0, abs=1.0e-5)


def test_perceive_and_seek_smoke() -> None:
    try:
        engine = BridgeModularRPNEngine()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialise ModularRPNEngine: {exc}")

    engine.bind_entity_soa(
        [
            {
                "star_id": "entity:trm:primary",
                "star_table_idx": 0,
                "physics_body_id": 0,
                "behavior_rpn_addr": 0,
                "house_x": 0.0,
                "house_y": 1.75,
                "house_z": 0.0,
                "sleep_state": 0,
                "faction": 0,
                "ai_tier": 0,
                "perception_flags": 0x1,
                "perception_radius": 50.0,
                "last_player_dist": 999.0,
                "awareness": 0.0,
                "blackboard_star_id": 0,
                "meta_rule_addr": 0,
                "cranial_origin": [0.0, 1.6, 0.0],
            },
            {
                "star_id": "entity:test:target",
                "star_table_idx": 1,
                "physics_body_id": 1,
                "behavior_rpn_addr": 0,
                "house_x": 5.0,
                "house_y": 1.75,
                "house_z": 0.0,
                "sleep_state": 0,
                "faction": 0,
                "ai_tier": 0,
                "perception_flags": 0x1,
                "perception_radius": 50.0,
                "last_player_dist": 999.0,
                "awareness": 0.0,
                "blackboard_star_id": 0,
                "meta_rule_addr": 0,
                "cranial_origin": [0.0, 1.6, 0.0],
            },
        ]
    )
    perceive_count = engine.execute_single(
        instance_id=0,
        op_codes=[0x00, OP_BH_PERCEIVE],
        scalars=[10.0],
        vectors=[],
    )
    assert perceive_count == pytest.approx(1.0, abs=1.0e-5)
    seek_result = engine.execute_single(
        instance_id=0,
        op_codes=[0x00, OP_BH_SEEK],
        scalars=[1.0],
        vectors=[],
    )
    assert seek_result > 0.0


def test_behavior_tick_stub_returns_running_status() -> None:
    try:
        engine = BridgeModularRPNEngine()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialise ModularRPNEngine: {exc}")
    result = engine.execute_single(
        instance_id=0,
        op_codes=[0x00, OP_BH_BT_TICK],
        scalars=[1.0],
        vectors=[],
    )
    assert result == pytest.approx(2.0, abs=1.0e-5)
