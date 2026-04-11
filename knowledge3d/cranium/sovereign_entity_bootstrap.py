"""Foundational entity stars and BEHAVIOR_PHASE hot-path projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


TRM_AVATAR_STAR = MeaningCentricStar(
    star_id="entity:trm:primary",
    meaning_class="entity",
    meaning_rpn="TRM AVATAR GAME_LOOP NINE_CHAIN_SWARM HALTING_GATE",
    domain="House/Avatars",
    taxonomy_refs=["concept_avatar", "concept_ai_entity"],
    visual_rpn=(
        "HANIM_LOA2_SKELETON "
        "k3d_cranial_origin k3d_tablet_grip k3d_thought_emitter "
        "DUAL_TEXTURE_BIND UV_MAP_0 UV_MAP_1"
    ),
    behavior_rpn="BH_PERCEIVE 50.0 BH_SLEEP_CHECK BH_BT_TICK",
    reality_refs=["physics:body:0"],
    grammar_refs=["rule:trm_game_loop"],
    meta_refs=["meta:sleep_consolidation"],
    house_position=(0.0, 1.75, 0.0),
    house_room="House",
    galaxy_ref="House",
    confidence=1,
    polarity=1,
)


def build_faction_blackboard(faction_id: int, faction_name: str, house_pos: tuple[float, float, float]) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=f"blackboard:faction:{faction_id}",
        meaning_class="blackboard",
        meaning_rpn=f"FACTION {faction_name.upper()} BLACKBOARD SHARED_MEMORY",
        domain="House/Entities/Factions",
        taxonomy_refs=["concept_faction", "concept_coordination"],
        behavior_rpn="BH_BLACKBOARD_READ BH_BLACKBOARD_WRITE",
        house_position=house_pos,
        house_room="House",
        galaxy_ref="House",
        confidence=1,
        polarity=0,
    )


FACTION_NEUTRAL_BLACKBOARD = build_faction_blackboard(0, "neutral", (10.0, 0.0, 0.0))
FACTION_ALLY_BLACKBOARD = build_faction_blackboard(1, "ally", (12.0, 0.0, 0.0))
FACTION_RIVAL_BLACKBOARD = build_faction_blackboard(2, "rival", (14.0, 0.0, 0.0))

FOUNDATIONAL_ENTITY_STARS = [
    TRM_AVATAR_STAR,
    FACTION_NEUTRAL_BLACKBOARD,
    FACTION_ALLY_BLACKBOARD,
    FACTION_RIVAL_BLACKBOARD,
]


def build_entity_stars() -> list[MeaningCentricStar]:
    return list(FOUNDATIONAL_ENTITY_STARS)


def _iter_candidate_galaxies(galaxy_manager: Any) -> Iterable[str]:
    names: list[str] = []
    default_names = getattr(getattr(galaxy_manager, "_knowledgeverse", None), "DEFAULT_GALAXIES", None)
    if isinstance(default_names, (list, tuple)):
        names.extend(str(name).strip() for name in default_names if str(name).strip())
    loaded = getattr(galaxy_manager, "_galaxies", None)
    if isinstance(loaded, dict):
        names.extend(str(name).strip() for name in loaded.keys() if str(name).strip())
    iter_storage = getattr(galaxy_manager, "iter_storage_jsonl_paths", None)
    if callable(iter_storage):
        for path in iter_storage():
            if isinstance(path, Path):
                names.append(path.stem)
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        yield name


def build_entity_hot_path_array(galaxy_manager: Any) -> list[dict[str, Any]]:
    hot_paths: list[dict[str, Any]] = []
    blackboard_indices: dict[str, int] = {}
    stars: list[MeaningCentricStar] = []

    for galaxy_name in _iter_candidate_galaxies(galaxy_manager):
        try:
            galaxy = galaxy_manager.get_galaxy(galaxy_name)
        except Exception:
            continue
        for entry in getattr(galaxy, "entries", []):
            try:
                star = MeaningCentricStar.from_galaxy_entry(entry)
            except Exception:
                continue
            if star.meaning_class not in {"entity", "blackboard"}:
                continue
            if star.meaning_class == "blackboard":
                blackboard_indices.setdefault(star.star_id, len(blackboard_indices) + 1)
            stars.append(star)

    entity_stars = [star for star in stars if star.meaning_class == "entity"]
    for idx, star in enumerate(entity_stars):
        physics_body_id = 0
        for ref in star.reality_refs:
            if str(ref).startswith("physics:body:"):
                try:
                    physics_body_id = int(str(ref).split(":")[-1])
                except ValueError:
                    physics_body_id = 0
                break
        if "ally" in star.star_id:
            faction = 1
            blackboard_star_id = blackboard_indices.get("blackboard:faction:1", 0)
        elif "rival" in star.star_id:
            faction = 2
            blackboard_star_id = blackboard_indices.get("blackboard:faction:2", 0)
        else:
            faction = 0
            blackboard_star_id = blackboard_indices.get("blackboard:faction:0", 0)

        hot_paths.append(
            {
                "star_id": star.star_id,
                "star_table_idx": idx,
                "physics_body_id": physics_body_id,
                "behavior_rpn": star.behavior_rpn or "",
                "behavior_rpn_addr": 0,
                "house_x": float(star.house_position[0]),
                "house_y": float(star.house_position[1]),
                "house_z": float(star.house_position[2]),
                "sleep_state": 0,
                "faction": faction,
                "ai_tier": 0,
                "perception_flags": 0x1,
                "perception_radius": 30.0,
                "last_player_dist": 999.0,
                "awareness": 0.0,
                "blackboard_star_id": int(blackboard_star_id),
                "meta_rule_addr": len(star.meta_refs),
                "cranial_origin": [0.0, 1.6, 0.0],
                "gaze_yaw": 0.0,
                "gaze_pitch": 0.0,
                "gaze_fov": 0.7853981633974483,
                "attention_entity_id": 0,
                "motor_output": [0.0, 0.0, 0.0],
                "current_goal_star": 0,
            }
        )
    return hot_paths


__all__ = [
    "TRM_AVATAR_STAR",
    "FACTION_NEUTRAL_BLACKBOARD",
    "FACTION_ALLY_BLACKBOARD",
    "FACTION_RIVAL_BLACKBOARD",
    "build_entity_stars",
    "build_entity_hot_path_array",
]
