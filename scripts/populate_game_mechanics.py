from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402


BOOTSTRAP_TAG = "phase_e40_game_mechanics_v2"
DEFAULT_HOUSE_DIR = Path("/K3D/Knowledge3D.local/house")


def _surface_forms(*forms: str) -> dict[str, Any]:
    ordered = [str(form).strip() for form in forms if str(form).strip()]
    if not ordered:
        ordered = ["game mechanic"]
    primary = ordered[0].lower().replace(" ", "_")
    return {
        "en": {
            "word_ref": f"word_{primary}",
            "char_refs": [],
            "surface_text": ordered,
        },
        "pt": {
            "word_ref": f"word_{primary}",
            "char_refs": [],
            "surface_text": ordered,
        },
    }


def _meaning_entry(
    entry_id: str,
    name: str,
    description: str,
    *,
    surface_forms: list[str],
    meaning_rpn: str,
    visual_rpn: str,
    behavior_rpn: str,
    meaning_class: str = "concept",
    properties: dict[str, Any] | None = None,
    taxonomy_refs: list[str] | None = None,
    component_refs: list[str] | None = None,
    visual_refs: list[str] | None = None,
    grammar_refs: list[str] | None = None,
    reality_refs: list[str] | None = None,
    math_refs: list[str] | None = None,
    meta_refs: list[str] | None = None,
    house_room: str = "Workshop",
    house_position: list[float] | None = None,
    confidence: int = 1,
    polarity: int = 1,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    refs = {
        "component_refs": list(component_refs or []),
        "visual_refs": list(visual_refs or []),
        "grammar_refs": list(grammar_refs or []),
        "reality_refs": list(reality_refs or []),
        "math_refs": list(math_refs or []),
        "meta_refs": list(meta_refs or []),
    }
    return {
        "id": entry_id,
        "star_id": entry_id,
        "name": name,
        "galaxy": "game_mechanics",
        "domain": "interactive_game",
        "category": "game_mechanic",
        "layer": 2,
        "meaning_class": meaning_class,
        "content": description,
        "summary": description,
        "description": description,
        "meaning_rpn": meaning_rpn,
        "visual_rpn": visual_rpn,
        "behavior_rpn": behavior_rpn,
        "taxonomy_refs": list(taxonomy_refs or []),
        "surface_forms": _surface_forms(*surface_forms),
        "house_room": house_room,
        "house_position": list(house_position or [0.0, 0.0, 0.0]),
        "confidence": int(confidence),
        "polarity": int(polarity),
        **refs,
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "surface_forms": {
                "en": list(surface_forms),
                "pt": list(surface_forms),
            },
            "properties": dict(properties or {}),
            "meaning_rpn": meaning_rpn,
            "visual_rpn": visual_rpn,
            "behavior_rpn": behavior_rpn,
            "taxonomy_refs": list(taxonomy_refs or []),
            "house_room": house_room,
            "house_position": list(house_position or [0.0, 0.0, 0.0]),
            **refs,
        },
        "tags": list(tags or []),
    }


GAME_MECHANICS: list[dict[str, Any]] = [
    _meaning_entry(
        "spatial_navigation_grid",
        "Spatial Navigation Grid",
        "Movement on a discrete two-dimensional grid via cardinal directions across walkable terrain.",
        surface_forms=["grid movement", "tile navigation", "discrete steps"],
        meaning_rpn="GRID_2D WALKABLE_TERRAIN CARDINAL_DIRECTION STEP_DISCRETE COMPOSE",
        visual_rpn="DRAW_GRID DRAW_WALKABLE_TILES DRAW_ENTITY DRAW_GOAL COMPOSE",
        behavior_rpn="CURRENT_POS RECALL ACTION_DIR RECALL STEP_CARDINAL WALKABLE_CHECK IF_MOVE",
        properties={"movement_model": "cardinal", "space": "2d_grid"},
        taxonomy_refs=["reality_position", "reality_translation"],
        visual_refs=["drawing_grid", "drawing_rect"],
        grammar_refs=["grammar_detect_walkable_terrain", "grammar_route_agent_on_walkable_grid"],
        reality_refs=["reality_walkable_terrain", "reality_grid_path_progress"],
        meta_refs=["meta_prefer_shortest_walkable_route"],
        house_position=[0.0, 0.0, 0.0],
        tags=["game", "navigation", "grid", "movement"],
    ),
    _meaning_entry(
        "switch_actuator",
        "Switch Actuator",
        "Walking over a switch changes the state of a linked object such as a door or bridge.",
        surface_forms=["switch", "trigger", "actuator", "pressure plate"],
        meaning_rpn="WALK_OVER SWITCH_CONTACT TARGET_STATE TOGGLE LINKED_OBJECT APPLY",
        visual_rpn="DRAW_CROSS DRAW_SWITCH_PLATE DRAW_LINK_INDICATOR COMPOSE",
        behavior_rpn="AGENT_POS SWITCH_POS EQUALS { TARGET_STATE TOGGLE } IF",
        properties={"trigger": "walk_over", "effect": "toggle_target_state"},
        component_refs=["spatial_navigation_grid"],
        visual_refs=["drawing_color_white", "drawing_cross"],
        grammar_refs=["grammar_toggle_switch_link", "grammar_infer_rule_from_state_delta"],
        reality_refs=["reality_switch_link", "reality_switch_state_toggle"],
        meta_refs=["meta_route_to_switch_when_locked", "meta_learn_from_visual_transition"],
        house_position=[1.0, 0.0, 0.0],
        tags=["game", "switch", "toggle", "cause_effect"],
    ),
    _meaning_entry(
        "lock_key_pattern_match",
        "Lock Key Pattern Match",
        "A locked passage opens when the current key pattern matches the target door pattern.",
        surface_forms=["key fits lock", "pattern match unlock", "shape key"],
        meaning_rpn="CURRENT_KEY_PATTERN TARGET_DOOR_PATTERN MATCH UNLOCK_GATE IF",
        visual_rpn="DRAW_KEY_BOX DRAW_TARGET_DOOR_PATTERN DRAW_MATCH_INDICATOR COMPOSE",
        behavior_rpn="KEY_PATTERN RECALL DOOR_PATTERN RECALL MATCH_CHECK { DOOR_UNLOCK } IF",
        properties={"condition": "pattern_equals_target", "effect": "passage_opens"},
        component_refs=["switch_actuator"],
        visual_refs=["drawing_shape", "drawing_pattern"],
        grammar_refs=["grammar_unlock_door_on_pattern_match", "grammar_compare_key_pattern_to_door"],
        reality_refs=["reality_key_pattern_state", "reality_door_lock_state"],
        math_refs=["math_equality"],
        meta_refs=["meta_match_key_before_door", "meta_probe_transform_blocks_on_mismatch"],
        house_position=[2.0, 0.0, 0.0],
        tags=["game", "lock", "key", "pattern", "unlock"],
    ),
    _meaning_entry(
        "level_progression",
        "Level Progression",
        "Completing the objective advances the game to the next challenge stage.",
        surface_forms=["next level", "stage clear", "level complete"],
        meaning_rpn="OBJECTIVE_COMPLETE LEVEL_SIGNAL ADVANCE_STAGE EMIT",
        visual_rpn="DRAW_EXIT_PORTAL DRAW_PROGRESS_SIGNAL COMPOSE",
        behavior_rpn="GOAL_REACHED { LEVELS_COMPLETED INCREMENT } IF",
        properties={"trigger": "objective_complete", "effect": "advance_stage"},
        component_refs=["lock_key_pattern_match"],
        grammar_refs=["grammar_complete_level_after_goal_cross"],
        reality_refs=["reality_level_completion_signal", "reality_grid_path_progress"],
        meta_refs=["meta_confirm_unlock_before_exit", "meta_click_to_advance_after_completion"],
        house_position=[3.0, 0.0, 0.0],
        tags=["game", "progression", "level", "objective"],
    ),
    _meaning_entry(
        "movement_recharge_block",
        "Movement Recharge Block",
        "Stepping on a yellow recharge block restores the available movement budget.",
        surface_forms=["recharge", "refuel", "energy pickup", "stamina restore"],
        meaning_rpn="WALK_OVER YELLOW_BLOCK MOVEMENT_BUDGET RESTORE",
        visual_rpn="DRAW_YELLOW_BLOCK DRAW_ENERGY_GLOW COMPOSE",
        behavior_rpn="AGENT_POS RECHARGE_POS EQUALS { MOVE_BUDGET MAX_RESTORE } IF",
        properties={"trigger": "walk_over", "effect": "restore_movement_points", "color_signature": "yellow"},
        component_refs=["spatial_navigation_grid"],
        visual_refs=["drawing_color_yellow"],
        grammar_refs=["grammar_restore_movement_budget"],
        reality_refs=["reality_movement_budget"],
        meta_refs=["meta_seek_recharge_when_budget_low"],
        house_position=[0.0, 1.0, 0.0],
        tags=["game", "resource", "recharge", "movement"],
    ),
    _meaning_entry(
        "color_transform_block",
        "Color Transform Block",
        "Stepping on a color transform block changes the entity color and may require repeated steps to reach the target color.",
        surface_forms=["color changer", "paint block", "dye station"],
        meaning_rpn="WALK_OVER COLOR_BLOCK ENTITY_COLOR TRANSFORM POSSIBLE_CYCLE",
        visual_rpn="DRAW_COLOR_WHEEL DRAW_ENTITY_COLOR_STATE COMPOSE",
        behavior_rpn="AGENT_POS COLOR_BLOCK_POS EQUALS { ENTITY_COLOR NEXT_COLOR } IF",
        properties={"trigger": "walk_over", "effect": "change_color", "cycle": "possible_multi_step"},
        component_refs=["lock_key_pattern_match", "movement_recharge_block"],
        visual_refs=["drawing_color_wheel"],
        grammar_refs=["grammar_apply_color_transform", "grammar_compare_key_pattern_to_door"],
        reality_refs=["reality_color_state"],
        meta_refs=["meta_probe_transform_blocks_on_mismatch"],
        house_position=[1.0, 1.0, 0.0],
        tags=["game", "color", "transform", "state"],
    ),
    _meaning_entry(
        "shape_transform_block",
        "Shape Transform Block",
        "Stepping on a transform block changes the entity shape so it can match a target door or slot.",
        surface_forms=["shape changer", "morph block", "transform station"],
        meaning_rpn="WALK_OVER SHAPE_BLOCK ENTITY_SHAPE TRANSFORM TARGET_MATCH_PREPARE",
        visual_rpn="DRAW_SHAPE_ICON DRAW_ENTITY_SHAPE_STATE COMPOSE",
        behavior_rpn="AGENT_POS SHAPE_BLOCK_POS EQUALS { ENTITY_SHAPE NEXT_SHAPE } IF",
        properties={"trigger": "walk_over", "effect": "change_shape"},
        component_refs=["lock_key_pattern_match", "color_transform_block"],
        visual_refs=["drawing_shape"],
        grammar_refs=["grammar_apply_shape_transform", "grammar_compare_key_pattern_to_door"],
        reality_refs=["reality_shape_state"],
        meta_refs=["meta_probe_transform_blocks_on_mismatch"],
        house_position=[2.0, 1.0, 0.0],
        tags=["game", "shape", "transform", "match"],
    ),
    _meaning_entry(
        "no_instruction_discovery",
        "No Instruction Discovery",
        "Rules are not stated explicitly and must be inferred from observed action-to-state changes.",
        surface_forms=["learn by doing", "trial and error", "implicit rules"],
        meaning_rpn="OBSERVE ACTION STATE_DELTA INFER_HIDDEN_RULE UPDATE_MODEL",
        visual_rpn="DRAW_BEFORE_STATE DRAW_AFTER_STATE DRAW_DIFF_OVERLAY COMPOSE",
        behavior_rpn="ACTION_LOG STATE_LOG PAIRWISE_DIFF RULE_HYPOTHESIS FORM",
        component_refs=["switch_actuator", "level_progression"],
        grammar_refs=["grammar_infer_rule_from_state_delta", "grammar_detect_walkable_terrain"],
        reality_refs=["reality_observation_trace"],
        meta_refs=["meta_learn_from_visual_transition"],
        house_position=[3.0, 1.0, 0.0],
        tags=["game", "discovery", "inference", "observation"],
    ),
    _meaning_entry(
        "visual_state_encoding",
        "Visual State Encoding",
        "Game state is encoded visually through colors, shapes, and spatial layout rather than explicit text instructions.",
        surface_forms=["visual logic", "color means state", "shape means type"],
        meaning_rpn="COLOR SHAPE POSITION STATE_ENCODE VISUAL_CHANNEL",
        visual_rpn="DRAW_COLOR_SHAPE_POSITION_LEGEND COMPOSE",
        behavior_rpn="FRAME_PARSE COLOR_MAP SHAPE_MAP POSITION_MAP STATE_BIND",
        component_refs=["color_transform_block", "shape_transform_block"],
        visual_refs=["drawing_color", "drawing_shape", "drawing_grid"],
        grammar_refs=["grammar_detect_walkable_terrain", "grammar_compare_key_pattern_to_door"],
        reality_refs=["reality_visual_state_encoding"],
        meta_refs=["meta_learn_from_visual_transition"],
        house_position=[0.0, 2.0, 0.0],
        tags=["game", "visual", "state", "encoding"],
    ),
    _meaning_entry(
        "multi_step_state_transform",
        "Multi Step State Transform",
        "Reaching a target state may require repeated or cyclic transformations instead of a single action.",
        surface_forms=["double step", "cyclic transform", "iterative approach"],
        meaning_rpn="STATE_TARGET REPEAT_TRANSFORM UNTIL_MATCH",
        visual_rpn="DRAW_TRANSFORM_CYCLE DRAW_TARGET_STATE COMPOSE",
        behavior_rpn="STATE_MISMATCH { TRANSFORM_STEP APPLY LOOP } IF",
        component_refs=["color_transform_block", "shape_transform_block"],
        grammar_refs=["grammar_apply_color_transform", "grammar_apply_shape_transform"],
        math_refs=["math_modular_arithmetic"],
        meta_refs=["meta_probe_transform_blocks_on_mismatch"],
        house_position=[1.0, 2.0, 0.0],
        tags=["game", "multi_step", "iteration", "transform"],
    ),
]


def _catalog_meaning_entry(
    *,
    entry_id: str,
    name: str,
    description: str,
    knowledge_category: str,
    surface_forms: list[str],
    default_grammar_refs: list[str],
    default_reality_refs: list[str],
    default_meta_refs: list[str],
    index: int,
    component_refs: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    token = entry_id.upper()
    properties = {
        "knowledge_category": knowledge_category,
        "catalog": "phase_e40_2d_game_corpus",
    }
    return _meaning_entry(
        entry_id,
        name,
        description,
        surface_forms=surface_forms,
        meaning_rpn=f"{token} GAME_MEANING STATE_AFFORDANCE BIND",
        visual_rpn=f"DRAW_{token} DRAW_GAME_CONTEXT COMPOSE",
        behavior_rpn=f"{token} GAME_STATE APPLY_INTERACTION",
        properties=properties,
        component_refs=list(component_refs or []),
        grammar_refs=list(default_grammar_refs),
        reality_refs=list(default_reality_refs),
        meta_refs=list(default_meta_refs),
        house_position=[float(index % 8), 4.0 + float(index // 8), 0.0],
        tags=["game", knowledge_category, *(tags or [])],
    )


def _extend_meaning_catalog(
    knowledge_category: str,
    specs: list[dict[str, Any]],
    *,
    grammar_refs: list[str],
    reality_refs: list[str],
    meta_refs: list[str],
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, spec in enumerate(specs):
        rows.append(
            _catalog_meaning_entry(
                entry_id=str(spec["id"]),
                name=str(spec["name"]),
                description=str(spec["description"]),
                knowledge_category=knowledge_category,
                surface_forms=list(spec.get("surface_forms") or [str(spec["name"]).lower()]),
                default_grammar_refs=grammar_refs,
                default_reality_refs=reality_refs,
                default_meta_refs=meta_refs,
                index=start_index + offset,
                component_refs=list(spec.get("component_refs") or []),
                tags=list(spec.get("tags") or []),
            )
        )
    return rows


MOVEMENT_MECHANICS = [
    {"id": "movement_gravity_field", "name": "Gravity Field", "description": "Downward force continuously pulls the avatar and loose objects toward the floor.", "surface_forms": ["gravity", "falling", "downward pull"], "tags": ["movement", "gravity"]},
    {"id": "movement_jump_impulse", "name": "Jump Impulse", "description": "A discrete upward impulse lets the avatar cross gaps or reach higher terrain.", "surface_forms": ["jump", "hop", "leap"], "tags": ["movement", "jump"]},
    {"id": "movement_double_jump", "name": "Double Jump", "description": "A second jump can be triggered while airborne to extend vertical reach.", "surface_forms": ["double jump", "air jump"], "tags": ["movement", "jump"]},
    {"id": "movement_wall_jump", "name": "Wall Jump", "description": "Jumping from wall contact redirects momentum upward and away from the wall.", "surface_forms": ["wall jump", "kick off wall"], "tags": ["movement", "wall"]},
    {"id": "movement_slide", "name": "Slide", "description": "Sliding lowers the avatar profile and preserves horizontal momentum under obstacles.", "surface_forms": ["slide", "duck slide"], "tags": ["movement", "momentum"]},
    {"id": "movement_dash_burst", "name": "Dash Burst", "description": "A short high-speed burst covers distance quickly or crosses dangerous tiles.", "surface_forms": ["dash", "burst", "quick step"], "tags": ["movement", "speed"]},
    {"id": "movement_ladder_climb", "name": "Ladder Climb", "description": "Vertical ladders provide constrained upward and downward traversal.", "surface_forms": ["ladder", "climb"], "tags": ["movement", "vertical"]},
    {"id": "movement_rope_swing", "name": "Rope Swing", "description": "A suspended rope or vine converts momentum into an arc across empty space.", "surface_forms": ["rope swing", "vine swing"], "tags": ["movement", "arc"]},
    {"id": "movement_teleporter_pair", "name": "Teleporter Pair", "description": "Entering one portal instantly relocates the avatar to its linked exit.", "surface_forms": ["teleporter", "portal", "warp"], "tags": ["movement", "teleport"]},
    {"id": "movement_conveyor_belt", "name": "Conveyor Belt", "description": "Conveyor surfaces apply automatic horizontal motion while the avatar stands on them.", "surface_forms": ["conveyor", "moving floor"], "tags": ["movement", "forced_motion"]},
    {"id": "movement_moving_platform", "name": "Moving Platform", "description": "A platform carries the avatar along a predefined route through the level.", "surface_forms": ["moving platform", "ride platform"], "tags": ["movement", "platform"]},
    {"id": "movement_swimming", "name": "Swimming", "description": "Water changes movement dynamics, slowing descent and enabling directional strokes.", "surface_forms": ["swimming", "water movement"], "tags": ["movement", "water"]},
    {"id": "movement_buoyancy_surface", "name": "Buoyancy Surface", "description": "A fluid surface pushes the avatar upward unless actively submerged.", "surface_forms": ["buoyancy", "float"], "tags": ["movement", "water"]},
    {"id": "movement_slippery_ice", "name": "Slippery Ice", "description": "Low-friction ground causes extended sliding and delayed stopping.", "surface_forms": ["ice", "slippery floor"], "tags": ["movement", "friction"]},
    {"id": "movement_crouch_tunnel", "name": "Crouch Tunnel", "description": "A low tunnel can only be traversed while the avatar is crouched or sliding.", "surface_forms": ["crouch tunnel", "low passage"], "tags": ["movement", "clearance"]},
    {"id": "movement_fall_damage", "name": "Fall Damage", "description": "Excessive drop height penalizes or resets the avatar on impact.", "surface_forms": ["fall damage", "hard landing"], "tags": ["movement", "risk"]},
]

OBJECT_MECHANICS = [
    {"id": "object_door_barrier", "name": "Door Barrier", "description": "A closed door blocks route progress until its unlock condition is satisfied.", "surface_forms": ["door", "gate"], "tags": ["object", "barrier"]},
    {"id": "object_key_pickup", "name": "Key Pickup", "description": "A collectible key grants permission to open or trigger a matching lock.", "surface_forms": ["key", "pickup"], "tags": ["object", "collectible"]},
    {"id": "object_pushable_crate", "name": "Pushable Crate", "description": "A movable crate can be pushed to bridge gaps, press switches, or reshape paths.", "surface_forms": ["crate", "box", "push block"], "tags": ["object", "pushable"]},
    {"id": "object_breakable_block", "name": "Breakable Block", "description": "A fragile block disappears after impact or after a specific interaction rule.", "surface_forms": ["breakable block", "fragile wall"], "tags": ["object", "breakable"]},
    {"id": "object_spring_launcher", "name": "Spring Launcher", "description": "A spring imparts a strong upward or directional launch on contact.", "surface_forms": ["spring", "launcher", "bounce pad"], "tags": ["object", "launch"]},
    {"id": "object_hazard_spikes", "name": "Hazard Spikes", "description": "Spike hazards punish contact and must be avoided or bypassed safely.", "surface_forms": ["spikes", "hazard"], "tags": ["object", "hazard"]},
    {"id": "object_hazard_lava", "name": "Hazard Lava", "description": "A lava pool destroys or resets the avatar unless protected.", "surface_forms": ["lava", "fire pit"], "tags": ["object", "hazard"]},
    {"id": "object_enemy_patrol", "name": "Enemy Patrol", "description": "A moving enemy occupies space, creates danger zones, and may enforce timing windows.", "surface_forms": ["enemy", "patrol"], "tags": ["object", "hazard"]},
    {"id": "object_checkpoint_beacon", "name": "Checkpoint Beacon", "description": "A checkpoint stores respawn progress after the avatar reaches it.", "surface_forms": ["checkpoint", "save point"], "tags": ["object", "progress"]},
    {"id": "object_powerup_pickup", "name": "Power-Up Pickup", "description": "A collectible power-up grants temporary or persistent movement or interaction abilities.", "surface_forms": ["power-up", "upgrade"], "tags": ["object", "powerup"]},
    {"id": "object_toggle_bridge", "name": "Toggle Bridge", "description": "A bridge appears or disappears according to linked switch state.", "surface_forms": ["bridge", "toggle bridge"], "tags": ["object", "toggle"]},
    {"id": "object_one_way_gate", "name": "One-Way Gate", "description": "A gate permits travel in one direction while blocking reverse traversal.", "surface_forms": ["one-way gate", "one-way door"], "tags": ["object", "constraint"]},
    {"id": "object_pressure_plate", "name": "Pressure Plate", "description": "A floor plate activates when weight is applied by the avatar or an object.", "surface_forms": ["pressure plate", "floor button"], "tags": ["object", "switch"]},
    {"id": "object_color_gate", "name": "Color Gate", "description": "A gate opens only when the avatar or carried key matches a required color state.", "surface_forms": ["color gate", "color lock"], "tags": ["object", "color"]},
    {"id": "object_shape_gate", "name": "Shape Gate", "description": "A gate opens only when the avatar or carried key matches a required shape state.", "surface_forms": ["shape gate", "shape lock"], "tags": ["object", "shape"]},
    {"id": "object_shrink_grow_pickup", "name": "Shrink Grow Pickup", "description": "A pickup changes avatar size, unlocking narrow passages or weight interactions.", "surface_forms": ["size pickup", "shrink", "grow"], "tags": ["object", "transform"]},
    {"id": "object_portal_exit", "name": "Portal Exit", "description": "A portal exit defines the destination location for a linked teleporter.", "surface_forms": ["portal exit", "teleport exit"], "tags": ["object", "teleport"]},
    {"id": "object_locked_chest", "name": "Locked Chest", "description": "A locked chest contains a reward or required item and opens only under a condition.", "surface_forms": ["locked chest", "treasure box"], "tags": ["object", "lock"]},
    {"id": "object_destructible_wall", "name": "Destructible Wall", "description": "A wall can be removed by force, bombs, or another state-dependent trigger.", "surface_forms": ["destructible wall", "cracked wall"], "tags": ["object", "breakable"]},
    {"id": "object_collectible_token", "name": "Collectible Token", "description": "A token contributes to score, unlocks progression, or gates the next room.", "surface_forms": ["token", "coin", "collectible"], "tags": ["object", "collectible"]},
    {"id": "object_timer_orb", "name": "Timer Orb", "description": "A timed object grants or removes time pressure within the level.", "surface_forms": ["timer orb", "time bonus"], "tags": ["object", "time"]},
    {"id": "object_wind_fan", "name": "Wind Fan", "description": "A fan applies directional force that changes jump arcs and movement planning.", "surface_forms": ["fan", "wind"], "tags": ["object", "force"]},
]

PUZZLE_MECHANICS = [
    {"id": "puzzle_lock_key_dependency", "name": "Lock Key Dependency", "description": "A goal remains inaccessible until a matching key state is acquired.", "surface_forms": ["lock and key", "dependency gate"], "tags": ["puzzle", "dependency"]},
    {"id": "puzzle_sokoban_alignment", "name": "Sokoban Alignment", "description": "Pushable objects must be arranged into specific target positions to proceed.", "surface_forms": ["sokoban", "crate puzzle"], "tags": ["puzzle", "crate"]},
    {"id": "puzzle_sequence_trigger_order", "name": "Sequence Trigger Order", "description": "Multiple switches must be activated in the correct order to unlock progress.", "surface_forms": ["sequence puzzle", "ordered switches"], "tags": ["puzzle", "sequence"]},
    {"id": "puzzle_state_toggle_dependency", "name": "State Toggle Dependency", "description": "Switching one object changes the accessibility of another object elsewhere.", "surface_forms": ["toggle puzzle", "linked states"], "tags": ["puzzle", "state"]},
    {"id": "puzzle_pattern_matching", "name": "Pattern Matching", "description": "A visible pattern must be recreated or matched through movement and transformations.", "surface_forms": ["pattern match", "symbol match"], "tags": ["puzzle", "pattern"]},
    {"id": "puzzle_timing_window", "name": "Timing Window", "description": "Actions must occur within a temporary window before the level closes again.", "surface_forms": ["timing puzzle", "time window"], "tags": ["puzzle", "timing"]},
    {"id": "puzzle_weight_balance", "name": "Weight Balance", "description": "Object placement or avatar weight must balance a mechanism to continue.", "surface_forms": ["weight puzzle", "balance"], "tags": ["puzzle", "physics"]},
    {"id": "puzzle_mirror_beam_alignment", "name": "Mirror Beam Alignment", "description": "Directional elements must be aligned so a beam reaches its target.", "surface_forms": ["mirror puzzle", "beam alignment"], "tags": ["puzzle", "alignment"]},
    {"id": "puzzle_color_cycle_match", "name": "Color Cycle Match", "description": "Repeated state changes are required until the correct color is reached.", "surface_forms": ["color cycle", "cycle match"], "tags": ["puzzle", "color"]},
    {"id": "puzzle_shape_fit_match", "name": "Shape Fit Match", "description": "An object or avatar must be transformed to fit a target outline or lock.", "surface_forms": ["shape fit", "shape match"], "tags": ["puzzle", "shape"]},
    {"id": "puzzle_route_optimization", "name": "Route Optimization", "description": "A limited resource requires choosing the most efficient path through the room.", "surface_forms": ["route optimization", "shortest route"], "tags": ["puzzle", "pathfinding"]},
    {"id": "puzzle_resource_budget", "name": "Resource Budget", "description": "A finite resource such as moves, time, or energy constrains the solution plan.", "surface_forms": ["resource budget", "limited moves"], "tags": ["puzzle", "resource"]},
    {"id": "puzzle_hidden_rule_discovery", "name": "Hidden Rule Discovery", "description": "The solution depends on inferring an unstated rule from visual feedback.", "surface_forms": ["hidden rule", "discovery puzzle"], "tags": ["puzzle", "discovery"]},
    {"id": "puzzle_reversible_path", "name": "Reversible Path", "description": "The player must revisit earlier rooms once new abilities or states are unlocked.", "surface_forms": ["reversible path", "backtrack puzzle"], "tags": ["puzzle", "backtracking"]},
    {"id": "puzzle_trap_avoidance", "name": "Trap Avoidance", "description": "A tempting route contains a punishment state that must be recognized and avoided.", "surface_forms": ["trap puzzle", "avoidance"], "tags": ["puzzle", "hazard"]},
    {"id": "puzzle_escort_object", "name": "Escort Object Puzzle", "description": "A movable object must be preserved or guided safely to a goal region.", "surface_forms": ["escort object", "protect crate"], "tags": ["puzzle", "escort"]},
    {"id": "puzzle_multi_room_dependency", "name": "Multi-Room Dependency", "description": "An action in one room changes the solvability of another room later in the level.", "surface_forms": ["multi-room dependency", "cross-room puzzle"], "tags": ["puzzle", "dependency"]},
    {"id": "puzzle_layered_switch_dependency", "name": "Layered Switch Dependency", "description": "Several switch-controlled systems stack and must be reasoned about together.", "surface_forms": ["layered switches", "compound toggle"], "tags": ["puzzle", "state"]},
]

CONTROL_MEANINGS = [
    {"id": "input_action_move_up", "name": "Input Meaning Move Up", "description": "ACTION1 means move the controlled entity upward in the current game space.", "surface_forms": ["action1", "move up", "up input"], "tags": ["input", "action1"]},
    {"id": "input_action_move_down", "name": "Input Meaning Move Down", "description": "ACTION2 means move the controlled entity downward in the current game space.", "surface_forms": ["action2", "move down", "down input"], "tags": ["input", "action2"]},
    {"id": "input_action_move_left", "name": "Input Meaning Move Left", "description": "ACTION3 means move the controlled entity leftward in the current game space.", "surface_forms": ["action3", "move left", "left input"], "tags": ["input", "action3"]},
    {"id": "input_action_move_right", "name": "Input Meaning Move Right", "description": "ACTION4 means move the controlled entity rightward in the current game space.", "surface_forms": ["action4", "move right", "right input"], "tags": ["input", "action4"]},
    {"id": "input_action_perform", "name": "Input Meaning Perform", "description": "ACTION5 triggers the primary contextual interaction such as start, confirm, or use.", "surface_forms": ["action5", "perform", "confirm"], "tags": ["input", "action5"]},
    {"id": "input_action_click", "name": "Input Meaning Click", "description": "ACTION6 performs a coordinate-specific click or tap on the game surface.", "surface_forms": ["action6", "click", "tap"], "tags": ["input", "action6"]},
    {"id": "input_action_undo", "name": "Input Meaning Undo", "description": "ACTION7 reverts the last reversible action or returns to a prior safe state.", "surface_forms": ["action7", "undo", "rewind"], "tags": ["input", "action7"]},
]

VISUAL_MECHANICS = [
    {"id": "visual_color_encodes_type", "name": "Color Encodes Type", "description": "Color differentiates object classes such as hazards, keys, doors, and terrain.", "surface_forms": ["color means type", "type by color"], "tags": ["visual", "color"]},
    {"id": "visual_color_encodes_affordance", "name": "Color Encodes Affordance", "description": "Color implies what interaction an object supports, such as pushable or collectible.", "surface_forms": ["color means affordance", "color cue"], "tags": ["visual", "color"]},
    {"id": "visual_flashing_temporary", "name": "Flashing Means Temporary", "description": "Flashing animation indicates a temporary or time-limited game state.", "surface_forms": ["flashing temporary", "blink means timed"], "tags": ["visual", "timing"]},
    {"id": "visual_flashing_danger", "name": "Flashing Means Danger", "description": "Rapid flashing highlights imminent danger or an unsafe interaction window.", "surface_forms": ["flashing danger", "blink hazard"], "tags": ["visual", "hazard"]},
    {"id": "visual_size_importance", "name": "Size Means Importance", "description": "Larger rendered objects usually indicate goals, bosses, or primary affordances.", "surface_forms": ["size means importance", "big means important"], "tags": ["visual", "salience"]},
    {"id": "visual_outline_interactable", "name": "Outline Means Interactable", "description": "An outline or border indicates an object that can be clicked, pushed, or used.", "surface_forms": ["outline interactable", "border means use"], "tags": ["visual", "interaction"]},
    {"id": "visual_icon_goal_marker", "name": "Icon Goal Marker", "description": "An icon or emblem marks the target exit, switch, or mission objective.", "surface_forms": ["goal icon", "objective marker"], "tags": ["visual", "goal"]},
    {"id": "visual_animation_active_state", "name": "Animation Active State", "description": "Motion or animation indicates that a system is active and likely changing over time.", "surface_forms": ["active animation", "moving means active"], "tags": ["visual", "state"]},
    {"id": "visual_brightness_charge", "name": "Brightness Means Charge", "description": "Brightness reflects energy, readiness, or power-up state.", "surface_forms": ["brightness charge", "glow means ready"], "tags": ["visual", "energy"]},
    {"id": "visual_shape_encodes_lock_class", "name": "Shape Encodes Lock Class", "description": "Shape class differentiates which doors, keys, or sockets correspond to one another.", "surface_forms": ["shape means class", "shape lock"], "tags": ["visual", "shape"]},
    {"id": "visual_pattern_encodes_state", "name": "Pattern Encodes State", "description": "Repeated visual motifs encode puzzle state, lock state, or object identity.", "surface_forms": ["pattern means state", "visual motif"], "tags": ["visual", "pattern"]},
    {"id": "visual_position_encodes_priority", "name": "Position Encodes Priority", "description": "Objects placed centrally, elevated, or isolated often deserve immediate attention.", "surface_forms": ["position means priority", "center is important"], "tags": ["visual", "layout"]},
]

LEVEL_DESIGN_MECHANICS = [
    {"id": "level_design_linear_progression", "name": "Linear Progression", "description": "The level advances through a mostly one-way route from start to finish.", "surface_forms": ["linear level", "straight progression"], "tags": ["level_design", "layout"]},
    {"id": "level_design_hub_spoke", "name": "Hub Spoke Layout", "description": "A central hub connects several branches that each unlock later progress.", "surface_forms": ["hub and spoke", "central hub"], "tags": ["level_design", "layout"]},
    {"id": "level_design_backtracking", "name": "Backtracking Route", "description": "Progress requires revisiting earlier areas after obtaining new state or abilities.", "surface_forms": ["backtracking", "return path"], "tags": ["level_design", "layout"]},
    {"id": "level_design_rising_difficulty", "name": "Rising Difficulty Curve", "description": "Challenge increases gradually as the player masters previously introduced mechanics.", "surface_forms": ["rising difficulty", "difficulty curve"], "tags": ["level_design", "progression"]},
    {"id": "level_design_tutorial_room", "name": "Tutorial Room", "description": "An early safe room introduces one mechanic before combining it with others.", "surface_forms": ["tutorial room", "safe introduction"], "tags": ["level_design", "teaching"]},
    {"id": "level_design_gated_shortcut", "name": "Gated Shortcut", "description": "A shortcut exists but only becomes usable after solving another part of the level.", "surface_forms": ["shortcut", "gated shortcut"], "tags": ["level_design", "layout"]},
    {"id": "level_design_secret_branch", "name": "Secret Branch", "description": "A hidden optional branch rewards observation or experimentation.", "surface_forms": ["secret branch", "hidden path"], "tags": ["level_design", "secret"]},
    {"id": "level_design_checkpoint_spacing", "name": "Checkpoint Spacing", "description": "Checkpoint placement controls retry friction and how much progress a mistake costs.", "surface_forms": ["checkpoint spacing", "save interval"], "tags": ["level_design", "progression"]},
    {"id": "level_design_dead_end_hint", "name": "Dead End With Hint", "description": "A dead end teaches by showing a blocked affordance or hidden dependency.", "surface_forms": ["dead end hint", "blocked clue"], "tags": ["level_design", "teaching"]},
    {"id": "level_design_multi_stage_gate", "name": "Multi Stage Gate", "description": "A large gate requires several prerequisite completions before the route opens.", "surface_forms": ["multi-stage gate", "grand lock"], "tags": ["level_design", "progression"]},
]

STATE_MACHINE_MECHANICS = [
    {"id": "state_machine_title_screen", "name": "Title Screen", "description": "The opening presentation state before gameplay begins.", "surface_forms": ["title screen", "opening screen"], "tags": ["state_machine", "ui"]},
    {"id": "state_machine_start_prompt", "name": "Start Prompt", "description": "A prompt instructs the player to begin play by pressing or clicking start.", "surface_forms": ["start prompt", "press start"], "tags": ["state_machine", "ui"]},
    {"id": "state_machine_gameplay_loop", "name": "Gameplay Loop", "description": "The active state where movement, interactions, and puzzle solving occur.", "surface_forms": ["gameplay", "active play"], "tags": ["state_machine", "loop"]},
    {"id": "state_machine_pause", "name": "Pause State", "description": "A temporary suspended state where inputs no longer affect world simulation.", "surface_forms": ["pause", "paused"], "tags": ["state_machine", "ui"]},
    {"id": "state_machine_failure_reset", "name": "Failure Reset", "description": "A failed attempt transitions the game to a reset or retry state.", "surface_forms": ["failure reset", "retry"], "tags": ["state_machine", "failure"]},
    {"id": "state_machine_completion_screen", "name": "Completion Screen", "description": "A solved level transitions to a completion acknowledgement state.", "surface_forms": ["completion screen", "level clear"], "tags": ["state_machine", "success"]},
    {"id": "state_machine_next_level_transition", "name": "Next Level Transition", "description": "After completion, the game waits for confirmation then loads the next stage.", "surface_forms": ["next level", "continue"], "tags": ["state_machine", "transition"]},
    {"id": "state_machine_hub_overworld", "name": "Hub Overworld State", "description": "A navigation hub lets the player choose among multiple puzzle branches.", "surface_forms": ["hub state", "overworld"], "tags": ["state_machine", "hub"]},
    {"id": "state_machine_inventory_overlay", "name": "Inventory Overlay", "description": "A transient overlay shows keys, forms, or collected state without ending the level.", "surface_forms": ["inventory overlay", "status overlay"], "tags": ["state_machine", "ui"]},
    {"id": "state_machine_cutscene_gate", "name": "Cutscene Gate State", "description": "A non-interactive transition communicates story or puzzle consequences.", "surface_forms": ["cutscene", "transition scene"], "tags": ["state_machine", "transition"]},
    {"id": "state_machine_respawn", "name": "Respawn State", "description": "After death or failure, the avatar returns to a prior checkpoint or start.", "surface_forms": ["respawn", "restart"], "tags": ["state_machine", "failure"]},
    {"id": "state_machine_save_resume", "name": "Save Resume State", "description": "The game can leave and later restore the last persisted progress state.", "surface_forms": ["save and resume", "resume state"], "tags": ["state_machine", "persistence"]},
]


GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "movement",
        MOVEMENT_MECHANICS,
        grammar_refs=["grammar_apply_gravity", "grammar_route_agent_on_walkable_grid"],
        reality_refs=["reality_gravity_field", "reality_walkable_terrain"],
        meta_refs=["meta_jump_over_gap", "meta_prefer_shortest_walkable_route"],
        start_index=len(GAME_MECHANICS),
    )
)
GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "objects",
        OBJECT_MECHANICS,
        grammar_refs=["grammar_push_object_until_blocked", "grammar_break_block_on_force"],
        reality_refs=["reality_pushable_mass", "reality_hazard_damage"],
        meta_refs=["meta_push_crate_to_enable_path", "meta_avoid_hazard_without_protection"],
        start_index=len(GAME_MECHANICS),
    )
)
GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "puzzles",
        PUZZLE_MECHANICS,
        grammar_refs=["grammar_validate_sequence_trigger", "grammar_compare_key_pattern_to_door"],
        reality_refs=["reality_sequence_trigger_state", "reality_pattern_gate_state"],
        meta_refs=["meta_backtrack_when_dependency_unsatisfied", "meta_probe_transform_blocks_on_mismatch"],
        start_index=len(GAME_MECHANICS),
    )
)
GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "controls",
        CONTROL_MEANINGS,
        grammar_refs=["grammar_transition_game_state", "grammar_route_agent_on_walkable_grid"],
        reality_refs=["reality_ui_game_state", "reality_grid_path_progress"],
        meta_refs=["meta_start_from_title_state"],
        start_index=len(GAME_MECHANICS),
    )
)
GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "visual_encoding",
        VISUAL_MECHANICS,
        grammar_refs=["grammar_decode_visual_signal", "grammar_infer_rule_from_state_delta"],
        reality_refs=["reality_visual_signal", "reality_visual_state_encoding"],
        meta_refs=["meta_read_visual_affordance_before_action", "meta_learn_from_visual_transition"],
        start_index=len(GAME_MECHANICS),
    )
)
GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "level_design",
        LEVEL_DESIGN_MECHANICS,
        grammar_refs=["grammar_follow_backtracking_route", "grammar_escalate_difficulty_curve"],
        reality_refs=["reality_level_topology", "reality_grid_path_progress"],
        meta_refs=["meta_backtrack_when_dependency_unsatisfied"],
        start_index=len(GAME_MECHANICS),
    )
)
GAME_MECHANICS.extend(
    _extend_meaning_catalog(
        "state_machine",
        STATE_MACHINE_MECHANICS,
        grammar_refs=["grammar_transition_game_state"],
        reality_refs=["reality_ui_game_state"],
        meta_refs=["meta_start_from_title_state", "meta_click_to_advance_after_completion"],
        start_index=len(GAME_MECHANICS),
    )
)


def _reality_entry(
    entry_id: str,
    name: str,
    description: str,
    *,
    behavior_rpn: str,
    meaning_rpn: str,
    visual_rpn: str,
    component_refs: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "galaxy": "Reality",
        "domain": "interactive_game",
        "category": "game_state",
        "layer": 2,
        "content": description,
        "summary": description,
        "description": description,
        "meaning_rpn": meaning_rpn,
        "visual_rpn": visual_rpn,
        "behavior_rpn": behavior_rpn,
        "component_refs": list(component_refs or []),
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "meaning_rpn": meaning_rpn,
            "visual_rpn": visual_rpn,
            "behavior_rpn": behavior_rpn,
        },
        "tags": list(tags or []),
    }


GAME_REALITY_ENTRIES: list[dict[str, Any]] = [
    _reality_entry(
        "reality_walkable_terrain",
        "Walkable Terrain",
        "Terrain cells that can be traversed by the entity during pathfinding.",
        meaning_rpn="CELL WALKABLE TRUE TERRAIN_CLASS",
        visual_rpn="DRAW_GREY_WALKABLE_TILE",
        behavior_rpn="NEXT_CELL WALKABLE? PERMIT_MOVE",
        tags=["game", "terrain", "walkable"],
    ),
    _reality_entry(
        "reality_switch_link",
        "Switch Link",
        "A switch is linked to a downstream stateful object such as a door or bridge.",
        meaning_rpn="SWITCH TARGET_OBJECT LINK_EXISTS",
        visual_rpn="DRAW_SWITCH DRAW_LINK DRAW_TARGET",
        behavior_rpn="SWITCH_TRIGGER TARGET_STATE TOGGLE",
        component_refs=["switch_actuator"],
        tags=["game", "switch", "link"],
    ),
    _reality_entry(
        "reality_switch_state_toggle",
        "Switch State Toggle",
        "Crossing the switch toggles the state of the linked game object.",
        meaning_rpn="STATE_A STATE_B TOGGLE_PAIR",
        visual_rpn="DRAW_STATE_A DRAW_STATE_B DRAW_TOGGLE_ARROW",
        behavior_rpn="TRIGGER_EVENT CURRENT_STATE TOGGLE NEXT_STATE STORE",
        component_refs=["switch_actuator"],
        tags=["game", "toggle", "state"],
    ),
    _reality_entry(
        "reality_key_pattern_state",
        "Key Pattern State",
        "The current key signature carried by the player, expressed as a visual pattern.",
        meaning_rpn="KEY_SIGNATURE VISUAL_PATTERN CURRENT_STATE",
        visual_rpn="DRAW_KEY_BOX DRAW_PATTERN",
        behavior_rpn="TRANSFORM_BLOCK CONTACT KEY_PATTERN UPDATE",
        component_refs=["lock_key_pattern_match", "color_transform_block", "shape_transform_block"],
        tags=["game", "key", "pattern"],
    ),
    _reality_entry(
        "reality_door_lock_state",
        "Door Lock State",
        "The door remains closed until its target pattern matches the current key pattern.",
        meaning_rpn="DOOR TARGET_PATTERN LOCKED_OR_UNLOCKED",
        visual_rpn="DRAW_DOOR DRAW_TARGET_PATTERN DRAW_LOCK_INDICATOR",
        behavior_rpn="KEY_PATTERN TARGET_PATTERN MATCH? { DOOR_UNLOCK } { DOOR_LOCK } IFELSE",
        component_refs=["lock_key_pattern_match"],
        tags=["game", "door", "lock"],
    ),
    _reality_entry(
        "reality_movement_budget",
        "Movement Budget",
        "A finite movement resource that can be restored by specific recharge tiles.",
        meaning_rpn="MOVE_POINTS RESOURCE_BUDGET",
        visual_rpn="DRAW_ENERGY_COUNTER DRAW_RECHARGE_TILE",
        behavior_rpn="STEP_CONSUME BUDGET_DEC RECHARGE_CONTACT BUDGET_RESET",
        component_refs=["movement_recharge_block"],
        tags=["game", "budget", "movement"],
    ),
    _reality_entry(
        "reality_grid_path_progress",
        "Grid Path Progress",
        "Progress toward the exit measured by traversing a valid route through the grid.",
        meaning_rpn="CURRENT_PATH GOAL_DISTANCE PROGRESS_SIGNAL",
        visual_rpn="DRAW_ROUTE DRAW_GOAL DRAW_PROGRESS_MARKERS",
        behavior_rpn="MOVE_VALID GOAL_DISTANCE UPDATE",
        component_refs=["spatial_navigation_grid", "level_progression"],
        tags=["game", "pathfinding", "progress"],
    ),
    _reality_entry(
        "reality_level_completion_signal",
        "Level Completion Signal",
        "Crossing the solved exit emits a completion signal and advances the stage.",
        meaning_rpn="EXIT_REACHED COMPLETION_SIGNAL STAGE_ADVANCE",
        visual_rpn="DRAW_EXIT DRAW_COMPLETION_FLASH",
        behavior_rpn="EXIT_CONTACT LEVELS_COMPLETED INC",
        component_refs=["level_progression"],
        tags=["game", "completion", "stage"],
    ),
    _reality_entry(
        "reality_visual_state_encoding",
        "Visual State Encoding",
        "Colors, shapes, and positions jointly encode interactive game state.",
        meaning_rpn="COLOR SHAPE POSITION STATE_BIND",
        visual_rpn="DRAW_COLOR_SHAPE_GRID",
        behavior_rpn="FRAME_PARSE FEATURE_MAP STATE_DECODE",
        component_refs=["visual_state_encoding"],
        tags=["game", "visual", "state"],
    ),
    _reality_entry(
        "reality_observation_trace",
        "Observation Trace",
        "An ordered before/after trace used to infer hidden mechanics from interaction outcomes.",
        meaning_rpn="OBSERVATION ACTION RESULT TRACE",
        visual_rpn="DRAW_BEFORE_AFTER_TRACE",
        behavior_rpn="APPEND_OBSERVATION APPEND_ACTION APPEND_RESULT",
        component_refs=["no_instruction_discovery"],
        tags=["game", "observation", "trace"],
    ),
]


def _extend_reality_catalog(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            _reality_entry(
                str(spec["id"]),
                str(spec["name"]),
                str(spec["description"]),
                meaning_rpn=f"{str(spec['id']).upper()} GAME_REALITY_STATE",
                visual_rpn=f"DRAW_{str(spec['id']).upper()} DRAW_STATE_VIEW",
                behavior_rpn=f"{str(spec['id']).upper()} APPLY_WORLD_UPDATE",
                component_refs=list(spec.get("component_refs") or []),
                tags=["game", *(spec.get("tags") or [])],
            )
        )
    return rows


GAME_REALITY_ENTRIES.extend(
    _extend_reality_catalog(
        [
            {"id": "reality_gravity_field", "name": "Gravity Field", "description": "A persistent downward acceleration field affecting avatars and movable bodies.", "tags": ["movement", "gravity"]},
            {"id": "reality_jump_arc", "name": "Jump Arc", "description": "A jump follows an arc shaped by upward impulse, gravity, and collision.", "tags": ["movement", "jump"]},
            {"id": "reality_teleporter_link", "name": "Teleporter Link", "description": "A portal pair binds entry and exit positions into one traversable route.", "tags": ["movement", "teleport"]},
            {"id": "reality_conveyor_force", "name": "Conveyor Force", "description": "A conveyor surface continuously applies directional velocity while occupied.", "tags": ["movement", "forced_motion"]},
            {"id": "reality_swim_buoyancy", "name": "Swim Buoyancy", "description": "A fluid medium changes gravity, drag, and upward float behavior.", "tags": ["movement", "water"]},
            {"id": "reality_wall_contact", "name": "Wall Contact", "description": "Side-wall contact creates a valid state for wall sliding or wall jumping.", "tags": ["movement", "wall"]},
            {"id": "reality_hazard_damage", "name": "Hazard Damage", "description": "Contact with a hazard triggers loss, reset, or damage state.", "tags": ["hazard", "failure"]},
            {"id": "reality_pushable_mass", "name": "Pushable Mass", "description": "A crate or block occupies space and can be translated by applied force.", "tags": ["object", "pushable"]},
            {"id": "reality_breakable_integrity", "name": "Breakable Integrity", "description": "A fragile object changes from intact to removed when its break condition is met.", "tags": ["object", "breakable"]},
            {"id": "reality_powerup_state", "name": "Power-Up State", "description": "Power-ups grant temporary or persistent changes to avatar capabilities.", "tags": ["object", "powerup"]},
            {"id": "reality_sequence_trigger_state", "name": "Sequence Trigger State", "description": "A trigger sequence tracks ordered activations across multiple objects.", "tags": ["puzzle", "sequence"]},
            {"id": "reality_pattern_gate_state", "name": "Pattern Gate State", "description": "A gate state depends on matching a required visible pattern or code.", "tags": ["puzzle", "pattern"]},
            {"id": "reality_visual_signal", "name": "Visual Signal", "description": "Flashing, color, size, and animation encode interaction-relevant state.", "tags": ["visual", "signal"]},
            {"id": "reality_level_topology", "name": "Level Topology", "description": "A level has a traversable layout with branches, hubs, dead ends, and shortcuts.", "tags": ["level_design", "layout"]},
            {"id": "reality_ui_game_state", "name": "UI Game State", "description": "The game transitions through title, prompt, gameplay, completion, and next states.", "tags": ["state_machine", "ui"]},
        ]
    )
)


def _grammar_rule(
    rule_id: str,
    name: str,
    pattern: str,
    rpn_program: str,
    *,
    domain: str = "interactive_game",
    word_refs: list[str] | None = None,
    component_refs: list[str] | None = None,
    reality_refs: list[str] | None = None,
    rule_strength: int = 0,
    superior_to: list[str] | None = None,
    trust_weight: float = 1.0,
    examples: list[dict[str, str]] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "rule_id": rule_id,
        "name": name,
        "galaxy": "Grammar",
        "layer": 3,
        "language": "game_logic",
        "pattern": pattern,
        "rpn_program": rpn_program,
        "domain": domain,
        "category": "game_rule",
        "symbol_refs": [],
        "word_refs": list(word_refs or []),
        "component_refs": list(component_refs or []),
        "reality_refs": list(reality_refs or []),
        "examples": list(examples or []),
        "rule_strength": int(rule_strength),
        "superior_to": list(superior_to or []),
        "trust_weight": float(trust_weight),
        "content": pattern,
        "summary": name,
        "description": rpn_program,
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": int(rule_strength),
            "superior_to": list(superior_to or []),
            "trust_weight": float(trust_weight),
        },
        "tags": list(tags or []),
    }


GAME_GRAMMAR_RULES: list[dict[str, Any]] = [
    _grammar_rule(
        "grammar_detect_walkable_terrain",
        "Detect Walkable Terrain",
        "walkable cells = dominant traversable region",
        "FRAME_PARSE TERRAIN_MASK EXTRACT_WALKABLE_COMPONENTS",
        word_refs=["walkable", "terrain", "path"],
        component_refs=["spatial_navigation_grid"],
        reality_refs=["reality_walkable_terrain"],
        rule_strength=1,
        tags=["game", "terrain", "vision"],
    ),
    _grammar_rule(
        "grammar_route_agent_on_walkable_grid",
        "Route Agent On Walkable Grid",
        "route entity across walkable grid toward target",
        "START_POS TARGET_POS WALKABLE_MASK LED_ASTAR NEXT_STEP_SELECT",
        word_refs=["route", "grid", "move"],
        component_refs=["spatial_navigation_grid"],
        reality_refs=["reality_walkable_terrain", "reality_grid_path_progress"],
        rule_strength=1,
        tags=["game", "pathfinding", "navigation"],
    ),
    _grammar_rule(
        "grammar_toggle_switch_link",
        "Toggle Switch Link",
        "stepping on switch toggles linked target state",
        "AGENT_POS SWITCH_POS EQUALS TARGET_STATE TOGGLE_IF_TRUE",
        word_refs=["switch", "toggle"],
        component_refs=["switch_actuator"],
        reality_refs=["reality_switch_link", "reality_switch_state_toggle"],
        tags=["game", "switch", "state"],
    ),
    _grammar_rule(
        "grammar_compare_key_pattern_to_door",
        "Compare Key Pattern To Door",
        "compare current key pattern to door target pattern",
        "KEY_PATTERN DOOR_PATTERN PATTERN_EQUALS",
        word_refs=["key", "pattern", "door", "match"],
        component_refs=["lock_key_pattern_match"],
        reality_refs=["reality_key_pattern_state", "reality_door_lock_state"],
        rule_strength=1,
        tags=["game", "pattern", "match"],
    ),
    _grammar_rule(
        "grammar_unlock_door_on_pattern_match",
        "Unlock Door On Pattern Match",
        "matching key pattern unlocks door",
        "KEY_PATTERN DOOR_PATTERN PATTERN_EQUALS { DOOR_UNLOCK } IF",
        word_refs=["unlock", "door", "match"],
        component_refs=["lock_key_pattern_match"],
        reality_refs=["reality_door_lock_state", "reality_key_pattern_state"],
        rule_strength=1,
        superior_to=["grammar_route_agent_on_walkable_grid"],
        tags=["game", "unlock", "door"],
    ),
    _grammar_rule(
        "grammar_restore_movement_budget",
        "Restore Movement Budget",
        "stepping on recharge tile restores movement points",
        "AGENT_POS RECHARGE_POS EQUALS { MOVE_BUDGET RESET } IF",
        word_refs=["recharge", "budget", "movement"],
        component_refs=["movement_recharge_block"],
        reality_refs=["reality_movement_budget"],
        tags=["game", "recharge", "resource"],
    ),
    _grammar_rule(
        "grammar_apply_color_transform",
        "Apply Color Transform",
        "stepping on color transform block changes current color state",
        "AGENT_POS COLOR_BLOCK_POS EQUALS { COLOR_STATE NEXT } IF",
        word_refs=["color", "transform"],
        component_refs=["color_transform_block"],
        reality_refs=["reality_key_pattern_state", "reality_visual_state_encoding"],
        tags=["game", "color", "transform"],
    ),
    _grammar_rule(
        "grammar_apply_shape_transform",
        "Apply Shape Transform",
        "stepping on shape transform block changes current shape state",
        "AGENT_POS SHAPE_BLOCK_POS EQUALS { SHAPE_STATE NEXT } IF",
        word_refs=["shape", "transform"],
        component_refs=["shape_transform_block"],
        reality_refs=["reality_key_pattern_state", "reality_visual_state_encoding"],
        tags=["game", "shape", "transform"],
    ),
    _grammar_rule(
        "grammar_complete_level_after_goal_cross",
        "Complete Level After Goal Cross",
        "crossing unlocked goal exit completes current level",
        "EXIT_CONTACT DOOR_UNLOCKED AND { LEVELS_COMPLETED INC } IF",
        word_refs=["level", "complete", "exit"],
        component_refs=["level_progression"],
        reality_refs=["reality_level_completion_signal", "reality_door_lock_state"],
        rule_strength=1,
        tags=["game", "completion", "goal"],
    ),
    _grammar_rule(
        "grammar_infer_rule_from_state_delta",
        "Infer Rule From State Delta",
        "derive hidden mechanic from before action and after state delta",
        "FRAME_BEFORE FRAME_AFTER DIFF ACTION_CAUSE HYPOTHESIS_FORM",
        word_refs=["infer", "observation", "rule"],
        component_refs=["no_instruction_discovery"],
        reality_refs=["reality_observation_trace"],
        tags=["game", "inference", "discovery"],
    ),
]


def _extend_grammar_catalog(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            _grammar_rule(
                str(spec["id"]),
                str(spec["name"]),
                str(spec["pattern"]),
                str(spec["rpn_program"]),
                word_refs=list(spec.get("word_refs") or []),
                component_refs=list(spec.get("component_refs") or []),
                reality_refs=list(spec.get("reality_refs") or []),
                rule_strength=int(spec.get("rule_strength", 0)),
                superior_to=list(spec.get("superior_to") or []),
                tags=["game", *(spec.get("tags") or [])],
            )
        )
    return rows


GAME_GRAMMAR_RULES.extend(
    _extend_grammar_catalog(
        [
            {"id": "grammar_apply_gravity", "name": "Apply Gravity", "pattern": "unsupported entity falls downward", "rpn_program": "ENTITY_SUPPORTED NOT { VEL_Y GRAVITY ADD POS_Y UPDATE } IF", "word_refs": ["gravity", "fall"], "reality_refs": ["reality_gravity_field"], "rule_strength": 1, "tags": ["movement", "gravity"]},
            {"id": "grammar_apply_jump_impulse", "name": "Apply Jump Impulse", "pattern": "jump input gives upward impulse", "rpn_program": "INPUT_JUMP GROUNDED AND { VEL_Y JUMP_IMPULSE SET } IF", "word_refs": ["jump", "impulse"], "reality_refs": ["reality_jump_arc"], "tags": ["movement", "jump"]},
            {"id": "grammar_apply_wall_jump", "name": "Apply Wall Jump", "pattern": "jump from wall contact redirects momentum", "rpn_program": "WALL_CONTACT INPUT_JUMP AND { VEL_X WALL_PUSH SET VEL_Y JUMP_IMPULSE SET } IF", "word_refs": ["wall", "jump"], "reality_refs": ["reality_wall_contact", "reality_jump_arc"], "tags": ["movement", "wall"]},
            {"id": "grammar_trigger_teleporter", "name": "Trigger Teleporter", "pattern": "entity entering portal moves to linked exit", "rpn_program": "ENTITY_POS PORTAL_ENTRY EQUALS { PORTAL_EXIT POS_SET } IF", "word_refs": ["teleporter", "portal"], "reality_refs": ["reality_teleporter_link"], "tags": ["movement", "teleport"]},
            {"id": "grammar_apply_conveyor_motion", "name": "Apply Conveyor Motion", "pattern": "standing on conveyor adds directional velocity", "rpn_program": "ON_CONVEYOR { VEL_X CONVEYOR_FORCE ADD } IF", "word_refs": ["conveyor", "belt"], "reality_refs": ["reality_conveyor_force"], "tags": ["movement", "forced_motion"]},
            {"id": "grammar_apply_swim_motion", "name": "Apply Swim Motion", "pattern": "movement in water uses buoyancy and drag", "rpn_program": "IN_WATER { BUOYANCY APPLY DRAG APPLY } IF", "word_refs": ["swim", "water"], "reality_refs": ["reality_swim_buoyancy"], "tags": ["movement", "water"]},
            {"id": "grammar_push_object_until_blocked", "name": "Push Object Until Blocked", "pattern": "pushable object moves while path ahead is clear", "rpn_program": "PUSH_FORCE APPLIED PATH_AHEAD_CLEAR AND { OBJECT_POS STEP } IF", "word_refs": ["push", "crate"], "reality_refs": ["reality_pushable_mass"], "tags": ["object", "pushable"]},
            {"id": "grammar_break_block_on_force", "name": "Break Block On Force", "pattern": "fragile object breaks when sufficient force or trigger applies", "rpn_program": "BREAK_CONDITION TRUE { BLOCK_STATE REMOVE } IF", "word_refs": ["break", "block"], "reality_refs": ["reality_breakable_integrity"], "tags": ["object", "breakable"]},
            {"id": "grammar_consume_powerup_state", "name": "Consume Power-Up State", "pattern": "touching power-up updates avatar state", "rpn_program": "POWERUP_CONTACT { AVATAR_STATE POWERUP_APPLY } IF", "word_refs": ["powerup", "upgrade"], "reality_refs": ["reality_powerup_state"], "tags": ["object", "powerup"]},
            {"id": "grammar_validate_sequence_trigger", "name": "Validate Sequence Trigger", "pattern": "ordered trigger sequence unlocks downstream gate", "rpn_program": "TRIGGER_TRACE REQUIRED_SEQUENCE EQUALS { GATE_UNLOCK } IF", "word_refs": ["sequence", "trigger"], "reality_refs": ["reality_sequence_trigger_state"], "tags": ["puzzle", "sequence"]},
            {"id": "grammar_decode_visual_signal", "name": "Decode Visual Signal", "pattern": "visual features map to game state cues", "rpn_program": "FRAME_FEATURES COLOR SHAPE FLASH SIZE MAP_TO_STATE", "word_refs": ["visual", "signal"], "reality_refs": ["reality_visual_signal", "reality_visual_state_encoding"], "tags": ["visual", "signal"]},
            {"id": "grammar_transition_game_state", "name": "Transition Game State", "pattern": "ui prompts and events move the game between lifecycle states", "rpn_program": "CURRENT_STATE EVENT LOOKUP NEXT_STATE TRANSITION", "word_refs": ["state", "transition"], "reality_refs": ["reality_ui_game_state"], "tags": ["state_machine", "ui"]},
            {"id": "grammar_follow_backtracking_route", "name": "Follow Backtracking Route", "pattern": "revisit earlier branch after unlocking new dependency", "rpn_program": "NEW_CAPABILITY PRIOR_BLOCKED_ROUTE REEVALUATE PATH_SELECT", "word_refs": ["backtrack", "route"], "reality_refs": ["reality_level_topology"], "tags": ["level_design", "layout"]},
            {"id": "grammar_escalate_difficulty_curve", "name": "Escalate Difficulty Curve", "pattern": "later rooms combine mastered mechanics into denser challenges", "rpn_program": "ROOM_INDEX PREVIOUS_MECHANICS COMBINE CHALLENGE_DENSITY INC", "word_refs": ["difficulty", "curve"], "reality_refs": ["reality_level_topology"], "tags": ["level_design", "progression"]},
        ]
    )
)


def _meta_rule(
    meta_id: str,
    name: str,
    category: str,
    condition: str,
    action: str,
    *,
    rule_refs: list[str] | None = None,
    component_refs: list[str] | None = None,
    priority: float = 1.0,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": meta_id,
        "meta_id": meta_id,
        "name": name,
        "galaxy": "Tool",
        "layer": 4,
        "domain": "interactive_game_strategy",
        "category": category,
        "condition": condition,
        "action": action,
        "rule_refs": list(rule_refs or []),
        "component_refs": list(component_refs or []),
        "priority": float(priority),
        "content": condition,
        "summary": name,
        "description": action,
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "priority": float(priority),
        },
        "tags": list(tags or []),
    }


GAME_META_RULES: list[dict[str, Any]] = [
    _meta_rule(
        "meta_prefer_shortest_walkable_route",
        "Prefer Shortest Walkable Route",
        "game_strategy",
        "TARGET_PRESENT WALKABLE_PATH_EXISTS AND",
        "ROUTE_CANDIDATES PATH_COST_MIN SELECT",
        rule_refs=["grammar_detect_walkable_terrain", "grammar_route_agent_on_walkable_grid"],
        component_refs=["spatial_navigation_grid"],
        priority=0.95,
        tags=["game", "strategy", "pathfinding"],
    ),
    _meta_rule(
        "meta_route_to_switch_when_locked",
        "Route To Switch When Locked",
        "game_strategy",
        "DOOR_LOCKED SWITCH_VISIBLE AND",
        "SWITCH_POS SET_TARGET grammar_route_agent_on_walkable_grid APPLY_RULE",
        rule_refs=["grammar_toggle_switch_link", "grammar_route_agent_on_walkable_grid"],
        component_refs=["switch_actuator", "lock_key_pattern_match"],
        priority=0.92,
        tags=["game", "switch", "lock"],
    ),
    _meta_rule(
        "meta_match_key_before_door",
        "Match Key Before Door",
        "game_strategy",
        "DOOR_LOCKED KEY_PATTERN_MISMATCH AND",
        "TRANSFORM_TARGET_SET grammar_compare_key_pattern_to_door APPLY_RULE",
        rule_refs=["grammar_compare_key_pattern_to_door", "grammar_unlock_door_on_pattern_match"],
        component_refs=["lock_key_pattern_match"],
        priority=0.96,
        tags=["game", "key", "door"],
    ),
    _meta_rule(
        "meta_seek_recharge_when_budget_low",
        "Seek Recharge When Budget Low",
        "game_strategy",
        "MOVE_BUDGET LOW_THRESHOLD_LT",
        "RECHARGE_TILE TARGET_SET grammar_restore_movement_budget APPLY_RULE",
        rule_refs=["grammar_restore_movement_budget", "grammar_route_agent_on_walkable_grid"],
        component_refs=["movement_recharge_block"],
        priority=0.9,
        tags=["game", "recharge", "resource"],
    ),
    _meta_rule(
        "meta_probe_transform_blocks_on_mismatch",
        "Probe Transform Blocks On Mismatch",
        "game_strategy",
        "DOOR_LOCKED KEY_PATTERN_MISMATCH AND",
        "COLOR_BLOCK SHAPE_BLOCK OBSERVE_TRANSFORM LOOP_UNTIL_MATCH",
        rule_refs=[
            "grammar_apply_color_transform",
            "grammar_apply_shape_transform",
            "grammar_compare_key_pattern_to_door",
        ],
        component_refs=["color_transform_block", "shape_transform_block", "multi_step_state_transform"],
        priority=0.91,
        tags=["game", "transform", "mismatch"],
    ),
    _meta_rule(
        "meta_confirm_unlock_before_exit",
        "Confirm Unlock Before Exit",
        "game_strategy",
        "EXIT_VISIBLE DOOR_UNLOCKED AND",
        "EXIT_POS SET_TARGET grammar_complete_level_after_goal_cross APPLY_RULE",
        rule_refs=["grammar_unlock_door_on_pattern_match", "grammar_complete_level_after_goal_cross"],
        component_refs=["level_progression", "lock_key_pattern_match"],
        priority=0.94,
        tags=["game", "exit", "completion"],
    ),
    _meta_rule(
        "meta_learn_from_visual_transition",
        "Learn From Visual Transition",
        "self_reflection",
        "STATE_CHANGED ACTION_OBSERVED AND",
        "grammar_infer_rule_from_state_delta APPLY_RULE CONSOLIDATE_PATTERN",
        rule_refs=["grammar_infer_rule_from_state_delta"],
        component_refs=["no_instruction_discovery", "visual_state_encoding"],
        priority=0.93,
        tags=["game", "learning", "observation"],
    ),
    _meta_rule(
        "meta_click_to_advance_after_completion",
        "Click To Advance After Completion",
        "game_strategy",
        "LEVEL_COMPLETE ADVANCE_PROMPT_VISIBLE AND",
        "CLICK_ADVANCE_BUTTON",
        rule_refs=["grammar_complete_level_after_goal_cross"],
        component_refs=["level_progression"],
        priority=0.88,
        tags=["game", "advance", "click"],
    ),
]


def _extend_meta_catalog(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            _meta_rule(
                str(spec["id"]),
                str(spec["name"]),
                str(spec["category"]),
                str(spec["condition"]),
                str(spec["action"]),
                rule_refs=list(spec.get("rule_refs") or []),
                component_refs=list(spec.get("component_refs") or []),
                priority=float(spec.get("priority", 0.9)),
                tags=["game", *(spec.get("tags") or [])],
            )
        )
    return rows


GAME_META_RULES.extend(
    _extend_meta_catalog(
        [
            {"id": "meta_jump_over_gap", "name": "Jump Over Gap", "category": "game_strategy", "condition": "GAP_AHEAD JUMP_CLEAR_PATH AND", "action": "grammar_apply_jump_impulse APPLY_RULE", "rule_refs": ["grammar_apply_jump_impulse"], "priority": 0.9, "tags": ["movement", "jump"]},
            {"id": "meta_use_teleporter_when_route_breaks", "name": "Use Teleporter When Route Breaks", "category": "game_strategy", "condition": "DIRECT_ROUTE_BLOCKED PORTAL_VISIBLE AND", "action": "grammar_trigger_teleporter APPLY_RULE", "rule_refs": ["grammar_trigger_teleporter"], "priority": 0.89, "tags": ["movement", "teleport"]},
            {"id": "meta_follow_conveyor_flow", "name": "Follow Conveyor Flow", "category": "game_strategy", "condition": "ON_CONVEYOR TRUE", "action": "grammar_apply_conveyor_motion APPLY_RULE", "rule_refs": ["grammar_apply_conveyor_motion"], "priority": 0.84, "tags": ["movement", "forced_motion"]},
            {"id": "meta_push_crate_to_enable_path", "name": "Push Crate To Enable Path", "category": "game_strategy", "condition": "PATH_BLOCKED PUSHABLE_OBJECT_VISIBLE AND", "action": "grammar_push_object_until_blocked APPLY_RULE", "rule_refs": ["grammar_push_object_until_blocked"], "priority": 0.9, "tags": ["object", "pushable"]},
            {"id": "meta_avoid_hazard_without_protection", "name": "Avoid Hazard Without Protection", "category": "game_strategy", "condition": "HAZARD_VISIBLE PROTECTION_ABSENT AND", "action": "REROUTE_TO_SAFE_TILE", "rule_refs": ["grammar_route_agent_on_walkable_grid"], "priority": 0.97, "tags": ["hazard", "safety"]},
            {"id": "meta_read_visual_affordance_before_action", "name": "Read Visual Affordance Before Action", "category": "game_strategy", "condition": "UNKNOWN_OBJECT_VISIBLE", "action": "grammar_decode_visual_signal APPLY_RULE", "rule_refs": ["grammar_decode_visual_signal"], "priority": 0.91, "tags": ["visual", "affordance"]},
            {"id": "meta_backtrack_when_dependency_unsatisfied", "name": "Backtrack When Dependency Unsatisfied", "category": "game_strategy", "condition": "GOAL_BLOCKED MISSING_DEPENDENCY AND", "action": "grammar_follow_backtracking_route APPLY_RULE", "rule_refs": ["grammar_follow_backtracking_route"], "priority": 0.93, "tags": ["level_design", "backtracking"]},
            {"id": "meta_use_checkpoint_after_progress", "name": "Use Checkpoint After Progress", "category": "game_strategy", "condition": "CHECKPOINT_VISIBLE RECENT_PROGRESS AND", "action": "MOVE_TO_CHECKPOINT", "rule_refs": ["grammar_route_agent_on_walkable_grid"], "priority": 0.83, "tags": ["progress", "checkpoint"]},
            {"id": "meta_start_from_title_state", "name": "Start From Title State", "category": "game_strategy", "condition": "CURRENT_STATE TITLE_OR_PROMPT", "action": "grammar_transition_game_state APPLY_RULE INPUT_ACTION_PERFORM", "rule_refs": ["grammar_transition_game_state"], "priority": 0.98, "tags": ["state_machine", "start"]},
        ]
    )
)


def build_game_mechanics_entries() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_MECHANICS]
    if len(rows) < 100:
        raise RuntimeError(f"Expected at least 100 game mechanic entries, generated {len(rows)}")
    return rows


def build_game_reality_entries() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_REALITY_ENTRIES]
    if len(rows) < 20:
        raise RuntimeError(f"Expected at least 20 game reality entries, generated {len(rows)}")
    return rows


def build_game_grammar_rules() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_GRAMMAR_RULES]
    if len(rows) < 20:
        raise RuntimeError(f"Expected at least 20 game grammar rules, generated {len(rows)}")
    return rows


def build_game_meta_rules() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_META_RULES]
    if len(rows) < 15:
        raise RuntimeError(f"Expected at least 15 game meta-rules, generated {len(rows)}")
    return rows


def populate_game_mechanics(*, house_dir: Path = DEFAULT_HOUSE_DIR) -> dict[str, int]:
    house_dir = Path(house_dir)
    house_dir.mkdir(parents=True, exist_ok=True)
    return upsert_entries(
        house_dir / "game_mechanics.jsonl",
        build_game_mechanics_entries(),
    )


def populate_game_knowledge(*, house_dir: Path = DEFAULT_HOUSE_DIR) -> dict[str, dict[str, int]]:
    house_dir = Path(house_dir)
    house_dir.mkdir(parents=True, exist_ok=True)
    return {
        "game_mechanics.jsonl": upsert_entries(house_dir / "game_mechanics.jsonl", build_game_mechanics_entries()),
        "Reality.jsonl": upsert_entries(house_dir / "Reality.jsonl", build_game_reality_entries()),
        "Grammar.jsonl": upsert_entries(house_dir / "Grammar.jsonl", build_game_grammar_rules()),
        "Tool.jsonl": upsert_entries(house_dir / "Tool.jsonl", build_game_meta_rules()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate House game mechanics knowledge for live game reasoning.")
    parser.add_argument(
        "--house-dir",
        type=Path,
        default=DEFAULT_HOUSE_DIR,
        help="Directory containing House JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_by_file = populate_game_knowledge(house_dir=args.house_dir)
    for filename, stats in stats_by_file.items():
        print(
            f"{filename}:"
            f" before={stats['before']}"
            f" after={stats['after']}"
            f" appended={stats['appended']}"
            f" replaced={stats['replaced']}"
            f" removed={stats['removed']}"
        )


if __name__ == "__main__":
    main()
