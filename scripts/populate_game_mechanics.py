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


def build_game_mechanics_entries() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_MECHANICS]
    if len(rows) < 10:
        raise RuntimeError(f"Expected at least 10 game mechanic entries, generated {len(rows)}")
    return rows


def build_game_reality_entries() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_REALITY_ENTRIES]
    if len(rows) < 8:
        raise RuntimeError(f"Expected at least 8 game reality entries, generated {len(rows)}")
    return rows


def build_game_grammar_rules() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_GRAMMAR_RULES]
    if len(rows) < 8:
        raise RuntimeError(f"Expected at least 8 game grammar rules, generated {len(rows)}")
    return rows


def build_game_meta_rules() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_META_RULES]
    if len(rows) < 6:
        raise RuntimeError(f"Expected at least 6 game meta-rules, generated {len(rows)}")
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
