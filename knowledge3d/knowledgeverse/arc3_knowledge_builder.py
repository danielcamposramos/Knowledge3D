"""Append universal spatial and ARC-AGI-3 interaction knowledge to Galaxy JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.cranium.action_primitives_bootstrap import spatial_action_atoms


GALAXY_DIR = Path("/K3D/Knowledge3D.local/galaxies")
BOOTSTRAP_TAG = "spatial_knowledge_builder_v1"


ACTION_ENTRY_OVERRIDES: dict[str, dict[str, Any]] = {
    "atom:action:move_up": {
        "id": "reality_action_move_up",
        "name": "Move Up - spatial translation",
        "category": "translation",
        "query_anchor": "move up upward above north spatial translation direction",
        "tags": ["spatial", "action", "translation", "up", "above", "north"],
    },
    "atom:action:move_down": {
        "id": "reality_action_move_down",
        "name": "Move Down - spatial translation",
        "category": "translation",
        "query_anchor": "move down downward below south spatial translation direction",
        "tags": ["spatial", "action", "translation", "down", "below", "south"],
    },
    "atom:action:move_left": {
        "id": "reality_action_move_left",
        "name": "Move Left - spatial translation",
        "category": "translation",
        "query_anchor": "move left west spatial translation direction",
        "tags": ["spatial", "action", "translation", "left", "west"],
    },
    "atom:action:move_right": {
        "id": "reality_action_move_right",
        "name": "Move Right - spatial translation",
        "category": "translation",
        "query_anchor": "move right east spatial translation direction",
        "tags": ["spatial", "action", "translation", "right", "east"],
    },
    "atom:action:perform": {
        "id": "reality_action_perform",
        "name": "Perform - spatial interaction",
        "category": "interaction",
        "query_anchor": "perform execute interact centered target spatial action",
        "tags": ["spatial", "action", "interaction", "perform", "execute"],
    },
    "atom:action:click": {
        "id": "reality_action_click",
        "name": "Click - coordinate selection",
        "category": "selection",
        "query_anchor": "click coordinates x y spatial selection interact",
        "tags": ["spatial", "action", "click", "coordinates", "selection"],
    },
    "atom:action:undo": {
        "id": "reality_action_undo",
        "name": "Undo - temporal recovery",
        "category": "recovery",
        "query_anchor": "undo recovery reverse previous action temporal spatial",
        "tags": ["spatial", "action", "undo", "recovery", "temporal"],
    },
    "atom:action:move_diagonal_ur": {
        "id": "reality_action_move_diagonal_ur",
        "name": "Move Diagonal Up Right - composed translation",
        "category": "translation_composed",
        "query_anchor": "move diagonal up right northeast composed spatial translation",
        "tags": ["spatial", "action", "translation", "diagonal", "up_right"],
    },
    "atom:action:reach": {
        "id": "reality_action_reach",
        "name": "Reach - object interaction",
        "category": "object_interaction",
        "query_anchor": "reach extend hand toward target object interaction spatial",
        "tags": ["spatial", "action", "reach", "object_interaction"],
    },
    "atom:action:grab": {
        "id": "reality_action_grab",
        "name": "Grab - object interaction",
        "category": "object_interaction",
        "query_anchor": "grab hold object close hand interaction spatial",
        "tags": ["spatial", "action", "grab", "object_interaction"],
    },
    "atom:action:hold": {
        "id": "reality_action_hold",
        "name": "Hold - object interaction",
        "category": "object_interaction",
        "query_anchor": "hold maintain grip object interaction spatial",
        "tags": ["spatial", "action", "hold", "object_interaction"],
    },
    "atom:action:release": {
        "id": "reality_action_release",
        "name": "Release - object interaction",
        "category": "object_interaction",
        "query_anchor": "release let go object interaction spatial",
        "tags": ["spatial", "action", "release", "object_interaction"],
    },
    "atom:action:use": {
        "id": "reality_action_use",
        "name": "Use - object interaction",
        "category": "object_interaction",
        "query_anchor": "use trigger tool object interaction spatial",
        "tags": ["spatial", "action", "use", "tool", "object_interaction"],
    },
    "atom:action:walk_to": {
        "id": "reality_action_walk_to",
        "name": "Walk To - spatial navigation",
        "category": "navigation",
        "query_anchor": "walk to target navigate path spatial movement",
        "tags": ["spatial", "action", "walk", "navigation", "path"],
    },
    "atom:action:teleport": {
        "id": "reality_action_teleport",
        "name": "Teleport - spatial navigation",
        "category": "navigation",
        "query_anchor": "teleport instant translation target spatial navigation",
        "tags": ["spatial", "action", "teleport", "navigation"],
    },
    "atom:action:look_at": {
        "id": "reality_action_look_at",
        "name": "Look At - spatial orientation",
        "category": "orientation",
        "query_anchor": "look at orient toward target spatial direction attention",
        "tags": ["spatial", "action", "look", "orientation", "target"],
    },
}


def _action_entry_id(legacy_id: str) -> str:
    return str(ACTION_ENTRY_OVERRIDES[str(legacy_id)]["id"])


def _action_entry_from_atom(atom: Any) -> dict[str, Any]:
    legacy_id = str(getattr(atom, "node_id", ""))
    override = ACTION_ENTRY_OVERRIDES[legacy_id]
    metadata = dict(getattr(atom, "metadata", {}) or {})
    entry_metadata: dict[str, Any] = {
        "bootstrap": BOOTSTRAP_TAG,
        "displacement": list(metadata.get("displacement") or [0, 0]),
        "action_type": str(metadata.get("action_type", "")),
        "query_anchor": str(override["query_anchor"]),
        "surface_forms": dict(metadata.get("surface_forms") or {}),
        "reusable_contexts": list(metadata.get("reusable_contexts") or []),
    }
    if metadata.get("inverse"):
        entry_metadata["inverse"] = _action_entry_id(str(metadata["inverse"]))
    if metadata.get("hanim_anchor") is not None:
        entry_metadata["hanim_anchor"] = metadata.get("hanim_anchor")
    if metadata.get("parameterized"):
        entry_metadata["parameterized"] = True
    if metadata.get("parameters"):
        entry_metadata["parameters"] = list(metadata.get("parameters") or [])
    if metadata.get("house_triggers"):
        entry_metadata["house_triggers"] = dict(metadata.get("house_triggers") or {})
    if metadata.get("arc3_action"):
        entry_metadata["action_bindings"] = {"arc3": str(metadata["arc3_action"])}
    entry: dict[str, Any] = {
        "id": str(override["id"]),
        "name": str(override["name"]),
        "domain": "spatial_action",
        "category": str(override["category"]),
        "content": str(metadata.get("description", "")),
        "description": str(metadata.get("description", "")),
        "visual_rpn": str(getattr(atom, "visual_rpn", "")),
        "behavior_rpn": str(getattr(atom, "behavior_rpn", "")),
        "law_rpn": str(getattr(atom, "law_rpn", "")),
        "metadata": entry_metadata,
        "tags": list(override.get("tags") or []),
    }
    component_refs = list(getattr(atom, "component_refs", []) or [])
    if component_refs:
        entry["component_refs"] = [_action_entry_id(str(ref)) for ref in component_refs]
    return entry


SPATIAL_ACTION_REALITY_ENTRIES: list[dict[str, Any]] = [
    _action_entry_from_atom(atom) for atom in spatial_action_atoms()
]


ARC3_GRAMMAR_RULES: list[dict[str, Any]] = [
    {
        "id": "grammar_spatial_move_toward_above",
        "name": "Spatial rule - move toward above",
        "domain": "spatial_reasoning",
        "category": "movement_rule",
        "description": "When the target lies above the current position, move upward.",
        "content": "universal spatial rule for moving toward a target above the current position",
        "rpn_program": "CURRENT_ROW TARGET_ROW SUB NEG THRESHOLD_GT ACTION_MOVE_UP",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 0,
            "action_name": "ACTION1",
            "action_label": "Move Up",
            "condition": "target_row < current_row",
            "query_anchor": "object above center top north navigate game frame move up direction spatial",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "move_up", "above", "north"],
        "word_refs": ["word_direction_above"],
        "reality_refs": ["reality_action_move_up"],
    },
    {
        "id": "grammar_spatial_move_toward_below",
        "name": "Spatial rule - move toward below",
        "domain": "spatial_reasoning",
        "category": "movement_rule",
        "description": "When the target lies below the current position, move downward.",
        "content": "universal spatial rule for moving toward a target below the current position",
        "rpn_program": "TARGET_ROW CURRENT_ROW SUB POS THRESHOLD_GT ACTION_MOVE_DOWN",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 1,
            "action_name": "ACTION2",
            "action_label": "Move Down",
            "condition": "target_row > current_row",
            "query_anchor": "object below center bottom south navigate game frame move down direction spatial",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "move_down", "below", "south"],
        "word_refs": ["word_direction_below"],
        "reality_refs": ["reality_action_move_down"],
    },
    {
        "id": "grammar_spatial_move_toward_left",
        "name": "Spatial rule - move toward left",
        "domain": "spatial_reasoning",
        "category": "movement_rule",
        "description": "When the target lies left of the current position, move left.",
        "content": "universal spatial rule for moving toward a target left of the current position",
        "rpn_program": "CURRENT_COL TARGET_COL SUB NEG THRESHOLD_GT ACTION_MOVE_LEFT",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 2,
            "action_name": "ACTION3",
            "action_label": "Move Left",
            "condition": "target_col < current_col",
            "query_anchor": "object left west navigate game frame move left direction spatial",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "move_left", "left", "west"],
        "word_refs": ["word_direction_left"],
        "reality_refs": ["reality_action_move_left"],
    },
    {
        "id": "grammar_spatial_move_toward_right",
        "name": "Spatial rule - move toward right",
        "domain": "spatial_reasoning",
        "category": "movement_rule",
        "description": "When the target lies right of the current position, move right.",
        "content": "universal spatial rule for moving toward a target right of the current position",
        "rpn_program": "TARGET_COL CURRENT_COL SUB POS THRESHOLD_GT ACTION_MOVE_RIGHT",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 3,
            "action_name": "ACTION4",
            "action_label": "Move Right",
            "condition": "target_col > current_col",
            "query_anchor": "object right east navigate game frame move right direction spatial",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "move_right", "right", "east"],
        "word_refs": ["word_direction_right"],
        "reality_refs": ["reality_action_move_right"],
    },
    {
        "id": "grammar_spatial_interact_centered",
        "name": "Spatial rule - interact when centered",
        "domain": "spatial_reasoning",
        "category": "interaction_rule",
        "description": "When the target is reached or aligned, perform the interaction action.",
        "content": "universal spatial rule for interacting once the target is centered or reached",
        "rpn_program": "TARGET_REACHED ACTION_PERFORM",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 4,
            "action_name": "ACTION5",
            "action_label": "Perform",
            "condition": "target_reached",
            "query_anchor": "object centered aligned target reached perform interact spatial",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "interact", "centered", "perform"],
        "word_refs": ["word_direction_centered"],
        "reality_refs": ["reality_action_perform"],
    },
    {
        "id": "grammar_spatial_click_coordinates",
        "name": "Spatial rule - click coordinates",
        "domain": "spatial_reasoning",
        "category": "interaction_rule",
        "description": "When a click target is inferred, emit a coordinate click action.",
        "content": "universal spatial rule for coordinate click interactions",
        "rpn_program": "CLICK_TARGET_X CLICK_TARGET_Y ACTION_CLICK",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 5,
            "action_name": "ACTION6",
            "action_label": "Click",
            "condition": "click_target_present",
            "query_anchor": "click coordinates x y target interact spatial selection",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "click", "coordinates"],
        "reality_refs": ["reality_action_click"],
    },
    {
        "id": "grammar_spatial_undo_recovery",
        "name": "Spatial rule - undo recovery",
        "domain": "spatial_reasoning",
        "category": "recovery_rule",
        "description": "When the state requires recovery or only undo is allowed, emit undo.",
        "content": "universal spatial recovery rule for undoing the previous action",
        "rpn_program": "UNDO_STATE ACTION_UNDO",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_index": 6,
            "action_name": "ACTION7",
            "action_label": "Undo",
            "condition": "undo_allowed",
            "query_anchor": "undo recovery loop stagnation previous action spatial",
            "rule_strength": 1,
        },
        "tags": ["spatial", "reasoning", "undo", "recovery"],
        "reality_refs": ["reality_action_undo"],
    },
    {
        "id": "grammar_spatial_keyboard_game",
        "name": "Spatial mode - keyboard navigation",
        "domain": "spatial_reasoning",
        "category": "interaction_mode_recognition",
        "description": "Recognize movement-based interactive tasks from directional actions being available.",
        "content": "infer keyboard movement mode from directional action availability",
        "rpn_program": "AVAILABLE_ACTIONS ANY_OF [0,1,2,3] CONTAINS MODE_KEYBOARD",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "keyboard",
            "query_anchor": "keyboard navigation move directional spatial action availability",
            "rule_strength": 1,
        },
        "tags": ["spatial", "mode", "keyboard", "navigation"],
        "reality_refs": ["reality_arc3_available_actions", "reality_arc3_keyboard_game"],
    },
    {
        "id": "grammar_spatial_click_game",
        "name": "Spatial mode - click interaction",
        "domain": "spatial_reasoning",
        "category": "interaction_mode_recognition",
        "description": "Recognize click-only interaction mode when coordinate click is available without movement.",
        "content": "infer click interaction mode from click-only action availability",
        "rpn_program": "AVAILABLE_ACTIONS 5 CONTAINS AVAILABLE_ACTIONS ANY_OF [0,1,2,3] NOT_CONTAINS AND MODE_CLICK",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "click",
            "query_anchor": "click interaction coordinates spatial action availability",
            "rule_strength": 1,
        },
        "tags": ["spatial", "mode", "click", "interaction"],
        "reality_refs": ["reality_arc3_available_actions", "reality_arc3_click_game"],
    },
    {
        "id": "grammar_spatial_keyboard_click_game",
        "name": "Spatial mode - keyboard click interaction",
        "domain": "spatial_reasoning",
        "category": "interaction_mode_recognition",
        "description": "Recognize mixed movement and click interaction mode.",
        "content": "infer mixed movement and click interaction mode from action availability",
        "rpn_program": "AVAILABLE_ACTIONS ANY_OF [0,1,2,3] CONTAINS AVAILABLE_ACTIONS 5 CONTAINS AND MODE_KEYBOARD_CLICK",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "keyboard_click",
            "query_anchor": "keyboard click navigation move click interact spatial action availability",
            "rule_strength": 1,
        },
        "tags": ["spatial", "mode", "keyboard_click", "interaction"],
        "reality_refs": ["reality_arc3_available_actions", "reality_arc3_keyboard_click_game"],
    },
    {
        "id": "grammar_spatial_detect_background",
        "name": "Spatial perception - detect background",
        "domain": "spatial_reasoning",
        "category": "frame_interpretation",
        "description": "Detect the background value as the most frequent value in the frame.",
        "content": "use argmax frequency to infer background in a grid frame",
        "rpn_program": "FRAME FREQUENCY_MAP ARGMAX BACKGROUND_ASSIGN",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "background_detection": "argmax_frequency",
        },
        "tags": ["spatial", "perception", "background"],
        "reality_refs": ["reality_arc3_game_frame"],
    },
    {
        "id": "grammar_spatial_detect_foreground_components",
        "name": "Spatial perception - detect foreground components",
        "domain": "spatial_reasoning",
        "category": "object_detection",
        "description": "Extract non-background connected components to identify foreground objects.",
        "content": "segment foreground objects by connected components after background removal",
        "rpn_program": "FRAME BACKGROUND FILTER_OUT CONNECTED_COMPONENTS FOREGROUND_OBJECTS",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "object_detection": "connected_components_non_background",
        },
        "tags": ["spatial", "perception", "foreground", "components"],
        "reality_refs": ["reality_arc3_game_frame"],
    },
    {
        "id": "grammar_spatial_foreground_centroid",
        "name": "Spatial perception - foreground centroid",
        "domain": "spatial_reasoning",
        "category": "frame_interpretation",
        "description": "Compute the centroid of non-background cells as a fallback target.",
        "content": "use centroid of visible foreground mass as fallback target in spatial navigation",
        "rpn_program": "FOREGROUND_OBJECTS CENTROID TARGET_ASSIGN",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 0,
            "targeting_mode": "foreground_centroid",
        },
        "tags": ["spatial", "perception", "centroid", "fallback"],
        "reality_refs": ["reality_arc3_game_frame"],
    },
]


ARC3_REALITY_ENTRIES: list[dict[str, Any]] = [
    *SPATIAL_ACTION_REALITY_ENTRIES,
    {
        "id": "reality_arc3_game_frame",
        "name": "ARC-AGI-3 Game Frame",
        "domain": "reality_interactive_game",
        "category": "game_perception",
        "content": "64x64 interactive ARC-AGI-3 frame with background and foreground objects",
        "description": "A 64x64 grid representing the current visual state of an ARC-AGI-3 interactive game. Background is the most frequent value. Foreground objects are non-background connected regions.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "frame_width": 64,
            "frame_height": 64,
            "background_detection": "argmax_frequency",
            "object_detection": "connected_components_non_background",
        },
        "tags": ["arc3", "frame", "perception", "interactive_game"],
    },
    {
        "id": "reality_arc3_available_actions",
        "name": "ARC-AGI-3 Available Actions",
        "domain": "reality_interactive_game",
        "category": "action_space",
        "content": "ARC-AGI-3 server returns integer action ids that map directly to internal action indices",
        "description": "The game server returns a list of valid action integers. The observed contract maps directly to internal 0-based action indices: 0=ACTION1 Move Up, 1=ACTION2 Move Down, 2=ACTION3 Move Left, 3=ACTION4 Move Right, 4=ACTION5 Perform, 5=ACTION6 Click, 6=ACTION7 Undo.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "action_map": {
                "0": "ACTION1_MOVE_UP",
                "1": "ACTION2_MOVE_DOWN",
                "2": "ACTION3_MOVE_LEFT",
                "3": "ACTION4_MOVE_RIGHT",
                "4": "ACTION5_PERFORM",
                "5": "ACTION6_CLICK",
                "6": "ACTION7_UNDO",
            },
        },
        "tags": ["arc3", "action_space", "contract", "server_api"],
    },
    {
        "id": "reality_arc3_levels_completed",
        "name": "ARC-AGI-3 Levels Completed",
        "domain": "reality_interactive_game",
        "category": "progress_signal",
        "content": "levels_completed increases when a sub-task is solved in an interactive ARC-AGI-3 game",
        "description": "levels_completed is the positive progress signal for live ARC-AGI-3 games. win_levels is the total needed. The game is won when levels_completed equals win_levels.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "signal_type": "positive_delta",
            "success_condition": "levels_completed == win_levels",
        },
        "tags": ["arc3", "progress", "levels_completed", "win_condition"],
    },
    {
        "id": "reality_arc3_keyboard_game",
        "name": "ARC-AGI-3 Keyboard Game",
        "domain": "reality_interactive_game",
        "category": "game_type",
        "content": "keyboard ARC-AGI-3 game uses directional movement and perform",
        "description": "Keyboard ARC-AGI-3 games expose directional movement actions and usually perform. The agent must navigate a cursor or object through the visual scene.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "keyboard",
            "canonical_actions": [0, 1, 2, 3, 4],
        },
        "tags": ["arc3", "keyboard", "game_type", "navigation"],
    },
    {
        "id": "reality_arc3_click_game",
        "name": "ARC-AGI-3 Click Game",
        "domain": "reality_interactive_game",
        "category": "game_type",
        "content": "click ARC-AGI-3 game uses coordinate selection rather than directional movement",
        "description": "Click ARC-AGI-3 games expose click actions and require choosing visual coordinates instead of directional motion.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "click",
            "canonical_actions": [5],
        },
        "tags": ["arc3", "click", "game_type", "coordinates"],
    },
    {
        "id": "reality_arc3_keyboard_click_game",
        "name": "ARC-AGI-3 Keyboard Click Game",
        "domain": "reality_interactive_game",
        "category": "game_type",
        "content": "mixed ARC-AGI-3 game uses both directional movement and click interactions",
        "description": "Keyboard-click ARC-AGI-3 games require combining movement and click actions in the same live puzzle.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "keyboard_click",
            "canonical_actions": [0, 1, 2, 3, 4, 5],
        },
        "tags": ["arc3", "keyboard_click", "game_type", "hybrid"],
    },
    {
        "id": "reality_arc3_undo_only_state",
        "name": "ARC-AGI-3 Undo Only State",
        "domain": "reality_interactive_game",
        "category": "state_constraint",
        "content": "undo-only ARC-AGI-3 state exposes only ACTION7 Undo in available_actions",
        "description": "Some ARC-AGI-3 states expose only the Undo action. This is a hard server constraint and should be treated as a recovery or initialization state.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "game_type": "undo_only",
            "canonical_actions": [6],
        },
        "tags": ["arc3", "undo", "constraint", "state"],
    },
    {
        "id": "reality_arc3_baseline_actions",
        "name": "ARC-AGI-3 Baseline Actions",
        "domain": "reality_interactive_game",
        "category": "benchmark_metadata",
        "content": "baseline_actions records counts per level in a reference solution, not action ids",
        "description": "baseline_actions values describe how many actions a baseline solution used per level. They are not action-type identifiers and should not be confused with available_actions.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "field_type": "per_level_action_counts",
        },
        "tags": ["arc3", "benchmark", "baseline_actions", "metadata"],
    },
]


ARC3_META_RULES: list[dict[str, Any]] = [
    {
        "id": "meta_spatial_seek_goal_when_present",
        "name": "Spatial meta-rule - seek goal when present",
        "domain": "spatial_strategy",
        "category": "navigation_strategy",
        "description": "Prefer explicit goal-directed navigation when a goal frame or target state is known.",
        "content": "goal-directed spatial strategy should override generic exploration",
        "rpn_program": "GOAL_PRESENT GOAL_OBJECT_POS CURRENT_POS DELTA DIRECTION_SELECT ACTION_EMIT",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "condition": "goal_present",
        },
        "tags": ["spatial", "meta_rule", "goal_seeking", "navigation"],
        "grammar_refs": [
            "grammar_spatial_move_toward_above",
            "grammar_spatial_move_toward_below",
            "grammar_spatial_move_toward_left",
            "grammar_spatial_move_toward_right",
        ],
        "reality_refs": ["reality_arc3_game_frame"],
    },
    {
        "id": "meta_spatial_seek_centroid_when_no_goal",
        "name": "Spatial meta-rule - seek centroid when no goal",
        "domain": "spatial_strategy",
        "category": "navigation_strategy",
        "description": "When no explicit goal is known, target the centroid of foreground activity as a defeasible strategy.",
        "content": "foreground centroid is the default spatial navigation target when no goal is available",
        "rpn_program": "GOAL_ABSENT FOREGROUND_CENTROID CURRENT_POS DELTA DIRECTION_SELECT ACTION_EMIT",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 0,
            "condition": "goal_absent",
            "superior_to": ["meta_spatial_default_action"],
        },
        "tags": ["spatial", "meta_rule", "centroid", "navigation"],
        "grammar_refs": [
            "grammar_spatial_foreground_centroid",
            "grammar_spatial_move_toward_above",
            "grammar_spatial_move_toward_below",
            "grammar_spatial_move_toward_left",
            "grammar_spatial_move_toward_right",
        ],
        "reality_refs": ["reality_arc3_game_frame"],
    },
    {
        "id": "meta_spatial_use_click_coordinates",
        "name": "Spatial meta-rule - use click coordinates",
        "domain": "spatial_strategy",
        "category": "interaction_strategy",
        "description": "For click or mixed tasks, derive coordinate targets from salient foreground structure before emitting click.",
        "content": "click strategy should convert salient visual targets into x y coordinates",
        "rpn_program": "MODE_CLICK_OR_KEYBOARD_CLICK SALIENT_TARGET COORDINATES ACTION_CLICK",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "condition": "click_available",
        },
        "tags": ["spatial", "meta_rule", "click", "coordinates"],
        "grammar_refs": [
            "grammar_spatial_click_coordinates",
            "grammar_spatial_click_game",
            "grammar_spatial_keyboard_click_game",
        ],
        "reality_refs": ["reality_arc3_click_game", "reality_arc3_keyboard_click_game"],
    },
    {
        "id": "meta_spatial_respect_available_actions",
        "name": "Spatial meta-rule - respect available actions",
        "domain": "spatial_strategy",
        "category": "constraint_compliance",
        "description": "Never emit actions outside the available_actions set returned by the live task server.",
        "content": "action availability is a hard constraint for interactive spatial control",
        "rpn_program": "AVAILABLE_ACTIONS HARD_FILTER ACTION_EMIT",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "condition": "always",
        },
        "tags": ["spatial", "meta_rule", "constraints", "server_api"],
        "reality_refs": ["reality_arc3_available_actions"],
    },
    {
        "id": "meta_spatial_undo_when_stuck",
        "name": "Spatial meta-rule - undo when stuck",
        "domain": "spatial_strategy",
        "category": "recovery_strategy",
        "description": "Prefer undo as a recovery move when action history is cycling and progress is stagnant.",
        "content": "use undo to break repeated failed action loops in interactive spatial tasks",
        "rpn_program": "ACTION_RING LAST_3 ALL_SAME LEVELS_STAGNANT AND ACTION_UNDO",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 0,
            "condition": "action_cycle_and_no_progress",
        },
        "tags": ["spatial", "meta_rule", "undo", "recovery", "stagnation"],
        "grammar_refs": ["grammar_spatial_undo_recovery"],
        "reality_refs": ["reality_arc3_levels_completed", "reality_arc3_undo_only_state"],
    },
    {
        "id": "meta_spatial_undo_only_constraint",
        "name": "Spatial meta-rule - undo only constraint",
        "domain": "spatial_strategy",
        "category": "constraint_compliance",
        "description": "When undo is the only action allowed, emit undo and treat the state as constrained recovery.",
        "content": "undo-only state forces undo emission in interactive spatial control",
        "rpn_program": "AVAILABLE_ACTIONS [6] EQUALS ACTION_UNDO",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "condition": "undo_only_available",
        },
        "tags": ["spatial", "meta_rule", "undo_only", "constraint"],
        "grammar_refs": ["grammar_spatial_undo_recovery"],
        "reality_refs": ["reality_arc3_undo_only_state", "reality_arc3_available_actions"],
    },
    {
        "id": "meta_spatial_reinforce_on_level_complete",
        "name": "Spatial meta-rule - reinforce on level complete",
        "domain": "spatial_strategy",
        "category": "consolidation",
        "description": "Apply positive consolidation when levels_completed increases so successful paths strengthen during sleep-time.",
        "content": "positive reinforcement for interactive spatial tasks should trigger on level completion increase",
        "rpn_program": "LEVELS_COMPLETED DELTA POSITIVE TERNARY_POSITIVE SLEEP_TIME_CONSOLIDATE",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "rule_strength": 1,
            "signal_value": 1,
            "trigger": "levels_completed_delta_gt_zero",
        },
        "tags": ["spatial", "meta_rule", "sleep_time", "reinforcement"],
        "reality_refs": ["reality_arc3_levels_completed"],
    },
]


ARC3_WORD_STARS: list[dict[str, Any]] = [
    {
        "id": "word_direction_above",
        "name": "Direction meaning - above",
        "domain": "spatial_direction",
        "category": "direction_meaning",
        "content": "Universal spatial meaning for above, up, north, or upward movement.",
        "description": "Meaning-centric star for upward displacement across language, navigation, physics, and interactive tasks.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "displacement": [0, -1],
            "behavior_rpn": "y RECALL dy RECALL - y STORE",
            "query_anchor": "above upward north up direction spatial movement",
        },
        "word_refs": ["above", "north", "top", "up", "upward"],
        "reality_refs": ["reality_action_move_up"],
        "tags": ["spatial", "direction", "above", "up", "north"],
    },
    {
        "id": "word_direction_below",
        "name": "Direction meaning - below",
        "domain": "spatial_direction",
        "category": "direction_meaning",
        "content": "Universal spatial meaning for below, down, south, or downward movement.",
        "description": "Meaning-centric star for downward displacement across language, navigation, physics, and interactive tasks.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "displacement": [0, 1],
            "behavior_rpn": "y RECALL dy RECALL + y STORE",
            "query_anchor": "below downward south down direction spatial movement",
        },
        "word_refs": ["below", "south", "bottom", "down", "downward"],
        "reality_refs": ["reality_action_move_down"],
        "tags": ["spatial", "direction", "below", "down", "south"],
    },
    {
        "id": "word_direction_left",
        "name": "Direction meaning - left",
        "domain": "spatial_direction",
        "category": "direction_meaning",
        "content": "Universal spatial meaning for leftward or westward movement.",
        "description": "Meaning-centric star for negative-X displacement across language, navigation, robotics, and interactive tasks.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "displacement": [-1, 0],
            "behavior_rpn": "x RECALL dx RECALL - x STORE",
            "query_anchor": "left west leftward direction spatial movement",
        },
        "word_refs": ["left", "west", "leftward"],
        "reality_refs": ["reality_action_move_left"],
        "tags": ["spatial", "direction", "left", "west"],
    },
    {
        "id": "word_direction_right",
        "name": "Direction meaning - right",
        "domain": "spatial_direction",
        "category": "direction_meaning",
        "content": "Universal spatial meaning for rightward or eastward movement.",
        "description": "Meaning-centric star for positive-X displacement across language, navigation, robotics, and interactive tasks.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "displacement": [1, 0],
            "behavior_rpn": "x RECALL dx RECALL + x STORE",
            "query_anchor": "right east rightward direction spatial movement",
        },
        "word_refs": ["right", "east", "rightward"],
        "reality_refs": ["reality_action_move_right"],
        "tags": ["spatial", "direction", "right", "east"],
    },
    {
        "id": "word_direction_centered",
        "name": "Direction meaning - centered",
        "domain": "spatial_direction",
        "category": "direction_meaning",
        "content": "Universal spatial meaning for centered, aligned, or target reached.",
        "description": "Meaning-centric star for zero displacement or being aligned with the target before interaction.",
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "displacement": [0, 0],
            "behavior_rpn": "state RECALL action_fn RECALL STORE",
            "query_anchor": "centered aligned target reached spatial perform interact",
        },
        "word_refs": ["centered", "aligned", "at target", "balanced"],
        "reality_refs": ["reality_action_perform"],
        "tags": ["spatial", "direction", "centered", "aligned", "target"],
    },
]


ARC3_GALAXY_PAYLOADS: dict[str, list[dict[str, Any]]] = {
    "Grammar.jsonl": ARC3_GRAMMAR_RULES,
    "Reality.jsonl": ARC3_REALITY_ENTRIES,
    "Tool.jsonl": ARC3_META_RULES,
    "Word.jsonl": ARC3_WORD_STARS,
}

LEGACY_ROW_IDS: dict[str, list[str]] = {
    "Grammar.jsonl": [
        "arc3_nav_move_up",
        "arc3_nav_move_down",
        "arc3_nav_move_left",
        "arc3_nav_move_right",
        "arc3_nav_perform",
        "arc3_nav_click",
        "arc3_nav_undo",
        "arc3_rule_keyboard_game",
        "arc3_rule_click_game",
        "arc3_rule_keyboard_click_game",
        "arc3_rule_detect_background",
        "arc3_rule_detect_foreground_components",
        "arc3_rule_foreground_centroid",
    ],
    "Tool.jsonl": [
        "meta_arc3_seek_goal_when_present",
        "meta_arc3_seek_centroid_when_no_goal",
        "meta_arc3_use_click_coordinates",
        "meta_arc3_respect_available_actions",
        "meta_arc3_undo_when_stuck",
        "meta_arc3_undo_only_constraint",
        "meta_arc3_reinforce_on_level_complete",
    ],
    "Word.jsonl": [
        "word_arc3_navigate_above",
        "word_arc3_navigate_below",
        "word_arc3_navigate_left",
        "word_arc3_navigate_right",
        "word_arc3_navigate_centered",
    ],
}


def _make_entry(entry_def: dict[str, Any]) -> dict[str, Any]:
    return dict(entry_def)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def _upsert_entries(
    path: Path,
    generated: list[dict[str, Any]],
    *,
    remove_ids: list[str] | None = None,
) -> dict[str, int]:
    existing = _read_jsonl(path)
    generated_rows = {
        str(row.get("id", "")).strip(): _make_entry(row)
        for row in generated
        if str(row.get("id", "")).strip()
    }
    remove_id_set = {str(row_id).strip() for row_id in list(remove_ids or []) if str(row_id).strip()}
    merged_rows: list[dict[str, Any]] = []
    replaced = 0
    appended = 0
    removed = 0
    seen_ids: set[str] = set()
    for row in existing:
        row_id = str(row.get("id", "")).strip()
        if row_id and row_id in generated_rows:
            merged_rows.append(generated_rows[row_id])
            seen_ids.add(row_id)
            replaced += 1
            continue
        if row_id and row_id in remove_id_set:
            removed += 1
            continue
        merged_rows.append(row)
        if row_id:
            seen_ids.add(row_id)
    for row_id, row in generated_rows.items():
        if row_id in seen_ids:
            continue
        merged_rows.append(row)
        appended += 1
    _write_jsonl(path, merged_rows)
    return {
        "before": len(existing),
        "generated": len(generated),
        "replaced": replaced,
        "appended": appended,
        "removed": removed,
        "after": len(merged_rows),
    }


def build_arc3_galaxy_knowledge(galaxy_dir: Path | str | None = None) -> dict[str, dict[str, int]]:
    target_dir = Path(galaxy_dir or GALAXY_DIR)
    results: dict[str, dict[str, int]] = {}
    for filename, payload in ARC3_GALAXY_PAYLOADS.items():
        results[filename] = _upsert_entries(
            target_dir / filename,
            payload,
            remove_ids=LEGACY_ROW_IDS.get(filename),
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append spatial and ARC-AGI-3 knowledge to Galaxy JSONL files.")
    parser.add_argument(
        "--galaxy-dir",
        default=str(GALAXY_DIR),
        help="Directory containing Grammar.jsonl, Reality.jsonl, Tool.jsonl, and Word.jsonl.",
    )
    args = parser.parse_args(argv)
    results = build_arc3_galaxy_knowledge(galaxy_dir=args.galaxy_dir)
    for filename in ("Grammar.jsonl", "Reality.jsonl", "Tool.jsonl", "Word.jsonl"):
        stats = results.get(filename, {})
        print(
            f"{filename}: replaced={stats.get('replaced', 0)} appended={stats.get('appended', 0)} "
            f"removed={stats.get('removed', 0)} before={stats.get('before', 0)} after={stats.get('after', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
