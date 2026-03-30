from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402


BOOTSTRAP_TAG = "phase_e39_game_mechanics_v1"
DEFAULT_HOUSE_DIR = Path("/K3D/Knowledge3D.local/house")


def _entry(
    entry_id: str,
    name: str,
    description: str,
    *,
    surface_forms: list[str],
    properties: dict[str, Any] | None = None,
    component_refs: list[str] | None = None,
    visual_refs: list[str] | None = None,
    grammar_refs: list[str] | None = None,
    reality_refs: list[str] | None = None,
    math_refs: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    refs = {
        "component_refs": list(component_refs or []),
        "visual_refs": list(visual_refs or []),
        "grammar_refs": list(grammar_refs or []),
        "reality_refs": list(reality_refs or []),
        "math_refs": list(math_refs or []),
    }
    return {
        "id": entry_id,
        "name": name,
        "galaxy": "game_mechanics",
        "domain": "interactive_game",
        "category": "game_mechanic",
        "layer": 2,
        "content": description,
        "summary": description,
        "description": description,
        **refs,
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "surface_forms": {
                "en": list(surface_forms),
                "pt": list(surface_forms),
            },
            "properties": dict(properties or {}),
            **refs,
        },
        "tags": list(tags or []),
    }


GAME_MECHANICS: list[dict[str, Any]] = [
    _entry(
        "spatial_navigation_grid",
        "Spatial Navigation Grid",
        "Movement on a discrete two-dimensional grid via cardinal directions across walkable terrain.",
        surface_forms=["grid movement", "tile navigation", "discrete steps"],
        properties={"movement_model": "cardinal", "space": "2d_grid"},
        visual_refs=["drawing_grid", "drawing_rect"],
        grammar_refs=["grammar_spatial_move_toward_above", "grammar_spatial_move_toward_below"],
        reality_refs=["reality_position", "reality_translation"],
        tags=["game", "navigation", "grid", "movement"],
    ),
    _entry(
        "switch_actuator",
        "Switch Actuator",
        "Walking over a switch changes the state of a linked object such as a door or bridge.",
        surface_forms=["switch", "trigger", "actuator", "pressure plate"],
        properties={"trigger": "walk_over", "effect": "toggle_target_state"},
        component_refs=["spatial_navigation_grid"],
        visual_refs=["drawing_color_white", "drawing_cross"],
        grammar_refs=["grammar_cause_effect", "grammar_state_transition"],
        reality_refs=["reality_mechanical_switch"],
        tags=["game", "switch", "toggle", "cause_effect"],
    ),
    _entry(
        "lock_key_pattern_match",
        "Lock Key Pattern Match",
        "A locked passage opens when the current key pattern matches the target door pattern.",
        surface_forms=["key fits lock", "pattern match unlock", "shape key"],
        properties={"condition": "pattern_equals_target", "effect": "passage_opens"},
        component_refs=["switch_actuator"],
        visual_refs=["drawing_shape", "drawing_pattern"],
        grammar_refs=["grammar_conditional_gate", "grammar_pattern_match"],
        reality_refs=["reality_lock", "reality_gate"],
        math_refs=["math_equality"],
        tags=["game", "lock", "key", "pattern", "unlock"],
    ),
    _entry(
        "level_progression",
        "Level Progression",
        "Completing the objective advances the game to the next challenge stage.",
        surface_forms=["next level", "stage clear", "level complete"],
        properties={"trigger": "objective_complete", "effect": "advance_stage"},
        component_refs=["lock_key_pattern_match"],
        grammar_refs=["grammar_sequence_progression"],
        reality_refs=["reality_stage_transition"],
        tags=["game", "progression", "level", "objective"],
    ),
    _entry(
        "movement_recharge_block",
        "Movement Recharge Block",
        "Stepping on a yellow recharge block restores the available movement budget.",
        surface_forms=["recharge", "refuel", "energy pickup", "stamina restore"],
        properties={"trigger": "walk_over", "effect": "restore_movement_points", "color_signature": "yellow"},
        component_refs=["spatial_navigation_grid"],
        visual_refs=["drawing_color_yellow"],
        grammar_refs=["grammar_resource_replenishment"],
        reality_refs=["reality_energy_restoration"],
        tags=["game", "resource", "recharge", "movement"],
    ),
    _entry(
        "color_transform_block",
        "Color Transform Block",
        "Stepping on a color transform block changes the entity color and may require repeated steps to reach the target color.",
        surface_forms=["color changer", "paint block", "dye station"],
        properties={"trigger": "walk_over", "effect": "change_color", "cycle": "possible_multi_step"},
        component_refs=["lock_key_pattern_match", "movement_recharge_block"],
        visual_refs=["drawing_color_wheel"],
        grammar_refs=["grammar_cyclic_state_machine", "grammar_state_transition"],
        reality_refs=["reality_color_state"],
        tags=["game", "color", "transform", "state"],
    ),
    _entry(
        "shape_transform_block",
        "Shape Transform Block",
        "Stepping on a transform block changes the entity shape so it can match a target door or slot.",
        surface_forms=["shape changer", "morph block", "transform station"],
        properties={"trigger": "walk_over", "effect": "change_shape"},
        component_refs=["lock_key_pattern_match", "color_transform_block"],
        visual_refs=["drawing_shape"],
        grammar_refs=["grammar_state_transition", "grammar_pattern_match"],
        reality_refs=["reality_shape_state"],
        tags=["game", "shape", "transform", "match"],
    ),
    _entry(
        "no_instruction_discovery",
        "No Instruction Discovery",
        "Rules are not stated explicitly and must be inferred from observed action-to-state changes.",
        surface_forms=["learn by doing", "trial and error", "implicit rules"],
        component_refs=["switch_actuator", "level_progression"],
        grammar_refs=["grammar_inductive_reasoning", "grammar_empirical_observation"],
        reality_refs=["reality_observation"],
        tags=["game", "discovery", "inference", "observation"],
    ),
    _entry(
        "visual_state_encoding",
        "Visual State Encoding",
        "Game state is encoded visually through colors, shapes, and spatial layout rather than explicit text instructions.",
        surface_forms=["visual logic", "color means state", "shape means type"],
        component_refs=["color_transform_block", "shape_transform_block"],
        visual_refs=["drawing_color", "drawing_shape", "drawing_grid"],
        grammar_refs=["grammar_visual_encoding"],
        reality_refs=["reality_state_representation"],
        tags=["game", "visual", "state", "encoding"],
    ),
    _entry(
        "multi_step_state_transform",
        "Multi Step State Transform",
        "Reaching a target state may require repeated or cyclic transformations instead of a single action.",
        surface_forms=["double step", "cyclic transform", "iterative approach"],
        component_refs=["color_transform_block", "shape_transform_block"],
        grammar_refs=["grammar_iteration", "grammar_cyclic_state_machine"],
        math_refs=["math_modular_arithmetic"],
        tags=["game", "multi_step", "iteration", "transform"],
    ),
]


def build_game_mechanics_entries() -> list[dict[str, Any]]:
    rows = [dict(row) for row in GAME_MECHANICS]
    if len(rows) < 10:
        raise RuntimeError(f"Expected at least 10 game mechanic entries, generated {len(rows)}")
    return rows


def populate_game_mechanics(*, house_dir: Path = DEFAULT_HOUSE_DIR) -> dict[str, int]:
    house_dir = Path(house_dir)
    house_dir.mkdir(parents=True, exist_ok=True)
    return upsert_entries(
        house_dir / "game_mechanics.jsonl",
        build_game_mechanics_entries(),
    )


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
    stats = populate_game_mechanics(house_dir=args.house_dir)
    print(
        "game_mechanics.jsonl:"
        f" before={stats['before']}"
        f" after={stats['after']}"
        f" appended={stats['appended']}"
        f" replaced={stats['replaced']}"
        f" removed={stats['removed']}"
    )


if __name__ == "__main__":
    main()
