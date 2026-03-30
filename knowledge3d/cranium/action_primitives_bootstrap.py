"""Bootstrap reusable spatial action primitives into Reality Galaxy."""

from __future__ import annotations

from pathlib import Path

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_nodes import RealityAtom


ACTION_MOVE_UP = "atom:action:move_up"
ACTION_MOVE_DOWN = "atom:action:move_down"
ACTION_MOVE_LEFT = "atom:action:move_left"
ACTION_MOVE_RIGHT = "atom:action:move_right"
ACTION_PERFORM = "atom:action:perform"
ACTION_CLICK = "atom:action:click"
ACTION_UNDO = "atom:action:undo"
ACTION_DIAGONAL_UR = "atom:action:move_diagonal_ur"
ACTION_REACH = "atom:action:reach"
ACTION_GRAB = "atom:action:grab"
ACTION_HOLD = "atom:action:hold"
ACTION_RELEASE = "atom:action:release"
ACTION_USE = "atom:action:use"
ACTION_WALK_TO = "atom:action:walk_to"
ACTION_TELEPORT = "atom:action:teleport"
ACTION_LOOK_AT = "atom:action:look_at"


def spatial_action_atoms() -> list[RealityAtom]:
    """Return the canonical reusable spatial action atoms."""
    return [
        RealityAtom(
            node_id=ACTION_MOVE_UP,
            visual_rpn="0 0 DRAW_MOVE 0 -1 DRAW_LINE DRAW_STROKE",
            behavior_rpn="y RECALL dy RECALL - y STORE",
            law_rpn="y RECALL y_min RECALL GTE",
            metadata={
                "description": "Move one unit in negative-Y direction",
                "displacement": [0, -1],
                "action_type": "spatial_translation",
                "arc3_action": "ACTION1",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "move up", "pt": "mover para cima"},
                "inverse": ACTION_MOVE_DOWN,
                "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
            },
        ),
        RealityAtom(
            node_id=ACTION_MOVE_DOWN,
            visual_rpn="0 0 DRAW_MOVE 0 1 DRAW_LINE DRAW_STROKE",
            behavior_rpn="y RECALL dy RECALL + y STORE",
            law_rpn="y RECALL y_max RECALL LT",
            metadata={
                "description": "Move one unit in positive-Y direction",
                "displacement": [0, 1],
                "action_type": "spatial_translation",
                "arc3_action": "ACTION2",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "move down", "pt": "mover para baixo"},
                "inverse": ACTION_MOVE_UP,
                "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
            },
        ),
        RealityAtom(
            node_id=ACTION_MOVE_LEFT,
            visual_rpn="0 0 DRAW_MOVE -1 0 DRAW_LINE DRAW_STROKE",
            behavior_rpn="x RECALL dx RECALL - x STORE",
            law_rpn="x RECALL x_min RECALL GTE",
            metadata={
                "description": "Move one unit in negative-X direction",
                "displacement": [-1, 0],
                "action_type": "spatial_translation",
                "arc3_action": "ACTION3",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "move left", "pt": "mover para a esquerda"},
                "inverse": ACTION_MOVE_RIGHT,
                "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
            },
        ),
        RealityAtom(
            node_id=ACTION_MOVE_RIGHT,
            visual_rpn="0 0 DRAW_MOVE 1 0 DRAW_LINE DRAW_STROKE",
            behavior_rpn="x RECALL dx RECALL + x STORE",
            law_rpn="x RECALL x_max RECALL LT",
            metadata={
                "description": "Move one unit in positive-X direction",
                "displacement": [1, 0],
                "action_type": "spatial_translation",
                "arc3_action": "ACTION4",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "move right", "pt": "mover para a direita"},
                "inverse": ACTION_MOVE_LEFT,
                "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
            },
        ),
        RealityAtom(
            node_id=ACTION_PERFORM,
            visual_rpn="0 0 0.1 circle fill",
            behavior_rpn="state RECALL action_fn RECALL STORE",
            law_rpn="action_available RECALL 1 EQ",
            metadata={
                "description": "Execute the context-dependent action at current position",
                "displacement": [0, 0],
                "action_type": "spatial_interaction",
                "arc3_action": "ACTION5",
                "hanim_anchor": "r_hand_tip",
                "surface_forms": {"en": "perform action", "pt": "executar ação"},
                "inverse": ACTION_UNDO,
                "reusable_contexts": ["arc3", "house_navigation", "grid_world"],
            },
        ),
        RealityAtom(
            node_id=ACTION_CLICK,
            visual_rpn="target_x target_y 0.05 circle fill",
            behavior_rpn="target_x RECALL target_y RECALL click_fn RECALL STORE",
            law_rpn="target_x RECALL x_max RECALL LT target_y RECALL y_max RECALL LT AND",
            metadata={
                "description": "Click at specific coordinates",
                "displacement": [0, 0],
                "action_type": "spatial_selection",
                "arc3_action": "ACTION6",
                "hanim_anchor": "r_hand_tip",
                "surface_forms": {"en": "click", "pt": "clicar"},
                "parameterized": True,
                "parameters": ["x", "y"],
                "reusable_contexts": ["arc3", "house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_UNDO,
            visual_rpn="0 0 DRAW_MOVE -0.3 0.3 DRAW_LINE -0.3 -0.3 DRAW_LINE DRAW_STROKE",
            behavior_rpn="history RECALL -1 index state STORE history RECALL pop STORE",
            law_rpn="history_length RECALL 0 GT",
            metadata={
                "description": "Undo the previous action",
                "displacement": [0, 0],
                "action_type": "temporal_reversal",
                "arc3_action": "ACTION7",
                "hanim_anchor": None,
                "surface_forms": {"en": "undo", "pt": "desfazer"},
                "inverse": None,
                "reusable_contexts": ["arc3", "house_navigation", "grid_world"],
            },
        ),
        RealityAtom(
            node_id=ACTION_DIAGONAL_UR,
            component_refs=[ACTION_MOVE_UP, ACTION_MOVE_RIGHT],
            visual_rpn="0 0 DRAW_MOVE 1 -1 DRAW_LINE DRAW_STROKE",
            behavior_rpn="x RECALL dx RECALL + x STORE y RECALL dy RECALL - y STORE",
            law_rpn="x RECALL x_max RECALL LT y RECALL y_min RECALL GTE AND",
            metadata={
                "description": "Diagonal move: up + right",
                "displacement": [1, -1],
                "action_type": "spatial_translation_composed",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "move diagonally up-right"},
                "reusable_contexts": ["house_navigation", "grid_world", "physics_sim"],
            },
        ),
        RealityAtom(
            node_id=ACTION_REACH,
            visual_rpn="0 0 DRAW_MOVE 0.5 0 DRAW_LINE DRAW_STROKE",
            behavior_rpn="target_pos RECALL hand_pos RECALL - reach_vec STORE",
            law_rpn="reach_vec RECALL VEC_L2_NORM arm_length RECALL LT",
            metadata={
                "description": "Extend hand toward target",
                "displacement": [0, 0],
                "action_type": "object_interaction",
                "hanim_anchor": "r_hand_tip",
                "surface_forms": {"en": "reach", "pt": "alcançar"},
                "reusable_contexts": ["house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_GRAB,
            component_refs=[ACTION_REACH],
            visual_rpn="0 0 0.08 circle fill",
            behavior_rpn="target_obj RECALL held_object STORE",
            law_rpn="reach_vec RECALL VEC_L2_NORM grip_range RECALL LT",
            metadata={
                "description": "Close hand on reachable object",
                "displacement": [0, 0],
                "action_type": "object_interaction",
                "hanim_anchor": "l_radiocarpal",
                "surface_forms": {"en": "grab", "pt": "agarrar"},
                "inverse": ACTION_RELEASE,
                "reusable_contexts": ["house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_HOLD,
            component_refs=[ACTION_GRAB],
            visual_rpn="0 0 0.08 circle stroke",
            behavior_rpn="held_object RECALL hand_pos RECALL STORE",
            law_rpn="held_object RECALL null EQ NOT",
            metadata={
                "description": "Maintain grip on held object",
                "displacement": [0, 0],
                "action_type": "object_interaction",
                "hanim_anchor": "k3d_tablet_grip",
                "surface_forms": {"en": "hold", "pt": "segurar"},
                "reusable_contexts": ["house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_RELEASE,
            visual_rpn="0 0 DRAW_MOVE 0.1 0.1 DRAW_LINE -0.1 0.1 DRAW_LINE DRAW_STROKE",
            behavior_rpn="held_object RECALL DROP",
            law_rpn="held_object RECALL null EQ NOT",
            metadata={
                "description": "Release held object",
                "displacement": [0, 0],
                "action_type": "object_interaction",
                "hanim_anchor": "l_radiocarpal",
                "surface_forms": {"en": "release", "pt": "soltar"},
                "inverse": ACTION_GRAB,
                "reusable_contexts": ["house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_USE,
            component_refs=[ACTION_HOLD],
            visual_rpn="0 0 0.06 circle fill 0.1 0 0.03 circle fill",
            behavior_rpn="held_object RECALL use_fn RECALL STORE",
            law_rpn="held_object RECALL usable RECALL AND",
            metadata={
                "description": "Trigger held object behavior (open book, activate tool)",
                "displacement": [0, 0],
                "action_type": "object_interaction",
                "hanim_anchor": "r_hand_tip",
                "surface_forms": {"en": "use", "pt": "usar"},
                "house_triggers": {"book": "load_galaxy", "door": "network_traverse", "tool": "tool_dispatch"},
                "reusable_contexts": ["house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_WALK_TO,
            component_refs=[ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT],
            visual_rpn="0 0 DRAW_MOVE target_x target_y DRAW_LINE DRAW_STROKE",
            behavior_rpn="target_pos RECALL current_pos RECALL - path_vec STORE",
            law_rpn="target_pos RECALL current_pos RECALL - VEC_L2_NORM 0 GT",
            metadata={
                "description": "Navigate to target via LED-A* (composed from cardinal moves)",
                "displacement": [0, 0],
                "action_type": "spatial_navigation_composed",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "walk to", "pt": "caminhar até"},
                "parameterized": True,
                "parameters": ["target_x", "target_y", "target_z"],
                "reusable_contexts": ["house_navigation", "grid_world"],
            },
        ),
        RealityAtom(
            node_id=ACTION_TELEPORT,
            visual_rpn="0 0 0.15 circle stroke 0 0 0.05 circle fill",
            behavior_rpn="target_pos RECALL current_pos STORE",
            law_rpn="target_pos RECALL accessible RECALL AND",
            metadata={
                "description": "Instant translation to target position",
                "displacement": [0, 0],
                "action_type": "spatial_navigation",
                "hanim_anchor": "humanoid_root",
                "surface_forms": {"en": "teleport", "pt": "teletransportar"},
                "reusable_contexts": ["house_navigation"],
            },
        ),
        RealityAtom(
            node_id=ACTION_LOOK_AT,
            visual_rpn="0 0 DRAW_MOVE 0.3 0 DRAW_LINE DRAW_STROKE",
            behavior_rpn="target_pos RECALL current_pos RECALL - VEC_NORMALIZE head_dir STORE",
            law_rpn="1",
            metadata={
                "description": "Orient head toward target (skullbase rotation)",
                "displacement": [0, 0],
                "action_type": "spatial_orientation",
                "hanim_anchor": "skullbase",
                "surface_forms": {"en": "look at", "pt": "olhar para"},
                "reusable_contexts": ["house_navigation", "arc3"],
            },
        ),
    ]


def bootstrap_spatial_actions(
    galaxy: RealityGalaxy,
    *,
    encode_embedding: bool = True,
) -> list[str]:
    """Load reusable action atoms into the supplied Reality Galaxy."""
    atoms = spatial_action_atoms()

    loaded: list[str] = []
    for atom in atoms:
        galaxy.add_node(atom, encode_embedding=encode_embedding)
        loaded.append(atom.node_id)
    return loaded


def build_default_action_galaxy(
    storage_root: str | Path = "../Knowledge3D.local",
    *,
    encode_embedding: bool = True,
) -> RealityGalaxy:
    """Create or load a default Reality Galaxy with spatial actions present."""
    galaxy = RealityGalaxy(galaxy_path=Path(storage_root) / "reality_galaxy")
    bootstrap_spatial_actions(galaxy, encode_embedding=encode_embedding)
    return galaxy


__all__ = [
    "ACTION_CLICK",
    "ACTION_DIAGONAL_UR",
    "ACTION_GRAB",
    "ACTION_HOLD",
    "ACTION_LOOK_AT",
    "ACTION_MOVE_DOWN",
    "ACTION_MOVE_LEFT",
    "ACTION_MOVE_RIGHT",
    "ACTION_MOVE_UP",
    "ACTION_PERFORM",
    "ACTION_REACH",
    "ACTION_RELEASE",
    "ACTION_TELEPORT",
    "ACTION_UNDO",
    "ACTION_USE",
    "ACTION_WALK_TO",
    "bootstrap_spatial_actions",
    "build_default_action_galaxy",
    "spatial_action_atoms",
]
