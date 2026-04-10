"""Seed foundational 2D game mechanics knowledge for the ARC-3 living path."""

from __future__ import annotations

from contextlib import nullcontext

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


GAME_MECHANICS_STARS: tuple[tuple[str, MeaningCentricStar, str], ...] = (
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:movement_budget",
            meaning_class="game_concept",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="MOVEMENT_BUDGET LIMITED_RESOURCE YELLOW_BAR CONSERVE_MOVES",
            taxonomy_refs=["arc3", "game_mechanics", "movement_budget"],
            reality_refs=["arc3", "movement_budget"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Movement Budget",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:lives_system",
            meaning_class="game_concept",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="LIVES_SYSTEM FINITE_ATTEMPTS RED_SQUARES AVOID_DEATH",
            taxonomy_refs=["arc3", "game_mechanics", "lives_system"],
            reality_refs=["arc3", "lives_system"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Lives System",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:level_goal",
            meaning_class="game_concept",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="LEVEL_GOAL REACH_TARGET COLLECT_OBJECT ADVANCE_LEVEL",
            taxonomy_refs=["arc3", "game_mechanics", "level_goal"],
            reality_refs=["arc3", "level_goal"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Level Goal",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:strategic_reset",
            meaning_class="game_concept",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="STRATEGIC_RESET ACTION7 LIFE_TRADE ESCAPE_TRAP",
            taxonomy_refs=["arc3", "game_mechanics", "strategic_reset"],
            reality_refs=["arc3", "strategic_reset"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Strategic Reset",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:exploration_vs_exploitation",
            meaning_class="game_concept",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="EXPLORE MAP_OBJECTS THEN EXPLOIT SAFE_PATHS",
            taxonomy_refs=["arc3", "game_mechanics", "exploration"],
            reality_refs=["arc3", "exploration_vs_exploitation"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Exploration vs Exploitation",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:action_refill",
            meaning_class="game_concept",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="ACTION_ITEM COLLECT YELLOW_BAR REFILL BUDGET_INCREASE",
            taxonomy_refs=["arc3", "game_mechanics", "action_refill"],
            reality_refs=["arc3", "action_refill", "yellow_bar"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Action Refill",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:key_switch",
            meaning_class="game_object",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="WHITE_CROSS GRID_ARTIFACT KEY_SWITCH INTERACT ACTION5 TOGGLE_KEY_ORIENTATION",
            taxonomy_refs=["arc3", "ls20", "key_switch", "white_cross"],
            reality_refs=["arc3", "key_switch", "artifact", "interact"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Key Switch (White Cross)",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:door_indicator",
            meaning_class="game_object",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="UPPER_BLUE_FIGURE DOOR_LOCATION NEXT_LEVEL_PORTAL KEY_ORIENTATION_TARGET",
            taxonomy_refs=["arc3", "ls20", "door", "blue_figure", "key_target"],
            reality_refs=["arc3", "door", "key_target", "level_exit"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Door Indicator (Upper Blue Figure)",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_game_mechanic:key_state_display",
            meaning_class="game_ui",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn="LEFT_STRIP_DISPLAY CURRENT_KEY_STATE ORIENTATION COLOR SHAPE FEEDBACK",
            taxonomy_refs=["arc3", "ls20", "ui", "key_display", "key_state"],
            reality_refs=["arc3", "key_state", "display", "ui_feedback"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Key State Display (Bottom-Left Strip)",
    ),
    (
        "Reality",
        MeaningCentricStar(
            star_id="arc3_ls20_strategy:level1_sequence",
            meaning_class="game_strategy",
            domain="reality",
            galaxy_ref="Reality",
            meaning_rpn=(
                "NAVIGATE_TO_KEY_SWITCH "
                "INTERACT_KEY_SWITCH ACTION5 "
                "CHECK_KEY_STATE_MATCHES_DOOR "
                "IF_NOT_MATCH INTERACT_AGAIN "
                "NAVIGATE_TO_DOOR_UPPER_BLUE_FIGURE "
                "ENTER_DOOR LEVEL_COMPLETE"
            ),
            taxonomy_refs=["arc3", "ls20", "level1", "winning_strategy"],
            reality_refs=["arc3", "ls20", "strategy", "level1"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "LS20 Level 1 Winning Strategy",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:move_into_obstacle",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="DIRECTIONAL_ACTION ADJACENT_SOLID",
            behavior_rpn="NO_POSITION_CHANGE",
            taxonomy_refs=["arc3", "game_rule", "blocked"],
            grammar_refs=["arc3", "movement", "obstacle"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Move Into Obstacle",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:move_into_empty",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="DIRECTIONAL_ACTION ADJACENT_WALKABLE",
            behavior_rpn="POSITION_CHANGES_BY_ACTION_VECTOR",
            taxonomy_refs=["arc3", "game_rule", "movement"],
            grammar_refs=["arc3", "movement", "walkable"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Move Into Empty",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:reach_goal",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="AGENT_POSITION_EQUALS_GOAL",
            behavior_rpn="LEVEL_COMPLETE_REWARD_PLUS_ONE",
            taxonomy_refs=["arc3", "game_rule", "goal"],
            grammar_refs=["arc3", "goal", "reward"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Reach Goal",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:contact_hazard",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="AGENT_MOVES_INTO_HAZARD",
            behavior_rpn="DEATH_LIVES_MINUS_ONE",
            taxonomy_refs=["arc3", "game_rule", "hazard"],
            grammar_refs=["arc3", "hazard", "death"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Contact Hazard",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:loop_detection",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="SAME_GRID_STATE_VISITED_THREE_TIMES",
            behavior_rpn="CURRENT_PATH_DEAD_END_TRY_DIFFERENT_ACTION",
            taxonomy_refs=["arc3", "game_rule", "loop_detection"],
            grammar_refs=["arc3", "loop_detection", "dead_end"],
            meta_refs=["bootstrap:arc3_game_mechanics_v1"],
            confidence=1,
            polarity=1,
        ),
        "Loop Detection",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:key_switch_interaction",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="AGENT_AT_WHITE_CROSS ACTION5",
            behavior_rpn="KEY_ORIENTATION_TOGGLES REPEAT_UNTIL_MATCHES_DOOR_TARGET",
            taxonomy_refs=["arc3", "ls20", "key_switch", "game_rule"],
            grammar_refs=["arc3", "key_switch", "orient", "interact"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Rule: Key Switch Interaction",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:door_entry_condition",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="KEY_ORIENTATION_MATCHES_DOOR_TARGET AND AGENT_AT_DOOR_POSITION",
            behavior_rpn="LEVEL_COMPLETE_ADVANCE_TO_NEXT",
            taxonomy_refs=["arc3", "ls20", "door", "win_condition"],
            grammar_refs=["arc3", "door", "key_match", "level_complete"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Rule: Door Entry Condition",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:door_entry_blocked",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="KEY_ORIENTATION_NOT_MATCHING AND AGENT_AT_DOOR_POSITION",
            behavior_rpn="BLOCKED_RETURN_TO_KEY_SWITCH_INTERACT_AGAIN",
            taxonomy_refs=["arc3", "ls20", "door", "blocked", "key_mismatch"],
            grammar_refs=["arc3", "door", "blocked", "key_orient_required"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Rule: Door Blocked When Key Mismatched",
    ),
    (
        "Grammar",
        MeaningCentricStar(
            star_id="arc3_game_rule:multi_step_artifact",
            meaning_class="game_rule",
            domain="grammar",
            galaxy_ref="Grammar",
            meaning_rpn="ARTIFACT_INTERACTION ACTION5 REPEAT CHECK_KEY_STATE",
            behavior_rpn="CONTINUE_INTERACTING_UNTIL_KEY_STATE_MATCHES_TARGET",
            taxonomy_refs=["arc3", "ls20", "artifact", "multi_step", "interact"],
            grammar_refs=["arc3", "artifact", "multi_step", "iterate"],
            meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
            confidence=1,
            polarity=1,
        ),
        "Rule: Multi-Step Artifact Interaction",
    ),
)


def seed_game_mechanics(knowledgeverse) -> int:
    galaxy_manager = getattr(knowledgeverse, "galaxy_manager", None)
    if galaxy_manager is None or not hasattr(galaxy_manager, "store_meaning_star"):
        return 0
    sync_context = galaxy_manager.bulk_disk_sync() if hasattr(galaxy_manager, "bulk_disk_sync") else nullcontext()
    with sync_context:
        for galaxy_name, star, label in GAME_MECHANICS_STARS:
            galaxy_manager.store_meaning_star(
                galaxy_name,
                star,
                name=label,
                category=str(star.meaning_class or "game_concept"),
                metadata={"bootstrap": "arc3_game_mechanics_v1"},
            )
    return len(GAME_MECHANICS_STARS)


__all__ = ["GAME_MECHANICS_STARS", "seed_game_mechanics"]
