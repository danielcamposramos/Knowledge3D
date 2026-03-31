from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402


BOOTSTRAP_TAG = "phase_e38_four_way_reading_v1"
DEFAULT_HOUSE_DIR = Path("/K3D/Knowledge3D.local/house")


def _surface_forms(*forms: str) -> dict[str, Any]:
    ordered = [str(form).strip() for form in forms if str(form).strip()]
    primary = (ordered[0] if ordered else "reasoning strategy").lower().replace(" ", "_")
    return {
        "en": {"word_ref": f"word_{primary}", "char_refs": [], "surface_text": ordered or [primary]},
        "pt": {"word_ref": f"word_{primary}", "char_refs": [], "surface_text": ordered or [primary]},
    }


def _meaning_entry(
    entry_id: str,
    name: str,
    description: str,
    *,
    surface_forms: list[str],
    meaning_rpn: str,
    behavior_rpn: str,
    grammar_refs: list[str] | None = None,
    reality_refs: list[str] | None = None,
    meta_refs: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "star_id": entry_id,
        "name": name,
        "galaxy": "reasoning_strategies",
        "domain": "reasoning",
        "category": "reasoning_strategy",
        "layer": 2,
        "content": description,
        "summary": description,
        "description": description,
        "meaning_rpn": meaning_rpn,
        "behavior_rpn": behavior_rpn,
        "surface_forms": _surface_forms(*surface_forms),
        "grammar_refs": list(grammar_refs or []),
        "reality_refs": list(reality_refs or []),
        "meta_refs": list(meta_refs or []),
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "knowledge_category": "reasoning",
            "meaning_rpn": meaning_rpn,
            "behavior_rpn": behavior_rpn,
            "grammar_refs": list(grammar_refs or []),
            "reality_refs": list(reality_refs or []),
            "meta_refs": list(meta_refs or []),
        },
        "tags": list(tags or []),
    }


def build_reasoning_meaning_entries() -> list[dict[str, Any]]:
    return [
        _meaning_entry(
            "forward_entity_extraction",
            "Forward Entity Extraction",
            "Read a problem left-to-right and bind quantities, units, entities, and action verbs into working memory.",
            surface_forms=["forward read", "entity extraction", "quantity binding"],
            meaning_rpn="TOKENS FOREACH IF NOUN THEN GALAXY_LOOKUP IF NUMBER THEN NUM_BIND IF VERB THEN ACTION_BIND",
            behavior_rpn="INPUT_SCAN LEFT_TO_RIGHT QUANTITY_BIND ENTITY_BIND ACTION_BIND",
            grammar_refs=["grammar_forward_entity_extraction", "grammar_quantity_unit_binding"],
            reality_refs=["reality_word_problem_goal_state", "reality_dependency_dag"],
            meta_refs=["meta_four_way_reading_strategy"],
            tags=["reasoning", "math", "reading", "forward"],
        ),
        _meaning_entry(
            "backward_goal_tracing",
            "Backward Goal Tracing",
            "Start from the asked goal and trace backward through dependencies to determine which quantities and operations are required.",
            surface_forms=["backward read", "goal tracing", "dependency backtrace"],
            meaning_rpn="QUESTION_ENTITY GOAL_BIND DEPENDENCY_TRACE REVERSE",
            behavior_rpn="GOAL_IDENTIFY TRACE_BACKWARD REQUIRED_QUANTITIES_MARK",
            grammar_refs=["grammar_backward_goal_tracing", "grammar_dependency_dag_build"],
            reality_refs=["reality_word_problem_goal_state", "reality_dependency_dag"],
            meta_refs=["meta_apply_backward_trace_before_emit"],
            tags=["reasoning", "math", "reading", "backward"],
        ),
        _meaning_entry(
            "operation_chain_construction",
            "Operation Chain Construction",
            "Compose a multi-step RPN program from the dependency graph instead of applying a single left-fold operation.",
            surface_forms=["operation chain", "multi-step program", "dependency execution"],
            meaning_rpn="DEPENDENCY_DAG TOPO_SORT OPERATION_BIND STORE RECALL CHAIN",
            behavior_rpn="SUBTASKS ORDER BUILD_RPN_CHAIN EXECUTE",
            grammar_refs=["grammar_operation_chain_construction", "grammar_recursive_subtask_decomposition"],
            reality_refs=["reality_operation_chain", "reality_dependency_dag"],
            meta_refs=["meta_decompose_multi_step_word_problem"],
            tags=["reasoning", "math", "chain", "rpn"],
        ),
        _meaning_entry(
            "result_normalization_validation",
            "Result Normalization Validation",
            "Validate intermediate and final results for unit consistency, magnitude plausibility, and complete quantity consumption.",
            surface_forms=["result validation", "normalization", "unit check"],
            meaning_rpn="INTERMEDIATE_RESULTS UNIT_CHECK MAGNITUDE_CHECK ORPHAN_CHECK VALIDATE",
            behavior_rpn="UNITS_COMPARE MAGNITUDE_COMPARE ANSWER_SANITY_GATE",
            grammar_refs=["grammar_result_normalization", "grammar_validate_units_and_magnitude"],
            reality_refs=["reality_unit_consistency", "reality_operation_chain"],
            meta_refs=["meta_validate_units_before_answer"],
            tags=["reasoning", "math", "validation", "normalization"],
        ),
        _meaning_entry(
            "word_problem_multi_step_reasoning",
            "Word Problem Multi-Step Reasoning",
            "A four-pass reasoning procedure for multi-step question solving: forward extraction, backward tracing, chain construction, and validation.",
            surface_forms=["four-way reading", "multi-step word problem", "recursive decomposition"],
            meaning_rpn="forward_entity_extraction CALL backward_goal_tracing CALL operation_chain_construction CALL result_normalization_validation CALL",
            behavior_rpn="FOUR_PASS_REASONING EXECUTE UNTIL_CONVERGED",
            grammar_refs=[
                "grammar_forward_entity_extraction",
                "grammar_backward_goal_tracing",
                "grammar_operation_chain_construction",
                "grammar_result_normalization",
            ],
            reality_refs=["reality_dependency_dag", "reality_operation_chain", "reality_unit_consistency"],
            meta_refs=["meta_four_way_reading_strategy", "meta_decompose_multi_step_word_problem"],
            tags=["reasoning", "math", "word_problem", "meta"],
        ),
    ]


def build_reasoning_reality_entries() -> list[dict[str, Any]]:
    return [
        {
            "id": "reality_word_problem_goal_state",
            "name": "Word Problem Goal State",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "goal_state",
            "content": "The explicit question target that determines what quantity or proof obligation must be produced.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "goal", "word_problem"],
        },
        {
            "id": "reality_dependency_dag",
            "name": "Reasoning Dependency DAG",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "dependency_structure",
            "content": "A directed acyclic graph of intermediate computations needed to reach a final answer.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "dag", "dependencies"],
        },
        {
            "id": "reality_operation_chain",
            "name": "Operation Chain",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "execution_structure",
            "content": "A multi-step ordered sequence of operations composed from the dependency graph and executed through RPN registers.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "rpn", "chain"],
        },
        {
            "id": "reality_unit_consistency",
            "name": "Unit Consistency",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "validation",
            "content": "A validation state asserting that units, scales, and answer type remain consistent across the reasoning chain.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "units", "validation"],
        },
    ]


def build_reasoning_grammar_rules() -> list[dict[str, Any]]:
    return [
        {
            "id": "grammar_forward_entity_extraction",
            "name": "Forward Entity Extraction",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Read the prompt left-to-right and bind nouns, numbers, units, and verbs into working slots.",
            "rpn_program": "TOKENS FOREACH IF_NOUN GALAXY_LOOKUP IF_NUMBER NUM_BIND IF_UNIT UNIT_BIND IF_VERB ACTION_BIND",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "reading", "forward"],
        },
        {
            "id": "grammar_quantity_unit_binding",
            "name": "Quantity Unit Binding",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Bind quantities to their local units and entities before arithmetic composition.",
            "rpn_program": "QUANTITY UNIT ENTITY BIND_TRIPLE STORE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "quantity", "units"],
        },
        {
            "id": "grammar_backward_goal_tracing",
            "name": "Backward Goal Tracing",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Trace backward from the requested answer type through the required dependencies.",
            "rpn_program": "QUESTION GOAL_BIND DEPENDENCY_TRACE REVERSE_STORE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "reading", "backward"],
        },
        {
            "id": "grammar_dependency_dag_build",
            "name": "Dependency DAG Build",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Construct a DAG of intermediate operations required by the final goal.",
            "rpn_program": "BOUND_QUANTITIES GOAL_STATE DEPENDENCY_DAG_BUILD",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "dag", "dependencies"],
        },
        {
            "id": "grammar_operation_chain_construction",
            "name": "Operation Chain Construction",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Topologically sort the dependency DAG and emit a multi-step RPN chain using STORE and RECALL.",
            "rpn_program": "DEPENDENCY_DAG TOPO_SORT FOREACH OP_BIND STORE RECALL CHAIN_BUILD",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "rpn", "chain"],
        },
        {
            "id": "grammar_recursive_subtask_decomposition",
            "name": "Recursive Subtask Decomposition",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Break a multi-step problem into sub-tasks, solve them individually, and compose their results.",
            "rpn_program": "QUERY PARTIAL_KNOWLEDGE DECOMPOSE SUBTASKS FOREACH SOLVE_SUBTASK COMPOSE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "recursive", "decomposition"],
        },
        {
            "id": "grammar_result_normalization",
            "name": "Result Normalization",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "validation_rule",
            "content": "Normalize the answer form, units, and magnitude before emitting a final result.",
            "rpn_program": "RESULT NORMALIZE_FORMAT UNIT_NORMALIZE MAGNITUDE_NORMALIZE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "normalization", "validation"],
        },
        {
            "id": "grammar_validate_units_and_magnitude",
            "name": "Validate Units And Magnitude",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "validation_rule",
            "content": "Check that units are consistent and the resulting magnitude is plausible for the requested answer.",
            "rpn_program": "UNITS_COMPARE MAGNITUDE_COMPARE PLAUSIBILITY_GATE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "units", "magnitude", "validation"],
        },
    ]


def build_reasoning_meta_rules() -> list[dict[str, Any]]:
    return [
        {
            "id": "meta_four_way_reading_strategy",
            "name": "Four Way Reading Strategy",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "For multi-step word problems, apply forward extraction, backward tracing, chain construction, then normalization.",
            "rpn_program": "IF WORD_PROBLEM THEN grammar_forward_entity_extraction CALL grammar_backward_goal_tracing CALL grammar_operation_chain_construction CALL grammar_result_normalization CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "word_problem"],
        },
        {
            "id": "meta_decompose_multi_step_word_problem",
            "name": "Decompose Multi-Step Word Problem",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "When direct resolution fails, decompose the problem into sub-tasks and solve recursively.",
            "rpn_program": "IF DIRECT_RESOLUTION_FAIL THEN grammar_recursive_subtask_decomposition CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "decomposition"],
        },
        {
            "id": "meta_apply_backward_trace_before_emit",
            "name": "Apply Backward Trace Before Emit",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Prevent answer emission until backward goal tracing has identified the required dependencies.",
            "rpn_program": "IF GOAL_DEPENDENCIES_MISSING THEN grammar_backward_goal_tracing CALL HOLD_EMIT",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "goal_trace"],
        },
        {
            "id": "meta_validate_units_before_answer",
            "name": "Validate Units Before Answer",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Require unit and magnitude validation before the final answer leaves the reasoning loop.",
            "rpn_program": "IF ANSWER_READY THEN grammar_validate_units_and_magnitude CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "validation"],
        },
    ]


def populate_reasoning_strategies(house_dir: str | Path = DEFAULT_HOUSE_DIR) -> dict[str, dict[str, int]]:
    root = Path(house_dir)
    root.mkdir(parents=True, exist_ok=True)
    stats = {
        "reasoning_strategies.jsonl": upsert_entries(root / "reasoning_strategies.jsonl", build_reasoning_meaning_entries()),
        "Reality.jsonl": upsert_entries(root / "Reality.jsonl", build_reasoning_reality_entries()),
        "Grammar.jsonl": upsert_entries(root / "Grammar.jsonl", build_reasoning_grammar_rules()),
        "Tool.jsonl": upsert_entries(root / "Tool.jsonl", build_reasoning_meta_rules()),
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate four-way reasoning strategy stars into the House")
    parser.add_argument("--house-dir", default=str(DEFAULT_HOUSE_DIR))
    args = parser.parse_args()
    stats = populate_reasoning_strategies(house_dir=args.house_dir)
    for name, payload in stats.items():
        print(
            f"{name}: before={payload['before']} after={payload['after']} "
            f"appended={payload['appended']} replaced={payload['replaced']} removed={payload['removed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
