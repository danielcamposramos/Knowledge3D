from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_meaning_class_routing_from_natural_queries() -> None:
    cases = [
        ("What is the capital of France?", {}, None, "DEFINITION_LOOKUP"),
        ("If all A are B and all B are C, what follows?", {}, None, "MULTI_HOP_INFERENCE"),
        ("What is 17 * 9?", {}, None, "NUMERIC_COMPUTE"),
        ("Transform this grid to match the pattern", {"input_grid": [[1]]}, None, "SPATIAL_TRANSFORM"),
        ("Which option best explains the evidence?", {}, ["a", "b", "c"], "COMPARATIVE_CHOICE"),
    ]
    for query_text, task, options, expected in cases:
        observed = Knowledgeverse._meaning_class_from_task_payload(
            task=task,
            query_text=query_text,
            options=options,
        )
        assert observed == expected

