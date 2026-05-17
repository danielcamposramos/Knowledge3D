from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


JANET_QUESTION = (
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the "
    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
    "every day at the farmers' market?"
)


def _janet_task(task_id: str) -> dict[str, object]:
    return {
        "type": "MATH_TASK",
        "task_id": task_id,
        "query": JANET_QUESTION,
        "question": JANET_QUESTION,
        "expected_answer": "18",
    }


def test_game_loop_query_path_returns_janet_via_ring(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_game_loop_janet")

    request_id = kv.enqueue_task(
        task=_janet_task("gsm8k_0"),
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
        use_enriched=True,
    )
    result = kv.wait_output_buffer(request_id, max_ticks=1)

    assert result["status"] == "ok"
    assert result["mode"] == "query_tick"
    assert result["trm_io"]["request_id"] == request_id
    assert int(result["trm_io"]["tick"]) > 0
    assert result["result"] == "18"


def test_game_loop_query_path_ticks_increase_across_requests(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_game_loop_sequence")

    ticks: list[int] = []
    request_ids: list[str] = []
    for index in range(3):
        request_id = kv.enqueue_task(
            task=_janet_task(f"gsm8k_seq_{index}"),
            route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
            specialist="math",
            domain_hint="math",
            use_enriched=True,
        )
        result = kv.wait_output_buffer(request_id, max_ticks=1)
        request_ids.append(request_id)
        ticks.append(int(result["trm_io"]["tick"]))

    assert len(set(request_ids)) == 3
    assert ticks[0] < ticks[1] < ticks[2]
