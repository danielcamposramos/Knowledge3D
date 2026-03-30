from __future__ import annotations

from knowledge3d.knowledgeverse.vram_task_buffer import VRAMTaskBuffer


def _sample_tasks(count: int = 10):
    tasks = []
    for index in range(count):
        query = [0.0] * 32
        query[index % 4] = 1.0
        option_embeddings = []
        for option_index in range(4):
            option = [0.0] * 32
            option[option_index] = 1.0
            option_embeddings.append(option)
        tasks.append(
            {
                "type": "MMLU_TASK",
                "query_embedding": query,
                "goal_embedding": list(reversed(query)),
                "option_embeddings": option_embeddings,
                "subject": f"subject_{index}",
                "domain_hint": f"domain_{index}",
                "thinking_budget": 10,
                "action_history": [3, 2, 1],
                "ternary_signal": -1,
            }
        )
    return tasks


def test_vram_task_buffer_round_trip():
    task_buffer = VRAMTaskBuffer(max_tasks=16)
    try:
        tasks = _sample_tasks(10)
        loaded = task_buffer.bulk_load(tasks)
        assert loaded == 10
        roundtrip = task_buffer.read_tasks(10)
        assert len(roundtrip) == 10
        assert roundtrip[0]["type"] == "MMLU_TASK"
        assert roundtrip[0]["option_count"] == 4
        assert roundtrip[0]["query_embedding"][0] == 1.0
        assert roundtrip[1]["query_embedding"][1] == 1.0
        assert roundtrip[0]["option_embeddings"][2][2] == 1.0
        assert roundtrip[0]["thinking_budget"] == 10
        assert roundtrip[0]["action_history"] == [3, 2, 1]
        assert roundtrip[0]["ternary_signal"] == -1
        assert roundtrip[0]["goal_embedding"][-1] == 1.0
    finally:
        task_buffer.close()


def test_vram_task_buffer_round_trip_seven_options():
    task_buffer = VRAMTaskBuffer(max_tasks=4)
    try:
        tasks = [
            {
                "type": "ARC3_TASK",
                "query_embedding": [1.0] + ([0.0] * 31),
                "goal_embedding": ([0.0] * 31) + [1.0],
                "option_embeddings": [[1.0 if i == j else 0.0 for i in range(32)] for j in range(7)],
                "subject": "arc3_subject",
                "domain_hint": "arc3_domain",
                "thinking_budget": 20,
                "action_history": [4, 4, 4, 2],
                "ternary_signal": -1,
            }
        ]
        loaded = task_buffer.bulk_load(tasks)
        assert loaded == 1
        roundtrip = task_buffer.read_tasks(1)
        assert len(roundtrip) == 1
        assert roundtrip[0]["type"] == "ARC3_TASK"
        assert roundtrip[0]["option_count"] == 7
        assert len(roundtrip[0]["option_embeddings"]) == 7
        assert roundtrip[0]["option_embeddings"][6][6] == 1.0
        assert roundtrip[0]["thinking_budget"] == 20
        assert roundtrip[0]["action_history"] == [4, 4, 4, 2]
        assert roundtrip[0]["ternary_signal"] == -1
        assert roundtrip[0]["goal_embedding"][-1] == 1.0
    finally:
        task_buffer.close()


def test_vram_result_buffer_round_trip():
    task_buffer = VRAMTaskBuffer(max_tasks=8)
    try:
        results = [
            {
                "answer_index": idx % 4,
                "confidence": float(idx) * 0.25,
                "convergence_signal": 1,
                "iterations_used": idx + 1,
                "answer_text": f"answer_{idx}",
                "goal_progress": (idx - 2) * 0.5,
            }
            for idx in range(5)
        ]
        written = task_buffer.write_results(results)
        assert written == 5
        roundtrip = task_buffer.read_results(5)
        assert len(roundtrip) == 5
        assert roundtrip[3]["answer_index"] == 3
        assert roundtrip[4]["iterations_used"] == 5
        assert roundtrip[2]["convergence_signal"] == 1
        assert roundtrip[0]["goal_progress"] == -1.0
        assert roundtrip[4]["goal_progress"] == 1.0
    finally:
        task_buffer.close()
