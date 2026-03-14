from __future__ import annotations

from knowledge3d.daemon.main import DaemonConfig, K3DDaemon
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


GSM8K_0_QUESTION = (
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the "
    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
    "every day at the farmers' market?"
)


def test_knowledgeverse_chat_query_uses_gpu_chat_profile(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_chat_profile")
    result = kv.execute_task(
        task={
            "type": "CHAT_TASK",
            "prompt": "cipher word validation",
            "messages": [{"role": "user", "content": "cipher word validation"}],
        },
        route={"specialist": "chat", "galaxy_names": ["Grammar", "Word", "Character"]},
        specialist="chat",
        domain_hint="general",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["program_id"] == Knowledgeverse.GPU_CHAT_REASONING_PROGRAM_ID
    assert result["route"]["galaxy_names"] == list(Knowledgeverse.GPU_CHAT_TARGET_GALAXIES)
    assert result["match"]["galaxy"] in {"Word", "Grammar", "Character"}
    assert result.get("shadow_event_id")
    assert any("Specialist route: chat" in step for step in result["reasoning_trace"])
    assert any("Path explored:" in step for step in result["reasoning_trace"])


def test_knowledgeverse_chat_query_can_promote_cross_domain_math_match(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_chat_cross_domain")
    result = kv.execute_task(
        task={
            "type": "CHAT_TASK",
            "prompt": GSM8K_0_QUESTION,
            "query": GSM8K_0_QUESTION,
            "messages": [{"role": "user", "content": GSM8K_0_QUESTION}],
            "expected_answer": "18",
        },
        route={"specialist": "chat", "galaxy_names": ["Grammar", "Word", "Character"]},
        specialist="chat",
        domain_hint="general",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["response"] == "18"
    assert result["match"]["galaxy"] == "Math"
    assert result["shadow_event_id"]
    assert any("Cross-domain promotion:" in step for step in result["reasoning_trace"])
    assert any("Path explored:" in step for step in result["reasoning_trace"])


def test_daemon_chat_command_uses_knowledgeverse_gpu_query(tmp_path) -> None:
    daemon = K3DDaemon(
        DaemonConfig(storage_root=tmp_path / "daemon_chat"),
    )
    response = daemon.handle_command(
        {
            "command": "CHAT",
            "prompt": GSM8K_0_QUESTION,
            "use_enriched": True,
        }
    )

    assert response["status"] == "ok"
    assert response["gpu_execution"] is True
    assert response["runtime"] == "knowledgeverse_gpu_query"
    assert response["program_id"] == Knowledgeverse.GPU_CHAT_REASONING_PROGRAM_ID
    assert response["response"] == "18"
    assert response["task_result"]["match"]["galaxy"] == "Math"
