from __future__ import annotations

from pathlib import Path

import numpy as np

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.trm_game_loop import TRMGameLoop


class _FakeSwarmBridge:
    def execute_swarm(self, embedding: np.ndarray, num_iterations: int = 1, reset_state: bool = False, readback_mode: str = "full"):
        weights = np.asarray([0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dtype=np.float32)
        return embedding, None, weights


class _FakeKnowledgeverse:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _execute_task_direct(self, *, task, route=None, specialist="auto", domain_hint=None, use_enriched=True):
        self.calls.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": specialist,
                "domain_hint": domain_hint,
                "use_enriched": use_enriched,
            }
        )
        return {
            "status": "ok",
            "answer": str(task.get("query") or task.get("prompt") or task.get("question") or ""),
        }

    def _jarvis_task_complexity(self, *, task_type: str, paths: list[dict[str, object]], options: list[str] | None) -> float:
        return 0.8 if task_type == "MATH_TASK" else 0.4

    def _jarvis_determine_swarm_count(self, task_complexity: float) -> int:
        return 4 if task_complexity >= 0.75 else 2

    def _embed_query_gpu(self, query_text: str, *, task: dict[str, object] | None = None) -> list[float]:
        base = [float((idx + 1) / 16.0) for idx in range(16)]
        return base

    def get_swarm_bridge(self):
        return _FakeSwarmBridge()

    def _jarvis_gpu_utilization(self) -> float:
        return 0.0

    def _jarvis_vram_free_bytes(self) -> int:
        return 8 * 1024 * 1024 * 1024


def test_trm_game_loop_buffers_and_dispatch_ticket():
    kv = _FakeKnowledgeverse()
    loop = TRMGameLoop(kv, input_size_mb=1, output_size_mb=1)
    loop.start()

    request_id = loop.enqueue_task(
        task={"type": "MATH_TASK", "query": "2+2=?"},
        route={"source": "test"},
        specialist="math",
        domain_hint="math",
        use_enriched=True,
    )

    assert loop.tick(max_tasks=1) == 1
    result = loop.wait_output(request_id, max_ticks=0)

    assert result is not None
    assert result["answer"] == "2+2=?"
    assert result["trm_io"]["request_id"] == request_id
    input_packet = loop.read_input_packet(request_id)
    output_packet = loop.read_output_packet(request_id)
    assert input_packet is not None
    assert output_packet is not None
    assert input_packet["task"]["query"] == "2+2=?"
    assert output_packet["result"]["answer"] == "2+2=?"
    dispatch = result["trm_dispatch"]
    assert dispatch["planned_swarm_groups"] == 4
    assert dispatch["recommended_swarm_groups"] == 0
    assert len(dispatch["worker_slots"]) == 36
    assert dispatch["resonance_weights"][:3] == [0.8999999761581421, 0.699999988079071, 0.5]
    assert dispatch["gpu_utilization"] >= 0.0
    assert dispatch["vram_free_bytes"] >= 0
    assert loop.snapshot()["completed_outputs"] == 1


def test_knowledgeverse_execute_task_uses_trm_game_loop(tmp_path: Path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_loop", eager_load_default_galaxies=False)

    called: list[dict[str, object]] = []

    def _fake_execute(*, task, route=None, specialist="auto", domain_hint=None, use_enriched=True):
        called.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": specialist,
                "domain_hint": domain_hint,
                "use_enriched": use_enriched,
            }
        )
        return {"status": "ok", "answer": "loop-shell"}

    monkeypatch.setattr(kv, "_execute_task_direct", _fake_execute)

    result = kv.execute_task(
        task={"type": "CHAT_TASK", "query": "hello"},
        specialist="chat",
        domain_hint="general",
        use_enriched=False,
    )

    assert result["answer"] == "loop-shell"
    assert result["trm_io"]["request_id"].startswith("trmio_")
    assert kv.trm_game_loop_status()["tick"] >= 1
    assert called and called[0]["specialist"] == "chat"
    raw_packet = kv._trm_game_loop.read_output_packet(result["trm_io"]["request_id"])
    assert raw_packet is not None
    assert raw_packet["result"]["answer"] == "loop-shell"


def test_knowledgeverse_adaptive_swarm_adapters_expose_no_cpu_escape_hatch(tmp_path: Path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gpu_required", eager_load_default_galaxies=False)
    for specialist in kv.adaptive_swarm.base.specialists.values():
        adapter = specialist.get("adapter") if isinstance(specialist, dict) else None
        assert adapter is not None
        assert not hasattr(adapter.config, "require_gpu")


def test_knowledgeverse_trm_game_loop_uses_house_regions(tmp_path: Path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_regions", eager_load_default_galaxies=False)
    assert kv._trm_game_loop.input_ring is kv.stargate.ring_buffer
    assert kv._trm_game_loop.output_ring is kv.shadow_copy.compressed_journal.buffer
