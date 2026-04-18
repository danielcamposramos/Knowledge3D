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

    def _dispatch_sovereign_task(self, *, task, route=None, specialist="auto", domain_hint=None, use_enriched=True):
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
            "trm_dispatch": {
                "planned_swarm_groups": 4,
                "recommended_swarm_groups": 0,
                "worker_slots": list(range(36)),
                "resonance_weights": [0.8999999761581421, 0.699999988079071, 0.5, 0.4],
                "gpu_utilization": 0.0,
                "vram_free_bytes": 8 * 1024 * 1024 * 1024,
            },
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


class _FakeFusedBridge:
    def __init__(self) -> None:
        self.background_calls = 0
        self.query_calls = 0
        self.start_calls = 0
        self.stop_calls = 0

    def start_tick_loop(self):
        self.start_calls += 1
        return self.tick_loop_status()

    def stop_tick_loop(self):
        self.stop_calls += 1
        return self.tick_loop_status()

    def tick_loop_status(self):
        return {"ticking": self.start_calls > self.stop_calls, "tick_count": self.background_calls}

    def launch_tick(self, **_kwargs):
        self.background_calls += 1
        return {"tick": self.background_calls + self.query_calls, "current_state": 1, "steps": 0, "drift": 0.0}

    def run_query_tick(self, **_kwargs):
        self.query_calls += 1
        return {"tick": self.background_calls + self.query_calls, "current_state": 1, "steps": 1, "drift": 0.0}

    def read_action_buffers_words(self):
        return [[0xFF] + [0] * 71]


def test_trm_game_loop_buffers_and_bridge_query_ticket():
    kv = _FakeKnowledgeverse()
    bridge = _FakeFusedBridge()
    loop = TRMGameLoop(kv, bridge=bridge, input_size_mb=1, output_size_mb=1)
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
    assert result["mode"] == "query_tick"
    assert result["trm_io"]["request_id"] == request_id
    assert result["trm_tick"]["steps"] == 1
    assert result["action_buffers"][0][0] == 0xFF
    assert result["answer"] == "2+2=?"
    assert result["task_result"]["answer"] == "2+2=?"
    input_packet = loop.read_input_packet(request_id)
    output_packet = loop.read_output_packet(request_id)
    assert input_packet is not None
    assert output_packet is not None
    assert input_packet["task"]["query"] == "2+2=?"
    assert output_packet["result"]["mode"] == "query_tick"
    assert bridge.query_calls == 1
    assert bridge.background_calls == 0
    assert len(kv.calls) == 1
    assert loop.snapshot()["completed_outputs"] == 1


def test_trm_game_loop_idle_tick_routes_to_bridge_not_dispatch():
    kv = _FakeKnowledgeverse()
    bridge = _FakeFusedBridge()
    loop = TRMGameLoop(kv, bridge=bridge, input_size_mb=1, output_size_mb=1)
    loop.start()

    assert loop.tick(max_tasks=1) == 1

    assert bridge.background_calls == 1
    assert bridge.query_calls == 0
    assert kv.calls == []


def test_knowledgeverse_execute_task_uses_trm_game_loop(tmp_path: Path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_loop", eager_load_default_galaxies=False)

    called: list[dict[str, object]] = []

    def _fake_dispatch(*, task, route=None, specialist="auto", domain_hint=None, use_enriched=True):
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

    monkeypatch.setattr(kv, "_dispatch_sovereign_task", _fake_dispatch)
    fake_bridge = _FakeFusedBridge()
    kv._trm_game_loop.bridge = fake_bridge
    kv._trm_game_loop.start()

    result = kv.execute_task(
        task={"type": "CHAT_TASK", "query": "hello"},
        specialist="chat",
        domain_hint="general",
        use_enriched=False,
    )

    assert result["mode"] == "query_tick"
    assert result["trm_io"]["request_id"].startswith("trmio_")
    assert result["answer"] == "loop-shell"
    assert result["task_result"]["answer"] == "loop-shell"
    assert kv.trm_game_loop_status()["tick"] >= 1
    assert len(called) == 1
    assert fake_bridge.query_calls == 1
    raw_packet = kv._trm_game_loop.read_output_packet(result["trm_io"]["request_id"])
    assert raw_packet is not None
    assert raw_packet["result"]["mode"] == "query_tick"


def test_knowledgeverse_start_stop_trm_game_loop_delegates_bridge_clock(tmp_path: Path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_loop_clock", eager_load_default_galaxies=False)
    fake_bridge = _FakeFusedBridge()
    kv._trm_game_loop.bridge = fake_bridge

    started = kv.start_trm_game_loop()
    stopped = kv.stop_trm_game_loop()

    assert fake_bridge.start_calls == 1
    assert fake_bridge.stop_calls == 1
    assert started["bridge"]["ticking"] is True
    assert stopped["bridge"]["ticking"] is False


def test_knowledgeverse_eager_boot_is_sovereign_only(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def _fake_boot(self, *, force_reload=False, force_rebuild=False):
        calls.append("sovereign_boot")
        return {"status": "ready", "mode": "stub", "force_rebuild": bool(force_rebuild)}

    monkeypatch.setattr(Knowledgeverse, "_boot_sovereign_runtime", _fake_boot)

    kv = Knowledgeverse(storage_root=tmp_path / "kv_eager_boot", eager_load_default_galaxies=True)

    assert kv._default_galaxies_loaded is True
    assert hasattr(Knowledgeverse, "bind_gpu_galaxy_runtime")
    assert calls == ["sovereign_boot"]


def test_knowledgeverse_execute_task_bypass_flag_skips_trm_game_loop(tmp_path: Path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_loop_bypass", eager_load_default_galaxies=False)

    called: list[dict[str, object]] = []

    def _fake_dispatch(*, task, route=None, specialist="auto", domain_hint=None, use_enriched=True):
        called.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": specialist,
                "domain_hint": domain_hint,
                "use_enriched": use_enriched,
            }
        )
        return {"status": "ok", "answer": "bypass-shell"}

    monkeypatch.setattr(kv, "_dispatch_sovereign_task", _fake_dispatch)
    monkeypatch.setenv("K3D_BYPASS_GAME_LOOP", "1")
    fake_bridge = _FakeFusedBridge()
    kv._trm_game_loop.bridge = fake_bridge
    kv._trm_game_loop.start()

    result = kv.execute_task(
        task={"type": "CHAT_TASK", "query": "hello"},
        specialist="chat",
        domain_hint="general",
        use_enriched=False,
    )

    assert result["status"] == "ok"
    assert result["answer"] == "bypass-shell"
    assert len(called) == 1
    assert fake_bridge.query_calls == 0


def test_knowledgeverse_run_ticks_pumps_bridge_clock(tmp_path: Path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_loop_ticks", eager_load_default_galaxies=False)
    fake_bridge = _FakeFusedBridge()
    kv._trm_game_loop.bridge = fake_bridge
    kv._trm_game_loop.start()

    processed = kv.run_ticks(3)

    assert processed == 3
    assert fake_bridge.background_calls == 3


def test_enqueue_task_wall_timeout(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("K3D_RING_TRACE_PATH", str(tmp_path / "ring_trace.jsonl"))
    kv = Knowledgeverse(storage_root=tmp_path / "kv_loop_timeout", eager_load_default_galaxies=False)
    fake_bridge = _FakeFusedBridge()
    kv._trm_game_loop.bridge = fake_bridge
    kv._trm_game_loop.start()
    kv._external_tick_driver_active = True

    request_id = kv.enqueue_task(
        task={"type": "CHAT_TASK", "task_id": "timeout_case", "query": "wait forever"},
        specialist="chat",
        domain_hint="general",
        use_enriched=False,
        max_wall_ms=50,
    )

    result = kv.wait_output_buffer(request_id, max_ticks=0, max_wall_ms=50)

    assert result["failure_code"] == "wall_timeout"
    assert result["task_result"]["failure_code"] == "wall_timeout"
    assert result["task_id"] == "timeout_case"
    assert result["trm_io"]["request_id"] == request_id
    assert kv._trm_game_loop.read_input_packet(request_id) is None
    assert kv.trm_game_loop_status()["pending_inputs"] == 0
    assert kv.trm_game_loop_status()["abandoned_requests"] == 0
    assert fake_bridge.query_calls == 0


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
