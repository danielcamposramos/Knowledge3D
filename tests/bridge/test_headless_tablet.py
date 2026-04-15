from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge3d.bridge.headless_tablet import (
    TABLET_WORD_OFFSET_DATA,
    ActionType,
    HeadlessTabletMPC,
    TabletEmit,
    TabletIngest,
    TabletSessionFrame,
    TabletSessionTape,
)
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class _FakeDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        task = dict(payload.get("task") or {})
        task_type = str(task.get("surface_kind") or task.get("type", ""))
        if task_type == "GAME_2D":
            return {
                "status": "ok",
                "route": {"specialist": "visual", "galaxy_names": ["Drawing", "Tool"], "domain": "visual"},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "grid",
                    "answer_materialized": True,
                    "output_grid": task.get("expected_output"),
                },
            }
        if task_type == "MATH":
            return {
                "status": "ok",
                "route": {"specialist": "math", "galaxy_names": ["Math", "Grammar"], "domain": "math"},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "numeric",
                    "answer_text": str(task.get("expected_answer") or ""),
                    "numeric_answer": 4.0,
                    "answer_materialized": True,
                },
            }
        if task_type == "QUESTION":
            return {
                "status": "ok",
                "route": {"specialist": "chat", "galaxy_names": ["Grammar", "Reality"], "domain": task.get("domain_hint", "multi")},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "choice",
                    "answer_choice": str(task.get("expected_answer") or "unknown"),
                    "answer_text": str(task.get("expected_answer") or "unknown"),
                    "answer_materialized": True,
                },
            }
        return {"status": "error", "error": "unsupported_task"}


class _FailingDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("daemon fallback should not run when bridge is available")


class _FakeTabletBridge:
    def __init__(self) -> None:
        self.bound_buffers: dict[str, Any] | None = None
        self.bound_galaxy_table: dict[str, Any] | None = None
        self.submitted_queries: list[dict[str, Any]] = []
        self.session_events: list[dict[str, Any]] = []
        self._session_open = False
        self._pending_frame: dict[str, Any] | None = None

    def bind_query_runtime_buffers(self, **kwargs: Any) -> dict[str, int]:
        self.bound_buffers = dict(kwargs)
        return {"bound": 1, "vector_dim": 512, "workspace_floats": 4096}

    def bind_galaxy_table(
        self,
        gpu_ptr: Any,
        star_count: int,
        *,
        embedding_dims: int = 64,
        host_stars: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        self.bound_galaxy_table = {
            "gpu_ptr": gpu_ptr,
            "star_count": int(star_count),
            "embedding_dims": int(embedding_dims),
            "host_stars": [dict(star) for star in list(host_stars or [])],
        }
        return {"bound": 1, "star_count": int(star_count), "embedding_dims": int(embedding_dims)}

    def submit_query(
        self,
        query_embedding: list[float],
        *,
        action_buffer_words: list[int] | None = None,
        delta_time: float = 0.02,
        tick: int | None = None,
    ) -> dict[str, Any]:
        action_words = list(action_buffer_words or [])
        self.submitted_queries.append(
            {
                "query_embedding": list(query_embedding),
                "action_buffer_words": action_words,
                "delta_time": float(delta_time),
                "tick": tick,
            }
        )
        return {
            "status": "ok",
            "mode": "submit_query",
            "tick": int(tick or 0),
            "steps": 1,
            "drift": 0.0,
            "current_state": 4,
            "sleep_state": 4,
            "query_embedding_512": [0.25] * 512,
            "y_new_vector_512": [0.5] * 512,
            "z_new_vector_512": [0.75] * 512,
            "trm_latency_us": 10.0,
            "action_buffers": [[int(ActionType.NO_ACTION.value)] + [0] * 71],
            "ring_event_payload": 1234,
            "tick_result": {"tick": int(tick or 0)},
        }

    def open_query_session(
        self,
        *,
        reset_runtime: bool = True,
        tick_hz: float = 50.0,
        delta_time: float = 0.02,
    ) -> dict[str, Any]:
        self._session_open = True
        self.session_events.append(
            {
                "event": "open",
                "reset_runtime": bool(reset_runtime),
                "tick_hz": float(tick_hz),
                "delta_time": float(delta_time),
            }
        )
        return {"active": True, "tick_hz": float(tick_hz), "delta_time": float(delta_time)}

    def query_session_status(self) -> dict[str, Any]:
        return {"active": bool(self._session_open), "pending": dict(self._pending_frame or {})}

    def queue_query_frame(
        self,
        query_embedding: list[float],
        *,
        action_buffer_words: list[int] | None = None,
        frame_id: str,
    ) -> dict[str, Any]:
        self._pending_frame = {
            "frame_id": str(frame_id),
            "query_embedding": list(query_embedding),
            "action_buffer_words": list(action_buffer_words or []),
        }
        self.session_events.append({"event": "queue", **self._pending_frame})
        return dict(self._pending_frame)

    def poll_query_result(
        self,
        *,
        timeout_s: float,
        frame_id: str,
    ) -> dict[str, Any]:
        assert self._pending_frame is not None
        self.session_events.append({"event": "poll", "frame_id": str(frame_id), "timeout_s": float(timeout_s)})
        pending = dict(self._pending_frame)
        self._pending_frame = None
        return {
            "status": "ok",
            "mode": "query_session",
            "frame_id": str(frame_id),
            "tick": 1,
            "steps": 1,
            "drift": 0.0,
            "current_state": 4,
            "sleep_state": 4,
            "query_embedding_512": list(pending["query_embedding"]),
            "y_new_vector_512": [0.5] * 512,
            "z_new_vector_512": [0.75] * 512,
            "trm_latency_us": 10.0,
            "action_buffers": [[int(ActionType.NO_ACTION.value)] + [0] * 71],
            "ring_event_payload": 4321,
            "tick_result": {"tick": 1},
            "answer_materialized": False,
            "failure_code": "not_materialized_from_y_new_yet",
        }

    def close_query_session(self) -> dict[str, Any]:
        self._session_open = False
        self.session_events.append({"event": "close"})
        return {"closed": True}


class _FakeKnowledgeverseSession:
    DEFAULT_GALAXIES = ("Drawing", "Math", "Grammar")

    def __init__(self) -> None:
        self.logged_events: list[tuple[str, dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []

    def ensure_default_galaxies_loaded(self) -> dict[str, int]:
        return {"Drawing": 10, "Math": 20, "Grammar": 30}

    def execute_task(
        self,
        *,
        task: dict[str, Any],
        route: dict[str, Any] | None = None,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": str(specialist),
                "domain_hint": domain_hint,
                "use_enriched": bool(use_enriched),
            }
        )
        return {
            "status": "ok",
            "route": dict(route or {}),
            "task_result": {
                "status": "ok",
                "answer_kind": "numeric",
                "answer_text": str(task.get("expected_answer") or "4"),
                "numeric_answer": 4.0,
                "answer_materialized": True,
                "gpu_execution": True,
                "runtime": "knowledgeverse_gpu_query",
                "route": dict(route or {}),
            },
        }

    def log_event(self, name: str, payload: dict[str, Any]) -> None:
        self.logged_events.append((str(name), dict(payload)))


class _FakeKnowledgeverseDirectSession(_FakeKnowledgeverseSession):
    def execute_task(
        self,
        *,
        task: dict[str, Any],
        route: dict[str, Any] | None = None,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": str(specialist),
                "domain_hint": domain_hint,
                "use_enriched": bool(use_enriched),
            }
        )
        return {
            "status": "ok",
            "answer": str(task.get("expected_answer") or "4"),
            "predicted_answer": str(task.get("expected_answer") or "4"),
            "numeric_answer": 4.0,
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": "gpu_task_dispatch_sovereign",
            "route": dict(route or {}),
        }


class _FakeKnowledgeverseGame2DSession(_FakeKnowledgeverseSession):
    def execute_task(
        self,
        *,
        task: dict[str, Any],
        route: dict[str, Any] | None = None,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": str(specialist),
                "domain_hint": domain_hint,
                "use_enriched": bool(use_enriched),
            }
        )
        return {
            "status": "ok",
            "route": dict(route or {}),
            "task_result": {
                "status": "ok",
                "answer_kind": "grid",
                "answer_materialized": True,
                "failure_code": "",
                "gpu_execution": True,
                "runtime": "knowledgeverse_gpu_query",
                "route": dict(route or {}),
                "output_grid": [[0, 2], [3, 0]],
            },
        }


def test_tablet_ingest_arc_builds_standard_route_payload_and_action_buffer():
    envelope = TabletIngest.arc_task(
        task_id="arc_demo",
        training_examples=[{"input": [[1]], "output": [[2]]}],
        input_grid=[[3]],
        expected_output=[[4]],
    )

    payload = envelope.to_route_payload(use_enriched=True)
    assert payload["command"] == "ROUTE"
    assert payload["specialist"] == "visual"
    assert payload["task"]["type"] == "ARC_TASK"
    assert payload["task"]["surface_kind"] == "GAME_2D"

    buf = envelope.to_action_buffer()
    mutation_type, mutation_payload = buf.extract_tablet_mutation()
    assert buf.get_action_type() == ActionType.UPDATE_TABLET
    assert mutation_type == 1
    assert mutation_payload.shape == (6,)


def test_tablet_ingest_game2d_route_keeps_all_live_galaxies_loaded():
    envelope = TabletIngest.game2d_task(
        task_id="arc_grid_demo",
        query="horizontal reflection grid transform",
        input_grid=[[1, 0], [0, 1]],
        expected_output=[[0, 1], [1, 0]],
        result_kind="grid",
    )

    assert "game_mechanics" in envelope.galaxies
    assert "Math" in envelope.galaxies
    assert "Language" in envelope.galaxies
    assert envelope.task["options"] == []
    assert envelope.task["expected_result_kind"] == "grid"


def test_headless_tablet_mpc_routes_math_through_standard_contract(tmp_path: Path):
    boundary = HeadlessTabletMPC(command_handler=_FakeDaemon(), storage_root=tmp_path)
    envelope = TabletIngest.math_problem(
        task_id="math_demo",
        question="What is 2 + 2?",
        competition="AMC",
        expected_answer="4",
    )

    result = boundary.submit(envelope, use_enriched=True)

    assert result["response"]["status"] == "ok"
    assert result["route_payload"]["task"]["type"] == "MATH_TASK"
    assert result["emitted"]["status"] == "success"
    assert result["emitted"]["numeric_answer"] == 4.0
    assert result["emitted"]["answer_text"] == "4"
    assert result["emitted"]["correct"] is True
    assert result["tablet_contract"]["action_type"] == "UPDATE_TABLET"


def test_headless_tablet_mpc_uses_bridge_ring_path_without_daemon(tmp_path: Path):
    bridge = _FakeTabletBridge()
    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        bridge=bridge,
        storage_root=tmp_path,
    )
    envelope = TabletIngest.math_problem(
        task_id="math_bridge_demo",
        question="What is 2 + 2?",
        competition="AMC",
        expected_answer="4",
    )

    result = boundary.submit(envelope, use_enriched=True)

    assert len(bridge.submitted_queries) == 1
    submitted = bridge.submitted_queries[0]
    assert submitted["delta_time"] == 0.02
    assert submitted["tick"] == 1
    assert len(submitted["query_embedding"]) == 512
    assert len(submitted["action_buffer_words"]) == 72
    assert submitted["action_buffer_words"][0] == int(ActionType.UPDATE_TABLET.value)
    assert result["tablet_contract"]["sovereign_path"] == "tablet_bridge_ring"
    assert result["tablet_contract"]["output_action_type"] == "NO_ACTION"
    assert result["response"]["task_result"]["gpu_execution"] is False
    assert result["response"]["task_result"]["answer_materialized"] is False
    assert result["response"]["task_result"]["failure_code"] == "not_materialized_from_y_new_yet"
    assert result["emitted"]["answer_materialized"] is False
    assert result["emitted"]["correct"] is False


def test_headless_tablet_mpc_runs_tape_session_through_bridge_session_api(tmp_path: Path):
    bridge = _FakeTabletBridge()
    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        bridge=bridge,
        storage_root=tmp_path,
    )
    tape = TabletSessionTape(
        session_id="session_demo",
        suite_name="math_demo",
        surface_kind="MATH",
        use_enriched=True,
        frames=(
            TabletSessionFrame(
                frame_id="frame_1",
                envelope=TabletIngest.math_problem(
                    task_id="math_bridge_session_demo",
                    question="What is 2 + 2?",
                    competition="AMC",
                    expected_answer="4",
                ),
                expected="4",
                source_meta={"suite": "math_demo"},
            ),
        ),
    )

    result = boundary.run_tape_session(tape, enforce_preflight=False)

    assert result["status"] == "ok"
    assert result["session_id"] == "session_demo"
    assert len(result["results"]) == 1
    row = result["results"][0]
    assert row["tablet_contract"]["sovereign_path"] == "tablet_bridge_session"
    assert row["tablet_contract"]["frame_id"] == "frame_1"
    assert [event["event"] for event in bridge.session_events] == ["open", "queue", "poll", "close"]


def test_headless_tablet_mpc_runs_tape_session_through_knowledgeverse_live_session(tmp_path: Path):
    kv = _FakeKnowledgeverseSession()
    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        knowledgeverse=kv,
        storage_root=tmp_path,
    )
    tape = TabletSessionTape(
        session_id="kv_session_demo",
        suite_name="math_demo",
        surface_kind="MATH",
        use_enriched=True,
        frames=(
            TabletSessionFrame(
                frame_id="frame_kv_1",
                envelope=TabletIngest.math_problem(
                    task_id="math_live_session_demo",
                    question="What is 2 + 2?",
                    competition="AMC",
                    expected_answer="4",
                ),
                expected="4",
                source_meta={"suite": "math_demo"},
            ),
        ),
    )

    result = boundary.run_tape_session(tape, enforce_preflight=False)

    assert result["status"] == "ok"
    assert result["session_id"] == "kv_session_demo"
    assert len(kv.calls) == 1
    call = kv.calls[0]
    assert call["specialist"] == "math"
    assert call["route"]["route_policy"] == "all_live_galaxies"
    row = result["results"][0]
    assert row["tablet_contract"]["sovereign_path"] == "knowledgeverse_dispatch_session"
    assert row["emitted"]["status"] == "success"
    assert row["emitted"]["correct"] is True


def test_headless_tablet_live_session_normalizes_direct_knowledgeverse_packets(tmp_path: Path):
    kv = _FakeKnowledgeverseDirectSession()
    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        knowledgeverse=kv,
        storage_root=tmp_path,
    )
    tape = TabletSessionTape(
        session_id="kv_session_direct",
        suite_name="math_demo",
        surface_kind="MATH",
        use_enriched=True,
        frames=(
            TabletSessionFrame(
                frame_id="frame_kv_direct_1",
                envelope=TabletIngest.math_problem(
                    task_id="math_live_session_direct",
                    question="What is 2 + 2?",
                    competition="AMC",
                    expected_answer="4",
                ),
                expected="4",
                source_meta={"suite": "math_demo"},
            ),
        ),
    )

    result = boundary.run_tape_session(tape, enforce_preflight=False)

    row = result["results"][0]
    assert row["response"]["task_result"]["gpu_execution"] is True
    assert row["response"]["task_result"]["program_id"] == "gpu_task_dispatch_sovereign"
    assert row["emitted"]["predicted_answer"] == 4.0
    assert row["emitted"]["correct"] is True


def test_headless_tablet_live_session_logs_trace_event(tmp_path: Path):
    kv = _FakeKnowledgeverseSession()
    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        knowledgeverse=kv,
        storage_root=tmp_path,
    )
    tape = TabletSessionTape(
        session_id="kv_trace_demo",
        suite_name="math_demo",
        surface_kind="MATH",
        use_enriched=True,
        frames=(
            TabletSessionFrame(
                frame_id="frame_trace_1",
                envelope=TabletIngest.math_problem(
                    task_id="math_trace_demo",
                    question="What is 2 + 2?",
                    competition="AMC",
                    expected_answer="4",
                ),
                expected="4",
                source_meta={"suite": "math_demo"},
            ),
        ),
    )

    boundary.run_tape_session(tape, enforce_preflight=False)

    event_types = [name for name, _payload in kv.logged_events]
    assert "tablet_session_trace" in event_types
    trace_payload = next(payload for name, payload in kv.logged_events if name == "tablet_session_trace")
    assert trace_payload["item_id"] == "frame_trace_1"
    assert trace_payload["suite"] == "math_demo"
    assert trace_payload["surface_kind"] == "MATH"
    assert trace_payload["specialist_lane"] == "math"
    assert trace_payload["halting_reason"] == "CONVERGED"


def test_headless_tablet_live_session_translates_materialized_game2d_grid(tmp_path: Path):
    kv = _FakeKnowledgeverseGame2DSession()
    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        knowledgeverse=kv,
        storage_root=tmp_path,
    )
    tape = TabletSessionTape(
        session_id="kv_game2d_materialize",
        suite_name="warmup_arc",
        surface_kind="GAME_2D",
        use_enriched=False,
        frames=(
            TabletSessionFrame(
                frame_id="frame_arc_1",
                envelope=TabletIngest.game2d_task(
                    task_id="warmup_arc_1",
                    query="horizontal reflection grid transform",
                    training_examples=[{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
                    input_grid=[[2, 0], [0, 3]],
                    expected_output=[[0, 2], [3, 0]],
                    result_kind="grid",
                ),
                expected=[[0, 2], [3, 0]],
                source_meta={"suite": "warmup_arc"},
            ),
        ),
    )

    result = boundary.run_tape_session(tape, enforce_preflight=False)

    row = result["results"][0]
    assert row["response"]["task_result"]["answer_materialized"] is True
    assert row["response"]["task_result"]["output_grid"] == [[0, 2], [3, 0]]
    assert row["emitted"]["output_grid"] == [[0, 2], [3, 0]]
    assert row["emitted"]["correct"] is True


def test_headless_tablet_mpc_discovers_knowledgeverse_bridge_and_binds_runtime(tmp_path: Path):
    class _Launcher:
        def __init__(self) -> None:
            self._step_fused_bridge = _FakeTabletBridge()

    kv = Knowledgeverse.__new__(Knowledgeverse)
    kv._trm = _Launcher()
    kv._trm_state_buffers = {
        "d_q": 1,
        "d_y": 2,
        "d_z": 3,
        "d_z_new": 4,
        "d_y_new": 5,
        "d_workspace": 6,
        "d_q_input": 7,
    }
    kv._trm_weight_buffers = {"W1": 11, "W2": 12, "W3": 13, "W4": 14}
    kv._matryoshka_bridge = object()
    kv._trm_matryoshka_weight_buffer = 15
    kv._sovereign_hot_path = type(
        "_Runtime",
        (),
        {
            "star_table": type("_StarTable", (), {"gpu_ptr": 123, "star_count": 1, "_host_stars": [{"answer_text": "42"}]})(),
            "_host_stars": [{"answer_text": "42"}],
        },
    )()

    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        knowledgeverse=kv,
        storage_root=tmp_path,
    )

    assert boundary._bridge is kv._trm._step_fused_bridge
    assert kv._trm._step_fused_bridge.bound_buffers is not None
    assert kv._trm._step_fused_bridge.bound_buffers["q_ptr"] == 1
    assert kv._trm._step_fused_bridge.bound_buffers["matryoshka_weight_ptr"] == 15
    assert kv._trm._step_fused_bridge.bound_galaxy_table is not None
    assert kv._trm._step_fused_bridge.bound_galaxy_table["star_count"] == 1


def test_headless_tablet_mpc_materializes_bridge_top_star_answer(tmp_path: Path):
    class _AnswerBridge(_FakeTabletBridge):
        def submit_query(
            self,
            query_embedding: list[float],
            *,
            action_buffer_words: list[int] | None = None,
            delta_time: float = 0.02,
            tick: int | None = None,
        ) -> dict[str, Any]:
            packet = super().submit_query(
                query_embedding,
                action_buffer_words=action_buffer_words,
                delta_time=delta_time,
                tick=tick,
            )
            packet.update(
                {
                    "answer_materialized": True,
                    "failure_code": "",
                    "top_star_idx": 7,
                    "top_star_score": 0.99,
                    "top_star_galaxy_id": 123,
                    "top_star_role": 4,
                    "top_star_hash": 456,
                    "top_star": {
                        "id": "math_answer_4",
                        "selection_role": "answer",
                        "metadata": {"answer_text": "4"},
                    },
                }
            )
            return packet

    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        bridge=_AnswerBridge(),
        storage_root=tmp_path,
    )
    envelope = TabletIngest.math_problem(
        task_id="math_bridge_answer_demo",
        question="What is 2 + 2?",
        competition="AMC",
        expected_answer="4",
    )

    result = boundary.submit(envelope)

    assert result["response"]["task_result"]["answer_materialized"] is True
    assert result["response"]["task_result"]["top_star_idx"] == 7
    assert result["emitted"]["answer_text"] == "4"
    assert result["emitted"]["numeric_answer"] == 4.0
    assert result["emitted"]["correct"] is True


def test_headless_tablet_mpc_decodes_signed_math_result_from_action_buffer(tmp_path: Path):
    class _SignedAnswerBridge(_FakeTabletBridge):
        def submit_query(
            self,
            query_embedding: list[float],
            *,
            action_buffer_words: list[int] | None = None,
            delta_time: float = 0.02,
            tick: int | None = None,
        ) -> dict[str, Any]:
            packet = super().submit_query(
                query_embedding,
                action_buffer_words=action_buffer_words,
                delta_time=delta_time,
                tick=tick,
            )
            words = [0] * 72
            words[0] = int(ActionType.UPDATE_TABLET.value)
            words[TABLET_WORD_OFFSET_DATA] = 0xFFFFFFFB
            words[TABLET_WORD_OFFSET_DATA + 1] = 7
            words[TABLET_WORD_OFFSET_DATA + 5] = 1
            packet.update(
                {
                    "action_buffers": [words],
                    "top_star": {
                        "id": "math_answer_negative_five",
                        "selection_role": "answer",
                    },
                }
            )
            return packet

    boundary = HeadlessTabletMPC(
        command_handler=_FailingDaemon(),
        bridge=_SignedAnswerBridge(),
        storage_root=tmp_path,
    )
    envelope = TabletIngest.math_problem(
        task_id="math_bridge_negative_demo",
        question="What is 2 - 7?",
        competition="AMC",
        expected_answer="-5",
    )

    result = boundary.submit(envelope)

    assert result["response"]["task_result"]["answer_materialized"] is True
    assert result["response"]["task_result"]["numeric_answer"] == -5
    assert result["emitted"]["numeric_answer"] == -5.0
    assert result["emitted"]["answer_text"] == "-5"


def test_tablet_emit_lhe_uses_typed_option_answer_from_task_result():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_demo",
        question="Choose the correct answer.",
        options=["A", "B", "C"],
        domain="logic",
        expected_answer="B",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Grammar"]},
        "task_result": {
            "status": "ok",
            "answer_kind": "choice",
            "answer_choice": "B",
            "answer_text": "B",
            "answer_materialized": True,
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)
    assert emitted["status"] == "success"
    assert emitted["predicted_answer"] == "B"
    assert emitted["correct"] is True
    assert envelope.task["type"] == "QUESTION_TASK"


def test_tablet_emit_flattens_nested_route_response_and_blocks_internal_answer_labels():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_nested_demo",
        question="Choose the correct answer.",
        options=["A", "B", "C"],
        domain="logic",
        expected_answer="B",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Grammar"]},
        "task_result": {
            "status": "ok",
            "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Reality"]},
            "task_result": {
                "status": "ok",
                "answer_kind": "choice",
                "answer_index": 1,
                "answer": "grammar_answer_validator",
                "response": "grammar_answer_validator",
                "winner_star_id": "grammar_answer_validator",
                "route_depth": 2,
            },
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)

    assert emitted["status"] == "success"
    assert emitted["predicted_answer"] == "B"
    assert emitted["answer_materialized"] is True
    assert emitted["route"]["galaxy_names"] == ["Reality"]


def test_tablet_emit_blocks_humanized_internal_labels_from_answer_text():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_internal_label_demo",
        question="Explain the best answer.",
        options=[],
        domain="logic",
        expected_answer="grounded explanation",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Grammar"]},
        "task_result": {
            "status": "ok",
            "answer_kind": "text",
            "answer_text": "Anti Pattern Missing Validator Traversal",
            "answer_materialized": True,
            "winner_star_id": "anti_pattern_missing_validator_traversal",
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)

    assert emitted["answer_text"] == ""
    assert emitted["predicted_answer"] == ""
    assert emitted["answer_materialized"] is False


def test_tablet_emit_prefers_task_specific_route_when_present():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_route_demo",
        question="Explain the best cybersecurity practice.",
        options=[],
        domain="cybersecurity",
        expected_answer="least privilege",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "grammar", "domain": "cybersecurity", "galaxy_names": ["Grammar", "Tool"]},
        "task_result": {
            "status": "ok",
            "answer_kind": "text",
            "answer_text": "least privilege",
            "answer_materialized": True,
            "route": {
                "specialist": "grammar",
                "domain": "cybersecurity",
                "galaxy_names": ["Grammar", "Reality", "Tool"],
            },
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)
    assert emitted["route"]["galaxy_names"] == ["Grammar", "Reality", "Tool"]


def test_dispatch_sovereign_task_preserves_typed_question_mode():
    class _FakeRuntime:
        def __init__(self) -> None:
            self.last_task: dict[str, object] | None = None

        def dispatch_task(self, task: dict[str, object]) -> dict[str, object]:
            self.last_task = dict(task)
            return {"status": "ok", "route": {"route_family": "QUESTION"}}

    kv = Knowledgeverse.__new__(Knowledgeverse)
    runtime = _FakeRuntime()
    kv._enter_query_activity = lambda: None
    kv._mark_runtime_activity = lambda: None
    kv._leave_query_activity = lambda: None
    kv._get_sovereign_hot_path = lambda: runtime
    kv._query_sequence = 0

    result = Knowledgeverse._dispatch_sovereign_task(
        kv,
        task={
            "type": "QUESTION_TASK",
            "surface_kind": "QUESTION",
            "task_id": "q_demo",
            "query": "Which element has atomic number 6?",
            "options": ["H", "C", "O", "N"],
        },
        route={"specialist": "auto"},
        specialist="auto",
        domain_hint="chemistry",
        use_enriched=False,
    )

    assert runtime.last_task is not None
    assert runtime.last_task["type"] == "QUESTION_TASK"
    assert runtime.last_task["surface_kind"] == "QUESTION"
    assert result["query_id"] == "kvq_00000001"
