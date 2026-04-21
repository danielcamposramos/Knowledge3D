"""Queued TRM game-loop transport around the sovereign runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from .ring_buffer import RingBuffer, RingWindow


@dataclass(frozen=True)
class TRMQueuedInput:
    request_id: str
    offset: int
    length: int
    payload: dict[str, Any]
    created_at: float
    max_wall_ms: int | None


@dataclass(frozen=True)
class TRMQueuedOutput:
    request_id: str
    offset: int
    length: int
    payload: dict[str, Any]
    emitted_at: float


class TRMGameLoop:
    """Queued shell that moves task ingress/egress into bounded buffers."""

    def __init__(
        self,
        knowledgeverse: Any,
        *,
        bridge: Any | None = None,
        input_ring: RingBuffer | None = None,
        output_ring: RingBuffer | None = None,
        input_size_mb: int = 512,
        output_size_mb: int = 256,
    ) -> None:
        self.knowledgeverse = knowledgeverse
        self.bridge = bridge
        self.input_ring = input_ring or RingBuffer(size_mb=input_size_mb)
        self.output_ring = output_ring or RingBuffer(size_mb=output_size_mb)
        self._lock = threading.RLock()
        self._pending_inputs: deque[TRMQueuedInput] = deque()
        self._completed_outputs: deque[TRMQueuedOutput] = deque()
        self._inputs_by_request: dict[str, TRMQueuedInput] = {}
        self._outputs_by_request: dict[str, TRMQueuedOutput] = {}
        self._inflight_requests: set[str] = set()
        self._abandoned_requests: set[str] = set()
        self._sequence = 0
        self._tick = 0
        self._active = False
        self._last_tick_result: dict[str, Any] = {}
        self._last_action_buffers: list[list[int]] = []
        self._trace_handle = self._open_trace_handle()

    def start(self) -> None:
        self._active = True
        bridge = self._bridge_or_none()
        if bridge is not None and hasattr(bridge, "start_tick_loop"):
            bridge.start_tick_loop()

    def stop(self) -> None:
        self._active = False
        bridge = self._bridge_or_none()
        if bridge is not None and hasattr(bridge, "stop_tick_loop"):
            bridge.stop_tick_loop()

    def is_active(self) -> bool:
        return bool(self._active)

    def enqueue_task(
        self,
        *,
        task: dict[str, Any],
        route: dict[str, Any] | None,
        specialist: str,
        domain_hint: str | None,
        use_enriched: bool,
        max_wall_ms: int | None = None,
    ) -> str:
        with self._lock:
            self._sequence += 1
            request_id = f"trmio_{self._sequence:08d}"
        payload = {
            "request_id": request_id,
            "task": dict(task or {}),
            "route": dict(route or {}),
            "specialist": str(specialist or "auto"),
            "domain_hint": str(domain_hint).strip() if domain_hint is not None else None,
            "use_enriched": bool(use_enriched),
            "queued_at": time.time(),
            "max_wall_ms": int(max_wall_ms) if max_wall_ms is not None else None,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        offset = self.input_ring.write(raw)
        record = TRMQueuedInput(
            request_id=request_id,
            offset=int(offset),
            length=int(len(raw)),
            payload=payload,
            created_at=float(payload["queued_at"]),
            max_wall_ms=int(max_wall_ms) if max_wall_ms is not None else None,
        )
        with self._lock:
            self._pending_inputs.append(record)
            self._inputs_by_request[request_id] = record
        self._trace_event(
            "enqueue",
            request_id=request_id,
            task_id=payload.get("task", {}).get("task_id"),
            elapsed_ms=0.0,
            max_wall_ms=record.max_wall_ms,
        )
        return request_id

    def tick(self, *, max_tasks: int = 1) -> int:
        if not self._active:
            return 0
        bridge = self._resolve_bridge()
        processed = 0
        while processed < max(1, int(max_tasks)):
            with self._lock:
                if not self._pending_inputs:
                    break
                record = self._pending_inputs.popleft()
                self._inflight_requests.add(record.request_id)
            result = self._run_query_tick(bridge, record)
            result.setdefault(
                "trm_io",
                {
                    "request_id": record.request_id,
                    "tick": self._tick + 1,
                    "input_offset": int(record.offset),
                    "input_length": int(record.length),
                },
            )
            output_payload = {
                "request_id": record.request_id,
                "result": result,
                "emitted_at": time.time(),
            }
            raw = json.dumps(
                output_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            offset = self.output_ring.write(raw)
            output = TRMQueuedOutput(
                request_id=record.request_id,
                offset=int(offset),
                length=int(len(raw)),
                payload=output_payload,
                emitted_at=float(output_payload["emitted_at"]),
            )
            with self._lock:
                self._inflight_requests.discard(record.request_id)
                if record.request_id in self._abandoned_requests:
                    self._abandoned_requests.discard(record.request_id)
                    self._inputs_by_request.pop(record.request_id, None)
                    self._tick += 1
                    processed += 1
                    continue
                self._completed_outputs.append(output)
                self._outputs_by_request[record.request_id] = output
            self._tick += 1
            processed += 1
            self._trace_event(
                "output",
                request_id=record.request_id,
                task_id=record.payload.get("task", {}).get("task_id"),
                tick=self._tick,
                elapsed_ms=max(0.0, (output.emitted_at - record.created_at) * 1000.0),
            )
        if processed <= 0:
            self._run_background_tick(bridge)
            with self._lock:
                self._tick += 1
            processed = 1
        return processed

    def wait_output(
        self,
        request_id: str,
        *,
        max_ticks: int = 1,
        max_wall_ms: int | None = None,
    ) -> dict[str, Any] | None:
        request_key = str(request_id)
        inline_ticks_remaining = max(0, int(max_ticks))
        while True:
            with self._lock:
                output = self._outputs_by_request.get(request_key)
            if output is not None:
                return dict(output.payload.get("result") or {})
            timeout_payload = self._maybe_timeout_request(request_key, max_wall_ms=max_wall_ms)
            if timeout_payload is not None:
                return timeout_payload
            if self._should_inline_wait_tick() and inline_ticks_remaining > 0:
                inline_ticks_remaining -= 1
                self.tick(max_tasks=1)
                continue
            if inline_ticks_remaining <= 0 and not self._has_live_request(request_key):
                return None
            time.sleep(0.005)

    def read_input_packet(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._inputs_by_request.get(str(request_id))
        if record is None:
            return None
        raw = self.input_ring.read_at(record.offset, record.length)
        return json.loads(raw.decode("utf-8"))

    def read_output_packet(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            output = self._outputs_by_request.get(str(request_id))
        if output is None:
            return None
        raw = self.output_ring.read_at(output.offset, output.length)
        return json.loads(raw.decode("utf-8"))

    def pump_until_idle(self, *, max_batch: int = 8) -> int:
        processed = 0
        while True:
            with self._lock:
                has_pending = bool(self._pending_inputs)
            if not has_pending:
                break
            processed += self.tick(max_tasks=max(1, int(max_batch)))
        return processed

    def snapshot(self) -> dict[str, Any]:
        input_window = self.input_ring.readable_window()
        output_window = self.output_ring.readable_window()
        with self._lock:
            pending_inputs = int(len(self._pending_inputs))
            completed_outputs = int(len(self._completed_outputs))
            inflight_requests = int(len(self._inflight_requests))
            abandoned_requests = int(len(self._abandoned_requests))
            tick = int(self._tick)
        return {
            "active": bool(self._active),
            "tick": tick,
            "pending_inputs": pending_inputs,
            "completed_outputs": completed_outputs,
            "inflight_requests": inflight_requests,
            "abandoned_requests": abandoned_requests,
            "input_size": int(self.input_ring.size()),
            "output_size": int(self.output_ring.size()),
            "input_window": self._window_payload(input_window),
            "output_window": self._window_payload(output_window),
            "bridge": self._bridge_status_payload(),
            "last_tick_result": dict(self._last_tick_result),
            "last_action_buffers": [list(row) for row in self._last_action_buffers],
        }

    @staticmethod
    def _window_payload(window: RingWindow) -> dict[str, int]:
        return {"start": int(window.start), "length": int(window.length)}

    def _bridge_or_none(self) -> Any | None:
        if self.bridge is not None:
            return self.bridge
        launcher = getattr(self.knowledgeverse, "_trm", None)
        if launcher is not None:
            bridge = getattr(launcher, "_step_fused_bridge", None)
            if bridge is not None:
                return bridge
        candidate = getattr(self.knowledgeverse, "bridge", None)
        if candidate is not None:
            return candidate
        return None

    def _resolve_bridge(self) -> Any:
        bridge = self._bridge_or_none()
        if bridge is None:
            raise RuntimeError("trm_game_loop_missing_fused_bridge")
        return bridge

    def _bridge_status_payload(self) -> dict[str, Any]:
        bridge = self._bridge_or_none()
        if bridge is None or not hasattr(bridge, "tick_loop_status"):
            return {}
        return dict(bridge.tick_loop_status())

    def _action_buffer_payload(self, bridge: Any) -> list[list[int]]:
        if not hasattr(bridge, "read_action_buffers_words"):
            return []
        return bridge.read_action_buffers_words()

    def _run_background_tick(self, bridge: Any) -> dict[str, Any]:
        tick_result = dict(bridge.launch_tick(delta_time=0.02))
        action_buffers = self._action_buffer_payload(bridge)
        self._last_tick_result = dict(tick_result)
        self._last_action_buffers = [list(row) for row in action_buffers]
        return {
            "status": "ok",
            "mode": "background_tick",
            "trm_tick": tick_result,
            "action_buffers": action_buffers,
        }

    def _run_query_tick(self, bridge: Any, record: TRMQueuedInput) -> dict[str, Any]:
        tick_result = dict(bridge.run_query_tick(delta_time=0.02))
        action_buffers = self._action_buffer_payload(bridge)
        # First-class halting scalar: pick up whatever the swarm halting
        # gate published during this task's reasoning path. The swarm
        # bridge writes a PTX-sourced Q15 fixed-point max-belief into a
        # mapped-host scalar; knowledgeverse stashes the ratio on
        # `_last_swarm_halting_value` at each swarm.tick() call.
        # See TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md §3.4
        halting_value = getattr(self.knowledgeverse, "_last_swarm_halting_value", None)
        if halting_value is not None:
            clamped = float(halting_value)
            if clamped < 0.0:
                clamped = 0.0
            elif clamped > 1.0:
                clamped = 1.0
            tick_result["halting_value"] = clamped
        self._last_tick_result = dict(tick_result)
        self._last_action_buffers = [list(row) for row in action_buffers]
        task_result = self._materialize_task_result(
            bridge=bridge,
            record=record,
            action_buffers=action_buffers,
            tick_result=tick_result,
        )
        answer_text = str(task_result.get("answer_text") or "")
        return {
            "status": str(task_result.get("status") or "ok"),
            "mode": "query_tick",
            "gpu_execution": True,
            "trm_tick": tick_result,
            "action_buffers": action_buffers,
            "task_result": task_result,
            "answer": answer_text,
            "predicted_answer": answer_text,
            "response": answer_text,
            "result": answer_text,
        }

    def _materialize_task_result(
        self,
        *,
        bridge: Any,
        record: TRMQueuedInput,
        action_buffers: list[list[int]],
        tick_result: dict[str, Any],
    ) -> dict[str, Any]:
        task_payload = dict(record.payload.get("task") or {})
        route_payload = dict(record.payload.get("route") or {})
        surface_kind = str(
            task_payload.get("surface_kind")
            or task_payload.get("type")
            or "GENERAL"
        ).strip().upper() or "GENERAL"
        meaning_class = str(
            task_payload.get("meaning_class") or "FACTUAL_RECALL"
        ).strip().upper() or "FACTUAL_RECALL"

        decode: dict[str, Any] = {}
        decoder = getattr(bridge, "_answer_decode_from_action_buffer", None)
        if callable(decoder):
            try:
                decode = dict(decoder(action_buffers) or {})
            except Exception:
                decode = {}

        top_star_raw = decode.get("top_star")
        top_star = dict(top_star_raw) if isinstance(top_star_raw, dict) else {}
        top_star_idx = int(decode.get("top_star_idx", -1) or -1)
        tablet_result_value = decode.get("tablet_result_value")
        decode_materialized = bool(decode.get("answer_materialized"))

        route_family = str(
            top_star.get("route_family")
            or route_payload.get("route_family")
            or surface_kind
        )

        stars = [top_star] if top_star else []
        runtime_packet: dict[str, Any] = {}
        kv = getattr(self, "knowledgeverse", None)
        if kv is not None and hasattr(kv, "materialize_runtime_result"):
            try:
                runtime_packet = dict(
                    kv.materialize_runtime_result(
                        task=task_payload,
                        route_family=route_family,
                        answer_kind=str(top_star.get("answer_kind") or ""),
                        answer_index=0,
                        stars=stars,
                    )
                    or {}
                )
            except Exception:
                runtime_packet = {}

        answer_text = str(runtime_packet.get("answer_text") or "").strip()
        numeric_answer = runtime_packet.get("numeric_answer")
        answer_choice = str(runtime_packet.get("answer_choice") or "").strip()
        output_grid = runtime_packet.get("output_grid")
        action_index = runtime_packet.get("action_index")
        action_name = str(runtime_packet.get("action_name") or "").strip()
        answer_kind = str(runtime_packet.get("answer_kind") or "").strip()

        if not answer_text and top_star:
            metadata = top_star.get("metadata") if isinstance(top_star.get("metadata"), dict) else {}
            for value in (
                metadata.get("answer_text"),
                metadata.get("resolved_answer"),
                metadata.get("boxed_answer"),
                top_star.get("answer_text"),
                top_star.get("answer"),
                top_star.get("response"),
            ):
                if value is None:
                    continue
                candidate = str(value).strip()
                if candidate:
                    answer_text = candidate
                    break

        if numeric_answer is None and tablet_result_value is not None:
            try:
                numeric_answer = int(tablet_result_value)
            except Exception:
                numeric_answer = None

        if not answer_text and numeric_answer is not None:
            answer_text = str(numeric_answer)

        materialized = bool(
            answer_text
            or answer_choice
            or numeric_answer is not None
            or output_grid is not None
            or action_index is not None
            or action_name
        )
        status = "ok" if materialized else "error"
        failure_code = "" if materialized else "no_answer_materialized_from_action_buffer"
        if not answer_kind:
            if output_grid is not None:
                answer_kind = "grid"
            elif action_index is not None or action_name:
                answer_kind = "action"
            elif answer_choice:
                answer_kind = "choice"
            elif numeric_answer is not None and not answer_text:
                answer_kind = "numeric"
            elif answer_text:
                answer_kind = "text"
            else:
                answer_kind = "none"

        return {
            "status": status,
            "mode": "query_tick",
            "runtime": "knowledgeverse_dispatch_session",
            "solver": "knowledgeverse_dispatch_session",
            "program_id": "trm_step_fused_query_tick",
            "gpu_execution": True,
            "answer": answer_text,
            "predicted_answer": answer_text,
            "response": answer_text,
            "result": answer_text,
            "answer_text": answer_text,
            "numeric_answer": numeric_answer,
            "answer_choice": answer_choice,
            "output_grid": output_grid,
            "action_index": action_index,
            "action_name": action_name,
            "answer_kind": answer_kind,
            "answer_materialized": materialized,
            "failure_code": failure_code,
            "route_family": route_family,
            "meaning_class": meaning_class,
            "top_star_idx": top_star_idx,
            "top_star_id": str(top_star.get("id", "")),
            "top_star": top_star,
            "winner_role": str(top_star.get("selection_role", "")),
            "task_id": task_payload.get("task_id"),
            "request_id": record.request_id,
            "tablet_result_value": tablet_result_value,
            "decode_materialized_from_action_buffer": decode_materialized,
            "trm_recursion_steps": int(tick_result.get("steps", 0) or 0),
        }

    def _has_live_request(self, request_id: str) -> bool:
        with self._lock:
            return (
                request_id in self._inputs_by_request
                or request_id in self._inflight_requests
                or request_id in self._outputs_by_request
            )

    def _should_inline_wait_tick(self) -> bool:
        return not bool(getattr(self.knowledgeverse, "_external_tick_driver_active", False))

    def _maybe_timeout_request(self, request_id: str, *, max_wall_ms: int | None) -> dict[str, Any] | None:
        with self._lock:
            output = self._outputs_by_request.get(request_id)
            if output is not None:
                return dict(output.payload.get("result") or {})
            record = self._inputs_by_request.get(request_id)
            inflight = request_id in self._inflight_requests
            tick = int(self._tick)
        if record is None:
            return None
        effective_max_wall_ms = record.max_wall_ms if max_wall_ms is None else int(max_wall_ms)
        if effective_max_wall_ms is None or effective_max_wall_ms <= 0:
            return None
        elapsed_ms = max(0.0, (time.time() - record.created_at) * 1000.0)
        if elapsed_ms < float(effective_max_wall_ms):
            return None
        with self._lock:
            self._inputs_by_request.pop(request_id, None)
            if inflight:
                self._abandoned_requests.add(request_id)
            else:
                self._pending_inputs = deque(
                    item for item in self._pending_inputs if item.request_id != request_id
                )
        payload = self._wall_timeout_payload(record, elapsed_ms=elapsed_ms, tick=tick)
        self._trace_event(
            "wall_timeout",
            request_id=request_id,
            task_id=record.payload.get("task", {}).get("task_id"),
            tick=tick,
            elapsed_ms=elapsed_ms,
            max_wall_ms=effective_max_wall_ms,
        )
        return payload

    def _wall_timeout_payload(
        self,
        record: TRMQueuedInput,
        *,
        elapsed_ms: float,
        tick: int,
    ) -> dict[str, Any]:
        task_payload = dict(record.payload.get("task") or {})
        task_id = task_payload.get("task_id")
        meaning_class = str(task_payload.get("meaning_class") or "FACTUAL_RECALL").strip().upper() or "FACTUAL_RECALL"
        low_confidence = bool(task_payload.get("low_confidence_routing", False))
        timeout_result = {
            "status": "error",
            "failure_code": "wall_timeout",
            "elapsed_ms": float(elapsed_ms),
            "task_id": task_id,
            "request_id": record.request_id,
            "meaning_class": meaning_class,
            "answer": "",
            "predicted_answer": "",
            "response": "",
            "result": "",
        }
        if low_confidence:
            timeout_result["low_confidence_routing"] = True
        payload = {
            "status": "error",
            "failure_code": "wall_timeout",
            "elapsed_ms": float(elapsed_ms),
            "task_id": task_id,
            "request_id": record.request_id,
            "mode": "query_tick",
            "gpu_execution": True,
            "meaning_class": meaning_class,
            "answer": "",
            "predicted_answer": "",
            "response": "",
            "result": "",
            "trm_io": {
                "request_id": record.request_id,
                "tick": int(tick),
                "input_offset": int(record.offset),
                "input_length": int(record.length),
            },
            "trm_tick": {"tick": int(tick)},
            "action_buffers": [],
            "task_result": timeout_result,
        }
        if low_confidence:
            payload["low_confidence_routing"] = True
        return payload

    def _open_trace_handle(self):
        trace_path = str(os.environ.get("K3D_RING_TRACE_PATH", "")).strip()
        if not trace_path:
            return None
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", buffering=1)

    def _trace_event(
        self,
        event: str,
        *,
        request_id: str,
        task_id: Any = None,
        tick: int | None = None,
        elapsed_ms: float | None = None,
        max_wall_ms: int | None = None,
    ) -> None:
        if self._trace_handle is None:
            return
        payload: dict[str, Any] = {
            "ts": time.time(),
            "event": str(event),
            "request_id": str(request_id),
        }
        if task_id not in (None, ""):
            payload["task_id"] = str(task_id)
        if tick is not None:
            payload["tick"] = int(tick)
        if elapsed_ms is not None:
            payload["elapsed_ms"] = float(elapsed_ms)
        if max_wall_ms is not None:
            payload["max_wall_ms"] = int(max_wall_ms)
        self._trace_handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
        self._trace_handle.flush()
