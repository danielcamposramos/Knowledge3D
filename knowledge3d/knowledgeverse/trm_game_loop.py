"""Queued TRM game-loop transport around the sovereign runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
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
        self._pending_inputs: deque[TRMQueuedInput] = deque()
        self._completed_outputs: deque[TRMQueuedOutput] = deque()
        self._inputs_by_request: dict[str, TRMQueuedInput] = {}
        self._outputs_by_request: dict[str, TRMQueuedOutput] = {}
        self._sequence = 0
        self._tick = 0
        self._active = False
        self._last_tick_result: dict[str, Any] = {}
        self._last_action_buffers: list[list[int]] = []

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
    ) -> str:
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
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        offset = self.input_ring.write(raw)
        record = TRMQueuedInput(
            request_id=request_id,
            offset=int(offset),
            length=int(len(raw)),
            payload=payload,
            created_at=float(payload["queued_at"]),
        )
        self._pending_inputs.append(record)
        self._inputs_by_request[request_id] = record
        return request_id

    def tick(self, *, max_tasks: int = 1) -> int:
        if not self._active:
            return 0
        bridge = self._resolve_bridge()
        processed = 0
        while self._pending_inputs and processed < max(1, int(max_tasks)):
            record = self._pending_inputs.popleft()
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
            self._completed_outputs.append(output)
            self._outputs_by_request[record.request_id] = output
            self._tick += 1
            processed += 1
        if processed <= 0:
            self._run_background_tick(bridge)
            self._tick += 1
            processed = 1
        return processed

    def wait_output(self, request_id: str, *, max_ticks: int = 1) -> dict[str, Any] | None:
        if request_id not in self._outputs_by_request:
            self.tick(max_tasks=max(1, int(max_ticks)))
        output = self._outputs_by_request.get(str(request_id))
        if output is None:
            return None
        return dict(output.payload.get("result") or {})

    def read_input_packet(self, request_id: str) -> dict[str, Any] | None:
        record = self._inputs_by_request.get(str(request_id))
        if record is None:
            return None
        raw = self.input_ring.read_at(record.offset, record.length)
        return json.loads(raw.decode("utf-8"))

    def read_output_packet(self, request_id: str) -> dict[str, Any] | None:
        output = self._outputs_by_request.get(str(request_id))
        if output is None:
            return None
        raw = self.output_ring.read_at(output.offset, output.length)
        return json.loads(raw.decode("utf-8"))

    def pump_until_idle(self, *, max_batch: int = 8) -> int:
        processed = 0
        while self._pending_inputs:
            processed += self.tick(max_tasks=max(1, int(max_batch)))
        return processed

    def snapshot(self) -> dict[str, Any]:
        input_window = self.input_ring.readable_window()
        output_window = self.output_ring.readable_window()
        return {
            "active": bool(self._active),
            "tick": int(self._tick),
            "pending_inputs": int(len(self._pending_inputs)),
            "completed_outputs": int(len(self._completed_outputs)),
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
        payload = dict(record.payload)
        dispatch = self.knowledgeverse._dispatch_sovereign_task(
            task=dict(payload.get("task") or {}),
            route=dict(payload.get("route") or {}),
            specialist=str(payload.get("specialist") or "auto"),
            domain_hint=payload.get("domain_hint"),
            use_enriched=bool(payload.get("use_enriched", True)),
        )
        dispatch_result = dict(dispatch or {})
        route_payload = dict(payload.get("route") or {})
        if route_payload and not isinstance(dispatch_result.get("route"), dict):
            dispatch_result["route"] = route_payload
        task_result = dispatch_result.get("task_result")
        if not isinstance(task_result, dict):
            task_result = {
                key: value
                for key, value in dispatch_result.items()
                if key not in {"task_result", "trm_tick", "action_buffers", "trm_io", "mode"}
            }
        else:
            task_result = dict(task_result)
        if route_payload and not isinstance(task_result.get("route"), dict):
            task_result["route"] = route_payload
        dispatch_result["task_result"] = task_result
        self._last_tick_result = dict(tick_result)
        self._last_action_buffers = [list(row) for row in action_buffers]
        dispatch_result.setdefault("status", "ok")
        dispatch_result["mode"] = "query_tick"
        dispatch_result["trm_tick"] = tick_result
        dispatch_result["action_buffers"] = action_buffers
        return dispatch_result
