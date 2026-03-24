"""Phase D.1 shell: queued TRM game-loop I/O around the current direct path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import time
from typing import Any

from .query_head_substrate import expand_embedding16_to128
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
        input_ring: RingBuffer | None = None,
        output_ring: RingBuffer | None = None,
        input_size_mb: int = 512,
        output_size_mb: int = 256,
    ) -> None:
        self.knowledgeverse = knowledgeverse
        self.input_ring = input_ring or RingBuffer(size_mb=input_size_mb)
        self.output_ring = output_ring or RingBuffer(size_mb=output_size_mb)
        self._pending_inputs: deque[TRMQueuedInput] = deque()
        self._completed_outputs: deque[TRMQueuedOutput] = deque()
        self._inputs_by_request: dict[str, TRMQueuedInput] = {}
        self._outputs_by_request: dict[str, TRMQueuedOutput] = {}
        self._sequence = 0
        self._tick = 0
        self._active = False

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

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
        processed = 0
        while self._pending_inputs and processed < max(1, int(max_tasks)):
            record = self._pending_inputs.popleft()
            task = dict(record.payload.get("task") or {})
            route = (
                dict(record.payload.get("route") or {})
                if isinstance(record.payload.get("route"), dict)
                else None
            )
            specialist = str(record.payload.get("specialist") or "auto")
            domain_hint = record.payload.get("domain_hint")
            use_enriched = bool(record.payload.get("use_enriched", True))
            dispatch_ticket = self._build_dispatch_ticket(task=task, specialist=specialist)
            result = self.knowledgeverse._execute_task_direct(
                task=task,
                route=route,
                specialist=specialist,
                domain_hint=domain_hint,
                use_enriched=use_enriched,
            )
            jarvis_brief = dict(result.get("jarvis_brief") or {}) if isinstance(result, dict) else {}
            if dispatch_ticket:
                trm_dispatch = result.setdefault("trm_dispatch", dict(dispatch_ticket))
                if jarvis_brief:
                    trm_dispatch.update(
                        {
                            "actual_swarm_groups": int(jarvis_brief.get("active_swarm_groups", 0) or 0),
                            "actual_worker_count": int(jarvis_brief.get("worker_count", 0) or 0),
                            "agreements": len(list(jarvis_brief.get("agreements") or [])),
                            "contradictions": len(list(jarvis_brief.get("contradictions") or [])),
                            "highest_confidence": str(jarvis_brief.get("highest_confidence", "")).strip(),
                        }
                    )
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
        }

    def _build_dispatch_ticket(self, *, task: dict[str, Any], specialist: str) -> dict[str, Any]:
        task_type = str(task.get("type", "")).strip().upper() or "GENERAL_TASK"
        options = task.get("options")
        option_list = list(options) if isinstance(options, list) else None
        task_complexity = self.knowledgeverse._jarvis_task_complexity(
            task_type=task_type,
            paths=[],
            options=option_list,
        )
        recommended_groups = self._recommended_groups_for_task(task_type)
        planned_groups = self.knowledgeverse._jarvis_determine_swarm_count(task_complexity)
        if recommended_groups > 0:
            planned_groups = max(int(planned_groups), int(recommended_groups))
        query_text = str(task.get("query") or task.get("prompt") or task.get("question") or "").strip()
        embedding = []
        if query_text:
            try:
                embedding = list(self.knowledgeverse._embed_query_gpu(query_text, task=task))
            except Exception:
                embedding = []
        resonance_weights: list[float] = []
        swarm = self.knowledgeverse.get_swarm_bridge()
        if swarm is not None and embedding:
            try:
                _, _, weights = swarm.execute_swarm(
                    expand_embedding16_to128(embedding),
                    num_iterations=1,
                    reset_state=True,
                    readback_mode="full",
                )
                if weights is not None:
                    resonance_weights = [float(value) for value in weights.tolist()]
            except Exception:
                resonance_weights = []
        worker_slots: list[dict[str, Any]] = []
        total_groups = max(1, int(planned_groups))
        for group_index in range(1, total_groups + 1):
            for worker_index in range(1, 10):
                weight = (
                    resonance_weights[(worker_index - 1) % len(resonance_weights)]
                    if resonance_weights
                    else 1.0
                )
                worker_slots.append(
                    {
                        "worker_id": f"g{group_index}.w{worker_index}",
                        "group": int(group_index),
                        "weight": float(weight),
                    }
                )
        return {
            "task_type": task_type,
            "specialist": str(specialist or "auto"),
            "planned_swarm_groups": total_groups,
            "recommended_swarm_groups": int(recommended_groups),
            "worker_slots": worker_slots,
            "resonance_weights": resonance_weights,
            "task_complexity": float(task_complexity),
            "gpu_utilization": float(self.knowledgeverse._jarvis_gpu_utilization()),
            "vram_free_bytes": int(self.knowledgeverse._jarvis_vram_free_bytes()),
        }

    def _recommended_groups_for_task(self, task_type: str) -> int:
        jarvis_state = getattr(self.knowledgeverse, "_jarvis_state", None)
        if not isinstance(jarvis_state, dict):
            return 0
        recommended = dict(jarvis_state.get("recommended_groups_by_task") or {})
        return max(0, int(recommended.get(str(task_type).strip(), 0) or 0))

    @staticmethod
    def _window_payload(window: RingWindow) -> dict[str, int]:
        return {"start": int(window.start), "length": int(window.length)}
