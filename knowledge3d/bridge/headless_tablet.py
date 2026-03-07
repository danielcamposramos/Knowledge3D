from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from knowledge3d.cranium.actions import ACTION_BUFFER_DTYPE, ActionBuffer, ActionType

from .memory_tablet import MemoryTablet


class CommandHandler(Protocol):
    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


_TABLET_MUTATION_TYPES = {
    "ARC": 1,
    "MATH": 2,
    "LHE": 3,
    "MMLU": 4,
}

_SPECIALIST_CODES = {
    "auto": 0,
    "visual": 1,
    "math": 2,
    "chat": 3,
    "grammar": 4,
    "any": 5,
}


def _hash_words(*parts: str) -> tuple[int, int]:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    lo = int.from_bytes(digest[:4], "little", signed=False)
    hi = int.from_bytes(digest[4:8], "little", signed=False)
    return lo, hi


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _normalise_text_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\\!", "")
    text = text.replace("\\$", "$")
    return re.sub(r"\s+", " ", text)


def _match_numeric_or_text(predicted: Any, expected: Any) -> bool:
    pred_text = _normalise_text_answer(predicted)
    exp_text = _normalise_text_answer(expected)
    if not pred_text or not exp_text:
        return False
    def _numeric_form(text: str) -> float | None:
        cleaned = (
            text.strip()
            .replace("$", "")
            .replace(",", "")
            .replace("%", "")
        )
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            return None
    try:
        pred_value = _numeric_form(pred_text)
        exp_value = _numeric_form(exp_text)
        if pred_value is not None and exp_value is not None:
            return abs(pred_value - exp_value) <= 1e-5
    except Exception:
        pass
    return pred_text == exp_text


@dataclass(frozen=True)
class TabletEnvelope:
    benchmark: str
    task_id: str
    query: str
    specialist: str
    task: dict[str, Any]
    domain_hint: str | None = None
    galaxies: tuple[str, ...] = ()
    user_lang: str = "en"
    document_langs: tuple[str, ...] = ("en",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_route_payload(self, *, use_enriched: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": "ROUTE",
            "specialist": self.specialist,
            "use_enriched": bool(use_enriched),
            "task": dict(self.task),
        }
        if self.query:
            payload["query"] = self.query
        if self.domain_hint:
            payload["domain_hint"] = self.domain_hint
        if self.galaxies:
            payload["galaxies"] = list(self.galaxies)
        return payload

    def to_action_buffer(self, *, confidence: float = 0.95, curiosity: float = 0.0) -> ActionBuffer:
        buf = ActionBuffer(np.zeros(1, dtype=ACTION_BUFFER_DTYPE))
        buf.buffer["action_type"][0] = np.uint32(ActionType.UPDATE_TABLET.value)
        buf.buffer["confidence"][0] = np.float32(confidence)
        buf.buffer["curiosity"][0] = np.float32(curiosity)
        buf.buffer["tablet_mutation_type"][0] = np.uint32(
            _TABLET_MUTATION_TYPES.get(self.benchmark.upper(), 0)
        )
        task_lo, task_hi = _hash_words(self.benchmark, self.task_id)
        query_lo, query_hi = _hash_words(self.query)
        specialist_code = _SPECIALIST_CODES.get(str(self.specialist).lower(), 0)
        payload = np.array(
            [
                task_lo,
                task_hi,
                query_lo,
                query_hi,
                min(len(self.query.encode("utf-8")), 0xFFFFFFFF),
                specialist_code,
            ],
            dtype=np.uint32,
        )
        buf.buffer["tablet_data"][0][:] = payload
        return buf


class TabletIngest:
    """Normalize external benchmark tasks into the standard tablet route contract."""

    @staticmethod
    def arc_task(
        *,
        task_id: str,
        training_examples: Sequence[dict[str, Any]],
        input_grid: Any,
        expected_output: Any | None = None,
    ) -> TabletEnvelope:
        return TabletEnvelope(
            benchmark="ARC",
            task_id=str(task_id),
            query="solve arc transformation task",
            specialist="visual",
            domain_hint="visual",
            galaxies=("Drawing", "Tool", "Grammar"),
            task={
                "type": "ARC_TASK",
                "task_id": str(task_id),
                "query": "solve arc transformation task",
                "training_examples": list(training_examples),
                "input_grid": input_grid,
                "expected_output": expected_output,
            },
            metadata={"expected_output": expected_output},
        )

    @staticmethod
    def math_problem(
        *,
        task_id: str,
        question: str,
        competition: str | None = None,
        expected_answer: Any | None = None,
    ) -> TabletEnvelope:
        return TabletEnvelope(
            benchmark="MATH",
            task_id=str(task_id),
            query=str(question),
            specialist="math",
            domain_hint="math",
            galaxies=("Math", "Grammar", "Tool"),
            task={
                "type": "MATH_TASK",
                "task_id": str(task_id),
                "query": str(question),
                "question": str(question),
                "competition": competition,
                "expected_answer": expected_answer,
            },
            metadata={"expected_answer": expected_answer, "competition": competition},
        )

    @staticmethod
    def lhe_question(
        *,
        task_id: str,
        question: str,
        options: Sequence[str] | None = None,
        domain: str = "multi",
        expected_answer: str | None = None,
    ) -> TabletEnvelope:
        option_list = [str(option) for option in (options or [])]
        return TabletEnvelope(
            benchmark="LHE",
            task_id=str(task_id),
            query=str(question),
            specialist="auto",
            domain_hint=str(domain or "multi"),
            galaxies=(),
            task={
                "type": "LHE_TASK",
                "task_id": str(task_id),
                "query": str(question),
                "prompt": str(question),
                "messages": [{"role": "user", "content": str(question)}],
                "options": option_list,
                "domain_hint": str(domain or "multi"),
                "expected_answer": expected_answer,
            },
            metadata={"expected_answer": expected_answer, "options": option_list},
        )


class TabletEmit:
    """Convert routed K3D results back into benchmark-native outputs."""

    @staticmethod
    def emit(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        benchmark = envelope.benchmark.upper()
        if benchmark == "ARC":
            return TabletEmit.arc_result(envelope, response)
        if benchmark == "MATH":
            return TabletEmit.math_result(envelope, response)
        if benchmark == "LHE":
            return TabletEmit.lhe_result(envelope, response)
        raise ValueError(f"unsupported benchmark envelope: {envelope.benchmark}")

    @staticmethod
    def arc_result(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        task_result = _as_dict(response.get("task_result"))  # type: ignore[arg-type]
        route_payload = _as_dict(task_result.get("route")) or _as_dict(response.get("route"))  # type: ignore[arg-type]
        predicted = task_result.get("output_grid")
        expected = envelope.metadata.get("expected_output")
        success = str(response.get("status", "")).lower() == "ok" and str(
            task_result.get("status", "")
        ).lower() in {"ok", "success"}
        return {
            "task_id": envelope.task_id,
            "status": "success" if success else "error",
            "predicted": predicted,
            "expected": expected,
            "correct": bool(expected is not None and predicted == expected),
            "route": route_payload,
            "task_result": task_result,
        }

    @staticmethod
    def math_result(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        task_result = _as_dict(response.get("task_result"))  # type: ignore[arg-type]
        route_payload = _as_dict(task_result.get("route")) or _as_dict(response.get("route"))  # type: ignore[arg-type]
        predicted = task_result.get("result")
        if predicted is None:
            predicted = task_result.get("predicted_answer")
        expected = envelope.metadata.get("expected_answer")
        success = str(response.get("status", "")).lower() == "ok" and str(
            task_result.get("status", "")
        ).lower() in {"ok", "success"}
        return {
            "task_id": envelope.task_id,
            "status": "success" if success else "error",
            "predicted_answer": predicted,
            "expected_answer": expected,
            "correct": bool(expected is not None and _match_numeric_or_text(predicted, expected)),
            "route": route_payload,
            "task_result": task_result,
        }

    @staticmethod
    def lhe_result(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        task_result = _as_dict(response.get("task_result"))  # type: ignore[arg-type]
        route_payload = _as_dict(task_result.get("route")) or _as_dict(response.get("route"))  # type: ignore[arg-type]
        raw_answer = task_result.get("response") or task_result.get("answer") or task_result.get("result") or ""
        options = [str(option) for option in envelope.metadata.get("options", [])]
        predicted = _normalise_text_answer(raw_answer)
        if options:
            lowered = predicted.lower()
            for option in options:
                pattern = rf"(?<![A-Za-z0-9_]){re.escape(option.lower())}(?![A-Za-z0-9_])"
                if re.search(pattern, lowered):
                    predicted = option
                    break
        expected = envelope.metadata.get("expected_answer")
        success = str(response.get("status", "")).lower() == "ok" and str(
            task_result.get("status", "")
        ).lower() in {"ok", "success"}
        return {
            "task_id": envelope.task_id,
            "status": "success" if success else "error",
            "predicted_answer": predicted,
            "correct_answer": expected,
            "correct": bool(expected is not None and _match_numeric_or_text(predicted, expected)),
            "route": route_payload,
            "task_result": task_result,
        }


class HeadlessTabletMPC:
    """
    Headless Tablet boundary for benchmark-style clients.

    This keeps external parsing/formatting at the boundary while routing the
    interior work through the same daemon `ROUTE` contract as the live system.
    """

    def __init__(
        self,
        *,
        command_handler: CommandHandler | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        knowledgeverse: Any | None = None,
        storage_root: str | Path | None = None,
        tablet: MemoryTablet | None = None,
        enable_sublex: bool = False,
    ) -> None:
        self.tablet = tablet or MemoryTablet(enable_sublex=enable_sublex)
        self._handler = command_handler or self._build_local_daemon_handler(
            knowledgeverse=knowledgeverse,
            storage_root=storage_root,
        )

    def _build_local_daemon_handler(
        self,
        *,
        knowledgeverse: Any | None,
        storage_root: str | Path | None,
    ) -> CommandHandler:
        from knowledge3d.daemon.main import DaemonConfig, K3DDaemon

        if storage_root is None:
            storage_root = getattr(knowledgeverse, "storage_root", None) or "../Knowledge3D.local"
        config = DaemonConfig(storage_root=Path(storage_root))
        return K3DDaemon(config=config, knowledgeverse=knowledgeverse)

    def _handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        handler = self._handler
        if hasattr(handler, "handle_command"):
            return handler.handle_command(payload)  # type: ignore[return-value]
        return handler(payload)  # type: ignore[misc]

    def submit(self, envelope: TabletEnvelope, *, use_enriched: bool = True) -> dict[str, Any]:
        self.tablet.prepare_headless_context(
            user_lang=envelope.user_lang,
            document_langs=list(envelope.document_langs),
        )
        action_buffer = envelope.to_action_buffer()
        mutation_type, payload_words = action_buffer.extract_tablet_mutation()
        response = self._handle_command(envelope.to_route_payload(use_enriched=use_enriched))
        emitted = TabletEmit.emit(envelope, response)
        return {
            "envelope": envelope,
            "route_payload": envelope.to_route_payload(use_enriched=use_enriched),
            "response": response,
            "emitted": emitted,
            "tablet_contract": {
                "action_type": action_buffer.get_action_type().name,
                "mutation_type": int(mutation_type),
                "payload_words": payload_words.tolist(),
            },
        }
