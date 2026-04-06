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


SURFACE_KIND_GAME_2D = "GAME_2D"
SURFACE_KIND_MATH = "MATH"
SURFACE_KIND_QUESTION = "QUESTION"
SURFACE_KIND_SPATIAL_3D = "SPATIAL_3D"
SURFACE_KIND_CHAT = "CHAT"
SURFACE_KIND_GENERAL = "GENERAL"
SURFACE_KIND_GRAMMAR = "GRAMMAR"

_TABLET_MUTATION_TYPES = {
    SURFACE_KIND_GAME_2D: 1,
    SURFACE_KIND_MATH: 2,
    SURFACE_KIND_QUESTION: 3,
    SURFACE_KIND_SPATIAL_3D: 4,
}

_SPECIALIST_CODES = {
    "auto": 0,
    "visual": 1,
    "math": 2,
    "chat": 3,
    "grammar": 4,
    "any": 5,
}


def _normalize_surface_kind(value: Any) -> str:
    token = str(value or "").strip().upper()
    mapping = {
        "ARC": SURFACE_KIND_GAME_2D,
        "ARC_TASK": SURFACE_KIND_GAME_2D,
        "SPATIAL_TASK": SURFACE_KIND_GAME_2D,
        "GAME_2D": SURFACE_KIND_GAME_2D,
        "MATH": SURFACE_KIND_MATH,
        "MATH_TASK": SURFACE_KIND_MATH,
        "GSM8K_TASK": SURFACE_KIND_MATH,
        "IMO_TASK": SURFACE_KIND_MATH,
        "QUESTION": SURFACE_KIND_QUESTION,
        "QUESTION_TASK": SURFACE_KIND_QUESTION,
        "MMLU_TASK": SURFACE_KIND_QUESTION,
        "LHE_TASK": SURFACE_KIND_QUESTION,
        "SPATIAL_3D": SURFACE_KIND_SPATIAL_3D,
        "CHAT": SURFACE_KIND_CHAT,
        "CHAT_TASK": SURFACE_KIND_CHAT,
        "GENERAL": SURFACE_KIND_GENERAL,
        "GENERAL_TASK": SURFACE_KIND_GENERAL,
        "GRAMMAR": SURFACE_KIND_GRAMMAR,
        "GRAMMAR_TASK": SURFACE_KIND_GRAMMAR,
    }
    return mapping.get(token, token or SURFACE_KIND_GENERAL)


def _hash_words(*parts: str) -> tuple[int, int]:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    lo = int.from_bytes(digest[:4], "little", signed=False)
    hi = int.from_bytes(digest[4:8], "little", signed=False)
    return lo, hi


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _route_internal_labels(
    *,
    route_payload: Mapping[str, Any] | None = None,
    task_result: Mapping[str, Any] | None = None,
) -> set[str]:
    labels: set[str] = set()
    for container in (route_payload, task_result):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "router_star",
            "executor_star",
            "validator_star",
            "winner_star",
            "router_star_id",
            "executor_star_id",
            "validator_star_id",
            "winner_star_id",
        ):
            value = str(container.get(key) or "").strip().lower()
            if value:
                labels.add(value)
                labels.add(re.sub(r"[^a-z0-9]+", "_", value).strip("_"))
    return labels


def _looks_like_internal_route_label(
    value: Any,
    *,
    route_payload: Mapping[str, Any] | None = None,
    task_result: Mapping[str, Any] | None = None,
) -> bool:
    text = _normalise_text_answer(value)
    if not text:
        return False
    lowered = text.lower()
    canonical = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if lowered in _route_internal_labels(route_payload=route_payload, task_result=task_result):
        return True
    if canonical in _route_internal_labels(route_payload=route_payload, task_result=task_result):
        return True
    if canonical.startswith("anti_pattern_"):
        return True
    if any(
        canonical == token or canonical.endswith(f"_{token}")
        for token in ("router", "executor", "validator", "materializer", "anti_pattern")
    ):
        return True
    return lowered in {"router", "executor", "validator", "materializer", "anti_pattern"}


def _materialized_answer(
    task_result: Mapping[str, Any] | None,
    *,
    route_payload: Mapping[str, Any] | None = None,
) -> str:
    payload = _as_dict(task_result)
    has_route_signal = _has_route_decision_signal(payload, route_payload=route_payload)
    live_sovereign_placeholder = bool(
        payload.get("trm_dispatch")
        or payload.get("query_type")
        or str(payload.get("program_type") or "").strip().lower() == "gpu_task_dispatch_sovereign"
    )
    for key in ("result", "predicted_answer", "answer", "response"):
        value = payload.get(key)
        text = _normalise_text_answer(value)
        if not text:
            continue
        if not has_route_signal and live_sovereign_placeholder:
            continue
        if _looks_like_internal_route_label(text, route_payload=route_payload, task_result=payload):
            continue
        return text
    return ""


def _has_route_decision_signal(
    task_result: Mapping[str, Any] | None,
    *,
    route_payload: Mapping[str, Any] | None = None,
) -> bool:
    payload = _as_dict(task_result)
    route_data = _as_dict(route_payload)
    if int(payload.get("route_depth", 0) or 0) > 0:
        return True
    if int(payload.get("convergence_signal", 0) or 0) > 0:
        return True
    for key in (
        "router_star",
        "executor_star",
        "validator_star",
        "winner_star",
        "winner_star_id",
    ):
        if str(payload.get(key) or route_data.get(key) or "").strip():
            return True
    return False


def _flatten_route_response(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(response)
    task_result = _as_dict(payload.get("task_result"))  # type: ignore[arg-type]
    nested = _as_dict(task_result.get("task_result"))  # type: ignore[arg-type]
    if nested:
        merged_task = dict(nested)
        for key, value in task_result.items():
            if key in {"task_result", "route"}:
                continue
            merged_task.setdefault(key, value)
        route_payload = _as_dict(merged_task.get("route")) or _as_dict(task_result.get("route")) or _as_dict(payload.get("route"))
        if route_payload:
            merged_task["route"] = route_payload
            payload["route"] = route_payload
        payload["task_result"] = merged_task
        payload["status"] = str(
            payload.get("status")
            or merged_task.get("status")
            or "ok"
        )
        return payload
    if task_result:
        route_payload = _as_dict(task_result.get("route")) or _as_dict(payload.get("route"))
        if route_payload:
            task_result["route"] = route_payload
            payload["route"] = route_payload
        payload["task_result"] = task_result
    return payload


def _normalise_text_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\\!", "")
    text = text.replace("\\$", "$")
    return re.sub(r"\s+", " ", text)


def _numeric_form(value: Any) -> float | None:
    text = _normalise_text_answer(value)
    cleaned = text.strip().replace("$", "").replace(",", "").replace("%", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _match_numeric_or_text(predicted: Any, expected: Any) -> bool:
    pred_text = _normalise_text_answer(predicted)
    exp_text = _normalise_text_answer(expected)
    if not pred_text or not exp_text:
        return False
    pred_value = _numeric_form(pred_text)
    exp_value = _numeric_form(exp_text)
    if pred_value is not None and exp_value is not None:
        return abs(pred_value - exp_value) <= 1e-5
    return pred_text == exp_text


def _task_result_packet(
    task_result: Mapping[str, Any] | None,
    *,
    route_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _as_dict(task_result)
    route_data = _as_dict(route_payload)
    answer_text = _normalise_text_answer(payload.get("answer_text"))
    answer_choice = _normalise_text_answer(payload.get("answer_choice"))
    if _looks_like_internal_route_label(answer_text, route_payload=route_payload, task_result=payload):
        answer_text = ""
    if _looks_like_internal_route_label(answer_choice, route_payload=route_payload, task_result=payload):
        answer_choice = ""
    try:
        answer_index = int(payload["answer_index"]) if payload.get("answer_index") is not None else None
    except Exception:
        answer_index = None
    raw_action_name = payload.get("action_name")
    action_name = _normalise_text_answer(raw_action_name)
    if _looks_like_internal_route_label(action_name, route_payload=route_payload, task_result=payload):
        action_name = ""
    try:
        action_index = int(payload["action_index"]) if payload.get("action_index") is not None else None
    except Exception:
        action_index = None
    if not answer_text:
        answer_text = _materialized_answer(payload, route_payload=route_payload)
    numeric_answer = payload.get("numeric_answer")
    answer_materialized = bool(
        answer_text
        or answer_choice
        or numeric_answer is not None
        or payload.get("output_grid") is not None
        or action_index is not None
        or action_name
    )
    return {
        "answer_kind": str(payload.get("answer_kind") or "none"),
        "answer_text": answer_text,
        "numeric_answer": numeric_answer,
        "answer_choice": answer_choice,
        "answer_index": answer_index,
        "output_grid": payload.get("output_grid"),
        "action_index": action_index,
        "action_name": action_name,
        "answer_materialized": answer_materialized,
        "failure_code": str(payload.get("failure_code") or payload.get("failure_reason") or ""),
        "route_family": str(payload.get("route_family") or route_data.get("route_family") or ""),
        "route_depth": int(payload.get("route_depth", route_data.get("route_depth", 0)) or 0),
        "trace_star_ids": list(payload.get("trace_star_ids") or []),
        "trace_roles": list(payload.get("trace_roles") or []),
        "anti_pattern_ids": list(payload.get("anti_pattern_ids") or []),
    }


@dataclass(frozen=True)
class TabletEnvelope:
    surface_kind: str
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
        task_payload = dict(self.task)
        task_payload.setdefault("surface_kind", _normalize_surface_kind(self.surface_kind))
        payload: dict[str, Any] = {
            "command": "ROUTE",
            "surface_kind": _normalize_surface_kind(self.surface_kind),
            "specialist": self.specialist,
            "use_enriched": bool(use_enriched),
            "task": task_payload,
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
        surface_kind = _normalize_surface_kind(self.surface_kind)
        buf.buffer["tablet_mutation_type"][0] = np.uint32(_TABLET_MUTATION_TYPES.get(surface_kind, 0))
        task_lo, task_hi = _hash_words(surface_kind, self.task_id)
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
    """Normalize external I/O into the generic tablet route contract."""

    @staticmethod
    def game2d_task(
        *,
        task_id: str,
        query: str,
        input_grid: Any | None = None,
        goal_grid: Any | None = None,
        training_examples: Sequence[dict[str, Any]] | None = None,
        available_actions: Sequence[int] | None = None,
        action_options: Sequence[str] | None = None,
        expected_output: Any | None = None,
        expected_game_action: Mapping[str, Any] | None = None,
        specialist: str = "visual",
        domain_hint: str | None = "game_2d",
        galaxies: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TabletEnvelope:
        route_galaxies = tuple(
            str(name)
            for name in (
                galaxies
                or ("Drawing", "game_mechanics", "Grammar", "Tool", "Reality")
            )
            if str(name).strip()
        )
        task_payload = {
            "surface_kind": SURFACE_KIND_GAME_2D,
            "task_id": str(task_id),
            "query": str(query),
            "prompt": str(query),
            "input_grid": input_grid,
            "goal_grid": goal_grid,
            "expected_output": expected_output,
            "training_examples": list(training_examples or []),
            "available_actions": list(available_actions or []),
            "action_options": [str(option) for option in (action_options or []) if str(option).strip()],
        }
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("expected_output", expected_output)
        merged_metadata.setdefault("expected_game_action", dict(expected_game_action or {}))
        merged_metadata.setdefault("action_options", list(task_payload["action_options"]))
        merged_metadata.setdefault("available_actions", list(task_payload["available_actions"]))
        return TabletEnvelope(
            surface_kind=SURFACE_KIND_GAME_2D,
            task_id=str(task_id),
            query=str(query),
            specialist=str(specialist or "visual"),
            domain_hint=str(domain_hint).strip() if domain_hint is not None else None,
            galaxies=route_galaxies,
            task=task_payload,
            metadata=merged_metadata,
        )

    @staticmethod
    def math_task(
        *,
        task_id: str,
        question: str,
        expected_answer: Any | None = None,
        competition: str | None = None,
        galaxies: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TabletEnvelope:
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("expected_answer", expected_answer)
        merged_metadata.setdefault("competition", competition)
        return TabletEnvelope(
            surface_kind=SURFACE_KIND_MATH,
            task_id=str(task_id),
            query=str(question),
            specialist="math",
            domain_hint=None,
            galaxies=tuple(str(name) for name in (galaxies or ()) if str(name).strip()),
            task={
                "surface_kind": SURFACE_KIND_MATH,
                "task_id": str(task_id),
                "query": str(question),
                "question": str(question),
                "expected_answer": expected_answer,
            },
            metadata=merged_metadata,
        )

    @staticmethod
    def question_task(
        *,
        task_id: str,
        question: str,
        options: Sequence[str] | None = None,
        domain: str = "general",
        expected_answer: str | None = None,
        specialist: str = "auto",
        galaxies: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TabletEnvelope:
        option_list = [str(option) for option in (options or []) if str(option).strip()]
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("expected_answer", expected_answer)
        merged_metadata.setdefault("options", list(option_list))
        merged_metadata.setdefault("question_domain", str(domain or "general"))
        return TabletEnvelope(
            surface_kind=SURFACE_KIND_QUESTION,
            task_id=str(task_id),
            query=str(question),
            specialist=str(specialist or "auto"),
            domain_hint=str(domain or "general"),
            galaxies=tuple(str(name) for name in (galaxies or ()) if str(name).strip()),
            task={
                "surface_kind": SURFACE_KIND_QUESTION,
                "task_id": str(task_id),
                "query": str(question),
                "prompt": str(question),
                "question": str(question),
                "messages": [{"role": "user", "content": str(question)}],
                "options": option_list,
                "domain_hint": str(domain or "general"),
                "expected_answer": expected_answer,
            },
            metadata=merged_metadata,
        )

    @staticmethod
    def arc_task(
        *,
        task_id: str,
        training_examples: Sequence[dict[str, Any]],
        input_grid: Any,
        expected_output: Any | None = None,
    ) -> TabletEnvelope:
        return TabletIngest.game2d_task(
            task_id=task_id,
            query="2d game transformation",
            training_examples=training_examples,
            input_grid=input_grid,
            goal_grid=expected_output,
            expected_output=expected_output,
        )

    @staticmethod
    def math_problem(
        *,
        task_id: str,
        question: str,
        competition: str | None = None,
        expected_answer: Any | None = None,
    ) -> TabletEnvelope:
        return TabletIngest.math_task(
            task_id=task_id,
            question=question,
            competition=competition,
            expected_answer=expected_answer,
        )

    @staticmethod
    def lhe_question(
        *,
        task_id: str,
        question: str,
        options: Sequence[str] | None = None,
        domain: str = "general",
        expected_answer: str | None = None,
    ) -> TabletEnvelope:
        return TabletIngest.question_task(
            task_id=task_id,
            question=question,
            options=options,
            domain=domain,
            expected_answer=expected_answer,
        )

    @staticmethod
    def mmlu_question(
        *,
        task_id: str,
        question: str,
        options: Sequence[str],
        subject: str,
        expected_answer: str | None = None,
    ) -> TabletEnvelope:
        return TabletIngest.question_task(
            task_id=task_id,
            question=question,
            options=options,
            domain=subject,
            expected_answer=expected_answer,
        )


class TabletEmit:
    """Convert routed K3D results back into generic tablet outputs."""

    @staticmethod
    def emit(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        response = _flatten_route_response(response)
        surface_kind = _normalize_surface_kind(envelope.surface_kind)
        if surface_kind == SURFACE_KIND_GAME_2D:
            return TabletEmit.game2d_result(envelope, response)
        if surface_kind == SURFACE_KIND_MATH:
            return TabletEmit.math_result(envelope, response)
        if surface_kind == SURFACE_KIND_QUESTION:
            return TabletEmit.question_result(envelope, response)
        raise ValueError(f"unsupported tablet surface kind: {envelope.surface_kind}")

    @staticmethod
    def game2d_result(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        normalized_response = _flatten_route_response(response)
        task_result = _as_dict(normalized_response.get("task_result"))  # type: ignore[arg-type]
        route_payload = _as_dict(task_result.get("route")) or _as_dict(normalized_response.get("route"))  # type: ignore[arg-type]
        packet = _task_result_packet(task_result, route_payload=route_payload)
        predicted_grid = packet.get("output_grid")
        action_index = packet.get("action_index")
        action_name = str(packet.get("action_name") or "")
        expected_output = envelope.metadata.get("expected_output")
        expected_game_action = (
            dict(envelope.metadata.get("expected_game_action") or {})
            if isinstance(envelope.metadata.get("expected_game_action"), Mapping)
            else {}
        )
        success = str(normalized_response.get("status", "")).lower() == "ok" and str(
            task_result.get("status", "")
        ).lower() in {"", "ok", "success"}
        game_action: dict[str, Any]
        if predicted_grid is not None:
            game_action = {
                "kind": "grid_submission",
                "output_grid": predicted_grid,
            }
        else:
            game_action = {
                "kind": "control",
                "action_index": action_index,
                "action_name": action_name,
            }
            action_input = task_result.get("action_input")
            if isinstance(action_input, Mapping):
                game_action["action_input"] = dict(action_input)
        correct = False
        if predicted_grid is not None and expected_output is not None:
            correct = predicted_grid == expected_output
        elif expected_game_action:
            expected_index = expected_game_action.get("action_index")
            expected_name = _normalise_text_answer(expected_game_action.get("action_name"))
            if expected_index is not None and action_index is not None:
                correct = int(expected_index) == int(action_index)
            elif expected_name and action_name:
                correct = expected_name == _normalise_text_answer(action_name)
        return {
            "task_id": envelope.task_id,
            "status": "success" if success else "error",
            "answer_kind": str(packet.get("answer_kind") or "none"),
            "route_family": str(packet.get("route_family") or route_payload.get("route_family") or ""),
            "route_depth": int(packet.get("route_depth") or 0),
            "output_grid": predicted_grid,
            "action_index": action_index,
            "action_name": action_name,
            "answer_materialized": bool(packet.get("answer_materialized")),
            "failure_code": str(packet.get("failure_code") or ""),
            "trace_star_ids": list(packet.get("trace_star_ids") or []),
            "trace_roles": list(packet.get("trace_roles") or []),
            "anti_pattern_ids": list(packet.get("anti_pattern_ids") or []),
            "game_action": game_action,
            "expected_output": expected_output,
            "expected_game_action": expected_game_action,
            "correct": bool(correct),
            "route": route_payload,
            "task_result": task_result,
            "predicted_action": action_name,
        }

    @staticmethod
    def math_result(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        normalized_response = _flatten_route_response(response)
        task_result = _as_dict(normalized_response.get("task_result"))  # type: ignore[arg-type]
        route_payload = _as_dict(task_result.get("route")) or _as_dict(normalized_response.get("route"))  # type: ignore[arg-type]
        packet = _task_result_packet(task_result, route_payload=route_payload)
        expected = envelope.metadata.get("expected_answer")
        success = str(normalized_response.get("status", "")).lower() == "ok" and str(
            task_result.get("status", "")
        ).lower() in {"", "ok", "success"}
        answer_text = _normalise_text_answer(packet.get("answer_text"))
        numeric_answer = packet.get("numeric_answer")
        if numeric_answer is None:
            numeric_answer = _numeric_form(answer_text)
        predicted = answer_text if numeric_answer is None else numeric_answer
        return {
            "task_id": envelope.task_id,
            "status": "success" if success else "error",
            "route_family": str(packet.get("route_family") or route_payload.get("route_family") or ""),
            "route_depth": int(packet.get("route_depth") or 0),
            "numeric_answer": numeric_answer,
            "answer_text": answer_text,
            "predicted_answer": predicted,
            "expected_answer": expected,
            "correct": bool(expected is not None and _match_numeric_or_text(predicted, expected)),
            "answer_kind": str(packet.get("answer_kind") or "none"),
            "answer_materialized": bool(packet.get("answer_materialized") or answer_text or numeric_answer is not None),
            "failure_code": str(packet.get("failure_code") or ""),
            "failure_reason": str(packet.get("failure_code") or ""),
            "trace_star_ids": list(packet.get("trace_star_ids") or []),
            "trace_roles": list(packet.get("trace_roles") or []),
            "anti_pattern_ids": list(packet.get("anti_pattern_ids") or []),
            "route": route_payload,
            "task_result": task_result,
        }

    @staticmethod
    def question_result(envelope: TabletEnvelope, response: Mapping[str, Any]) -> dict[str, Any]:
        normalized_response = _flatten_route_response(response)
        task_result = _as_dict(normalized_response.get("task_result"))  # type: ignore[arg-type]
        route_payload = _as_dict(task_result.get("route")) or _as_dict(normalized_response.get("route"))  # type: ignore[arg-type]
        options = [str(option) for option in envelope.metadata.get("options", [])]
        packet = _task_result_packet(task_result, route_payload=route_payload)
        answer_text = _normalise_text_answer(packet.get("answer_text"))
        answer_choice = _normalise_text_answer(packet.get("answer_choice"))
        if not answer_choice and options and answer_text in options:
            answer_choice = answer_text
        if not answer_choice:
            raw_answer_index = packet.get("answer_index")
            answer_index = int(raw_answer_index) if isinstance(raw_answer_index, int) else -1
            if 0 <= answer_index < len(options):
                answer_choice = options[answer_index]
        predicted = answer_choice or answer_text
        expected = envelope.metadata.get("expected_answer")
        success = str(normalized_response.get("status", "")).lower() == "ok" and str(
            task_result.get("status", "")
        ).lower() in {"", "ok", "success"}
        return {
            "task_id": envelope.task_id,
            "status": "success" if success else "error",
            "route_family": str(packet.get("route_family") or route_payload.get("route_family") or ""),
            "route_depth": int(packet.get("route_depth") or 0),
            "answer_choice": answer_choice,
            "answer_text": answer_text,
            "predicted_answer": predicted,
            "correct_answer": expected,
            "correct": bool(expected is not None and _match_numeric_or_text(predicted, expected)),
            "answer_kind": str(packet.get("answer_kind") or "none"),
            "answer_materialized": bool(packet.get("answer_materialized") or predicted),
            "failure_code": str(packet.get("failure_code") or ""),
            "failure_reason": str(packet.get("failure_code") or ""),
            "trace_star_ids": list(packet.get("trace_star_ids") or []),
            "trace_roles": list(packet.get("trace_roles") or []),
            "anti_pattern_ids": list(packet.get("anti_pattern_ids") or []),
            "route": route_payload,
            "task_result": task_result,
        }

    mmlu_result = question_result
    lhe_result = question_result


class HeadlessTabletMPC:
    """
    Headless Tablet boundary for benchmark-style clients.

    External dataset I/O is normalized here; the interior work stays on the
    same daemon `ROUTE` contract as the live system.
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
        route_payload = envelope.to_route_payload(use_enriched=use_enriched)
        action_buffer = envelope.to_action_buffer()
        mutation_type, payload_words = action_buffer.extract_tablet_mutation()
        raw_response = self._handle_command(route_payload)
        response = _flatten_route_response(raw_response)
        emitted = TabletEmit.emit(envelope, response)
        return {
            "envelope": envelope,
            "route_payload": route_payload,
            "response": response,
            "raw_response": raw_response,
            "emitted": emitted,
            "tablet_contract": {
                "action_type": action_buffer.get_action_type().name,
                "mutation_type": int(mutation_type),
                "payload_words": payload_words.tolist(),
                "surface_kind": _normalize_surface_kind(envelope.surface_kind),
            },
        }
