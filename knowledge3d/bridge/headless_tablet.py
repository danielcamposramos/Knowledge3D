from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .memory_tablet import MemoryTablet

_ACTION_TYPES_PATH = Path(__file__).resolve().parent.parent / "cranium" / "actions" / "action_types.py"
_ACTION_TYPES_SPEC = importlib.util.spec_from_file_location("_k3d_action_types_direct", _ACTION_TYPES_PATH)
if _ACTION_TYPES_SPEC is None or _ACTION_TYPES_SPEC.loader is None:
    raise ImportError(f"unable_to_load_action_types:{_ACTION_TYPES_PATH}")
_ACTION_TYPES_MODULE = sys.modules.get(str(_ACTION_TYPES_SPEC.name))
if _ACTION_TYPES_MODULE is None:
    _ACTION_TYPES_MODULE = importlib.util.module_from_spec(_ACTION_TYPES_SPEC)
    sys.modules[str(_ACTION_TYPES_SPEC.name)] = _ACTION_TYPES_MODULE
    _ACTION_TYPES_SPEC.loader.exec_module(_ACTION_TYPES_MODULE)
ACTION_BUFFER_DTYPE = _ACTION_TYPES_MODULE.ACTION_BUFFER_DTYPE
ActionBuffer = _ACTION_TYPES_MODULE.ActionBuffer
ActionType = _ACTION_TYPES_MODULE.ActionType


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
ROUTE_POLICY_ALL_LIVE_GALAXIES = "all_live_galaxies"

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

TABLET_WORD_OFFSET_MUTATION_TYPE = 60
TABLET_WORD_OFFSET_DATA = 61
TABLET_WORD_OFFSET_RESERVED = 67
TABLET_MUTATION_MATH_RESULT = 2


@dataclass(frozen=True)
class _MathOperands:
    left: int = 0
    right: int = 0
    count: int = 0
    operator_hint: int = 0


def _extract_math_operands(query: str) -> _MathOperands:
    text = str(query or "").strip()
    match = re.search(r"(-?\d+)\s*([+\-*/^])\s*(-?\d+)", text)
    if match is None:
        return _MathOperands()
    try:
        left = int(match.group(1))
        right = int(match.group(3))
    except Exception:
        return _MathOperands()
    operator = str(match.group(2) or "")[:1]
    return _MathOperands(
        left=left,
        right=right,
        count=2,
        operator_hint=ord(operator) if operator else 0,
    )


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
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES
    result_kind: str | None = None
    user_lang: str = "en"
    document_langs: tuple[str, ...] = ("en",)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_route_payload(self, *, use_enriched: bool = True) -> dict[str, Any]:
        task_payload = dict(self.task)
        task_payload.setdefault("surface_kind", _normalize_surface_kind(self.surface_kind))
        task_payload.setdefault("route_policy", str(self.route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES))
        if self.result_kind:
            task_payload.setdefault("expected_result_kind", str(self.result_kind))
        payload: dict[str, Any] = {
            "command": "ROUTE",
            "surface_kind": _normalize_surface_kind(self.surface_kind),
            "specialist": self.specialist,
            "use_enriched": bool(use_enriched),
            "route_policy": str(self.route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
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
        if surface_kind == SURFACE_KIND_MATH:
            operands = _extract_math_operands(self.query)
            buf.buffer["tablet_reserved"][0][:] = np.array(
                [
                    operands.left & 0xFFFFFFFF,
                    operands.right & 0xFFFFFFFF,
                    operands.count & 0xFFFFFFFF,
                    operands.operator_hint & 0xFFFFFFFF,
                ],
                dtype=np.uint32,
            )
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
        route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
        result_kind: str | None = None,
        task_context: Mapping[str, Any] | None = None,
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
            "type": "ARC_TASK",
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
        task_payload["options"] = list(task_payload["action_options"])
        for key, value in dict(task_context or {}).items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            task_payload[normalized_key] = value
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("expected_output", expected_output)
        merged_metadata.setdefault("expected_game_action", dict(expected_game_action or {}))
        merged_metadata.setdefault("action_options", list(task_payload["action_options"]))
        merged_metadata.setdefault("options", list(task_payload["options"]))
        merged_metadata.setdefault("available_actions", list(task_payload["available_actions"]))
        if result_kind is not None:
            merged_metadata.setdefault("expected_result_kind", str(result_kind))
        return TabletEnvelope(
            surface_kind=SURFACE_KIND_GAME_2D,
            task_id=str(task_id),
            query=str(query),
            specialist=str(specialist or "visual"),
            domain_hint=str(domain_hint).strip() if domain_hint is not None else None,
            galaxies=route_galaxies,
            route_policy=str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
            result_kind=str(result_kind) if result_kind is not None else None,
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
        route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
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
            route_policy=str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
            task={
                "type": "MATH_TASK",
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
        route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
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
            route_policy=str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
            task={
                "type": "QUESTION_TASK",
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
        expected_result_kind = str(
            envelope.metadata.get("expected_result_kind") or envelope.result_kind or ""
        ).strip().lower()
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
        actual_result_kind = "grid" if predicted_grid is not None else "control"
        wrong_result_kind = bool(expected_result_kind) and actual_result_kind != expected_result_kind
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
        if wrong_result_kind:
            success = False
            correct = False
        failure_code = str(packet.get("failure_code") or "")
        if wrong_result_kind:
            failure_code = f"unexpected_result_kind:{actual_result_kind}"
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
            "failure_code": failure_code,
            "trace_star_ids": list(packet.get("trace_star_ids") or []),
            "trace_roles": list(packet.get("trace_roles") or []),
            "anti_pattern_ids": list(packet.get("anti_pattern_ids") or []),
            "game_action": game_action,
            "expected_output": expected_output,
            "expected_game_action": expected_game_action,
            "expected_result_kind": expected_result_kind,
            "actual_result_kind": actual_result_kind,
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
        bridge: Any | None = None,
        storage_root: str | Path | None = None,
        tablet: MemoryTablet | None = None,
        enable_sublex: bool = False,
    ) -> None:
        self.tablet = tablet or MemoryTablet(enable_sublex=enable_sublex)
        self._knowledgeverse = knowledgeverse
        self._bridge = bridge
        self._tick_seq = 0
        self._handler = command_handler or self._build_local_daemon_handler(
            knowledgeverse=knowledgeverse,
            storage_root=storage_root,
        )
        self._resolve_sovereign_bridge()

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

    def _resolve_sovereign_bridge(self) -> Any | None:
        if self._bridge is not None:
            self._bind_bridge_query_runtime()
            return self._bridge
        kv = self._knowledgeverse
        launcher = getattr(kv, "_trm", None)
        bridge = getattr(launcher, "_step_fused_bridge", None)
        if bridge is not None:
            self._bridge = bridge
            self._bind_bridge_query_runtime()
        return self._bridge

    def _bind_bridge_query_runtime(self) -> None:
        bridge = self._bridge
        kv = self._knowledgeverse
        if bridge is None or kv is None or not hasattr(bridge, "bind_query_runtime_buffers"):
            return
        state_buffers = getattr(kv, "_trm_state_buffers", {}) or {}
        weight_buffers = getattr(kv, "_trm_weight_buffers", {}) or {}
        required_state = ("d_q", "d_y", "d_z", "d_z_new", "d_y_new", "d_workspace")
        required_weights = ("W1", "W2", "W3", "W4")
        if not all(name in state_buffers for name in required_state):
            return
        if not all(name in weight_buffers for name in required_weights):
            return
        bridge.bind_query_runtime_buffers(
            q_ptr=state_buffers["d_q"],
            y_ptr=state_buffers["d_y"],
            z_ptr=state_buffers["d_z"],
            W1_ptr=weight_buffers["W1"],
            W2_ptr=weight_buffers["W2"],
            W3_ptr=weight_buffers["W3"],
            W4_ptr=weight_buffers["W4"],
            z_new_ptr=state_buffers["d_z_new"],
            y_new_ptr=state_buffers["d_y_new"],
            workspace_ptr=state_buffers["d_workspace"],
            q_input_ptr=state_buffers.get("d_q_input"),
            matryoshka_bridge=getattr(kv, "_matryoshka_bridge", None),
            matryoshka_weight_ptr=getattr(kv, "_trm_matryoshka_weight_buffer", None),
        )
        if hasattr(bridge, "bind_galaxy_table"):
            runtime = getattr(kv, "_sovereign_hot_path", None)
            star_table = getattr(runtime, "star_table", None)
            star_count = int(getattr(star_table, "star_count", 0) or 0) if star_table is not None else 0
            gpu_ptr = getattr(star_table, "gpu_ptr", None) if star_table is not None else None
            if gpu_ptr is not None and star_count > 0:
                host_stars = list(getattr(runtime, "_host_stars", None) or getattr(star_table, "_host_stars", []) or [])
                bridge.bind_galaxy_table(
                    gpu_ptr,
                    star_count,
                    embedding_dims=64,
                    host_stars=[dict(star) for star in host_stars if isinstance(star, Mapping)],
                )
            program_table = getattr(runtime, "program_table", None)
            program_ptr = getattr(program_table, "gpu_ptr", None) if program_table is not None else None
            program_size = int(getattr(program_table, "size_bytes", 0) or 0) if program_table is not None else 0
            if hasattr(bridge, "bind_program_table") and program_ptr is not None and program_size > 0:
                bridge.bind_program_table(program_ptr, program_size)

    @staticmethod
    def _action_buffer_words(action_buffer: ActionBuffer) -> list[int]:
        host = np.asarray(action_buffer.buffer)
        return [int(value) for value in host.view(np.uint32).reshape(-1)[:72]]

    @staticmethod
    def _fallback_query_embedding(envelope: TabletEnvelope) -> list[float]:
        digest = hashlib.sha512(
            "|".join(
                [
                    _normalize_surface_kind(envelope.surface_kind),
                    str(envelope.specialist),
                    str(envelope.query),
                ]
            ).encode("utf-8")
        ).digest()
        values: list[float] = []
        while len(values) < 512:
            for byte in digest:
                values.append((float(byte) / 127.5) - 1.0)
                if len(values) >= 512:
                    break
            digest = hashlib.sha512(digest).digest()
        return values

    def _query_embedding_for_envelope(self, envelope: TabletEnvelope) -> list[float]:
        for key in ("query_embedding_512", "query_embedding", "embedding"):
            value = envelope.metadata.get(key)
            if value is not None:
                return [float(item) for item in list(value)]
        kv = self._knowledgeverse
        embed = getattr(kv, "_embed_query_gpu", None)
        if callable(embed):
            return [float(value) for value in list(embed(envelope.query, task=dict(envelope.task)))]
        return self._fallback_query_embedding(envelope)

    def _decode_bridge_top_galaxies(self, y_new_vector: list[float]) -> dict[str, Any]:
        kv = self._knowledgeverse
        decoder = getattr(kv, "_decode_trm_galaxy_distribution", None)
        order_fn = getattr(kv, "_current_live_galaxy_order", None)
        if not callable(decoder) or not callable(order_fn) or not y_new_vector:
            return {}
        try:
            logits, distribution, decoder_source = decoder(y_new_vector)
            galaxy_order = list(order_fn())
        except Exception:
            return {}
        top_indexes = sorted(
            range(min(len(distribution), len(galaxy_order))),
            key=lambda index: float(distribution[index]),
            reverse=True,
        )[:3]
        return {
            "decoder_source": str(decoder_source),
            "y_new_top3_galaxies": [
                {
                    "galaxy": str(galaxy_order[index]),
                    "weight": float(distribution[index]),
                    "logit": float(logits[index]) if index < len(logits) else 0.0,
                }
                for index in top_indexes
            ],
        }

    def _materialized_answer_from_star(
        self,
        bridge_result: Mapping[str, Any],
        envelope: TabletEnvelope,
    ) -> dict[str, Any]:
        if not bool(bridge_result.get("answer_materialized")):
            return {
                "answer_materialized": False,
                "failure_code": str(bridge_result.get("failure_code") or "not_materialized_from_y_new_yet"),
            }
        star = bridge_result.get("top_star")
        star_payload = dict(star) if isinstance(star, Mapping) else {}
        metadata = star_payload.get("metadata") if isinstance(star_payload.get("metadata"), Mapping) else {}

        answer_text = ""
        for value in (
            metadata.get("answer_text"),
            metadata.get("resolved_answer"),
            metadata.get("boxed_answer"),
            metadata.get("definition"),
            star_payload.get("answer_text"),
            star_payload.get("answer"),
            star_payload.get("response"),
            star_payload.get("definition"),
            star_payload.get("content"),
            star_payload.get("name"),
            star_payload.get("id"),
        ):
            text = _normalise_text_answer(value)
            if text and not _looks_like_internal_route_label(text):
                answer_text = text
                break

        options = [str(option) for option in list(envelope.metadata.get("options") or []) if str(option).strip()]
        answer_choice = ""
        if options and answer_text:
            for option in options:
                if _normalise_text_answer(option) == answer_text:
                    answer_choice = option
                    break

        output_grid = metadata.get("output_grid", star_payload.get("output_grid"))
        if not isinstance(output_grid, list):
            output_grid = None

        action_index = None
        action_name = ""
        raw_index = metadata.get("action_index", star_payload.get("action_index"))
        try:
            if raw_index is not None:
                action_index = int(raw_index)
        except Exception:
            action_index = None
        action_name = _normalise_text_answer(metadata.get("action_name", star_payload.get("action_name")))
        numeric_answer = _numeric_form(answer_text)
        answer_materialized = bool(answer_text or answer_choice or output_grid is not None or action_index is not None or action_name)
        return {
            "answer_materialized": answer_materialized,
            "failure_code": "" if answer_materialized else "top_star_has_no_materialized_answer",
            "answer_text": answer_text,
            "numeric_answer": numeric_answer,
            "answer_choice": answer_choice,
            "output_grid": output_grid,
            "action_index": action_index,
            "action_name": action_name,
            "top_star": star_payload,
        }

    def _materialize_from_action_buffer(
        self,
        *,
        bridge_result: Mapping[str, Any],
        action_words: list[list[int]],
        envelope: TabletEnvelope,
    ) -> dict[str, Any]:
        first_words = list(action_words[0]) if action_words else []
        action_type_value = int(first_words[0]) if first_words else int(ActionType.NO_ACTION.value)
        try:
            action_type_name = ActionType(action_type_value).name
        except ValueError:
            action_type_name = ActionType.NO_ACTION.name
        y_new = [float(value) for value in list(bridge_result.get("y_new_vector_512") or [])]
        decode_meta = self._decode_bridge_top_galaxies(y_new)
        action_buffer_numeric_answer = None
        action_buffer_top_star = {}
        action_buffer_answer_materialized = False
        if (
            action_type_value == int(ActionType.UPDATE_TABLET.value)
            and len(first_words) >= (TABLET_WORD_OFFSET_DATA + 6)
            and int(first_words[TABLET_WORD_OFFSET_DATA + 5]) == 1
        ):
            action_buffer_numeric_answer = int(np.int32(first_words[TABLET_WORD_OFFSET_DATA]))
            action_buffer_answer_materialized = True
            top_star_idx = int(first_words[TABLET_WORD_OFFSET_DATA + 1])
            if 0 <= top_star_idx:
                top_star = bridge_result.get("top_star")
                if isinstance(top_star, Mapping):
                    action_buffer_top_star = dict(top_star)
        answer_packet = self._materialized_answer_from_star(bridge_result, envelope)
        if action_buffer_answer_materialized:
            answer_packet = {
                "answer_materialized": True,
                "failure_code": "",
                "answer_text": str(action_buffer_numeric_answer),
                "numeric_answer": int(action_buffer_numeric_answer),
                "answer_choice": "",
                "output_grid": None,
                "action_index": None,
                "action_name": "",
                "top_star": dict(action_buffer_top_star),
            }
        route_payload = envelope.to_route_payload(use_enriched=True)
        route = {
            "specialist": str(envelope.specialist),
            "domain": str(envelope.domain_hint or _normalize_surface_kind(envelope.surface_kind).lower()),
            "galaxy_names": list(envelope.galaxies),
            "route_family": _normalize_surface_kind(envelope.surface_kind),
            "runtime": "tablet_bridge_ring",
        }
        task_result = {
            "status": "ok",
            "runtime": "tablet_bridge_ring_query",
            "program_type": "trm_step_fused_submit_query",
            "gpu_execution": True,
            "answer_kind": "embedding",
            "answer_materialized": bool(answer_packet.get("answer_materialized")),
            "answer_text": str(answer_packet.get("answer_text") or ""),
            "numeric_answer": answer_packet.get("numeric_answer"),
            "answer_choice": str(answer_packet.get("answer_choice") or ""),
            "output_grid": answer_packet.get("output_grid"),
            "action_index": answer_packet.get("action_index"),
            "action_name": str(answer_packet.get("action_name") or ""),
            "failure_code": str(answer_packet.get("failure_code") or ""),
            "trm_recursion_steps": int(bridge_result.get("steps", 0) or 0),
            "trm_drift": float(bridge_result.get("drift", 0.0) or 0.0),
            "trm_latency_us": float(bridge_result.get("trm_latency_us", 0.0) or 0.0),
            "query_embedding_512": list(bridge_result.get("query_embedding_512") or []),
            "y_new_vector_512": y_new,
            "action_type": action_type_name,
            "action_buffer_words": first_words,
            "ring_event_payload": int(bridge_result.get("ring_event_payload", 0) or 0),
            "top_star_idx": int(bridge_result.get("top_star_idx", -1) or -1),
            "top_star_score": float(bridge_result.get("top_star_score", 0.0) or 0.0),
            "top_star_galaxy_id": int(bridge_result.get("top_star_galaxy_id", 0) or 0),
            "top_star_role": int(bridge_result.get("top_star_role", 0) or 0),
            "top_star_hash": int(bridge_result.get("top_star_hash", 0) or 0),
            "top_star": dict(answer_packet.get("top_star") or {}),
            "route": route,
            **decode_meta,
        }
        if task_result["answer_materialized"]:
            if task_result["output_grid"] is not None:
                task_result["answer_kind"] = "grid"
            elif task_result["action_index"] is not None or task_result["action_name"]:
                task_result["answer_kind"] = "action"
            elif task_result["answer_choice"]:
                task_result["answer_kind"] = "choice"
            elif task_result["numeric_answer"] is not None:
                task_result["answer_kind"] = "numeric"
            else:
                task_result["answer_kind"] = "text"
        return {
            "status": "ok",
            "route": route,
            "route_payload": route_payload,
            "task_result": task_result,
            "bridge_result": dict(bridge_result),
        }

    def submit(self, envelope: TabletEnvelope, *, use_enriched: bool = True) -> dict[str, Any]:
        self.tablet.prepare_headless_context(
            user_lang=envelope.user_lang,
            document_langs=list(envelope.document_langs),
        )
        route_payload = envelope.to_route_payload(use_enriched=use_enriched)
        action_buffer = envelope.to_action_buffer()
        mutation_type, payload_words = action_buffer.extract_tablet_mutation()
        bridge = self._resolve_sovereign_bridge()
        if bridge is not None and hasattr(bridge, "submit_query"):
            self._tick_seq += 1
            action_words = self._action_buffer_words(action_buffer)
            bridge_result = bridge.submit_query(
                self._query_embedding_for_envelope(envelope),
                action_buffer_words=action_words,
                delta_time=0.02,
                tick=self._tick_seq,
            )
            action_buffer_words = bridge_result.get("action_buffers") or []
            response = self._materialize_from_action_buffer(
                bridge_result=bridge_result,
                action_words=action_buffer_words,
                envelope=envelope,
            )
            emitted = TabletEmit.emit(envelope, response)
            return {
                "envelope": envelope,
                "route_payload": route_payload,
                "response": response,
                "raw_response": response,
                "emitted": emitted,
                "tablet_contract": {
                    "action_type": action_buffer.get_action_type().name,
                    "mutation_type": int(mutation_type),
                    "payload_words": payload_words.tolist(),
                    "surface_kind": _normalize_surface_kind(envelope.surface_kind),
                    "sovereign_path": "tablet_bridge_ring",
                    "output_action_type": response["task_result"]["action_type"],
                },
            }
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
