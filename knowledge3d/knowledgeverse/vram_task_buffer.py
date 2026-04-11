"""VRAM task/result slot buffers for the sovereign benchmark game loop."""

from __future__ import annotations

import ctypes
import json
import struct
from typing import Any

from knowledge3d.cranium.sovereign import loader


EMBEDDING_DIMS = 64
EMBEDDING32_DIMS = EMBEDDING_DIMS
INPUT_SLOT_BYTES = 2688
OUTPUT_SLOT_BYTES = 640
OPTION_EMBEDDING_SLOTS = 7
OPTION_EMBEDDING_BYTES = EMBEDDING_DIMS * 4
OPTION_HASH_BYTES = 8

QUERY_EMBEDDING_OFFSET = 0
TASK_TYPE_OFFSET = 256
OPTION_COUNT_OFFSET = 260
OPTION_EMBEDDINGS_OFFSET = 264
OPTION_HASHES_OFFSET = 2056
SUBJECT_ID_OFFSET = 2112
DOMAIN_HINT_ID_OFFSET = 2116
THINKING_BUDGET_OFFSET = 2120
ACTION_HISTORY_OFFSET = 2124
ACTION_HISTORY_LEN_OFFSET = 2131
TERNARY_SIGNAL_OFFSET = 2132
GOAL_EMBEDDING_OFFSET = 2136
EXPECTED_HASH_OFFSET = 2392
EXPECTED_INDEX_OFFSET = 2400

ANSWER_INDEX_OFFSET = 0
CONFIDENCE_OFFSET = 4
CONVERGENCE_OFFSET = 8
ITERATIONS_OFFSET = 12
ANSWER_HASH_OFFSET = 16
GOAL_PROGRESS_OFFSET = 24
WINNER_STAR_INDEX_OFFSET = 28
WINNER_ROLE_ID_OFFSET = 32
ROUTE_DEPTH_OFFSET = 36
ANTI_PATTERN_SIGNAL_OFFSET = 40
ROUTER_STAR_INDEX_OFFSET = 44
EXECUTOR_STAR_INDEX_OFFSET = 48
VALIDATOR_STAR_INDEX_OFFSET = 52
ROUTE_BUDGET_USED_OFFSET = 56
ROUTE_BUDGET_MIN_OFFSET = 60
RECURSION_DEPTH_USED_OFFSET = 64
ROUTE_TRACE_STAR_INDICES_OFFSET = 68
ROUTE_TRACE_ROLE_IDS_OFFSET = 100
ROUTE_TRACE_LIMIT = 8

TASK_TYPE_IDS = {
    "GAME_2D": 1,
    "MATH": 2,
    "QUESTION": 3,
    "CHAT": 4,
    "GENERAL": 5,
    "GRAMMAR": 6,
    "INTERACTION": 7,
}
TASK_TYPE_ALIASES = {
    "ARC": "GAME_2D",
    "ARC_TASK": "GAME_2D",
    "ARC3": "GAME_2D",
    "ARC3_TASK": "GAME_2D",
    "SPATIAL_TASK": "GAME_2D",
    "GAME_2D": "GAME_2D",
    "MATH": "MATH",
    "MATH_TASK": "MATH",
    "GSM8K_TASK": "MATH",
    "IMO_TASK": "MATH",
    "QUESTION": "QUESTION",
    "QUESTION_TASK": "QUESTION",
    "MMLU_TASK": "QUESTION",
    "LHE_TASK": "QUESTION",
    "CHAT": "CHAT",
    "CHAT_TASK": "CHAT",
    "GENERAL": "GENERAL",
    "GENERAL_TASK": "GENERAL",
    "GRAMMAR": "GRAMMAR",
    "GRAMMAR_TASK": "GRAMMAR",
    "INTERACTION": "INTERACTION",
    "INTERACTION_TASK": "INTERACTION",
}
TASK_TYPE_NAMES = {value: key for key, value in TASK_TYPE_IDS.items()}


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value)


def _stable_hash_payload(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _fnv1a64(value)
    if isinstance(value, (int, float, bool)):
        return _fnv1a64(str(value))
    try:
        normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        normalized = str(value)
    return _fnv1a64(normalized)


def _embedding64(values: object) -> list[float]:
    if isinstance(values, (list, tuple)):
        flat = [float(value) for value in values[:EMBEDDING_DIMS]]
    else:
        flat = [float(value) for value in list(values or [])[:EMBEDDING_DIMS]]
    if len(flat) < EMBEDDING_DIMS:
        flat.extend([0.0] * (EMBEDDING_DIMS - len(flat)))
    return flat


def _option_embeddings(task: dict[str, Any]) -> list[list[float]]:
    raw = task.get("option_embeddings")
    if not isinstance(raw, list):
        return []
    return [_embedding64(item) for item in raw[:OPTION_EMBEDDING_SLOTS]]


def _bytes_ptr(payload: bytearray) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))


class VRAMTaskBuffer:
    """Contiguous task/result slot storage backed by sovereign CUDA allocations."""

    def __init__(self, max_tasks: int = 6000) -> None:
        self.max_tasks = max(1, int(max_tasks))
        self.input_bytes = self.max_tasks * INPUT_SLOT_BYTES
        self.output_bytes = self.max_tasks * OUTPUT_SLOT_BYTES
        self.input_buffer = loader.gpu_malloc(self.input_bytes)
        self.output_buffer = loader.gpu_malloc(self.output_bytes)
        self._last_loaded = 0
        self.clear_outputs()

    def close(self) -> None:
        if getattr(self, "input_buffer", None):
            loader.gpu_free(self.input_buffer)
            self.input_buffer = None
        if getattr(self, "output_buffer", None):
            loader.gpu_free(self.output_buffer)
            self.output_buffer = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def normalize_task_type(cls, task_type: Any) -> str:
        normalized = str(task_type or "").strip().upper()
        return str(TASK_TYPE_ALIASES.get(normalized, normalized if normalized in TASK_TYPE_IDS else "GENERAL"))

    @classmethod
    def task_type_id(cls, task_type: Any) -> int:
        normalized = cls.normalize_task_type(task_type)
        return int(TASK_TYPE_IDS.get(normalized, TASK_TYPE_IDS["GENERAL"]))

    @staticmethod
    def task_type_name(task_type_id: int) -> str:
        return str(TASK_TYPE_NAMES.get(int(task_type_id), "GENERAL"))

    def clear_outputs(self) -> None:
        if not getattr(self, "output_buffer", None):
            return
        payload = bytearray(self.output_bytes)
        loader.memcpy_htod(self.output_buffer, _bytes_ptr(payload), len(payload))

    def bulk_load(self, tasks: list[dict[str, Any]]) -> int:
        count = min(len(tasks), self.max_tasks)
        payload = bytearray(self.input_bytes)
        for task_index in range(count):
            self._pack_task_slot(payload, task_index, tasks[task_index])
        loader.memcpy_htod(self.input_buffer, _bytes_ptr(payload), len(payload))
        self._last_loaded = count
        self.clear_outputs()
        return count

    def read_tasks(self, count: int | None = None) -> list[dict[str, Any]]:
        read_count = min(self._last_loaded, self.max_tasks if count is None else max(0, int(count)))
        payload = bytearray(self.input_bytes)
        loader.memcpy_dtoh(_bytes_ptr(payload), self.input_buffer, len(payload))
        return [self._unpack_task_slot(payload, index) for index in range(read_count)]

    def read_results(self, count: int) -> list[dict[str, Any]]:
        read_count = min(max(0, int(count)), self.max_tasks)
        payload = bytearray(self.output_bytes)
        loader.memcpy_dtoh(_bytes_ptr(payload), self.output_buffer, len(payload))
        return [self._unpack_result_slot(payload, index) for index in range(read_count)]

    def write_results(self, results: list[dict[str, Any]]) -> int:
        count = min(len(results), self.max_tasks)
        payload = bytearray(self.output_bytes)
        for result_index in range(count):
            self._pack_result_slot(payload, result_index, results[result_index])
        loader.memcpy_htod(self.output_buffer, _bytes_ptr(payload), len(payload))
        return count

    def _pack_task_slot(self, payload: bytearray, task_index: int, task: dict[str, Any]) -> None:
        base = int(task_index) * INPUT_SLOT_BYTES
        query_embedding = _embedding64(task.get("query_embedding") or task.get("embedding") or [])
        struct.pack_into(f"<{EMBEDDING_DIMS}f", payload, base + QUERY_EMBEDDING_OFFSET, *query_embedding)
        family = self.normalize_task_type(task.get("surface_kind") or task.get("type", ""))
        struct.pack_into("<I", payload, base + TASK_TYPE_OFFSET, self.task_type_id(family))
        option_embeddings = _option_embeddings(task)
        struct.pack_into("<I", payload, base + OPTION_COUNT_OFFSET, len(option_embeddings))
        option_hashes = list(task.get("option_hashes") or [])
        for option_index, option_embedding in enumerate(option_embeddings):
            option_offset = base + OPTION_EMBEDDINGS_OFFSET + (option_index * OPTION_EMBEDDING_BYTES)
            struct.pack_into(f"<{EMBEDDING_DIMS}f", payload, option_offset, *option_embedding)
            option_hash = int(option_hashes[option_index]) if option_index < len(option_hashes) else _stable_hash_payload(
                (task.get("options") or [])[option_index] if isinstance(task.get("options"), list) and option_index < len(task.get("options")) else ""
            )
            hash_offset = base + OPTION_HASHES_OFFSET + (option_index * OPTION_HASH_BYTES)
            struct.pack_into("<Q", payload, hash_offset, option_hash & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into(
            "<I",
            payload,
            base + SUBJECT_ID_OFFSET,
            _fnv1a32(str(task.get("subject") or task.get("subject_id") or "")),
        )
        struct.pack_into(
            "<I",
            payload,
            base + DOMAIN_HINT_ID_OFFSET,
            _fnv1a32(str(task.get("domain_hint") or task.get("domain") or "")),
        )
        thinking_budget = max(5, min(20, int(task.get("thinking_budget", 10))))
        struct.pack_into("<I", payload, base + THINKING_BUDGET_OFFSET, thinking_budget)
        action_history = [int(value) for value in list(task.get("action_history") or [])[:7]]
        for history_index, action_index in enumerate(action_history):
            struct.pack_into("<B", payload, base + ACTION_HISTORY_OFFSET + history_index, action_index & 0xFF)
        struct.pack_into("<B", payload, base + ACTION_HISTORY_LEN_OFFSET, min(len(action_history), 7))
        ternary_signal = max(-1, min(1, int(task.get("ternary_signal", 0))))
        struct.pack_into("<b", payload, base + TERNARY_SIGNAL_OFFSET, ternary_signal)
        goal_embedding = _embedding64(task.get("goal_embedding") or [])
        struct.pack_into(f"<{EMBEDDING_DIMS}f", payload, base + GOAL_EMBEDDING_OFFSET, *goal_embedding)
        expected_hash = int(task.get("expected_hash") or _stable_hash_payload(task.get("expected_answer") or task.get("expected_output")))
        struct.pack_into("<Q", payload, base + EXPECTED_HASH_OFFSET, expected_hash & 0xFFFFFFFFFFFFFFFF)
        expected_index = max(-1, int(task.get("expected_index", -1)))
        struct.pack_into("<i", payload, base + EXPECTED_INDEX_OFFSET, expected_index)

    def _unpack_task_slot(self, payload: bytearray, task_index: int) -> dict[str, Any]:
        base = int(task_index) * INPUT_SLOT_BYTES
        query_embedding = list(struct.unpack_from(f"<{EMBEDDING_DIMS}f", payload, base + QUERY_EMBEDDING_OFFSET))
        task_type_id = struct.unpack_from("<I", payload, base + TASK_TYPE_OFFSET)[0]
        option_count = struct.unpack_from("<I", payload, base + OPTION_COUNT_OFFSET)[0]
        option_embeddings: list[list[float]] = []
        option_hashes: list[int] = []
        for option_index in range(min(option_count, OPTION_EMBEDDING_SLOTS)):
            option_offset = base + OPTION_EMBEDDINGS_OFFSET + (option_index * OPTION_EMBEDDING_BYTES)
            option_embeddings.append(list(struct.unpack_from(f"<{EMBEDDING_DIMS}f", payload, option_offset)))
            option_hash_offset = base + OPTION_HASHES_OFFSET + (option_index * OPTION_HASH_BYTES)
            option_hashes.append(int(struct.unpack_from("<Q", payload, option_hash_offset)[0]))
        subject_id = struct.unpack_from("<I", payload, base + SUBJECT_ID_OFFSET)[0]
        domain_hint_id = struct.unpack_from("<I", payload, base + DOMAIN_HINT_ID_OFFSET)[0]
        thinking_budget = struct.unpack_from("<I", payload, base + THINKING_BUDGET_OFFSET)[0]
        action_history_len = struct.unpack_from("<B", payload, base + ACTION_HISTORY_LEN_OFFSET)[0]
        action_history = [
            struct.unpack_from("<B", payload, base + ACTION_HISTORY_OFFSET + history_index)[0]
            for history_index in range(min(action_history_len, 7))
        ]
        ternary_signal = struct.unpack_from("<b", payload, base + TERNARY_SIGNAL_OFFSET)[0]
        goal_embedding = list(struct.unpack_from(f"<{EMBEDDING_DIMS}f", payload, base + GOAL_EMBEDDING_OFFSET))
        expected_hash = int(struct.unpack_from("<Q", payload, base + EXPECTED_HASH_OFFSET)[0])
        expected_index = int(struct.unpack_from("<i", payload, base + EXPECTED_INDEX_OFFSET)[0])
        return {
            "type_id": int(task_type_id),
            "type": self.task_type_name(task_type_id),
            "query_embedding": query_embedding,
            "option_count": int(option_count),
            "option_embeddings": option_embeddings,
            "option_hashes": option_hashes,
            "subject_id": int(subject_id),
            "domain_hint_id": int(domain_hint_id),
            "thinking_budget": int(thinking_budget),
            "action_history": action_history,
            "ternary_signal": int(ternary_signal),
            "goal_embedding": goal_embedding,
            "expected_hash": expected_hash,
            "expected_index": expected_index,
        }

    def _unpack_result_slot(self, payload: bytearray, task_index: int) -> dict[str, Any]:
        base = int(task_index) * OUTPUT_SLOT_BYTES
        answer_index = struct.unpack_from("<I", payload, base + ANSWER_INDEX_OFFSET)[0]
        confidence = struct.unpack_from("<f", payload, base + CONFIDENCE_OFFSET)[0]
        convergence_signal = struct.unpack_from("<b", payload, base + CONVERGENCE_OFFSET)[0]
        iterations_used = struct.unpack_from("<I", payload, base + ITERATIONS_OFFSET)[0]
        answer_text_hash = struct.unpack_from("<Q", payload, base + ANSWER_HASH_OFFSET)[0]
        goal_progress = struct.unpack_from("<f", payload, base + GOAL_PROGRESS_OFFSET)[0]
        winner_star_index = struct.unpack_from("<I", payload, base + WINNER_STAR_INDEX_OFFSET)[0]
        winner_role_id = struct.unpack_from("<I", payload, base + WINNER_ROLE_ID_OFFSET)[0]
        route_depth = struct.unpack_from("<I", payload, base + ROUTE_DEPTH_OFFSET)[0]
        anti_pattern_signal = struct.unpack_from("<i", payload, base + ANTI_PATTERN_SIGNAL_OFFSET)[0]
        router_star_index = struct.unpack_from("<I", payload, base + ROUTER_STAR_INDEX_OFFSET)[0]
        executor_star_index = struct.unpack_from("<I", payload, base + EXECUTOR_STAR_INDEX_OFFSET)[0]
        validator_star_index = struct.unpack_from("<I", payload, base + VALIDATOR_STAR_INDEX_OFFSET)[0]
        route_budget_used = struct.unpack_from("<I", payload, base + ROUTE_BUDGET_USED_OFFSET)[0]
        route_budget_min = struct.unpack_from("<I", payload, base + ROUTE_BUDGET_MIN_OFFSET)[0]
        recursion_depth_used = struct.unpack_from("<I", payload, base + RECURSION_DEPTH_USED_OFFSET)[0]
        route_trace_star_indices = list(struct.unpack_from(f"<{ROUTE_TRACE_LIMIT}I", payload, base + ROUTE_TRACE_STAR_INDICES_OFFSET))
        route_trace_role_ids = list(struct.unpack_from(f"<{ROUTE_TRACE_LIMIT}I", payload, base + ROUTE_TRACE_ROLE_IDS_OFFSET))
        return {
            "answer_index": int(answer_index),
            "confidence": float(confidence),
            "convergence_signal": int(convergence_signal),
            "iterations_used": int(iterations_used),
            "answer_text_hash": int(answer_text_hash),
            "goal_progress": float(goal_progress),
            "winner_star_index": int(winner_star_index),
            "winner_role_id": int(winner_role_id),
            "route_depth": int(route_depth),
            "anti_pattern_signal": int(anti_pattern_signal),
            "router_star_index": int(router_star_index),
            "executor_star_index": int(executor_star_index),
            "validator_star_index": int(validator_star_index),
            "route_budget_used": int(route_budget_used),
            "route_budget_min": int(route_budget_min),
            "recursion_depth_used": int(recursion_depth_used),
            "route_trace_star_indices": [int(value) for value in route_trace_star_indices],
            "route_trace_role_ids": [int(value) for value in route_trace_role_ids],
        }

    def _pack_result_slot(self, payload: bytearray, task_index: int, result: dict[str, Any]) -> None:
        base = int(task_index) * OUTPUT_SLOT_BYTES
        struct.pack_into("<I", payload, base + ANSWER_INDEX_OFFSET, int(result.get("answer_index", 0)) & 0xFFFFFFFF)
        struct.pack_into("<f", payload, base + CONFIDENCE_OFFSET, float(result.get("confidence", 0.0) or 0.0))
        struct.pack_into("<b", payload, base + CONVERGENCE_OFFSET, int(result.get("convergence_signal", 0) or 0))
        struct.pack_into("<I", payload, base + ITERATIONS_OFFSET, int(result.get("iterations_used", 0) or 0) & 0xFFFFFFFF)
        answer_hash = int(result.get("answer_text_hash") or _stable_hash_payload(result.get("answer_text")))
        struct.pack_into("<Q", payload, base + ANSWER_HASH_OFFSET, answer_hash & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<f", payload, base + GOAL_PROGRESS_OFFSET, float(result.get("goal_progress", 0.0) or 0.0))
        struct.pack_into("<I", payload, base + WINNER_STAR_INDEX_OFFSET, int(result.get("winner_star_index", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + WINNER_ROLE_ID_OFFSET, int(result.get("winner_role_id", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + ROUTE_DEPTH_OFFSET, int(result.get("route_depth", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<i", payload, base + ANTI_PATTERN_SIGNAL_OFFSET, int(result.get("anti_pattern_signal", 0) or 0))
        struct.pack_into("<I", payload, base + ROUTER_STAR_INDEX_OFFSET, int(result.get("router_star_index", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + EXECUTOR_STAR_INDEX_OFFSET, int(result.get("executor_star_index", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + VALIDATOR_STAR_INDEX_OFFSET, int(result.get("validator_star_index", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + ROUTE_BUDGET_USED_OFFSET, int(result.get("route_budget_used", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + ROUTE_BUDGET_MIN_OFFSET, int(result.get("route_budget_min", 0) or 0) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + RECURSION_DEPTH_USED_OFFSET, int(result.get("recursion_depth_used", 0) or 0) & 0xFFFFFFFF)


__all__ = [
    "ANSWER_HASH_OFFSET",
    "ANSWER_INDEX_OFFSET",
    "ANTI_PATTERN_SIGNAL_OFFSET",
    "CONFIDENCE_OFFSET",
    "CONVERGENCE_OFFSET",
    "DOMAIN_HINT_ID_OFFSET",
    "EMBEDDING_DIMS",
    "EMBEDDING32_DIMS",
    "EXPECTED_HASH_OFFSET",
    "EXPECTED_INDEX_OFFSET",
    "EXECUTOR_STAR_INDEX_OFFSET",
    "GOAL_EMBEDDING_OFFSET",
    "GOAL_PROGRESS_OFFSET",
    "INPUT_SLOT_BYTES",
    "ITERATIONS_OFFSET",
    "OPTION_COUNT_OFFSET",
    "OPTION_EMBEDDINGS_OFFSET",
    "OPTION_HASHES_OFFSET",
    "OUTPUT_SLOT_BYTES",
    "QUERY_EMBEDDING_OFFSET",
    "RECURSION_DEPTH_USED_OFFSET",
    "ROUTE_DEPTH_OFFSET",
    "ROUTE_BUDGET_MIN_OFFSET",
    "ROUTE_BUDGET_USED_OFFSET",
    "ROUTE_TRACE_LIMIT",
    "ROUTE_TRACE_ROLE_IDS_OFFSET",
    "ROUTE_TRACE_STAR_INDICES_OFFSET",
    "ROUTER_STAR_INDEX_OFFSET",
    "SUBJECT_ID_OFFSET",
    "TASK_TYPE_IDS",
    "TASK_TYPE_NAMES",
    "TASK_TYPE_OFFSET",
    "TERNARY_SIGNAL_OFFSET",
    "VALIDATOR_STAR_INDEX_OFFSET",
    "VRAMTaskBuffer",
    "WINNER_ROLE_ID_OFFSET",
    "WINNER_STAR_INDEX_OFFSET",
]
