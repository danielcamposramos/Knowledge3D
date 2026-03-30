"""VRAM task/result slot buffers for the GPU benchmark game loop."""

from __future__ import annotations

import ctypes
import struct
from typing import Any

from knowledge3d.cranium.sovereign import loader


INPUT_SLOT_BYTES = 1280
OUTPUT_SLOT_BYTES = 512
EMBEDDING32_DIMS = 32
OPTION_EMBEDDING_SLOTS = 7
OPTION_EMBEDDING_BYTES = EMBEDDING32_DIMS * 4
QUERY_EMBEDDING_OFFSET = 0
TASK_TYPE_OFFSET = 128
OPTION_COUNT_OFFSET = 132
OPTION_EMBEDDINGS_OFFSET = 136
SUBJECT_ID_OFFSET = 1032
DOMAIN_HINT_ID_OFFSET = 1036
THINKING_BUDGET_OFFSET = 1040
ACTION_HISTORY_OFFSET = 1044
ACTION_HISTORY_LEN_OFFSET = 1051
TERNARY_SIGNAL_OFFSET = 1052
GOAL_EMBEDDING_OFFSET = 1056

ANSWER_INDEX_OFFSET = 0
CONFIDENCE_OFFSET = 4
CONVERGENCE_OFFSET = 8
ITERATIONS_OFFSET = 12
ANSWER_HASH_OFFSET = 16
GOAL_PROGRESS_OFFSET = 24

TASK_TYPE_IDS = {
    "ARC_TASK": 0,
    "MATH_TASK": 1,
    "GSM8K_TASK": 2,
    "LHE_TASK": 3,
    "MMLU_TASK": 4,
    "CHAT_TASK": 5,
    "GENERAL_TASK": 6,
    "GRAMMAR_TASK": 7,
    "ARC3_TASK": 8,
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


def _embedding32(values: object) -> list[float]:
    if isinstance(values, (list, tuple)):
        flat = [float(value) for value in values[:EMBEDDING32_DIMS]]
    else:
        flat = [float(value) for value in list(values or [])[:EMBEDDING32_DIMS]]
    if len(flat) < EMBEDDING32_DIMS:
        flat.extend([0.0] * (EMBEDDING32_DIMS - len(flat)))
    return flat


def _option_embeddings(task: dict[str, Any]) -> list[list[float]]:
    raw = task.get("option_embeddings")
    if not isinstance(raw, list):
        return []
    return [_embedding32(item) for item in raw[:OPTION_EMBEDDING_SLOTS]]


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

    @staticmethod
    def task_type_id(task_type: str) -> int:
        return int(TASK_TYPE_IDS.get(str(task_type or "").strip().upper(), TASK_TYPE_IDS["GENERAL_TASK"]))

    @staticmethod
    def task_type_name(task_type_id: int) -> str:
        return str(TASK_TYPE_NAMES.get(int(task_type_id), "GENERAL_TASK"))

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

    def write_results(self, results: list[dict[str, Any]]) -> int:
        count = min(len(results), self.max_tasks)
        payload = bytearray(self.output_bytes)
        for task_index in range(count):
            self._pack_result_slot(payload, task_index, results[task_index])
        loader.memcpy_htod(self.output_buffer, _bytes_ptr(payload), len(payload))
        return count

    def read_results(self, count: int) -> list[dict[str, Any]]:
        read_count = min(max(0, int(count)), self.max_tasks)
        payload = bytearray(self.output_bytes)
        loader.memcpy_dtoh(_bytes_ptr(payload), self.output_buffer, len(payload))
        return [self._unpack_result_slot(payload, index) for index in range(read_count)]

    def _pack_task_slot(self, payload: bytearray, task_index: int, task: dict[str, Any]) -> None:
        base = int(task_index) * INPUT_SLOT_BYTES
        query_embedding = _embedding32(task.get("query_embedding") or task.get("embedding") or [])
        struct.pack_into("<32f", payload, base + QUERY_EMBEDDING_OFFSET, *query_embedding)
        struct.pack_into("<I", payload, base + TASK_TYPE_OFFSET, self.task_type_id(str(task.get("type", ""))))
        option_embeddings = _option_embeddings(task)
        struct.pack_into("<I", payload, base + OPTION_COUNT_OFFSET, len(option_embeddings))
        for option_index, option_embedding in enumerate(option_embeddings):
            option_offset = base + OPTION_EMBEDDINGS_OFFSET + (option_index * OPTION_EMBEDDING_BYTES)
            struct.pack_into("<32f", payload, option_offset, *option_embedding)
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
        goal_embedding = _embedding32(task.get("goal_embedding") or [])
        struct.pack_into("<32f", payload, base + GOAL_EMBEDDING_OFFSET, *goal_embedding)

    def _unpack_task_slot(self, payload: bytearray, task_index: int) -> dict[str, Any]:
        base = int(task_index) * INPUT_SLOT_BYTES
        query_embedding = list(struct.unpack_from("<32f", payload, base + QUERY_EMBEDDING_OFFSET))
        task_type_id = struct.unpack_from("<I", payload, base + TASK_TYPE_OFFSET)[0]
        option_count = struct.unpack_from("<I", payload, base + OPTION_COUNT_OFFSET)[0]
        option_embeddings: list[list[float]] = []
        for option_index in range(min(option_count, OPTION_EMBEDDING_SLOTS)):
            option_offset = base + OPTION_EMBEDDINGS_OFFSET + (option_index * OPTION_EMBEDDING_BYTES)
            option_embeddings.append(list(struct.unpack_from("<32f", payload, option_offset)))
        subject_id = struct.unpack_from("<I", payload, base + SUBJECT_ID_OFFSET)[0]
        domain_hint_id = struct.unpack_from("<I", payload, base + DOMAIN_HINT_ID_OFFSET)[0]
        thinking_budget = struct.unpack_from("<I", payload, base + THINKING_BUDGET_OFFSET)[0]
        action_history_len = struct.unpack_from("<B", payload, base + ACTION_HISTORY_LEN_OFFSET)[0]
        action_history = [
            struct.unpack_from("<B", payload, base + ACTION_HISTORY_OFFSET + history_index)[0]
            for history_index in range(min(action_history_len, 7))
        ]
        ternary_signal = struct.unpack_from("<b", payload, base + TERNARY_SIGNAL_OFFSET)[0]
        goal_embedding = list(struct.unpack_from("<32f", payload, base + GOAL_EMBEDDING_OFFSET))
        return {
            "type_id": int(task_type_id),
            "type": self.task_type_name(task_type_id),
            "query_embedding": query_embedding,
            "option_count": int(option_count),
            "option_embeddings": option_embeddings,
            "subject_id": int(subject_id),
            "domain_hint_id": int(domain_hint_id),
            "thinking_budget": int(thinking_budget),
            "action_history": action_history,
            "ternary_signal": int(ternary_signal),
            "goal_embedding": goal_embedding,
        }

    def _pack_result_slot(self, payload: bytearray, task_index: int, result: dict[str, Any]) -> None:
        base = int(task_index) * OUTPUT_SLOT_BYTES
        answer_index = max(0, int(result.get("answer_index", 0)))
        confidence = float(result.get("confidence", 0.0))
        convergence_signal = int(result.get("convergence_signal", 0))
        iterations_used = max(0, int(result.get("iterations_used", 0)))
        answer_text_hash = int(
            result.get("answer_text_hash")
            or _fnv1a64(str(result.get("answer_text") or result.get("answer") or ""))
        )
        goal_progress = float(result.get("goal_progress", 0.0))
        struct.pack_into("<I", payload, base + ANSWER_INDEX_OFFSET, answer_index)
        struct.pack_into("<f", payload, base + CONFIDENCE_OFFSET, confidence)
        struct.pack_into("<b", payload, base + CONVERGENCE_OFFSET, convergence_signal)
        struct.pack_into("<I", payload, base + ITERATIONS_OFFSET, iterations_used)
        struct.pack_into("<Q", payload, base + ANSWER_HASH_OFFSET, answer_text_hash)
        struct.pack_into("<f", payload, base + GOAL_PROGRESS_OFFSET, goal_progress)

    def _unpack_result_slot(self, payload: bytearray, task_index: int) -> dict[str, Any]:
        base = int(task_index) * OUTPUT_SLOT_BYTES
        answer_index = struct.unpack_from("<I", payload, base + ANSWER_INDEX_OFFSET)[0]
        confidence = struct.unpack_from("<f", payload, base + CONFIDENCE_OFFSET)[0]
        convergence_signal = struct.unpack_from("<b", payload, base + CONVERGENCE_OFFSET)[0]
        iterations_used = struct.unpack_from("<I", payload, base + ITERATIONS_OFFSET)[0]
        answer_text_hash = struct.unpack_from("<Q", payload, base + ANSWER_HASH_OFFSET)[0]
        goal_progress = struct.unpack_from("<f", payload, base + GOAL_PROGRESS_OFFSET)[0]
        return {
            "answer_index": int(answer_index),
            "confidence": float(confidence),
            "convergence_signal": int(convergence_signal),
            "iterations_used": int(iterations_used),
            "answer_text_hash": int(answer_text_hash),
            "goal_progress": float(goal_progress),
        }


__all__ = [
    "ANSWER_HASH_OFFSET",
    "ANSWER_INDEX_OFFSET",
    "CONFIDENCE_OFFSET",
    "CONVERGENCE_OFFSET",
    "DOMAIN_HINT_ID_OFFSET",
    "EMBEDDING32_DIMS",
    "GOAL_EMBEDDING_OFFSET",
    "GOAL_PROGRESS_OFFSET",
    "INPUT_SLOT_BYTES",
    "ITERATIONS_OFFSET",
    "OPTION_COUNT_OFFSET",
    "OPTION_EMBEDDINGS_OFFSET",
    "OPTION_EMBEDDING_SLOTS",
    "OUTPUT_SLOT_BYTES",
    "QUERY_EMBEDDING_OFFSET",
    "SUBJECT_ID_OFFSET",
    "TASK_TYPE_IDS",
    "TASK_TYPE_OFFSET",
    "THINKING_BUDGET_OFFSET",
    "ACTION_HISTORY_OFFSET",
    "ACTION_HISTORY_LEN_OFFSET",
    "TERNARY_SIGNAL_OFFSET",
    "VRAMTaskBuffer",
]
