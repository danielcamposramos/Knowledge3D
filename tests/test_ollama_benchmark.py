from __future__ import annotations

import json
import urllib.error

from knowledge3d.ingestion.ollama_manager import OllamaModelManager
from knowledge3d.tools.ollama_benchmark import (
    SUITE_MODEL_MAP,
    SYSTEM_PROMPTS,
    build_rag_context,
    create_ollama_query_fn,
    extract_answer,
    get_model_for_suite,
)


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_get_model_for_suite() -> None:
    assert get_model_for_suite("mmlu") == "qwen3:8b"
    assert get_model_for_suite("gsm8k") == "qwen2.5:32b"
    assert get_model_for_suite("unknown") == "qwen2.5:32b"
    assert SUITE_MODEL_MAP["arc"] == "qwen2.5:32b"


def test_system_prompts_exist_for_all_suites() -> None:
    assert set(SYSTEM_PROMPTS.keys()) == {"mmlu", "gsm8k", "math", "lhe", "arc"}
    assert "ONLY the final letter" in SYSTEM_PROMPTS["mmlu"]
    assert "ONLY the JSON grid" in SYSTEM_PROMPTS["arc"]


def test_extract_answer_mmlu() -> None:
    assert extract_answer("Reasoning...\nAnswer: C", "mmlu") == "C"


def test_extract_answer_gsm8k() -> None:
    assert extract_answer("Steps...\nFinal answer: 144", "gsm8k") == "144"


def test_extract_answer_arc() -> None:
    assert extract_answer("```json\n[[1,2],[3,4]]\n```", "arc") == "[[1, 2], [3, 4]]"


def test_extract_answer_strips_thinking() -> None:
    assert extract_answer("<think>hidden</think>\nB", "mmlu") == "B"


def test_build_rag_context_chemistry() -> None:
    row = {
        "question": "What is the atomic number of oxygen?",
        "payload": {"subject": "college_chemistry"},
    }

    context = build_rag_context(row, "mmlu")

    assert "REFERENCE FACTS" in context
    assert "Oxygen (O)" in context


def test_build_rag_context_physics() -> None:
    row = {
        "question": "Which constant represents the speed of light?",
        "payload": {"subject": "college_physics"},
    }

    context = build_rag_context(row, "mmlu")

    assert "Physical constants" in context
    assert "speed of light" in context


def test_build_rag_context_math_for_gsm8k() -> None:
    row = {
        "question": "Convert 3 kilometres to metres.",
        "payload": {"subject": "elementary_mathematics"},
    }

    context = build_rag_context(row, "gsm8k")

    assert "Measurement and conversion references" in context
    assert "length" in context


def test_build_rag_context_no_match() -> None:
    row = {
        "question": "Who wrote this poem?",
        "payload": {"subject": "literature"},
    }

    context = build_rag_context(row, "mmlu")

    assert context == ""


def test_ollama_chat_success(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        captured["timeout"] = timeout
        captured["body"] = request.data
        return _FakeHttpResponse({"message": {"content": "A"}})

    monkeypatch.setattr("knowledge3d.ingestion.ollama_manager.urllib.request.urlopen", _fake_urlopen)

    manager = OllamaModelManager(default_timeout=2.0)
    result = manager.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": "Guide"},
            {"role": "user", "content": "a\x00b"},
        ],
        timeout=1.5,
    )

    assert result.returncode == 0
    assert result.output == "A"
    payload = json.loads(captured["body"].decode("utf-8"))  # type: ignore[union-attr]
    assert payload["model"] == "qwen3:8b"
    assert payload["messages"][1]["content"] == "ab"


def test_ollama_chat_failure(monkeypatch) -> None:
    def _fake_urlopen(request, timeout):  # noqa: ANN001
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("knowledge3d.ingestion.ollama_manager.urllib.request.urlopen", _fake_urlopen)

    manager = OllamaModelManager(default_timeout=2.0)
    result = manager.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result.returncode == 1
    assert "offline" in result.stderr


def test_create_ollama_query_fn_routes_mmlu_and_extracts_letter(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_chat(self, model, messages, **kwargs):  # noqa: ANN001
        seen["model"] = model
        seen["messages"] = messages
        return type(
            "_Result",
            (),
            {"output": "<think>hidden</think>\nB", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(OllamaModelManager, "chat", _fake_chat)

    query_fn = create_ollama_query_fn()
    response = query_fn(
        {
            "id": "mmlu_1",
            "suite": "mmlu",
            "question": "What is 2 + 2?",
            "payload": {
                "subject": "elementary_mathematics",
                "options": ["3", "4", "5", "6"],
            },
        }
    )

    assert response["answer"] == "B"
    assert response["model"] == "qwen3:8b"
    assert response["suite"] == "mmlu"
    assert seen["model"] == "qwen3:8b"
    assert any("A. 3" in message["content"] for message in seen["messages"])  # type: ignore[index]


def test_create_ollama_query_fn_routes_arc_and_extracts_json(monkeypatch) -> None:
    def _fake_chat(self, model, messages, **kwargs):  # noqa: ANN001
        return type(
            "_Result",
            (),
            {"output": "```json\n[[9,0],[1,2]]\n```", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(OllamaModelManager, "chat", _fake_chat)

    query_fn = create_ollama_query_fn()
    response = query_fn(
        {
            "id": "arc_1",
            "suite": "arc",
            "question": "ARC task synthetic_flip_h",
            "payload": {"id": "synthetic_flip_h"},
        }
    )

    assert response["answer"] == "[[9, 0], [1, 2]]"
    assert response["model"] == "qwen2.5:32b"


def test_create_ollama_query_fn_marks_used_rag(monkeypatch) -> None:
    def _fake_chat(self, model, messages, **kwargs):  # noqa: ANN001
        return type(
            "_Result",
            (),
            {"output": "8", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(OllamaModelManager, "chat", _fake_chat)

    query_fn = create_ollama_query_fn()
    response = query_fn(
        {
            "id": "chem_1",
            "suite": "mmlu",
            "question": "What is the atomic number of oxygen?",
            "payload": {"subject": "college_chemistry", "options": ["6", "7", "8", "9"]},
        }
    )

    assert response["used_rag"] is True
