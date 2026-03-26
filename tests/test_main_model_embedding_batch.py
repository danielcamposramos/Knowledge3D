from __future__ import annotations

from pathlib import Path

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_rpn_embedding_engine_embed_sentences_gpu_reuses_unique_tokens(monkeypatch):
    engine = RPNEmbeddingEngine(embedding_dim=4)
    engine._gpu_bridge = object()

    calls: list[str] = []

    def _fake_embed_word_gpu(token: str) -> np.ndarray:
        calls.append(token)
        base = float(len(token))
        return np.asarray([base, base + 1.0, base + 2.0, base + 3.0], dtype=np.float32)

    monkeypatch.setattr(engine, "embed_word_gpu", _fake_embed_word_gpu)

    outputs = engine.embed_sentences_gpu(
        [
            "alpha beta",
            "beta gamma",
            "alpha gamma beta",
        ]
    )

    assert len(outputs) == 3
    assert sorted(calls) == ["alpha", "beta", "gamma"]
    assert all(getattr(row, "shape", None) == (4,) for row in outputs)
    assert all([float(value) for value in row] for row in outputs)


def test_knowledgeverse_embed_query_gpu_delegates_to_batch_helper(tmp_path: Path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_batch_delegate", eager_load_default_galaxies=False)

    captured: list[list[str]] = []

    def _fake_batch(texts: list[str], *, task=None):
        captured.append(list(texts))
        return [[0.1, 0.2, 0.3, 0.4]]

    monkeypatch.setattr(kv, "_embed_query_batch_gpu", _fake_batch)

    result = kv._embed_query_gpu("cipher word validation", task={"type": "CHAT_TASK"})

    assert result == [0.1, 0.2, 0.3, 0.4]
    assert captured == [["cipher word validation"]]


def test_parse_bundle_embeddings_uses_main_model_batch_path(tmp_path: Path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_parse_batch", eager_load_default_galaxies=False)

    seen_texts: list[list[str]] = []

    def _raise_single(*_args, **_kwargs):
        raise AssertionError("_embed_query_gpu should not be called from parse bundle batch path")

    def _fake_batch(texts: list[str], *, task=None):
        seen_texts.append(list(texts))
        rows = []
        for idx, _text in enumerate(texts, start=1):
            rows.append([float(idx)] * 16)
        return rows

    monkeypatch.setattr(kv, "_embed_query_gpu", _raise_single)
    monkeypatch.setattr(kv, "_embed_query_batch_gpu", _fake_batch)

    parse_context = kv._parse_bundle_embeddings(
        query_embedding=[0.25] * 16,
        parse_bundle={
            "query_text": "base question",
            "route_plan": [
                {"strategy": "fusion", "query_variant": "fusion text"},
                {"strategy": "forward", "query_variant": "forward text"},
                {"strategy": "backward", "query_variant": "backward text"},
                {"strategy": "fusion", "query_variant": "duplicate fusion text"},
            ],
        },
        task={"type": "MATH_TASK"},
    )

    assert seen_texts == [["fusion text", "forward text", "backward text"]]
    assert parse_context["fusion_embedding"]
    assert parse_context["navigation_embedding"]
    assert parse_context["directional_embedding"]


def test_build_option_embedding_cache_batches_unique_option_texts(tmp_path: Path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_option_batch", eager_load_default_galaxies=False)

    seen_texts: list[list[str]] = []

    def _raise_single(*_args, **_kwargs):
        raise AssertionError("_embed_query_gpu should not be called from option cache batch path")

    def _fake_batch(texts: list[str], *, task=None):
        seen_texts.append(list(texts))
        rows = []
        for text in texts:
            base = 1.0 if text == "Option A" else 2.0
            rows.append([base, 0.0, 0.0, 0.0])
        return rows

    class _FakeResonator:
        def __init__(self) -> None:
            self.calls: list[tuple[list[float], list[float], float]] = []

        def resonate_list(self, query_embedding, option_embedding, alpha: float):
            self.calls.append((list(query_embedding), list(option_embedding), float(alpha)))
            return [float(option_embedding[0]), float(alpha), 0.0, 0.0]

    fake_resonator = _FakeResonator()

    monkeypatch.setattr(kv, "_embed_query_gpu", _raise_single)
    monkeypatch.setattr(kv, "_embed_query_batch_gpu", _fake_batch)
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: fake_resonator)

    cache = kv._build_option_embedding_cache(
        query_embedding=[0.25, 0.5, 0.75, 1.0],
        paths=[
            {"option_text": "Option A", "query_text": "stem alpha"},
            {"option_text": "Option B", "query_text": "stem beta"},
            {"option_text": "Option A", "query_text": "stem gamma"},
            {"option_text": "Option A", "query_text": ""},
        ],
        task_type="MMLU_TASK",
    )

    assert seen_texts == [["Option A", "Option B"]]
    assert len(fake_resonator.calls) == 3
    assert sorted(cache) == ["Option A", "stem alpha", "stem beta", "stem gamma"]
    assert cache["stem alpha"] == cache["stem gamma"]
    assert cache["stem alpha"] != cache["Option A"]
