from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import importlib.util

from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse import knowledgeverse as kvmod

_ENRICH_SPEC = importlib.util.spec_from_file_location(
    "enrich_embeddings_module",
    Path(__file__).resolve().parents[1] / "scripts" / "enrich_embeddings.py",
)
assert _ENRICH_SPEC is not None and _ENRICH_SPEC.loader is not None
_ENRICH_MODULE = importlib.util.module_from_spec(_ENRICH_SPEC)
_ENRICH_SPEC.loader.exec_module(_ENRICH_MODULE)
enrich_embeddings = _ENRICH_MODULE.enrich_embeddings


class _FakeBatchEmbedEngine:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def has_gpu_bridge(self) -> bool:
        return True

    def embed_sentences_gpu(self, texts: list[str]):
        self.calls.append(list(texts))
        outputs = []
        for index, _text in enumerate(texts):
            row = [0.0] * 16
            row[index % 16] = 1.0
            outputs.append(row)
        return outputs


class _FakeGraph:
    signature = "fake_graph_signature"

    def ensure_device_buffers(self) -> None:
        return None


class _FakeBindingEngine:
    def __init__(self) -> None:
        self.bound_payloads: list[object] = []

    def bind_galaxy_buffer(self, flat_entries, **kwargs):
        self.bound_payloads.append(flat_entries)
        return {
            "entry_count": int(kwargs["entry_count"]),
            "entry_stride": int(kwargs["entry_stride"]),
            "embedding_offset": int(kwargs["embedding_offset"]),
            "embedding_dim": int(kwargs["embedding_dim"]),
        }


def test_flatten_galaxies_for_gpu_batch_enriches_missing_embeddings(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_embed_batch", eager_load_default_galaxies=False)
    kv.include_runtime_artifacts = False
    galaxy_name = "UnitTestEmbed"
    kv.galaxy_manager.add_entry(
        galaxy_name,
        {"id": "entry_a", "name": "alpha vector", "content": "alpha meaning"},
    )
    kv.galaxy_manager.add_entry(
        galaxy_name,
        {"id": "entry_b", "name": "beta vector", "content": "beta meaning"},
    )

    fake_engine = _FakeBatchEmbedEngine()
    monkeypatch.setattr(kv, "get_gpu_query_embedding_engine", lambda: fake_engine)

    flat, catalog, enriched_count = kv._flatten_galaxies_for_gpu(galaxy_names=[galaxy_name])

    assert enriched_count == 2
    assert len(fake_engine.calls) == 1
    assert len(fake_engine.calls[0]) == 2
    assert len(catalog) == 2
    assert len(flat) == 2 * kv.GPU_GALAXY_ENTRY_STRIDE
    assert all(len(entry["embedding16"]) == 16 for entry in catalog)

    restored_entries = kv.galaxy_manager.get_galaxy(galaxy_name).entries
    assert all(len(entry["embedding16"]) == 16 for entry in restored_entries)


def test_bind_gpu_galaxy_runtime_persists_enriched_checkpoint(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_bind_enrich", eager_load_default_galaxies=False)
    kv.include_runtime_artifacts = False
    kv._gpu_reasoning_engine = _FakeBindingEngine()

    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math"])
    monkeypatch.setattr(
        kv,
        "_flatten_galaxies_for_gpu",
        lambda **_kwargs: (
            [0.0] * kv.GPU_GALAXY_ENTRY_STRIDE,
            [
                {
                    "index": 0,
                    "galaxy": "Math",
                    "id": "entry_a",
                    "name": "alpha",
                    "category": "unit_test",
                    "domain": "math",
                    "confidence": 0.5,
                    "domain_hash": 1.0,
                    "subject_hash": 2.0,
                    "answer_text": "",
                    "embedding_text": "alpha",
                    "embedding16": [1.0] + [0.0] * 15,
                    "rpn_program": "",
                    "metadata": {},
                    "template_ref": "",
                    "template_params": {},
                    "answer_format": "",
                    "subject": "",
                    "gpu_category_class": 0.0,
                    "gpu_source_class": 0.0,
                    "gpu_galaxy_index": kv._gpu_galaxy_index("Math"),
                    "gpu_has_template_ref": 0.0,
                    "output_grid": None,
                    "arc_transform_chain": [],
                    "arc_color_mapping": {},
                    "arc_primitive_plan": [],
                    "arc_task_id": "",
                }
            ],
            3,
        ),
    )

    monkeypatch.setattr(kvmod, "load_or_build_semantic_csr_graph", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(kvmod.QueryHeadSubstrate, "build", lambda **_kwargs: SimpleNamespace(close=lambda: None))

    saved: list[bool] = []
    monkeypatch.setattr(
        kv,
        "save_consolidated_state",
        lambda: saved.append(True) or {"path": str(tmp_path / "checkpoint.json")},
    )

    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math"], force=True)

    assert saved == [True]


def test_bind_gpu_galaxy_runtime_reuses_flat_buffer_cache(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_bind_cache", eager_load_default_galaxies=False)
    kv.include_runtime_artifacts = False
    galaxy_name = "UnitTestCache"
    kv.galaxy_manager.add_entry(
        galaxy_name,
        {
            "id": "entry_cache_a",
            "name": "cache alpha",
            "content": "cache meaning",
            "embedding16": [1.0] + [0.0] * 15,
        },
    )
    kv._gpu_reasoning_engine = _FakeBindingEngine()
    monkeypatch.setattr(kvmod, "load_or_build_semantic_csr_graph", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(kvmod.QueryHeadSubstrate, "build", lambda **_kwargs: SimpleNamespace(close=lambda: None))

    first_binding = kv.bind_gpu_galaxy_runtime(galaxy_names=[galaxy_name], force=True)
    flat_path, catalog_path = kv._gpu_flat_cache_paths(first_binding["flat_cache_signature"])

    assert first_binding["flat_cache_hit"] is False
    assert flat_path.exists()
    assert catalog_path.exists()
    assert hasattr(kv._gpu_reasoning_engine.bound_payloads[0], "dtype")

    kv.invalidate_gpu_galaxy_binding()
    monkeypatch.setattr(
        kv,
        "_flatten_galaxies_for_gpu",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("flatten should not run on cache hit")),
    )

    second_binding = kv.bind_gpu_galaxy_runtime(galaxy_names=[galaxy_name], force=True)

    assert second_binding["flat_cache_hit"] is True
    assert hasattr(kv._gpu_reasoning_engine.bound_payloads[-1], "dtype")


def test_bind_gpu_galaxy_runtime_caches_slim_catalog_and_resolves_full_entry(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_bind_slim_cache", eager_load_default_galaxies=False)
    kv.include_runtime_artifacts = False
    galaxy_name = "Math"
    kv.galaxy_manager.add_entry(
        galaxy_name,
        {
            "id": "entry_slim_a",
            "name": "alpha anchor",
            "domain": "math",
            "category": "benchmark_fact",
            "content": "long content body that should not be copied into the slim catalog cache",
            "summary": "resolved summary",
            "rpn_program": "PUSH 1",
            "embedding16": [1.0] + [0.0] * 15,
            "metadata": {
                "subject": "algebra",
                "subfield": "benchmark_question_anchor",
                "query_anchor": "alpha benchmark query",
                "answer": "42",
                "template_params": {"n": 2},
                "arc_primitive_plan": [{"op": "paint"}],
            },
        },
    )
    kv._gpu_reasoning_engine = _FakeBindingEngine()
    monkeypatch.setattr(kvmod, "load_or_build_semantic_csr_graph", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(kvmod.QueryHeadSubstrate, "build", lambda **_kwargs: SimpleNamespace(close=lambda: None))

    binding = kv.bind_gpu_galaxy_runtime(galaxy_names=[galaxy_name], force=True)
    _flat_path, catalog_path = kv._gpu_flat_cache_paths(binding["flat_cache_signature"])
    with catalog_path.open("rb") as handle:
        cached_catalog = pickle.load(handle)

    cached_row = next(
        row
        for row in cached_catalog
        if isinstance(row, dict) and row.get("id") == "entry_slim_a"
    )
    assert int(cached_row["entry_idx"]) >= 0
    assert cached_row["metadata"]["subject"] == "algebra"
    assert cached_row["metadata"]["query_anchor"] == "alpha benchmark query"
    assert "content" not in cached_row
    assert "summary" not in cached_row
    assert "rpn_program" not in cached_row
    assert "answer_text" not in cached_row
    assert "template_params" not in cached_row
    assert kv._catalog_source_entry(cached_row)["id"] == "entry_slim_a"

    resolved = kv._catalog_entry_by_id("entry_slim_a")
    assert resolved is not None
    assert resolved["metadata"]["template_params"] == {"n": 2}
    assert resolved["rpn_program"] == "PUSH 1"
    assert resolved["answer_text"] == "42"


def test_gpu_flat_cache_signature_changes_when_entries_change(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_bind_signature", eager_load_default_galaxies=False)
    kv.include_runtime_artifacts = False
    galaxy_name = "UnitTestSig"
    kv.galaxy_manager.add_entry(
        galaxy_name,
        {
            "id": "entry_sig_a",
            "name": "sig alpha",
            "content": "sig meaning",
            "embedding16": [1.0] + [0.0] * 15,
        },
    )
    signature_before = kv._gpu_flat_cache_signature([galaxy_name])

    kv.galaxy_manager.add_entry(
        galaxy_name,
        {
            "id": "entry_sig_b",
            "name": "sig beta",
            "content": "sig meaning",
            "embedding16": [0.0, 1.0] + [0.0] * 14,
        },
    )
    signature_after = kv._gpu_flat_cache_signature([galaxy_name])

    assert signature_before != signature_after


def test_enrich_embeddings_script_updates_checkpoint_payloads(tmp_path):
    storage_root = tmp_path / "kv_enrich_script"
    kv = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    kv.galaxy_manager.add_entry(
        "Math",
        {"id": "entry_missing_embedding", "name": "gamma", "content": "gamma meaning"},
    )
    kv.save_consolidated_state()

    binary_path = storage_root / "house" / "galaxy_state.bin"
    with binary_path.open("rb") as handle:
        payload = pickle.load(handle)
    for entries in payload["galaxies"].values():
        for entry in entries:
            if isinstance(entry, dict):
                entry.pop("embedding16", None)
                metadata = entry.get("metadata")
                if isinstance(metadata, dict):
                    metadata.pop("embedding16", None)
    with binary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    latest_path = storage_root / "checkpoints" / "galaxy_consolidated_latest.json"
    target_path = latest_path.resolve() if latest_path.is_symlink() else latest_path
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    summary = enrich_embeddings(storage_root)

    assert summary["enriched"] >= 1

    with binary_path.open("rb") as handle:
        enriched_binary = pickle.load(handle)
    math_entries = enriched_binary["galaxies"]["Math"]
    assert any(len(entry.get("embedding16", [])) == 16 for entry in math_entries if isinstance(entry, dict))

    enriched_json = json.loads(target_path.read_text(encoding="utf-8"))
    assert any(
        len(entry.get("embedding16", [])) == 16
        for entry in enriched_json["galaxies"]["Math"]
        if isinstance(entry, dict)
    )
