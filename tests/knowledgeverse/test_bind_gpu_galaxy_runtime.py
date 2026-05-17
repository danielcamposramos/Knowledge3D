from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from knowledge3d.knowledgeverse import knowledgeverse as kvmod
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class _FakeGraph:
    signature = "fake_graph_signature"

    def ensure_device_buffers(self) -> None:
        return None


class _FakeBindingEngine:
    def __init__(self) -> None:
        self.bound_payloads: list[np.ndarray] = []

    def bind_galaxy_buffer(self, flat_entries, **kwargs):
        payload = np.asarray(flat_entries, dtype=np.float32).reshape(-1)
        self.bound_payloads.append(payload)
        return {
            "entry_count": int(kwargs["entry_count"]),
            "entry_stride": int(kwargs["entry_stride"]),
            "embedding_offset": int(kwargs["embedding_offset"]),
            "embedding_dim": int(kwargs["embedding_dim"]),
        }


def test_bind_gpu_galaxy_runtime_snapshot_and_idempotency(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_bind_runtime",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
        include_runtime_artifacts=False,
        include_runtime_language_enrichment=False,
    )
    kv._gpu_reasoning_engine = _FakeBindingEngine()
    monkeypatch.setattr(
        kv,
        "_flatten_galaxies_for_gpu",
        lambda **_kwargs: (
            [0.0] * kv.GPU_GALAXY_ENTRY_STRIDE,
            [
                {
                    "index": 0,
                    "entry_idx": 0,
                    "galaxy": "Math",
                    "id": "math_anchor",
                    "name": "math anchor",
                    "metadata": {},
                    "embedding16": [1.0] + [0.0] * 15,
                    "gpu_source_class": 0.0,
                    "gpu_galaxy_index": kv._gpu_galaxy_index("Math"),
                }
            ],
            0,
        ),
    )
    monkeypatch.setattr(kvmod, "load_or_build_semantic_csr_graph", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(kvmod.QueryHeadSubstrate, "build", lambda **_kwargs: SimpleNamespace(close=lambda: None))

    binding = kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])

    assert binding["bound"] == ["Math", "Grammar", "Number", "Word"]
    assert binding["total"] == 4
    assert binding["requested_galaxies"] == ["Math", "Grammar", "Number", "Word"]
    assert len(kv._gpu_reasoning_engine.bound_payloads) == 1
    initial_rebuilds = int(kv.metrics.gpu_bind_rebuilds)

    rebound = kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])

    assert rebound["bound"] == ["Math", "Grammar", "Number", "Word"]
    assert len(kv._gpu_reasoning_engine.bound_payloads) == 1
    assert int(kv.metrics.gpu_bind_rebuilds) == initial_rebuilds


def test_bind_gpu_galaxy_runtime_unknown_name_raises_value_error(tmp_path):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_bind_runtime_unknown",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
        include_runtime_artifacts=False,
        include_runtime_language_enrichment=False,
    )

    with pytest.raises(ValueError, match="unknown_galaxy_names:UnknownGalaxy"):
        kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "UnknownGalaxy"])
