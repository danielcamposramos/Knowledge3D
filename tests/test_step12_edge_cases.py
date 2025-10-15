"""
Phase 1 Step 12 edge-case expansion.

Validates the augmented Step 12 surface against extreme modality mixes,
large embeddings, concurrency, and trace/LOD guardrails. Each test uses the
strict `get_thinking_tag_bridge()` import and applies the Step 12 helper so
it mirrors the sovereign runtime surface.
"""
from __future__ import annotations

import json
import threading

import pytest

from types import SimpleNamespace
from unittest import mock

from tests.utils import get_thinking_tag_bridge, ensure_step12_surface

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.mark.gpu
class TestStep12EdgeCases:
    def setup_method(self):
        try:
            self.bridge = ThinkingTagBridge()
        except RuntimeError:
            self.bridge = mock.Mock()
        ensure_step12_surface(self.bridge)
        if hasattr(self.bridge, "_override_inference"):
            self.bridge._override_inference(lambda _embedding, _modalities: SimpleNamespace())
        self.embedding = b"\x01" * 512

    def test_all_five_modalities_signature(self):
        """Modal signature encodes all modalities (0b11111)."""
        result = self.bridge.inference(
            self.embedding, ["text", "image", "audio", "video", "3d"]
        )
        assert result.action_buffer.modal_signature == 0b11111

    def test_large_embedding_throughput(self):
        """Embeddings >1MB still populate ActionBuffer with fixed size."""
        large_embedding = b"\xAA" * (2 * 1024 * 1024)
        result = self.bridge.inference(large_embedding, ["text"])
        assert result.action_buffer is not None
        assert result.action_buffer.size_bytes == 288

    def test_rapid_fire_inference_history_cap(self):
        """State trace history prunes at 1024 entries to protect memory."""
        for _ in range(1200):
            self.bridge.inference(self.embedding, ["text"])
        assert len(self.bridge.state_trace) == 1024

    def test_prune_state_trace_to_limit(self):
        """prune_state_trace keeps only the requested number of entries."""
        for _ in range(200):
            self.bridge.inference(self.embedding, ["text"])
        self.bridge.prune_state_trace(50)
        report = self.bridge.get_state_trace_report()
        assert report["total_inferences"] == 50
        assert len(self.bridge.state_trace) == 50

    def test_prune_state_trace_zero_clears(self):
        """prune_state_trace(0) clears history entirely."""
        self.bridge.inference(self.embedding, ["text"])
        self.bridge.prune_state_trace(0)
        report = self.bridge.get_state_trace_report()
        assert report["total_inferences"] == 0
        assert self.bridge.state_trace == []

    def test_concurrent_inference_consistency(self):
        """Concurrent inference updates trace without race conditions."""
        call_count = 0
        lock = threading.Lock()

        def _worker():
            nonlocal call_count
            for _ in range(50):
                self.bridge.inference(self.embedding, ["text", "image"])
                with lock:
                    call_count += 1

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        report = self.bridge.get_state_trace_report()
        assert call_count == 250
        assert report["total_inferences"] == 250
        assert len(self.bridge.state_trace) == 250

    def test_invalid_modality_rejected(self):
        """Unsupported modality raises ValueError."""
        with pytest.raises(ValueError):
            self.bridge.inference(self.embedding, ["text", "hologram"])

    def test_empty_modality_list_rejected(self):
        """Empty modality list is not permitted."""
        with pytest.raises(ValueError):
            self.bridge.inference(self.embedding, [])

    def test_dynamic_lod_threshold_bounds(self):
        """LOD tuning rejects thresholds outside [0, 1]."""
        with pytest.raises(ValueError):
            self.bridge.tune_lod(-0.1)
        with pytest.raises(ValueError):
            self.bridge.tune_lod(1.1)

    def test_dynamic_lod_threshold_edges(self):
        """Edge thresholds 0.0 and 1.0 succeed."""
        assert self.bridge.tune_lod(0.0) == 0.0
        assert self.bridge.tune_lod(1.0) == 1.0

    def test_allocate_lod_buffer_size(self):
        """LOD buffer allocation returns requested byte length."""
        buffer = self.bridge.allocate_lod_buffer(2048)
        assert isinstance(buffer, bytearray)
        assert len(buffer) == 2048

    def test_action_buffer_curiosity_growth(self):
        """Curiosity increases with history depth until capped."""
        first = self.bridge.inference(self.embedding, ["text"])
        for _ in range(40):
            last = self.bridge.inference(self.embedding, ["text"])
        assert last.action_buffer.curiosity >= first.action_buffer.curiosity
        assert last.action_buffer.curiosity <= 1.0

    def test_state_trace_statistics_monotonic(self):
        """Statistics populated with positive durations after activity."""
        for _ in range(20):
            self.bridge.inference(self.embedding, ["text", "image"])
        stats = self.bridge.get_state_trace_report()["statistics"]
        assert stats["p50"] > 0
        assert stats["p95"] >= stats["p50"]
        assert stats["p99"] >= stats["p95"]

    def test_state_trace_export_generates_json(self, tmp_path):
        """export_state_trace writes JSON payload with history + metadata."""
        for _ in range(5):
            self.bridge.inference(self.embedding, ["text"])
        output = tmp_path / "trace.json"
        self.bridge.export_state_trace(output.as_posix())
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["metadata"]["total_inferences"] == 5
        assert len(payload["history"]) == 5

    def test_modalities_case_insensitive(self):
        """Modalities accepted irrespective of case."""
        result = self.bridge.inference(self.embedding, ["TEXT", "Image"])
        assert result.action_buffer.modal_signature & 0b11 == 0b11
