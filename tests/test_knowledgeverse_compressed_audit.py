from __future__ import annotations

import json
import time

from knowledge3d.knowledgeverse.compressed_audit import CompressedAuditJournal
from knowledge3d.knowledgeverse.ring_buffer import RingBuffer


def create_test_event(i: int) -> dict:
    specialist = "math" if i % 2 == 0 else "visual"
    confidence = (i % 100) / 100.0
    return {
        "type": "shadow_update" if i % 3 else "verify",
        "timestamp": time.time() + (i * 1e-4),
        "data": {
            "id": i,
            "payload": ("knowledgeverse_event_" + str(i % 10)) * 80,
            "score": confidence,
            "route": ["router", specialist, "executor"],
        },
        "confidence": confidence,
        "specialist": specialist,
        "galaxy": "math_galaxy" if specialist == "math" else "visual_galaxy",
        "verification": "passed" if i % 5 else "failed",
    }


def test_compression_ratio(tmp_path):
    journal = CompressedAuditJournal(
        region_buffer=RingBuffer(size_mb=64),
        index_path=tmp_path / "audit_index.json",
    )

    events = [create_test_event(i) for i in range(10_000)]
    for event in events:
        journal.append_event(event)

    compressed_size = journal.buffer.size()
    json_size = sum(
        len(json.dumps(event, separators=(",", ":"), sort_keys=True))
        for event in events
    )
    ratio = json_size / compressed_size
    assert ratio >= 10.0, f"Compression ratio too low: {ratio:.2f}"


def test_query_performance(tmp_path):
    journal = CompressedAuditJournal(
        region_buffer=RingBuffer(size_mb=128),
        index_path=tmp_path / "audit_index_perf.json",
    )

    for i in range(100_000):
        journal.append_event(create_test_event(i))

    start = time.perf_counter()
    results = journal.query_by_specialist("math", limit=100)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.01, f"Query took {elapsed:.6f}s (expected <0.01s)"
    assert len(results) <= 100
    assert all(r["specialist"] == "math" for r in results)


def test_ternary_quantization(tmp_path):
    journal = CompressedAuditJournal(
        region_buffer=RingBuffer(size_mb=8),
        index_path=tmp_path / "audit_index_quant.json",
    )

    cases = [
        (0.10, -1),
        (0.50, 0),
        (0.90, 1),
        (0.33, 0),
        (0.66, 1),
    ]
    for confidence, expected in cases:
        assert journal._quantize_confidence(confidence) == expected


def test_binary_serialization_roundtrip(tmp_path):
    journal = CompressedAuditJournal(
        region_buffer=RingBuffer(size_mb=8),
        index_path=tmp_path / "audit_index_roundtrip.json",
    )

    original = create_test_event(42)
    offset = journal.append_event(original)
    unpacked = journal._unpack_event_at(offset)

    assert unpacked["type"] == original["type"]
    assert unpacked["specialist"] == original["specialist"]
    assert unpacked["galaxy"] == original["galaxy"]
    assert unpacked["verification"] == original["verification"]
    assert unpacked["confidence_ternary"] == journal._quantize_confidence(
        original["confidence"]
    )
    assert len(unpacked["data_hash"]) == 64

