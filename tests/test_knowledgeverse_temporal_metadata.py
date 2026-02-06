from __future__ import annotations

import time

from knowledge3d.knowledgeverse.temporal_metadata import TemporalMetadataManager


def test_create_metadata_unique_ids():
    manager = TemporalMetadataManager("v5.0", "test_region")
    events = [manager.create_metadata() for _ in range(100)]
    event_ids = [e.event_id for e in events]
    assert len(event_ids) == len(set(event_ids))


def test_lamport_clock_monotonic():
    manager = TemporalMetadataManager("v5.0", "test_region")
    events = [manager.create_metadata() for _ in range(10)]
    for i in range(1, len(events)):
        assert events[i].lamport_clock > events[i - 1].lamport_clock


def test_causality_chain():
    manager = TemporalMetadataManager("v5.0", "test_region")
    parent = manager.create_metadata()
    child = manager.create_metadata(parent_event_id=parent.event_id)

    assert child.parent_event_id == parent.event_id
    assert manager.is_causally_before(parent, child)
    assert not manager.is_causally_before(child, parent)


def test_vector_clock_merge():
    manager1 = TemporalMetadataManager("v5.0", "region_1")
    manager2 = TemporalMetadataManager("v5.0", "region_2")

    manager1.create_metadata()
    event2 = manager2.create_metadata()
    manager1.merge_vector_clock(event2.vector_clock)

    assert manager1.vector_clock["region_1"] > 0
    assert manager1.vector_clock["region_2"] > 0


def test_concurrent_events_not_causally_ordered():
    manager1 = TemporalMetadataManager("v5.0", "region_1")
    manager2 = TemporalMetadataManager("v5.0", "region_2")

    event_a = manager1.create_metadata()
    event_b = manager2.create_metadata()

    assert not manager1.is_causally_before(event_a, event_b)
    assert not manager1.is_causally_before(event_b, event_a)


def test_manifest_version_attached():
    manager = TemporalMetadataManager("kv-2026-02-06", "test_region")
    events = [manager.create_metadata() for _ in range(10)]
    for event in events:
        assert event.manifest_version == "kv-2026-02-06"


def test_timestamp_ordering():
    manager = TemporalMetadataManager("v5.0", "test_region")
    events = []
    for _ in range(5):
        events.append(manager.create_metadata())
        time.sleep(0.01)

    for i in range(1, len(events)):
        assert events[i].timestamp > events[i - 1].timestamp

