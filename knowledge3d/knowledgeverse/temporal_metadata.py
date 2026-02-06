"""Temporal metadata primitives for Knowledgeverse causality tracking."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TemporalMetadata:
    """Temporal metadata for audit trail and causality reconstruction."""

    event_id: str
    timestamp: float
    lamport_clock: int
    vector_clock: dict[str, int]
    parent_event_id: Optional[str]
    manifest_version: str

    def __repr__(self) -> str:
        return (
            f"TemporalMetadata(id={self.event_id[:8]}..., "
            f"t={self.timestamp:.3f}, L={self.lamport_clock})"
        )


class TemporalMetadataManager:
    """Manage temporal metadata generation and causal ordering checks."""

    def __init__(self, manifest_version: str, region_id: str):
        self.manifest_version = manifest_version
        self.region_id = region_id
        self.lamport_clock = 0
        self.vector_clock: dict[str, int] = {region_id: 0}

    def create_metadata(self, parent_event_id: Optional[str] = None) -> TemporalMetadata:
        """Create temporal metadata and increment local logical clocks."""
        self.lamport_clock += 1
        self.vector_clock[self.region_id] = self.vector_clock.get(self.region_id, 0) + 1

        return TemporalMetadata(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            lamport_clock=self.lamport_clock,
            vector_clock=self.vector_clock.copy(),
            parent_event_id=parent_event_id,
            manifest_version=self.manifest_version,
        )

    def merge_vector_clock(self, other_vector_clock: dict[str, int]) -> None:
        """Merge another region's vector clock into local state."""
        for region_id, counter in other_vector_clock.items():
            current = self.vector_clock.get(region_id, 0)
            self.vector_clock[region_id] = max(current, int(counter))

        self.vector_clock[self.region_id] = self.vector_clock.get(self.region_id, 0) + 1
        self.lamport_clock += 1

    @staticmethod
    def is_causally_before(event_a: TemporalMetadata, event_b: TemporalMetadata) -> bool:
        """Return True if `event_a` happens-before `event_b` via vector clocks."""
        vc_a = event_a.vector_clock
        vc_b = event_b.vector_clock

        all_regions = set(vc_a.keys()) | set(vc_b.keys())
        at_least_one_less = False

        for region in all_regions:
            a_count = vc_a.get(region, 0)
            b_count = vc_b.get(region, 0)
            if a_count > b_count:
                return False
            if a_count < b_count:
                at_least_one_less = True

        return at_least_one_less

