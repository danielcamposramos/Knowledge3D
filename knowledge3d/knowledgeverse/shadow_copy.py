"""Shadow Copy integration with compressed audit journal."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

from .compressed_audit import CompressedAuditJournal
from .ring_buffer import RingBuffer
from .temporal_metadata import TemporalMetadataManager


class ShadowCopyLearning:
    """Shadow Copy recorder that writes compressed audit events."""

    def __init__(
        self,
        audit_region_buffer: RingBuffer | None = None,
        trm_manager: Any | None = None,
        index_path: str | Path = "../Knowledge3D.local/audit_index.json",
        manifest_version: str = "unknown",
    ):
        self.compressed_journal = CompressedAuditJournal(
            region_buffer=audit_region_buffer,
            index_path=index_path,
        )
        self.trm_manager = trm_manager
        self.temporal_manager = TemporalMetadataManager(
            manifest_version=manifest_version,
            region_id="shadow_copy",
        )
        self.event_buffer: list[dict[str, Any]] = []

    def record_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Record Shadow Copy event with temporal metadata.

        Returns:
            event_id for causality chain building.
        """
        temporal = self.temporal_manager.create_metadata(parent_event_id=parent_event_id)

        event = {
            "type": event_type,
            "timestamp": float(event_data.get("timestamp", temporal.timestamp)),
            "data": event_data,
            "confidence": float(event_data.get("confidence", 0.5)),
            "specialist": str(event_data.get("specialist", "unknown")),
            "galaxy": str(event_data.get("galaxy", "")),
            "verification": str(event_data.get("verification", "")),
            "temporal": {
                "event_id": temporal.event_id,
                "timestamp": temporal.timestamp,
                "lamport_clock": temporal.lamport_clock,
                "vector_clock": temporal.vector_clock,
                "parent_event_id": temporal.parent_event_id,
                "manifest_version": temporal.manifest_version,
            },
        }
        self.compressed_journal.append_event(event)
        self.event_buffer.append(event)
        return temporal.event_id

    def query_events_by_specialist(self, specialist: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.compressed_journal.query_by_specialist(specialist=specialist, limit=limit)

    def flush(self) -> None:
        self.compressed_journal.flush()
