"""Compressed audit journal for Knowledgeverse Region 6."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

from .ring_buffer import RingBuffer


class BTreeIndex:
    """MVP simplified in-memory index persisted as JSON.

    The implementation keeps specialist buckets and supports fast lookup by
    specialist with bounded result sets. This is intentionally lightweight for
    MVP and can be replaced by a true on-disk B-tree in Post-MVP.
    """

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.by_specialist: dict[str, list[dict[str, Any]]] = {}
        self.by_offset: dict[str, dict[str, Any]] = {}
        self._load()

    def insert(
        self,
        key: int,
        offset: int,
        timestamp: float,
        specialist: str,
        record_length: int,
        event_type: str,
        confidence_ternary: int,
    ) -> None:
        entry = {
            "key": key,
            "offset": offset,
            "timestamp": timestamp,
            "record_length": record_length,
            "event_type": event_type,
            "confidence_ternary": confidence_ternary,
        }
        bucket = self.by_specialist.setdefault(specialist, [])
        bucket.append(entry)
        # Offset can be reused in ring wrap-around. Latest write wins.
        self.by_offset[str(offset)] = {"specialist": specialist, **entry}

    def query(self, specialist: str, limit: int = 100) -> list[dict[str, Any]]:
        entries = self.by_specialist.get(specialist, [])
        if limit <= 0:
            return []
        # Newest first to favor latest retained records.
        return entries[-limit:][::-1]

    def lookup_offset(self, offset: int) -> dict[str, Any] | None:
        return self.by_offset.get(str(offset))

    def flush(self) -> None:
        payload = {
            "by_specialist": self.by_specialist,
            "by_offset": self.by_offset,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.by_specialist = payload.get("by_specialist", {})
        self.by_offset = payload.get("by_offset", {})


class CompressedAuditJournal:
    """Ternary-compressed audit journal with specialist index."""

    # [u8 event_type][u32 timestamp_delta][i8 confidence_ternary]
    # [32 bytes SHA256][u16 metadata_len][metadata bytes]
    _HEADER_STRUCT = struct.Struct("<B I b 32s H")
    _LENGTH_STRUCT = struct.Struct("<I")

    def __init__(
        self,
        region_buffer: RingBuffer | None = None,
        index_path: str | Path = "../Knowledge3D.local/audit_index.json",
    ):
        self.buffer = region_buffer or RingBuffer(size_mb=256)
        self.index = BTreeIndex(index_path)
        self.boot_time = time.time()
        self.last_timestamp_us = 0
        self.event_counter = 0
        self._event_type_to_code: dict[str, int] = {}
        self._code_to_event_type: dict[int, str] = {}
        self._autosave_mod = 128

    @staticmethod
    def _quantize_confidence(confidence: float) -> int:
        """Map confidence [0.0, 1.0] to ternary {-1, 0, +1}."""
        if confidence < 0.33:
            return -1
        if confidence < 0.66:
            return 0
        return 1

    @staticmethod
    def _dequantize_confidence(ternary: int) -> float:
        if ternary <= -1:
            return 0.165
        if ternary >= 1:
            return 0.835
        return 0.495

    def _event_type_code(self, event_type: str) -> int:
        code = self._event_type_to_code.get(event_type)
        if code is not None:
            return code
        if len(self._event_type_to_code) >= 255:
            # Reserve 255 as overflow fallback.
            return 255
        code = len(self._event_type_to_code)
        self._event_type_to_code[event_type] = code
        self._code_to_event_type[code] = event_type
        return code

    def append_event(self, event: dict[str, Any]) -> int:
        """Append event to ring buffer and index."""
        event_type = str(event.get("type", "unknown"))
        timestamp = float(event.get("timestamp", time.time()))
        confidence = float(event.get("confidence", 0.5))
        specialist = str(event.get("specialist", "unknown"))

        event_type_code = self._event_type_code(event_type)
        confidence_ternary = self._quantize_confidence(confidence)

        ts_us = int((timestamp - self.boot_time) * 1_000_000)
        if ts_us < 0:
            ts_us = 0
        delta_us = ts_us - self.last_timestamp_us
        if delta_us < 0:
            delta_us = 0
        if delta_us > 0xFFFFFFFF:
            delta_us = 0xFFFFFFFF
        self.last_timestamp_us = ts_us

        data_blob = json.dumps(
            event.get("data", {}),
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        data_hash = hashlib.sha256(data_blob).digest()

        metadata = {
            "specialist": specialist,
            "galaxy": event.get("galaxy", ""),
            "verification": event.get("verification", ""),
        }
        temporal = event.get("temporal")
        if isinstance(temporal, dict):
            metadata["temporal_event_id"] = temporal.get("event_id", "")
            metadata["temporal_timestamp"] = temporal.get("timestamp", 0.0)
            metadata["temporal_parent_event_id"] = temporal.get("parent_event_id", "")
            metadata["temporal_lamport"] = temporal.get("lamport_clock", 0)
            metadata["temporal_vector_clock"] = temporal.get("vector_clock", {})
            metadata["temporal_manifest_version"] = temporal.get("manifest_version", "")
        metadata_bytes = json.dumps(
            metadata, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(metadata_bytes) > 0xFFFF:
            raise ValueError("Metadata too large for u16 length field")

        payload = self._HEADER_STRUCT.pack(
            event_type_code,
            delta_us,
            confidence_ternary,
            data_hash,
            len(metadata_bytes),
        ) + metadata_bytes
        record = self._LENGTH_STRUCT.pack(len(payload)) + payload

        offset = self.buffer.write(record)

        self.index.insert(
            key=self.event_counter,
            offset=offset,
            timestamp=timestamp,
            specialist=specialist,
            record_length=len(record),
            event_type=event_type,
            confidence_ternary=confidence_ternary,
        )
        self.event_counter += 1
        if self.event_counter % self._autosave_mod == 0:
            self.index.flush()

        return offset

    def _unpack_event_at(self, offset: int) -> dict[str, Any]:
        """Read and decode a record at ring offset."""
        length_blob = self.buffer.read_at(offset, self._LENGTH_STRUCT.size)
        if len(length_blob) != self._LENGTH_STRUCT.size:
            raise ValueError("Invalid record length prefix")
        (payload_len,) = self._LENGTH_STRUCT.unpack(length_blob)
        if payload_len <= 0:
            raise ValueError("Invalid payload length")
        if payload_len + self._LENGTH_STRUCT.size > self.buffer.size_bytes:
            raise ValueError("Payload length exceeds buffer capacity")

        payload = self.buffer.read_at(offset + self._LENGTH_STRUCT.size, payload_len)
        header_size = self._HEADER_STRUCT.size
        if len(payload) < header_size:
            raise ValueError("Payload shorter than header")

        event_type_code, delta_us, conf_ternary, data_hash, metadata_len = (
            self._HEADER_STRUCT.unpack(payload[:header_size])
        )
        metadata_start = header_size
        metadata_end = metadata_start + metadata_len
        if metadata_end > len(payload):
            raise ValueError("Metadata length exceeds payload")

        metadata_blob = payload[metadata_start:metadata_end]
        metadata = json.loads(metadata_blob.decode("utf-8")) if metadata_blob else {}
        index_entry = self.index.lookup_offset(offset)

        event_type = self._code_to_event_type.get(
            event_type_code,
            (index_entry or {}).get("event_type", "unknown"),
        )
        timestamp = (index_entry or {}).get("timestamp", self.boot_time + delta_us / 1e6)

        return {
            "type": event_type,
            "timestamp": timestamp,
            "confidence_ternary": conf_ternary,
            "confidence": self._dequantize_confidence(conf_ternary),
            "specialist": metadata.get("specialist", "unknown"),
            "galaxy": metadata.get("galaxy", ""),
            "verification": metadata.get("verification", ""),
            "temporal": {
                "event_id": metadata.get("temporal_event_id", ""),
                "timestamp": metadata.get("temporal_timestamp", 0.0),
                "parent_event_id": metadata.get("temporal_parent_event_id", ""),
                "lamport_clock": metadata.get("temporal_lamport", 0),
                "vector_clock": metadata.get("temporal_vector_clock", {}),
                "manifest_version": metadata.get("temporal_manifest_version", ""),
            },
            "data_hash": data_hash.hex(),
            "offset": offset,
            "record_length": payload_len + self._LENGTH_STRUCT.size,
        }

    def query_by_specialist(self, specialist: str, limit: int = 100) -> list[dict[str, Any]]:
        """Query latest events by specialist."""
        entries = self.index.query(specialist=specialist, limit=limit)
        results: list[dict[str, Any]] = []
        for entry in entries:
            try:
                results.append(self._unpack_event_at(int(entry["offset"])))
            except Exception:
                # Ring offsets may have been overwritten; skip stale entries.
                continue
        return results

    def flush(self) -> None:
        self.index.flush()
