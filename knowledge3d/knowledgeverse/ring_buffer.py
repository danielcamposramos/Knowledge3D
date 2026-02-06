"""Fixed-size ring buffer for Knowledgeverse Region 6 audit storage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RingWindow:
    """Readable window describing the valid data range in ring order."""

    start: int
    length: int


class RingBuffer:
    """Byte-addressable fixed-size ring buffer.

    The buffer overwrites oldest data when writes wrap, which is suitable for
    bounded telemetry and audit journals.
    """

    def __init__(self, size_mb: int = 256):
        if size_mb <= 0:
            raise ValueError("size_mb must be positive")
        self.size_bytes = size_mb * 1024 * 1024
        self.buffer = bytearray(self.size_bytes)
        self.write_offset = 0
        self.data_size = 0

    def write(self, data: bytes) -> int:
        """Write bytes and return the start offset."""
        data_len = len(data)
        if data_len == 0:
            return self.write_offset
        if data_len > self.size_bytes:
            raise ValueError(f"Data too large: {data_len} > {self.size_bytes}")

        start_offset = self.write_offset
        end_offset = start_offset + data_len

        if end_offset <= self.size_bytes:
            self.buffer[start_offset:end_offset] = data
        else:
            first_chunk = self.size_bytes - start_offset
            self.buffer[start_offset:] = data[:first_chunk]
            self.buffer[: data_len - first_chunk] = data[first_chunk:]

        self.write_offset = (start_offset + data_len) % self.size_bytes
        self.data_size = min(self.size_bytes, self.data_size + data_len)
        return start_offset

    def read_at(self, offset: int, length: int) -> bytes:
        """Read `length` bytes starting at `offset` (wrap-aware)."""
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return b""
        if length > self.size_bytes:
            raise ValueError("length exceeds ring capacity")

        offset %= self.size_bytes
        end_offset = offset + length
        if end_offset <= self.size_bytes:
            return bytes(self.buffer[offset:end_offset])

        first_chunk = self.size_bytes - offset
        return bytes(self.buffer[offset:] + self.buffer[: length - first_chunk])

    def size(self) -> int:
        """Return current valid bytes retained in the ring."""
        return self.data_size

    def clear(self) -> None:
        """Reset ring metadata; leaves underlying bytes as-is."""
        self.write_offset = 0
        self.data_size = 0

    def readable_window(self) -> RingWindow:
        """Return ring window describing oldest retained segment."""
        if self.data_size == 0:
            return RingWindow(start=self.write_offset, length=0)
        if self.data_size < self.size_bytes:
            return RingWindow(start=0, length=self.data_size)
        # Full buffer: oldest byte starts at current write offset.
        return RingWindow(start=self.write_offset, length=self.size_bytes)

