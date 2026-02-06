"""Numbered context provider for ingestion-time RAG extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextChunk:
    """One numbered context chunk."""

    chunk_id: int
    line_start: int
    line_end: int
    content: str
    char_count: int


class NumberedContextProvider:
    """Split document text into numbered chunks for deterministic retrieval."""

    def __init__(self, content: str, chunk_size: int = 2000):
        if chunk_size < 64:
            raise ValueError("chunk_size must be >= 64")
        self.content = content
        self.chunk_size = chunk_size
        self.chunks = self._create_chunks()

    def _create_chunks(self) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        lines = self.content.splitlines()
        if not lines:
            return [
                ContextChunk(
                    chunk_id=1,
                    line_start=1,
                    line_end=1,
                    content="",
                    char_count=0,
                )
            ]

        bucket: list[str] = []
        start_line = 1

        for line_num, line in enumerate(lines, start=1):
            bucket.append(line)
            char_count = sum(len(x) + 1 for x in bucket)
            if char_count >= self.chunk_size or line_num == len(lines):
                chunks.append(
                    ContextChunk(
                        chunk_id=len(chunks) + 1,
                        line_start=start_line,
                        line_end=line_num,
                        content="\n".join(bucket),
                        char_count=char_count,
                    )
                )
                bucket = []
                start_line = line_num + 1
        return chunks

    def get_initial_context(self, num_chunks: int = 1) -> dict[str, Any]:
        n = max(1, int(num_chunks))
        selected = self.chunks[:n]
        return {
            "total_chunks": len(self.chunks),
            "total_chars": len(self.content),
            "provided_chunks": [self._as_dict(chunk) for chunk in selected],
            "instructions": (
                'You may request additional context using JSON: '
                '{"request_more": true, "chunk_ids": [2,3]}.'
            ),
        }

    def get_chunks(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        wanted = set(chunk_ids)
        return [self._as_dict(chunk) for chunk in self.chunks if chunk.chunk_id in wanted]

    def get_lines(self, line_start: int, line_end: int) -> str:
        lines = self.content.splitlines()
        start = max(1, line_start)
        end = max(start, line_end)
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _as_dict(chunk: ContextChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "content": chunk.content,
            "char_count": chunk.char_count,
        }
