"""
Navigation supervision dataset backed by Log Galaxy binary files.
"""

from __future__ import annotations

from array import array
import json
import mmap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class NavigationDataset:
    """
    Loads Log Galaxy binary traces with O(1) access via offsets.

    Returns (problem_embedding, target_sequence_ids).
    """

    def __init__(self, *, bin_path: str, meta_path: str):
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for NavigationDataset.") from exc

        self._torch = torch
        self._bin_path = Path(bin_path)
        self._meta_path = Path(meta_path)
        self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))

        self._embedding_dim = int(self._meta["counts"]["embedding_dim"])
        self._trace_count = int(self._meta["counts"]["traces"])

        self._mm = self._bin_path.open("rb")
        self._mmap = mmap.mmap(self._mm.fileno(), 0, access=mmap.ACCESS_READ)

        self._trace_offsets = self._load_uint32_array("trace_offsets")

        self._rule_ids_offset = int(self._meta["offsets"]["step_rule_ids"])
        self._kind_offset = int(self._meta["offsets"]["step_kind"])
        self._embed_offset = int(self._meta["offsets"]["problem_embeddings"])

    def __len__(self) -> int:
        return self._trace_count

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._trace_count:
            raise IndexError(idx)

        start = self._trace_offsets[idx]
        end = self._trace_offsets[idx + 1]
        step_count = end - start

        embed = self._read_embedding(idx)
        rule_ids = self._read_rule_ids(start, step_count)
        return embed, rule_ids

    def _load_uint32_array(self, name: str) -> array:
        offset = int(self._meta["offsets"][name])
        length = int(self._meta["lengths"][name])
        byte_len = length * 4
        mv = memoryview(self._mmap)[offset : offset + byte_len]
        data = array("I")
        data.frombytes(mv.tobytes())
        return data

    def _read_embedding(self, idx: int):
        offset = self._embed_offset + idx * self._embedding_dim * 4
        mv = memoryview(self._mmap)[offset : offset + self._embedding_dim * 4]
        data = array("f")
        data.frombytes(mv.tobytes())
        return self._torch.tensor(data, dtype=self._torch.float32)

    def _read_rule_ids(self, start: int, step_count: int):
        offset = self._rule_ids_offset + start * 2
        mv = memoryview(self._mmap)[offset : offset + step_count * 2]
        data = array("H")
        data.frombytes(mv.tobytes())
        return self._torch.tensor(data, dtype=self._torch.long)

    def close(self) -> None:
        try:
            self._mmap.close()
        finally:
            self._mm.close()


__all__ = ["NavigationDataset"]
