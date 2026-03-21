"""Shared deterministic sampling utilities for benchmark loaders."""

from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def _even_pick(items: Sequence[T], count: int) -> list[T]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    step = len(items) / count
    picked: list[T] = []
    for index in range(count):
        item_index = min(len(items) - 1, int(index * step))
        picked.append(items[item_index])
    return picked


def stratified_sample(items: Sequence[T], limit: int | None) -> list[T]:
    """Sample evenly across easy/mid/hard thirds while staying deterministic."""
    rows = list(items)
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        return []

    total = len(rows)
    third = total // 3
    easy = rows[:third]
    mid = rows[third : 2 * third]
    hard = rows[2 * third :]

    per_region = limit // 3
    extra = limit - per_region * 3

    sampled: list[T] = []
    sampled.extend(_even_pick(easy, per_region))
    sampled.extend(_even_pick(mid, per_region))
    sampled.extend(_even_pick(hard, per_region + extra))
    return sampled
