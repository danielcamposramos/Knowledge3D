from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter, _GeneratedPattern


def _sample_task() -> dict:
    return {
        "id": "adapter_test",
        "train": [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}],
        "test": [{"input": [[9, 0], [1, 2]], "output": [[0, 9], [2, 1]]}],
    }


def _phase_tile_task() -> dict:
    return {
        "id": "phase_tile_test",
        "train": [
            {
                "input": [[8, 6], [6, 4]],
                "output": [
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                    [6, 8, 6, 8, 6, 8],
                    [4, 6, 4, 6, 4, 6],
                    [8, 6, 8, 6, 8, 6],
                    [6, 4, 6, 4, 6, 4],
                ],
            },
            {
                "input": [[7, 9], [4, 3]],
                "output": [
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                    [9, 7, 9, 7, 9, 7],
                    [3, 4, 3, 4, 3, 4],
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                ],
            },
        ],
        "test": [
            {
                "input": [[3, 2], [7, 8]],
                "output": [
                    [3, 2, 3, 2, 3, 2],
                    [7, 8, 7, 8, 7, 8],
                    [2, 3, 2, 3, 2, 3],
                    [8, 7, 8, 7, 8, 7],
                    [3, 2, 3, 2, 3, 2],
                    [7, 8, 7, 8, 7, 8],
                ],
            }
        ],
    }


def _self_pattern_nonzero_mask_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_input = [[0, 1], [1, 0]]
    test_input = [[2, 0], [2, 2]]
    return {
        "id": "0692e18c_like",
        "train": [
            {
                "input": train_input,
                "output": adapter._grid_self_pattern_nonzero_mask(train_input),
            }
        ],
        "test": [
            {
                "input": test_input,
                "output": adapter._grid_self_pattern_nonzero_mask(test_input),
            }
        ],
    }


def _self_pattern_complement_mask_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_input = [[0, 7, 0], [7, 7, 7], [0, 7, 0]]
    test_input = [[0, 0, 3], [3, 3, 0], [0, 3, 0]]
    return {
        "id": "0692e18c_complement_like",
        "train": [
            {
                "input": train_input,
                "output": adapter._grid_self_pattern_complement_mask(train_input),
            }
        ],
        "test": [
            {
                "input": test_input,
                "output": adapter._grid_self_pattern_complement_mask(test_input),
            }
        ],
    }


def _enclosed_fill_count_mod_task() -> dict:
    return {
        "id": "00dbd492_like",
        "train": [
            {
                "input": [
                    [2, 2, 2, 2, 2, 0, 0],
                    [2, 0, 0, 0, 2, 0, 0],
                    [2, 0, 2, 0, 2, 0, 0],
                    [2, 0, 0, 0, 2, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [2, 2, 2, 2, 2, 0, 0],
                    [2, 8, 8, 8, 2, 0, 0],
                    [2, 8, 2, 8, 2, 0, 0],
                    [2, 8, 8, 8, 2, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
            }
        ],
        "test": [
            {
                "input": [
                    [0, 0, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 0, 0, 0, 2, 0, 0],
                    [0, 0, 2, 0, 2, 0, 2, 0, 0],
                    [0, 0, 2, 0, 0, 0, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 8, 8, 8, 2, 0, 0],
                    [0, 0, 2, 8, 2, 8, 2, 0, 0],
                    [0, 0, 2, 8, 8, 8, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            }
        ],
    }


def _enclosed_fill_count_lookup_task() -> dict:
    return {
        "id": "00dbd492_lookup_like",
        "train": [
            {
                "input": [
                    [2, 2, 2, 2, 2, 0, 0],
                    [2, 0, 0, 0, 2, 0, 0],
                    [2, 0, 2, 0, 2, 0, 0],
                    [2, 0, 0, 0, 2, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [2, 2, 2, 2, 2, 0, 0],
                    [2, 8, 8, 8, 2, 0, 0],
                    [2, 8, 2, 8, 2, 0, 0],
                    [2, 8, 8, 8, 2, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 2, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 4, 4, 2, 4, 4, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 2, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0, 0, 0],
                    [2, 0, 0, 0, 2, 0, 0, 0, 0],
                    [2, 0, 2, 0, 2, 0, 0, 0, 0],
                    [2, 0, 0, 0, 2, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0, 0, 0],
                ],
                "output": [
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 4, 4, 2, 4, 4, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 4, 4, 4, 4, 4, 2, 0, 0],
                    [2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0, 0, 0],
                    [2, 8, 8, 8, 2, 0, 0, 0, 0],
                    [2, 8, 2, 8, 2, 0, 0, 0, 0],
                    [2, 8, 8, 8, 2, 0, 0, 0, 0],
                    [2, 2, 2, 2, 2, 0, 0, 0, 0],
                ],
            }
        ],
    }


def _connect_color_pairs_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_input = [
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    test_input = [
        [0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    return {
        "id": "070dd51e_like",
        "train": [{"input": train_input, "output": adapter._grid_connect_color_pairs(train_input)}],
        "test": [{"input": test_input, "output": adapter._grid_connect_color_pairs(test_input)}],
    }


def _separator_bridge_projection_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_input = [
        [0, 0, 0, 4, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    test_input = [
        [0, 4, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 0],
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return {
        "id": "05a7bcf2_like",
        "train": [{"input": train_input, "output": adapter._grid_separator_bridge_projection(train_input)}],
        "test": [{"input": test_input, "output": adapter._grid_separator_bridge_projection(test_input)}],
    }


def _anchor_spiral_pair_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_a = [
        [5, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    train_b = [
        [3, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    test_input = [
        [2, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return {
        "id": "08573cc6_like",
        "train": [
            {"input": train_a, "output": adapter._grid_anchor_spiral_pair(train_a)},
            {"input": train_b, "output": adapter._grid_anchor_spiral_pair(train_b)},
        ],
        "test": [{"input": test_input, "output": adapter._grid_anchor_spiral_pair(test_input)}],
    }


def _anchor_spiral_arc_eval_like_task() -> dict:
    return {
        "id": "08573cc6_eval_like",
        "train": [
            {
                "input": [
                    [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [5, 5, 5, 5, 5, 6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 5, 5, 1, 0, 6, 0, 0, 0, 0, 0],
                    [0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 6, 5, 5, 5, 5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 3, 3, 3, 3, 3, 3, 2, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 2, 0, 3, 3, 1, 0, 2, 0, 0],
                    [0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
                    [0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
                    [0, 0, 2, 0, 2, 3, 3, 3, 3, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 0, 8],
                    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
                    [8, 0, 2, 2, 2, 2, 2, 2, 8, 0, 8, 0, 8],
                    [8, 0, 8, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8],
                    [8, 0, 8, 0, 2, 2, 1, 0, 8, 0, 8, 0, 8],
                    [8, 0, 8, 0, 8, 0, 0, 0, 8, 0, 8, 0, 8],
                    [8, 0, 8, 0, 8, 0, 0, 0, 8, 0, 8, 0, 8],
                    [8, 0, 8, 0, 8, 2, 2, 2, 2, 0, 8, 0, 8],
                    [8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
                    [8, 0, 8, 2, 2, 2, 2, 2, 2, 2, 2, 0, 8],
                    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                    [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
            }
        ],
    }


def _diagonal_component_pack_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_input = [
        [0, 0, 0, 0, 0, 0, 0],
        [4, 4, 0, 0, 0, 0, 0],
        [4, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 3, 3],
        [0, 0, 0, 2, 0, 3, 3],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    test_input = [
        [0, 0, 0, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0],
        [7, 0, 8, 8, 0, 0, 0],
        [7, 0, 8, 8, 0, 6, 0],
        [7, 0, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    return {
        "id": "03560426_like",
        "train": [{"input": train_input, "output": adapter._grid_diagonal_component_pack(train_input)}],
        "test": [{"input": test_input, "output": adapter._grid_diagonal_component_pack(test_input)}],
    }


def _marker_shape_lookup_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_a = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0],
        [0, 8, 0, 8, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]
    train_b = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0],
        [0, 8, 0, 8, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    train_c = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0],
        [0, 8, 0, 8, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    test_input = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 8, 8, 0, 0],
        [0, 0, 8, 0, 8, 0, 0],
        [0, 0, 8, 8, 8, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]
    return {
        "id": "009d5c81_like",
        "train": [
            {
                "input": train_a,
                "output": adapter._grid_marker_shape_color_lookup_recolor(
                    train_a,
                    params={
                        "marker_color": 1,
                        "object_color": 8,
                        "shape_to_color": {"0:1;1:0;1:1;1:2;2:1": 2},
                    },
                ),
            },
            {
                "input": train_b,
                "output": adapter._grid_marker_shape_color_lookup_recolor(
                    train_b,
                    params={
                        "marker_color": 1,
                        "object_color": 8,
                        "shape_to_color": {"0:0;1:0;2:0;2:1": 3},
                    },
                ),
            },
            {
                "input": train_c,
                "output": adapter._grid_marker_shape_color_lookup_recolor(
                    train_c,
                    params={
                        "marker_color": 1,
                        "object_color": 8,
                        "shape_to_color": {"0:0;0:1;0:2;1:1;2:1": 7},
                    },
                ),
            },
        ],
        "test": [
            {
                "input": test_input,
                "output": adapter._grid_marker_shape_color_lookup_recolor(
                    test_input,
                    params={
                        "marker_color": 1,
                        "object_color": 8,
                        "shape_to_color": {"0:0;0:1;0:2;1:1;2:1": 7},
                    },
                ),
            }
        ],
    }


def _repeated_tile_consensus_task() -> dict:
    return {
        "id": "0607ce86_consensus_like",
        "train": [
            {
                "input": [
                    [2, 8, 0, 2, 2],
                    [3, 3, 0, 3, 8],
                    [0, 1, 0, 0, 0],
                    [2, 2, 0, 8, 2],
                    [3, 8, 0, 3, 3],
                ],
                "output": [
                    [2, 2, 0, 2, 2],
                    [3, 3, 0, 3, 3],
                    [0, 0, 0, 0, 0],
                    [2, 2, 0, 2, 2],
                    [3, 3, 0, 3, 3],
                ],
            }
        ],
        "test": [
            {
                "input": [
                    [7, 4, 0, 7, 9],
                    [5, 5, 0, 9, 5],
                    [0, 2, 0, 0, 1],
                    [9, 4, 0, 7, 4],
                    [5, 9, 0, 5, 5],
                ],
                "output": [
                    [7, 4, 0, 7, 4],
                    [5, 5, 0, 5, 5],
                    [0, 0, 0, 0, 0],
                    [7, 4, 0, 7, 4],
                    [5, 5, 0, 5, 5],
                ],
            }
        ],
    }


def _marker_opposite_crop_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)

    def build_original(rotated_grid: list[list[int]]) -> list[list[int]]:
        return adapter._grid_rotate_90(adapter._grid_rotate_90(rotated_grid))

    train_rotated = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 6, 4, 0],
        [0, 0, 0, 0, 0, 0, 6, 9, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    test_rotated = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 7, 3, 0],
        [0, 0, 0, 0, 0, 0, 7, 1, 7, 0],
        [0, 0, 0, 0, 0, 0, 3, 7, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    params = {"mode": "marker_opposite_crop", "marker_color": 8, "mirror_margin": 2}
    return {
        "id": "0934a4d8_marker_crop_like",
        "train": [
            {
                "input": build_original(train_rotated),
                "output": adapter._grid_marker_opposite_crop(train_rotated, params=params),
            }
        ],
        "test": [
            {
                "input": build_original(test_rotated),
                "output": adapter._grid_marker_opposite_crop(test_rotated, params=params),
            }
        ],
    }


def _marker_axis_crop_task() -> dict:
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train_input = [
        [0, 0, 0, 0, 0, 6, 4, 6, 0, 0],
        [0, 0, 0, 0, 0, 4, 9, 4, 0, 0],
        [0, 0, 0, 0, 0, 6, 4, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 8, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 8, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 8, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    test_input = [
        [0, 0, 0, 0, 7, 3, 7, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, 3, 0, 0, 0],
        [0, 0, 0, 0, 7, 3, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    params = {"mode": "marker_axis_crop", "marker_color": 8, "mirror_margin": 2}
    return {
        "id": "0934a4d8_axis_crop_like",
        "train": [{"input": train_input, "output": adapter._grid_marker_axis_crop(train_input, params=params)}],
        "test": [{"input": test_input, "output": adapter._grid_marker_axis_crop(test_input, params=params)}],
    }


def test_adapter_ignores_legacy_fallback_and_uses_sovereign_solver():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    adapter.pipeline = None

    def fallback(task: dict, use_enriched: bool) -> dict:
        raise AssertionError("legacy fallback should not be called")

    result = adapter.solve_task(_sample_task(), fallback_solver=fallback)
    assert result["solver"] == "arc_sovereign"
    assert result["correct"] is True


def test_adapter_strict_mode_still_uses_sovereign_solver():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    adapter.pipeline = None
    adapter.strict_legacy = True
    result = adapter.solve_task(_sample_task())
    assert result["solver"] == "arc_sovereign"
    assert result["correct"] is True


def test_phase_tile_pattern_is_discovered_and_applied_compositionally():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _phase_tile_task()
    patterns = adapter.discover_patterns(task["train"])
    phase_patterns = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("tile_pattern", "phase_shift")]
    assert phase_patterns
    assert max(int(p.composition_depth) for p in phase_patterns) >= 2
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], phase_patterns[0])
    assert predicted == task["test"][0]["output"]


def test_phase_tile_task_stays_correct_under_benchmark_arc_flags():
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        enable_contrastive_learning=True,
        enable_validity_gates=True,
        enable_fuzzy_oracle=True,
        enable_figure_ground_reversal=True,
        enable_object_aware_generation=True,
        enable_ptx_ranking=False,
        enable_full_ptx=False,
    )
    result = adapter.solve_task(_phase_tile_task())
    assert result["correct"] is True
    assert result["predicted"] == _phase_tile_task()["test"][0]["output"]


def test_self_pattern_nonzero_mask_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _self_pattern_nonzero_mask_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert target
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], target[0])
    assert predicted == task["test"][0]["output"]


def test_self_pattern_complement_mask_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _self_pattern_complement_mask_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "self_pattern_complement_mask" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "self_pattern_complement_mask")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_enclosed_zero_count_fill_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _enclosed_fill_count_mod_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [
        p
        for p in patterns
        if tuple(getattr(p, "ops", ())) == ("object_extract", "connected_components", "conditional_fill")
    ]
    assert target
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], target[0])
    assert predicted == task["test"][0]["output"]


def test_enclosed_zero_count_lookup_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _enclosed_fill_count_lookup_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [
        p
        for p in patterns
        if tuple(getattr(p, "ops", ())) == ("object_extract", "connected_components", "conditional_fill")
        and getattr(p, "params", {}).get("mode") == "enclosed_zero_count_lookup"
    ]
    assert target
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], target[0])
    assert predicted == task["test"][0]["output"]


def test_connect_color_pairs_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _connect_color_pairs_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "connect_color_pairs" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "connect_color_pairs")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_separator_bridge_projection_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _separator_bridge_projection_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "separator_bridge_projection" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "separator_bridge_projection")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_connect_color_pairs_survives_benchmark_flags_with_negative_forms():
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        enable_contrastive_learning=True,
        enable_validity_gates=True,
        enable_fuzzy_oracle=True,
        enable_figure_ground_reversal=True,
        enable_object_aware_generation=True,
        enable_ptx_ranking=False,
        enable_full_ptx=False,
    )
    result = adapter.solve_task(_connect_color_pairs_task())
    assert result["correct"] is True


def test_anchor_spiral_pair_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _anchor_spiral_pair_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "anchor_spiral_pair" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "anchor_spiral_pair")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_anchor_spiral_pair_matches_arc_eval_family():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _anchor_spiral_arc_eval_like_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "anchor_spiral_pair" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "anchor_spiral_pair")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_diagonal_component_pack_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _diagonal_component_pack_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "diagonal_component_pack" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "diagonal_component_pack")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_marker_shape_lookup_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _marker_shape_lookup_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "lookup_color_remap")]
    assert target
    assert any(getattr(p, "params", {}).get("mode") == "marker_shape_color_lookup" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "marker_shape_color_lookup")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_repeated_tile_consensus_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _repeated_tile_consensus_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [p for p in patterns if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")]
    assert any(getattr(p, "params", {}).get("mode") == "repeated_tile_consensus" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "repeated_tile_consensus")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_marker_opposite_crop_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _marker_opposite_crop_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [
        p
        for p in patterns
        if "window_extract" in tuple(getattr(p, "ops", ()))
    ]
    assert any(
        getattr(p, "params", {}).get("mode") in {"marker_opposite_crop", "marker_axis_crop"}
        for p in target
    )
    pattern = next(
        p
        for p in target
        if getattr(p, "params", {}).get("mode") in {"marker_opposite_crop", "marker_axis_crop"}
    )
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_marker_axis_crop_is_discovered_and_applied():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    task = _marker_axis_crop_task()
    patterns = adapter.discover_patterns(task["train"])
    target = [
        p
        for p in patterns
        if tuple(getattr(p, "ops", ())) == ("object_extract", "window_extract")
    ]
    assert any(getattr(p, "params", {}).get("mode") == "marker_axis_crop" for p in target)
    pattern = next(p for p in target if getattr(p, "params", {}).get("mode") == "marker_axis_crop")
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], pattern)
    assert predicted == task["test"][0]["output"]


def test_marker_axis_crop_solves_real_0934a4d8_when_arc_corpus_present():
    task_path = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation/0934a4d8.json")
    if not task_path.exists():
        pytest.skip("local ARC evaluation corpus not present")
    task = json.loads(task_path.read_text())
    task["id"] = "0934a4d8"
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    patterns = adapter.discover_patterns(task["train"])
    target = [
        p
        for p in patterns
        if tuple(getattr(p, "ops", ())) == ("object_extract", "window_extract")
        and getattr(p, "params", {}).get("mode") == "marker_axis_crop"
    ]
    assert target
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], target[0])
    assert predicted == task["test"][0]["output"]


def test_separator_bridge_projection_solves_real_05a7bcf2_when_arc_corpus_present():
    task_path = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation/05a7bcf2.json")
    if not task_path.exists():
        pytest.skip("local ARC evaluation corpus not present")
    task = json.loads(task_path.read_text())
    task["id"] = "05a7bcf2"
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    patterns = adapter.discover_patterns(task["train"])
    target = [
        p
        for p in patterns
        if tuple(getattr(p, "ops", ())) == ("object_extract", "object_place")
        and getattr(p, "params", {}).get("mode") == "separator_bridge_projection"
    ]
    assert target
    predicted = adapter._generate_candidate_from_pattern(task["test"][0]["input"], target[0])
    assert predicted == task["test"][0]["output"]


def test_verified_arc_four_pass_candidate_is_not_rejected_for_family_mismatch():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    passes, reason = adapter._candidate_passes_generation_constraints(
        candidate_grid=[[1, 1], [1, 1]],
        input_grid=[[1, 0], [0, 1]],
        profile={},
        constraint_scores={
            "family_match": True,
            "family_score": 0.85,
            "shape_score": 1.0,
            "palette_score": 1.0,
            "object_score": 1.0,
        },
    )
    assert passes is True
    assert reason == ""


def test_low_precision_autonomous_navigation_does_not_outrank_precise_four_pass():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, enable_ptx_ranking=False)
    four_pass = {
        "candidate": [[1]],
        "score": 0.0,
        "components": {
            "source_precision": 0.98,
            "quality_prior": 0.5,
            "train_similarity": 0.675,
            "novelty": 1.0,
            "grammar_confidence": 0.897,
            "cross_modal": 0.4,
            "compositional": 0.4,
            "reuse": 0.0,
            "family_bonus": 0.1,
            "navigation_multiplier": 1.0,
            "family_score": 1.0,
            "shape_score": 1.0,
            "palette_score": 1.0,
            "object_score": 1.0,
            "generation_pass": True,
        },
        "pattern": {"source": "arc_four_pass"},
    }
    autonomous = {
        "candidate": [[2]],
        "score": 0.0,
        "components": {
            "source_precision": 0.19,
            "quality_prior": 0.5,
            "train_similarity": 0.675,
            "novelty": 1.0,
            "grammar_confidence": 0.9,
            "cross_modal": 0.4,
            "compositional": 0.4,
            "reuse": 0.1,
            "family_bonus": 0.1,
            "navigation_multiplier": 1.45,
            "family_score": 1.0,
            "shape_score": 1.0,
            "palette_score": 1.0,
            "object_score": 1.0,
            "generation_pass": True,
        },
        "pattern": {"source": "autonomous_generation"},
    }
    ranked = adapter._score_and_sort_candidates_sovereign([four_pass, autonomous])
    assert ranked[0]["pattern"]["source"] == "arc_four_pass"


def test_describe_visual_transformation_reflection_and_color():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    desc = adapter._describe_visual_transformation(
        [[1, 2], [3, 4]],
        [[2, 1], [4, 3]],
    )
    assert "reflect across vertical axis" in desc

    recolor_desc = adapter._describe_visual_transformation(
        [[1, 1], [2, 2]],
        [[3, 3], [4, 4]],
    )
    assert "color transformation" in recolor_desc or "recolor" in recolor_desc


class _FakeNavigator:
    def generate_from_procedural(self, **kwargs):
        return {
            "id": "gen_rule_1",
            "metadata": {
                "source_galaxy": kwargs.get("source_galaxy", "3DObjects"),
                "confidence": 0.82,
            },
        }

    def navigate_and_compose(self, **_kwargs):
        return {
            "candidates": [
                {"entry": {"id": "cross_modal_rule_1"}, "confidence": 0.76},
                {"entry": {"id": "cross_modal_rule_2"}, "score": 0.71},
            ]
        }


class _FakeKV:
    def __init__(self):
        self.trm_navigator = _FakeNavigator()
        self.events = []

    def log_event(self, event_type: str, event_data: dict):
        self.events.append((event_type, event_data))


def test_discover_patterns_includes_all_sources():
    kv = _FakeKV()
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, knowledgeverse=kv)
    patterns = adapter.discover_patterns(_sample_task()["train"])
    assert patterns
    sources = {pattern.source for pattern in patterns}
    assert "traditional" in sources
    assert "autonomous_generation" in sources
    assert "multi_galaxy_composition" in sources


def test_prepare_discovery_examples_adds_negative_form_pairs():
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        enable_figure_ground_reversal=True,
    )
    train = [{"input": [[1, 0], [2, 3]], "output": [[0, 1], [3, 2]]}]
    prepared = adapter._prepare_discovery_examples(train)
    assert len(prepared) >= 2
    assert any(
        (row.get("metadata", {}) or {}).get("form_polarity") == "negative"
        for row in prepared
    )


def test_discover_patterns_contrastive_adds_anti_patterns():
    kv = _FakeKV()
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        knowledgeverse=kv,
        enable_contrastive_learning=True,
    )
    patterns = adapter.discover_patterns(_sample_task()["train"])
    sources = {pattern.source for pattern in patterns}
    assert "contrastive_anti" in sources


def test_forced_navigation_injection_adds_curriculum_patterns():
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        enable_forced_navigation_curriculum=True,
        forced_navigation_ratio=1.0,
        forced_navigation_required_galaxies="Math,Reality",
    )
    base = [
        _GeneratedPattern(
            pattern_id="base_0",
            source_galaxy="Drawing",
            target_galaxy="Grammar",
            confidence=0.6,
            query="traditional visual rule: reflect across vertical axis",
            source="traditional",
            pair_index=0,
        )
    ]
    injected = adapter._inject_forced_navigation_patterns(
        train_examples=_sample_task()["train"],
        patterns=base,
    )
    assert len(injected) >= len(base)
    assert any(pattern.source == "curriculum_forced_navigation" for pattern in injected)


def test_forced_navigation_source_expands_galaxy_participation():
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_forced_navigation_curriculum=True,
        forced_navigation_ratio=0.5,
        forced_navigation_required_galaxies="Math,Reality",
    )
    galaxies = adapter._extract_pattern_galaxy_set(
        {
            "source": "curriculum_forced_navigation",
            "metadata": {},
        }
    )
    assert "Drawing" in galaxies
    assert "Grammar" in galaxies
    assert "Math" in galaxies
    assert "Reality" in galaxies


def test_rank_candidates_prefers_autonomous_and_cross_modal_signals():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, knowledgeverse=_FakeKV())
    candidates = [
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2]],
    ]
    patterns = [
        {
            "pattern_id": "traditional_low",
            "source": "traditional",
            "confidence": 0.55,
            "metadata": {"composition_depth": 1, "reuse_count": 1},
        },
        {
            "pattern_id": "autonomous_high",
            "source": "autonomous_generation",
            "confidence": 0.82,
            "metadata": {
                "composition_depth": 3,
                "reuse_count": 8,
                "source_galaxy": "Drawing+Math+Reality",
                "cross_modal": True,
            },
        },
    ]
    ranked = adapter._rank_candidates(candidates, patterns)
    assert ranked
    assert ranked[0]["pattern"]["pattern_id"] == "autonomous_high"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_palette_distribution_score_discriminates_candidates():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    profile = {
        "inferred_family": "spatial_or_recolor",
        "output_palette": [1, 2],
        "output_palette_distribution": {1: 0.75, 2: 0.25},
        "stable_output_palette_size": 2,
    }
    input_grid = [[1, 2], [1, 2]]
    good = [[1, 1], [1, 2]]
    bad = [[1, 2], [1, 2]]
    good_score = adapter._compute_generation_constraint_scores(
        candidate_grid=good,
        input_grid=input_grid,
        profile=profile,
    )["palette_score"]
    bad_score = adapter._compute_generation_constraint_scores(
        candidate_grid=bad,
        input_grid=input_grid,
        profile=profile,
    )["palette_score"]
    assert float(good_score) > float(bad_score)


def test_palette_penalty_weight_increases_penalty_strength():
    components = {
        "family_score": 1.0,
        "shape_score": 1.0,
        "palette_score": 0.4,
        "object_score": 1.0,
    }
    baseline = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    palette_heavy = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        palette_penalty_weight=2.0,
    )
    baseline_score = baseline._apply_constraint_penalty(base_score=1.0, components=components)
    palette_heavy_score = palette_heavy._apply_constraint_penalty(base_score=1.0, components=components)
    assert palette_heavy_score < baseline_score


class _FakePipelineResult:
    def __init__(self, output_grid):
        self.output_grid = output_grid
        self.correct = True
        self.score = 0.9
        self.fuzzy_score = 0.9
        self.best_program = "GRID 2 2 FILL"
        self.program_type = "test_program"
        self.signature = "sig:test"


class _FakePipeline:
    def process_task(self, **kwargs):
        test_input = kwargs["test_input"]
        # Predict horizontal flip to match _sample_task expected output.
        output = [list(reversed(row)) for row in test_input]
        return _FakePipelineResult(output)


def test_solve_task_emits_oracle_and_ranking_diagnostics():
    kv = _FakeKV()
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, knowledgeverse=kv)
    adapter.pipeline = _FakePipeline()

    result = adapter.solve_task(_sample_task())

    assert result["correct"] is True
    assert result["legacy_correct"] is False
    assert result["solver"] == "arc_sovereign"
    assert "oracle_at_3" in result
    assert "oracle_at_10" in result
    assert "oracle_at_all" in result
    assert result["oracle_at_all"] is True
    assert result["correct_rank"] is not None
    assert "ranking_changed_top1" in result
    assert "ranking_score_range" in result
    assert "ranking_score_stddev" in result
    assert isinstance(result["ranking_top_5_scores"], list)
    assert isinstance(result["ranking_top_5_sources"], list)

    event_types = [event_type for event_type, _ in kv.events]
    assert "arc_pattern_discovery" in event_types
    assert "arc_candidate_contrast" in event_types


def test_validity_profile_infers_family_and_expected_shape():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
        {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
    ]
    profile = adapter._build_validity_profile(train_examples=train, test_input=[[9, 0], [1, 2]])
    assert profile["inferred_family"] in {"spatial", "spatial_or_recolor"}
    assert profile["expected_shape"] == (2, 2)


def test_candidate_validity_rejects_family_mismatch():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}]
    profile = adapter._build_validity_profile(train_examples=train, test_input=[[9, 0], [1, 2]])
    # Mismatch family: scaling output when family inferred as spatial.
    scaled = [[9, 9, 0, 0], [9, 9, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2]]
    ok, reason = adapter._candidate_passes_validity(scaled, profile)
    assert ok is False
    assert reason in {"family", "shape"}


def test_oracle_metrics_include_stratified_fuzzy_keys():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    expected = [[1, 0], [0, 1]]
    candidates = [
        {"candidate": [[1, 0], [0, 1]]},
        {"candidate": [[1, 1], [0, 1]]},
    ]
    metrics = adapter._compute_oracle_metrics(candidates, expected, fuzzy_threshold=0.95)
    assert metrics["oracle_at_all"] is True
    assert "oracle_fuzzy_0_80" in metrics
    assert "oracle_fuzzy_0_85" in metrics
    assert "oracle_fuzzy_0_90" in metrics
    assert "oracle_fuzzy_0_95" in metrics
    assert "oracle_exact" in metrics


def test_full_ptx_validity_path_is_used(monkeypatch):
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_full_ptx=True,
        ptx_validity_strictness="relaxed",
    )
    adapter._full_ptx_available = True

    class _StubPTX:
        def apply_validity_gates_relaxed_ptx(self, *, ranked_candidates, validity_profile, strictness):
            assert strictness == "relaxed"
            return ranked_candidates[:1], {
                "enabled": True,
                "mode": "ptx_validity",
                "strictness": strictness,
                "pre_count": len(ranked_candidates),
                "post_count": 1,
                "filtered_count": max(0, len(ranked_candidates) - 1),
                "fallback_to_ungated": False,
                "family_rejects": 0,
                "shape_rejects": 0,
                "palette_rejects": 0,
                "object_rejects": 0,
                "validity_reject_rate": 0.5,
            }

    monkeypatch.setattr("benchmarks.arc_agi_2_adapter.ARC_PTX_OPS", _StubPTX())
    filtered, report = adapter._apply_validity_gates(
        ranked_candidates=[
            {"candidate": [[1, 0], [0, 1]], "pattern": {"pattern_id": "a"}},
            {"candidate": [[0, 1], [1, 0]], "pattern": {"pattern_id": "b"}},
        ],
        validity_profile={"inferred_family": "spatial"},
    )
    assert len(filtered) == 1
    assert report["mode"] == "ptx_validity"
    assert report["strictness"] == "relaxed"


def test_full_ptx_oracle_path_is_used(monkeypatch):
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_full_ptx=True,
    )
    adapter._full_ptx_available = True

    class _StubPTX:
        def check_oracle_fuzzy_ptx(self, **_kwargs):
            return {
                "oracle_at_3": False,
                "oracle_at_10": True,
                "oracle_at_all": True,
                "correct_rank": 4,
                "oracle_fuzzy_0_80": True,
                "oracle_fuzzy_0_85": True,
                "oracle_fuzzy_0_90": False,
                "oracle_fuzzy_0_95": False,
                "oracle_exact": True,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": True,
                "fuzzy_oracle_at_all": True,
                "fuzzy_best_score": 0.91,
                "fuzzy_best_rank": 4,
            }

    monkeypatch.setattr("benchmarks.arc_agi_2_adapter.ARC_PTX_OPS", _StubPTX())
    metrics = adapter._compute_oracle_metrics(
        ranked_candidates=[{"candidate": [[1, 0], [0, 1]]}],
        expected_output=[[1, 0], [0, 1]],
        fuzzy_threshold=0.95,
    )
    assert metrics["oracle_at_all"] is True
    assert metrics["oracle_at_10"] is True
    assert metrics["ptx_oracle_used"] is True


def test_oracle_rejected_rescue_augments_oracle_metrics_exact():
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_oracle_rejected_rescue=True,
        oracle_rejected_rescue_size=4,
        enable_fuzzy_oracle=True,
        fuzzy_oracle_threshold=0.95,
    )
    base_metrics = {
        "oracle_at_3": False,
        "oracle_at_10": False,
        "oracle_at_all": False,
        "correct_rank": None,
        "oracle_fuzzy_0_80": False,
        "oracle_fuzzy_0_85": False,
        "oracle_fuzzy_0_90": False,
        "oracle_fuzzy_0_95": False,
        "oracle_exact": False,
        "fuzzy_oracle_at_3": False,
        "fuzzy_oracle_at_10": False,
        "fuzzy_oracle_at_all": False,
        "fuzzy_best_score": 0.40,
        "fuzzy_best_rank": 0,
        "ptx_oracle_used": False,
    }
    rescue_candidates = [
        {"candidate": [[1, 0], [0, 1]], "score": 0.1, "pattern": {}, "components": {"generation_pass": False}},
    ]
    merged = adapter._augment_oracle_metrics_with_rejected_rescue(
        oracle_metrics=base_metrics,
        rejected_rescue_candidates=rescue_candidates,
        expected_output=[[1, 0], [0, 1]],
        ranked_candidate_count=10,
    )
    assert merged["oracle_rejected_rescue_enabled"] is True
    assert merged["oracle_rejected_rescue_candidate_count"] == 1
    assert merged["oracle_rejected_rescue_exact"] is True
    assert merged["oracle_at_all"] is True
    assert merged["correct_rank"] == 10


def test_build_oracle_rejected_rescue_candidates_skips_existing_signatures():
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_oracle_rejected_rescue=True,
        oracle_rejected_rescue_size=8,
    )
    existing_grid = [[1, 1], [0, 0]]
    candidate_map = {
        adapter._grid_signature(existing_grid): (existing_grid, {"pattern_id": "existing"}),
    }
    rejected_reserve = [
        (0.9, existing_grid, {"pattern_id": "dup", "generation_constraint": {"reason": "shape"}}),
        (0.8, [[1, 0], [0, 1]], {"pattern_id": "unique", "generation_constraint": {"reason": "palette"}}),
    ]
    rescue = adapter._build_oracle_rejected_rescue_candidates(
        rejected_reserve=rejected_reserve,
        candidate_map=candidate_map,
    )
    assert len(rescue) == 1
    assert rescue[0]["pattern"]["pattern_id"] == "unique"
    assert rescue[0]["components"]["generation_pass"] is False
