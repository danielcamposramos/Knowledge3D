
"""Utilities to expose dynamic LOD saliency data to the viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import numpy as np


def _normalize_node_ids(node_ids: Sequence[Union[int, str]], count: int) -> Sequence[str]:
    if not node_ids:
        return [str(i) for i in range(count)]
    if len(node_ids) != count:
        raise ValueError(f"node id count ({len(node_ids)}) does not match saliency rows ({count})")
    return [str(node_id) for node_id in node_ids]


def build_saliency_payload(
    node_ids: Sequence[Union[int, str]],
    saliency_map: np.ndarray,
    morton_levels: np.ndarray,
) -> Mapping[str, object]:
    """Create a JSON-serialisable payload for viewer consumption."""

    if saliency_map.ndim != 2 or saliency_map.shape[1] < 2:
        raise ValueError("saliency_map must be shaped (N, 2)")

    node_keys = _normalize_node_ids(node_ids, saliency_map.shape[0])
    payload = {
        "extensions": {
            "K3D_saliency": {
                "nodes": {}
            }
        }
    }

    nodes_section = payload["extensions"]["K3D_saliency"]["nodes"]
    lod_as_int = morton_levels.astype(np.int32)

    for idx, key in enumerate(node_keys):
        nodes_section[key] = {
            "cosine": float(saliency_map[idx, 0]),
            "lod": int(lod_as_int[idx]),
        }

    return payload


def write_saliency_manifest(
    output_path: Path,
    node_ids: Sequence[Union[int, str]],
    saliency_map: np.ndarray,
    morton_levels: np.ndarray,
) -> None:
    """Persist the saliency payload to disk for the viewer to ingest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_saliency_payload(node_ids, saliency_map, morton_levels)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

