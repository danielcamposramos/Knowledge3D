"""Load action embeddings from reusable Galaxy action atoms."""

from __future__ import annotations

import base64
from array import array
from typing import Any

from knowledge3d.cranium.action_primitives_bootstrap import (
    ACTION_CLICK,
    ACTION_GRAB,
    ACTION_HOLD,
    ACTION_LOOK_AT,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_PERFORM,
    ACTION_REACH,
    ACTION_RELEASE,
    ACTION_TELEPORT,
    ACTION_UNDO,
    ACTION_USE,
    ACTION_WALK_TO,
    build_default_action_galaxy,
)
from knowledge3d.knowledgeverse.vram_task_buffer import EMBEDDING32_DIMS


ARC3_ACTION_ATOM_IDS = [
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
]
ARC3_EXTENDED_ACTION_ATOM_IDS = [
    *ARC3_ACTION_ATOM_IDS,
    ACTION_PERFORM,
    ACTION_CLICK,
    ACTION_UNDO,
]
AVATAR_ACTION_ATOM_IDS = [
    ACTION_REACH,
    ACTION_GRAB,
    ACTION_HOLD,
    ACTION_RELEASE,
    ACTION_USE,
    ACTION_WALK_TO,
    ACTION_TELEPORT,
    ACTION_LOOK_AT,
]


def load_action_embeddings_from_galaxy(
    galaxy: Any,
    action_ids: list[str] | None = None,
) -> list[list[float]]:
    """Load action star embeddings from Reality Galaxy."""
    ids = list(action_ids or ARC3_ACTION_ATOM_IDS)
    embeddings: list[list[float]] = []
    for atom_id in ids:
        node = galaxy.get_node(atom_id) if galaxy is not None and hasattr(galaxy, "get_node") else None
        if node is not None:
            embedding = _node_to_embedding(node)
            if not any(abs(float(value)) > 1.0e-8 for value in embedding):
                decoded = _decode_embedding(node.embedding)
                embedding = decoded[:EMBEDDING32_DIMS] if decoded else embedding
        else:
            embedding = _displacement_to_embedding(_get_displacement(galaxy, atom_id))
        if len(embedding) < EMBEDDING32_DIMS:
            embedding.extend([0.0] * (EMBEDDING32_DIMS - len(embedding)))
        embeddings.append(embedding[:EMBEDDING32_DIMS])
    return embeddings


def load_default_action_embeddings() -> list[list[float]]:
    galaxy = build_default_action_galaxy()
    return load_action_embeddings_from_galaxy(galaxy, ARC3_EXTENDED_ACTION_ATOM_IDS)


def _decode_embedding(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(value) for value in raw]
    if not isinstance(raw, dict):
        return []
    payload = raw.get("program")
    codec = str(raw.get("codec", "")).strip().lower()
    if codec != "raw_f32" or not payload:
        return []
    try:
        decoded = base64.b64decode(str(payload).encode("ascii"))
        floats = array("f")
        floats.frombytes(decoded)
        return [float(value) for value in floats]
    except Exception:
        return []


def _normalize_action_id(atom_id: str) -> str:
    raw = str(atom_id or "").strip()
    if raw.startswith("atom:"):
        return raw
    if raw.startswith("action:"):
        return f"atom:{raw}"
    return raw


def _get_displacement(galaxy: Any, atom_id: str) -> list[float]:
    node = galaxy.get_node(_normalize_action_id(atom_id)) if galaxy is not None and hasattr(galaxy, "get_node") else None
    if node is not None and hasattr(node, "metadata") and isinstance(node.metadata, dict):
        disp = node.metadata.get("displacement", [0.0, 0.0])
        return [float(value) for value in list(disp)[:2]]
    defaults = {
        ACTION_MOVE_UP: [0.0, -1.0],
        ACTION_MOVE_DOWN: [0.0, 1.0],
        ACTION_MOVE_LEFT: [-1.0, 0.0],
        ACTION_MOVE_RIGHT: [1.0, 0.0],
        ACTION_PERFORM: [0.0, 0.0],
        ACTION_CLICK: [0.0, 0.0],
        ACTION_UNDO: [0.0, 0.0],
        ACTION_REACH: [0.0, 0.0],
        ACTION_GRAB: [0.0, 0.0],
        ACTION_HOLD: [0.0, 0.0],
        ACTION_RELEASE: [0.0, 0.0],
        ACTION_USE: [0.0, 0.0],
        ACTION_WALK_TO: [0.0, 0.0],
        ACTION_TELEPORT: [0.0, 0.0],
        ACTION_LOOK_AT: [0.0, 0.0],
    }
    return defaults.get(_normalize_action_id(atom_id), [0.0, 0.0])


def _displacement_to_embedding(displacement: list[float]) -> list[float]:
    dx = displacement[0] if len(displacement) > 0 else 0.0
    dy = displacement[1] if len(displacement) > 1 else 0.0
    magnitude = (dx * dx + dy * dy) ** 0.5
    norm_dx = dx / (magnitude + 1.0e-8)
    norm_dy = dy / (magnitude + 1.0e-8)
    embedding = [0.0] * EMBEDDING32_DIMS
    embedding[0] = dx
    embedding[1] = dy
    embedding[4] = magnitude
    embedding[5] = norm_dx
    embedding[6] = norm_dy
    embedding[7] = 1.0 if magnitude > 0.0 else 0.0
    return embedding


def _node_to_embedding(node: Any) -> list[float]:
    displacement = _get_displacement(None, getattr(node, "node_id", ""))
    embedding = _displacement_to_embedding(displacement)
    metadata = getattr(node, "metadata", {}) or {}
    action_type = str(metadata.get("action_type", ""))

    action_type_signatures = {
        "spatial_translation": 0.9,
        "spatial_translation_composed": 0.8,
        "spatial_navigation": 0.7,
        "spatial_navigation_composed": 0.7,
        "spatial_orientation": 0.6,
        "spatial_interaction": 0.3,
        "spatial_selection": 0.2,
        "temporal_reversal": -0.9,
        "object_interaction": 0.1,
    }
    embedding[2] = action_type_signatures.get(action_type, 0.0)
    action_mode_signatures = {
        "spatial_translation": 0.1,
        "spatial_translation_composed": 0.2,
        "spatial_navigation": 0.35,
        "spatial_navigation_composed": 0.45,
        "spatial_orientation": 0.25,
        "spatial_interaction": 0.7,
        "spatial_selection": 0.95,
        "temporal_reversal": -1.0,
        "object_interaction": 0.6,
    }
    embedding[3] = action_mode_signatures.get(action_type, 0.0)
    if metadata.get("parameterized"):
        embedding[3] += 0.25
    if metadata.get("inverse") is not None:
        embedding[3] -= 0.1

    token_stream = " ".join(
        [
            str(getattr(node, "visual_rpn", "")),
            str(getattr(node, "behavior_rpn", "")),
            str(getattr(node, "law_rpn", "")),
        ]
    ).strip()
    tokens = [token for token in token_stream.split() if token]
    for token in tokens:
        bucket = 8 + (_fnv1a32(token) % max(1, EMBEDDING32_DIMS - 8))
        embedding[bucket] += 0.25
    if displacement[0] != 0.0 or displacement[1] != 0.0:
        embedding[8] += 0.5
    if action_type in {"spatial_interaction", "spatial_selection", "object_interaction"}:
        embedding[9] += 0.5
    if action_type == "temporal_reversal":
        embedding[10] -= 0.75
    return embedding


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


__all__ = [
    "ARC3_ACTION_ATOM_IDS",
    "ARC3_EXTENDED_ACTION_ATOM_IDS",
    "AVATAR_ACTION_ATOM_IDS",
    "_displacement_to_embedding",
    "_get_displacement",
    "load_action_embeddings_from_galaxy",
    "load_default_action_embeddings",
]
