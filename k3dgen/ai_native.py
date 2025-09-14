"""
AI-native helpers for constructing glTF primitives that embed K3D node data
directly in `primitive.extras["k3d"]`.

This module is aligned with spec/k3d_node_schema.json and supports efficient
storage of the embedding as base64-encoded Float32 bytes when higher-level
GLTF container wiring (Buffer/BufferView) is not in scope.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import base64
import numpy as np
import warnings
from pygltflib import Primitive  # type: ignore


def _as_float32_array(values: Any) -> np.ndarray:
    """Return values as a contiguous float32 numpy array (1D).

    Accepts list/tuple/np.ndarray.
    """
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float32, copy=False)
        return arr.reshape(-1)
    return np.asarray(values, dtype=np.float32).reshape(-1)


warnings.warn(
    "k3dgen.ai_native is deprecated as of Cranium Core v3.0: "
    "store embeddings in a Float32 BufferView and reference it via extras.k3d.embeddingsView.",
    DeprecationWarning,
    stacklevel=2,
)


def _embedding_to_b64(embedding: Any) -> Dict[str, Any]:
    """Convert embedding (list/np.ndarray) to a compact base64 payload.

    The payload includes dtype and dims for robust decoding on the client.
    """
    arr = _as_float32_array(embedding)
    raw = arr.tobytes(order="C")
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "data": b64,
        "dtype": "f32",
        "dims": int(arr.size),
        "endianness": "little",
    }


def create_ai_native_gltf_primitive(node_data: Mapping[str, Any]) -> Primitive:
    """Create a pygltflib.Primitive with AI-native K3D payload embedded in extras.

    The full `node_data` (conforming to spec/k3d_node_schema.json) is written
    into `primitive.extras["k3d"]`. If an `embedding` vector is present, a
    compact `embedding_b64` structure is also provided to avoid large JSON
    overhead. Both forms are kept for compatibility unless the caller has
    already supplied an `embedding_b64` field.

    Parameters
    - node_data: Mapping[str, Any]
        Node descriptor with keys like id, vector, embedding, metadata,
        neighbors, ai_interaction_protocol, ai_state_flags, embedding_version.

    Returns
    - Primitive: a glTF primitive with the payload in extras. Geometry
      attributes are intentionally omitted; callers can set them later or keep
      this as a metadata-only carrier.

    Example
    >>> from k3dgen.ai_native import create_ai_native_gltf_primitive
    >>> node = {
    ...   "id": "node-1",
    ...   "vector": [0.0, 1.0, 2.0],
    ...   "embedding": [0.1, 0.2, 0.3, 0.4],
    ...   "embedding_version": 1,
    ...   "metadata": {"label": "demo", "type": "concept"},
    ...   "neighbors": ["node-2"],
    ...   "ai_interaction_protocol": "direct_vector_manipulation",
    ...   "ai_state_flags": {"is_active": True, "is_traversable": True, "has_new_information": False},
    ... }
    >>> prim = create_ai_native_gltf_primitive(node)
    >>> isinstance(prim.extras.get("k3d"), dict)
    True
    """

    payload: Dict[str, Any] = deepcopy(dict(node_data))

    # Add compact embedding_b64 if an embedding exists and no compact form yet
    if "embedding" in payload and "embedding_b64" not in payload:
        try:
            payload["embedding_b64"] = _embedding_to_b64(payload["embedding"])  # type: ignore[arg-type]
        except Exception:
            # Fallback: leave only the JSON form if conversion fails
            pass

    primitive = Primitive(
        attributes={},  # geometry can be attached by caller as needed
        extras={"k3d": payload},
    )
    return primitive


__all__ = [
    "create_ai_native_gltf_primitive",
]
