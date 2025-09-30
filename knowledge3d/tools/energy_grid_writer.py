from __future__ import annotations

"""Minimal GLB writer for an energy grid volume.

Stores a 3D grid of float32 as a binary buffer and records shape metadata in
extras.k3d on the root node. No geometry is included; consumers read extras and
bufferViews directly.
"""

from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np  # type: ignore


def write_energy_grid(out_path: Path, grid: np.ndarray) -> Path:
    from pygltflib import GLTF2, Scene, Node, Buffer, BufferView, Asset  # type: ignore

    g = GLTF2(asset=Asset(generator="k3d-energy-grid"), scenes=[Scene(nodes=[0])], scene=0)
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError("grid must be 3D")
    blob = grid.tobytes()
    g.buffers.append(Buffer(byteLength=len(blob)))
    g.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=len(blob)))
    node = Node(extras={
        "k3d": {
            "type": "energy_grid",
            "shape": list(grid.shape),
            "dtype": "float32",
            "bufferView": 0,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    })
    g.nodes.append(node)
    g.set_binary_blob(blob)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(out_path))
    return out_path

