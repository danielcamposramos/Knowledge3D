"""Bridge for graph crystallizer kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class GraphCrystallizer:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_graph_crystallizer.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_graph_crystallizer")

    def crystallize(
        self,
        nodes: cp.ndarray,
        neighbors: cp.ndarray,
        ema_rate: float = 0.1,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        node_values = cp.asarray(nodes, dtype=cp.float32).ravel()
        neighbor_values = cp.asarray(neighbors, dtype=cp.float32).ravel()
        if node_values.size != neighbor_values.size:
            raise ValueError("Node and neighbor arrays must align")
        out = cp.empty_like(node_values)
        self.kernel(
            (1,),
            (128,),
            (
                node_values.data.ptr,
                neighbor_values.data.ptr,
                out.data.ptr,
                cp.uint32(node_values.size),
                cp.float32(ema_rate),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out.reshape(nodes.shape)


__all__ = ["GraphCrystallizer"]
