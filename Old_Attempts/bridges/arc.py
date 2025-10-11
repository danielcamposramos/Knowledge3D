"""ARC Reasoner bridge for gre_arc_reasoner.ptx."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class ArcReasoner:
    """Wrapper around the ARC PTX kernel.

    The kernel expects a flattened int32 grid and produces a vector of three
    int32 values describing a simple rule hypothesis.
    """

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_arc_reasoner.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_arc_reasoner")

    def infer(self, grid: cp.ndarray, stream: Optional[cp.cuda.Stream] = None) -> cp.ndarray:
        """Infer ARC rule metadata from a grid.

        Parameters
        ----------
        grid:
            CuPy array of shape (H, W) or (N,) with dtype int32.
        stream:
            Optional CUDA stream to run on.

        Returns
        -------
        cp.ndarray shape (3,) dtype=int32
        """

        grid_cp = cp.asarray(grid, dtype=cp.int32)
        flat = grid_cp.ravel()
        out = cp.zeros(3, dtype=cp.int32)
        grid_size = cp.uint32(flat.size)

        self.kernel(
            (1,),
            (32,),
            (flat.data.ptr, grid_size, out.data.ptr),
            stream=stream,
        )

        if stream is not None:
            stream.synchronize()
        return out


__all__ = ["ArcReasoner"]
