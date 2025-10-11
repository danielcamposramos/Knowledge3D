"""Multimodal geometry router bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class GeometryRouter:
    """Route modality tensors through the gre_geometry_router kernel."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_geometry_router.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_geometry_router")

    def route(self, tensor: cp.ndarray, shape_id: int, stream: Optional[cp.cuda.Stream] = None) -> cp.ndarray:
        data = cp.asarray(tensor, dtype=cp.float32)
        out = cp.empty_like(data)
        length = cp.uint32(data.size)
        self.kernel(
            (1,),
            (128,),
            (
                data.data.ptr,
                out.data.ptr,
                length,
                cp.uint32(shape_id),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out


__all__ = ["GeometryRouter"]
