"""Bridge for the fractal emitter kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class FractalEmitter:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_fractal_emitter.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_fractal_emitter")

    def emit(
        self,
        atoms: cp.ndarray,
        scale: float = 1.0,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        atoms_cp = cp.asarray(atoms, dtype=cp.float32)
        count = atoms_cp.size
        coords = cp.empty((count, 3), dtype=cp.float32)
        self.kernel(
            (1,),
            (128,),
            (
                atoms_cp.data.ptr,
                coords.data.ptr,
                cp.uint32(count),
                cp.float32(scale),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return coords


__all__ = ["FractalEmitter"]
