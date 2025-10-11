"""Bridge for atomic fission/fusion kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class AtomicEvolution:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_atomic_fission_fusion.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_atomic_fission_fusion")

    def apply(
        self,
        atoms: cp.ndarray,
        mode: int,
        ratio: float,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        data = cp.asarray(atoms, dtype=cp.float32).ravel()
        out = cp.empty_like(data)
        self.kernel(
            (1,),
            (128,),
            (
                data.data.ptr,
                out.data.ptr,
                cp.uint32(data.size),
                cp.uint32(mode),
                cp.float32(ratio),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out.reshape(atoms.shape)


__all__ = ["AtomicEvolution"]
