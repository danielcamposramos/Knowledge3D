"""Bridge for the temporal reasoning kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class TemporalReasoner:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_temporal_reasoning.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_temporal_reasoning")

    def deltas(
        self,
        sequence: cp.ndarray,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        seq = cp.asarray(sequence, dtype=cp.float32)
        if seq.ndim != 2:
            raise ValueError("Sequence must be 2-D (time, features)")
        time_steps, feature_dim = seq.shape
        out = cp.empty_like(seq)
        self.kernel(
            (1,),
            (min(feature_dim, 128),),
            (
                seq.data.ptr,
                out.data.ptr,
                cp.uint32(time_steps),
                cp.uint32(feature_dim),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out


__all__ = ["TemporalReasoner"]
