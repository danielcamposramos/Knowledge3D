"""Bridge for the multimodal halting gate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class MultimodalHaltingGate:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_multimodal_halting_gate.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_multimodal_halting_gate")

    def evaluate(
        self,
        logits: cp.ndarray,
        mask: cp.ndarray,
        threshold: float,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        logits_cp = cp.asarray(logits, dtype=cp.float32).ravel()
        mask_cp = cp.asarray(mask, dtype=cp.uint32).ravel()
        if logits_cp.size != mask_cp.size:
            raise ValueError("Logits and mask must align")
        out = cp.zeros_like(mask_cp)
        self.kernel(
            (1,),
            (128,),
            (
                logits_cp.data.ptr,
                mask_cp.data.ptr,
                out.data.ptr,
                cp.uint32(logits_cp.size),
                cp.float32(threshold),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out.reshape(mask.shape)


__all__ = ["MultimodalHaltingGate"]
