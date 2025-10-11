"""Resonance-related GPU bridges."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class GalaxyResonanceEngine:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "galaxy_resonance_engine.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("galaxy_resonance_engine")

    def run(
        self,
        embeddings: cp.ndarray,
        latent: cp.ndarray,
        alpha: float = 0.5,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        emb = cp.asarray(embeddings, dtype=cp.float32)
        lat = cp.asarray(latent, dtype=cp.float32)
        if emb.shape != lat.shape:
            raise ValueError("Embeddings and latent tensors must match")
        batch, dim = emb.shape
        out = cp.empty_like(emb)
        self.kernel(
            (batch,),
            (1,),
            (
                emb.data.ptr,
                lat.data.ptr,
                out.data.ptr,
                cp.uint32(dim),
                cp.uint32(batch),
                cp.float32(alpha),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out


class ResonanceField:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_resonance_field.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_resonance_field")

    def compute(
        self,
        positions: cp.ndarray,
        density: cp.ndarray,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        pos = cp.asarray(positions, dtype=cp.float32).reshape(-1, 3)
        den = cp.asarray(density, dtype=cp.float32).ravel()
        if pos.shape[0] != den.size:
            raise ValueError("Positions and density must align")
        out = cp.empty_like(den)
        self.kernel(
            (1,),
            (128,),
            (
                pos.data.ptr,
                den.data.ptr,
                out.data.ptr,
                cp.uint32(den.size),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out


class VectorResonator:
    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_vector_resonator.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_vector_resonator")

    def blend(
        self,
        vector_a: cp.ndarray,
        vector_b: cp.ndarray,
        alpha: float = 0.5,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        a = cp.asarray(vector_a, dtype=cp.float32).ravel()
        b = cp.asarray(vector_b, dtype=cp.float32).ravel()
        if a.size != b.size:
            raise ValueError("Vectors must be the same size")
        out = cp.empty_like(a)
        self.kernel(
            (1,),
            (128,),
            (
                a.data.ptr,
                b.data.ptr,
                out.data.ptr,
                cp.uint32(a.size),
                cp.float32(alpha),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()
        return out.reshape(vector_a.shape)


__all__ = [
    "GalaxyResonanceEngine",
    "ResonanceField",
    "VectorResonator",
]
