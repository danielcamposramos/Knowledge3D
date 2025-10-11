"""Cognitive executive bridge coordinating multiple GPU kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cupy as cp

from .router import GeometryRouter
from .resonance import GalaxyResonanceEngine
from .fractal import FractalEmitter
from .dual_client_sync import DualClientSync
from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class CognitiveExecutive:
    """Minimal orchestrator joining the sensory, cognitive, and motor stages."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_cognitive_executive.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_cognitive_executive")

        self.router = GeometryRouter()
        self.resonance = GalaxyResonanceEngine()
        self.fractal = FractalEmitter()
        self.sync = DualClientSync()

    def process(
        self,
        sensory_tensor: cp.ndarray,
        media_shape_id: int,
        weights: cp.ndarray,
        bias: float = 0.0,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> cp.ndarray:
        """Execute the three-stage pipeline and return motor outputs."""

        # Sensory routing
        routed = self.router.route(sensory_tensor, media_shape_id, stream=stream)

        # Cognitive resonance (dummy latent identical to routed for now)
        latent = cp.zeros_like(routed)
        refined = self.resonance.run(routed[cp.newaxis, :], latent[cp.newaxis, :], stream=stream)[0]

        # Motor emission and synchronization baseline
        coords = self.fractal.emit(refined, stream=stream)
        self.sync.stage(coords, coords)
        synced_human, _ = self.sync.sync()

        # Cognitive executive kernel for final scoring
        input_cp = cp.asarray(refined, dtype=cp.float32).ravel()
        weights_cp = cp.asarray(weights, dtype=cp.float32).ravel()
        if input_cp.size != weights_cp.size:
            raise ValueError("Input and weight vectors must align")
        out = cp.empty_like(input_cp)

        self.kernel(
            (1,),
            (input_cp.size if input_cp.size < 256 else 256,),
            (
                input_cp.data.ptr,
                weights_cp.data.ptr,
                out.data.ptr,
                cp.uint32(input_cp.size),
                cp.float32(bias),
            ),
            stream=stream,
        )
        if stream is not None:
            stream.synchronize()

        return out.reshape(refined.shape), synced_human


__all__ = ["CognitiveExecutive"]
