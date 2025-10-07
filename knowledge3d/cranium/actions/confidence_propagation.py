from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:  # Torch is optional at import time; we fall back to CPU when absent.
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

from knowledge3d.gpu.perf_counters import gpu_utilisation
from knowledge3d.gpu.ptx_utils import launch_ptx_kernel


class ConfidencePropagator:
    """
    GPU-first confidence propagator with adaptive curiosity bias.

    The PTX kernel performs the heavy lifting; we fall back to a NumPy
    implementation when CUDA is unavailable so unit tests can exercise the
    logic in CPU-only environments.
    """

    def __init__(
        self,
        curiosity_bias_factor: float = 0.1,
        safety_margin: float = 0.1,
        *,
        alpha_floor: float = 0.05,
        alpha_ceiling: float = 0.2,
    ) -> None:
        self.curiosity_bias_factor = float(curiosity_bias_factor)
        self.alpha_floor = float(alpha_floor)
        self.alpha_ceiling = float(alpha_ceiling)
        self.safety_margin = float(safety_margin)
        self._cubin_path = str(
            Path(__file__).resolve().parent.parent / "ptx" / "confidence_propagation.ptx"
        )

    # ------------------------------------------------------------------
    def _gpu_ready(self) -> bool:
        return bool(_TORCH_AVAILABLE and torch.cuda.is_available() and _CUPY_AVAILABLE)

    def _compute_alpha(self, system_load: Optional[float]) -> float:
        load = system_load
        if load is None or not np.isfinite(load):
            load = gpu_utilisation(default=0.5)
        load = float(max(0.0, min(1.0, load)))
        alpha = self.curiosity_bias_factor * (1.0 - load) + self.alpha_floor * load
        return float(max(self.alpha_floor, min(self.alpha_ceiling, alpha)))

    # ------------------------------------------------------------------
    def propagate_confidence(
        self,
        base_confidences: Sequence[float],
        curiosity_scores: Sequence[float],
        input_confidence: float,
        system_load: Optional[float] = None,
    ) -> np.ndarray:
        """
        Propagate confidence scores on the GPU when possible.

        Args:
            base_confidences: Iterable of base confidences
            curiosity_scores: Iterable of curiosity signals
            input_confidence: Upper bound passed from the reasoning stack
            system_load: Optional external load hint (0-1)
        """

        alpha = self._compute_alpha(system_load)

        if not self._gpu_ready():
            base = np.asarray(base_confidences, dtype=np.float32)
            curiosity = np.asarray(curiosity_scores, dtype=np.float32)
            if base.shape != curiosity.shape:
                raise ValueError("curiosity_scores must match base_confidences")
            biased = base * (1.0 + alpha * curiosity)
            max_allowed = input_confidence * (1.0 - self.safety_margin)
            return np.clip(biased, 0.0, max_allowed)

        assert torch is not None  # for type-checkers
        base_t = torch.as_tensor(base_confidences, dtype=torch.float32, device="cuda")
        cur_t = torch.as_tensor(curiosity_scores, dtype=torch.float32, device="cuda")
        if base_t.numel() != cur_t.numel():
            raise ValueError("curiosity_scores must match base_confidences")

        out_t = torch.empty_like(base_t)
        mask_t = torch.empty(base_t.numel(), dtype=torch.uint8, device="cuda")

        launch_ptx_kernel(
            self._cubin_path,
            "propagate_confidence_kernel",
            base_t.data_ptr(),
            cur_t.data_ptr(),
            out_t.data_ptr(),
            mask_t.data_ptr(),
            np.float32(input_confidence),
            np.float32(alpha),
            np.float32(self.safety_margin),
            np.uint32(base_t.numel()),
            grid=(1, 1, 1),
            block=(32, 1, 1),
        )
        torch.cuda.synchronize()

        valid = mask_t.bool()
        if not torch.any(valid):
            return np.empty(0, dtype=np.float32)
        return out_t[valid].detach().cpu().numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def verify_confidence_invariant(
        biased_confidences: Sequence[float],
        input_confidence: float,
    ) -> bool:
        arr = np.asarray(biased_confidences, dtype=np.float32)
        return bool(np.all(arr <= input_confidence + 1e-6))
