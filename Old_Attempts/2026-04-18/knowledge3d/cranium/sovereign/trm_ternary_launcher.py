"""
TRM Ternary Launcher

Extends TRMLauncher with ternary attention masks:
- +1 (attract): amplify outputs (×2)
-  0 (neutral): leave unchanged
- -1 (repel):  dampen outputs (×0.1)

Masks are computed GPU-side via TernaryAttention (packed 2-bit trits).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.tools.ternary_attention import TernaryAttention


class TRMTernaryLauncher(TRMLauncher):
    """TRM launcher that applies ternary attention masks to modulation."""

    def __init__(
        self,
        ptx_path: Optional[str] = None,
        use_rpn: Optional[bool] = None,
        use_fused: Optional[bool] = None,
    ):
        super().__init__(ptx_path=ptx_path, use_rpn=use_rpn, use_fused=use_fused)
        self.attn = TernaryAttention(adaptive_thresholds=True)

    @staticmethod
    def _mask_factor(trit: int) -> float:
        if trit > 0:
            return 2.0
        if trit < 0:
            return 0.1
        return 1.0

    def _compute_mask_trit(self, q: np.ndarray) -> int:
        # For single-token self-attention, force a contrastive key to reveal sign
        Q = q.reshape(1, 1, -1)
        key = q if np.sum(q) >= 0 else -q
        K = key.reshape(1, 1, -1)
        masks = self.attn.compute_masks(Q, K, attract_thresh=0.1, repel_thresh=-0.1)
        trits = self.attn.unpack_masks(masks, seq_len=1)
        return int(trits[0, 0, 0])

    def refine(
        self,
        q: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        W1: np.ndarray,
        W2: np.ndarray,
        W3: np.ndarray,
        W4: np.ndarray,
        n_steps: int = 6,
        eps: float = 1e-4,
        ternary_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        trit = self._compute_mask_trit(q) if ternary_mask is None else int(
            self.attn.unpack_masks(ternary_mask, seq_len=1)[0, 0, 0]
        )
        factor = self._mask_factor(trit)
        if trit < 0:
            # Skip heavy refine when repel — early exit (Round 6 prep)
            return np.zeros_like(y, dtype=np.float32), np.zeros_like(z, dtype=np.float32)
        y_base, z_base = super().refine(q, y, z, W1, W2, W3, W4, n_steps=n_steps, eps=eps)
        return y_base * factor, z_base * factor

    def refine_batch(
        self,
        Q: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        W1: np.ndarray,
        W2: np.ndarray,
        W3: np.ndarray,
        W4: np.ndarray,
        n_steps: int = 6,
        eps: float = 1e-4,
        ternary_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = Q.shape[0]
        outputs_y = np.zeros_like(Y, dtype=np.float32)
        outputs_z = np.zeros_like(Z, dtype=np.float32)
        # Compute masks for batch
        if ternary_masks is None:
            Q_reshaped = Q.reshape(batch_size, 1, -1)
            K_reshaped = np.stack([
                (q if np.sum(q) >= 0 else -q) for q in Q
            ], axis=0).reshape(batch_size, 1, -1)
            masks = self.attn.compute_masks(Q_reshaped, K_reshaped, attract_thresh=0.1, repel_thresh=-0.1)
        else:
            masks = ternary_masks
        trits = self.attn.unpack_masks(masks, seq_len=1).reshape(batch_size)
        # Fast path: zero-out repels without launching TRM kernels
        keep_indices = []
        for i in range(batch_size):
            if int(trits[i]) < 0:
                outputs_y[i] = 0.0
                outputs_z[i] = 0.0
            else:
                keep_indices.append(i)

        # Launch TRM only for non-repel entries
        for i in keep_indices:
            factor = self._mask_factor(int(trits[i]))
            y_base, z_base = super().refine(
                Q[i], Y[i], Z[i], W1, W2, W3, W4, n_steps=n_steps, eps=eps
            )
            outputs_y[i] = y_base * factor
            outputs_z[i] = z_base * factor
        return outputs_y, outputs_z


__all__ = ["TRMTernaryLauncher"]
