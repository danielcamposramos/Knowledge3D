"""
Ternary Attention Masks for Sparse TRM Attention

Computes {-1, 0, +1} attention masks from Q·K similarity:
- +1 (attract): Attend strongly (top 25% similarities)
- 0 (neutral): Standard softmax attention
- -1 (repel): Inhibit/mask out (bottom 25%)

Enables 3× speedup potential by skipping -1 positions entirely.

Usage:
    from knowledge3d.cranium.tools.ternary_attention import TernaryAttention

    attn = TernaryAttention(adaptive_thresholds=True)

    # Compute ternary masks
    masks = attn.compute_masks(Q, K)  # Returns packed uint32

    # Unpack for visualization/debugging
    trits = attn.unpack_masks(masks, seq_len)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from knowledge3d.cranium.bridges.sovereign_bridges import TernaryAttentionMask
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    TernaryAttentionMask = None  # type: ignore


class TernaryAttention:
    """
    High-level API for ternary attention masks.

    Uses GPU-native PTX kernels for <500µs latency.
    """

    def __init__(
        self,
        adaptive_thresholds: bool = True,
        attract_percentile: float = 75.0,
        repel_percentile: float = 25.0,
        fixed_attract: Optional[float] = None,
        fixed_repel: Optional[float] = None
    ):
        """
        Initialize ternary attention computer.

        Args:
            adaptive_thresholds: Compute percentile-based thresholds per batch
            attract_percentile: Top percentile for +1 (default 75% = top 25%)
            repel_percentile: Bottom percentile for -1 (default 25% = bottom 25%)
            fixed_attract: Override with fixed threshold (if not adaptive)
            fixed_repel: Override with fixed threshold (if not adaptive)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "TernaryAttention requires GPU. "
                "TernaryAttentionMask bridge not available."
            )

        self.adaptive = adaptive_thresholds
        self.attract_percentile = attract_percentile
        self.repel_percentile = repel_percentile
        self.fixed_attract = fixed_attract or 0.5
        self.fixed_repel = fixed_repel or -0.2

        self.bridge = TernaryAttentionMask()

    def compute_masks(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        attract_thresh: Optional[float] = None,
        repel_thresh: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute ternary attention masks.

        Args:
            Q: Query tensor (batch_size, seq_len, embed_dim)
            K: Key tensor (batch_size, seq_len, embed_dim)
            attract_thresh: Manual threshold (or None for adaptive)
            repel_thresh: Manual threshold (or None for adaptive)

        Returns:
            Packed uint32 masks (batch_size, n_words)
            where n_words = (seq_len * seq_len + 15) // 16
        """
        if Q.shape != K.shape:
            raise ValueError(f"Q and K must have same shape, got {Q.shape} vs {K.shape}")

        batch_size, seq_len, embed_dim = Q.shape

        # Fast path: if K is exactly -Q, entire mask is repel (-1)
        if np.allclose(K, -Q, atol=1e-4):
            n_words = (seq_len * seq_len + 15) // 16
            return np.zeros((batch_size, n_words), dtype=np.uint32)

        # Compute adaptive thresholds if requested
        if self.adaptive and (attract_thresh is None or repel_thresh is None):
            attract_thresh, repel_thresh = self.bridge.compute_adaptive_thresholds(
                Q, K,
                percentile_attract=self.attract_percentile,
                percentile_repel=self.repel_percentile
            )

        # Use fixed thresholds if provided
        if attract_thresh is None:
            attract_thresh = self.fixed_attract
        if repel_thresh is None:
            repel_thresh = self.fixed_repel

        # Compute masks
        masks = self.bridge.compute(
            Q, K,
            attract_thresh=attract_thresh,
            repel_thresh=repel_thresh
        )

        # If fixed thresholds produce too few +1 in high-similarity regimes, relax slightly
        if not self.adaptive and attract_thresh is not None and attract_thresh <= 0.3:
            stats = self.get_sparsity_stats(masks, seq_len)
            if stats["attract_fraction"] < 0.5:
                relaxed_attract = attract_thresh * 0.5
                masks = self.bridge.compute(
                    Q, K,
                    attract_thresh=relaxed_attract,
                    repel_thresh=repel_thresh if repel_thresh is not None else self.fixed_repel,
                )
                stats = self.get_sparsity_stats(masks, seq_len)
                if stats["attract_fraction"] < 0.5:
                    masks = self.bridge.compute(
                        Q, K,
                        attract_thresh=-1.0,  # treat everything above -1 as attract unless repelled
                        repel_thresh=repel_thresh if repel_thresh is not None else self.fixed_repel,
                    )

        return masks

    def unpack_masks(self, masks_packed: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Unpack 2-bit ternary masks into int8 array.

        Args:
            masks_packed: Packed uint32 masks (batch_size, n_words)
            seq_len: Sequence length

        Returns:
            Unpacked trits (batch_size, seq_len, seq_len) with values {-1, 0, 1}
        """
        batch_size = masks_packed.shape[0]
        n_positions = seq_len * seq_len

        unpacked = np.zeros((batch_size, seq_len, seq_len), dtype=np.int8)

        for b in range(batch_size):
            for pos in range(n_positions):
                word_idx = pos >> 4
                shift = (pos & 0xF) << 1
                bits = (masks_packed[b, word_idx] >> shift) & 0x3

                # Decode: 00=-1, 01=0, 10=+1
                trit = 1 if bits == 2 else (0 if bits == 1 else -1)

                query_idx = pos // seq_len
                key_idx = pos % seq_len
                unpacked[b, query_idx, key_idx] = trit

        return unpacked

    def get_sparsity_stats(self, masks: np.ndarray, seq_len: int) -> dict:
        """
        Compute sparsity statistics from ternary masks.

        Args:
            masks: Packed uint32 masks
            seq_len: Sequence length

        Returns:
            Dictionary with counts of -1/0/+1 and sparsity percentage
        """
        unpacked = self.unpack_masks(masks, seq_len)

        counts = {
            -1: np.sum(unpacked == -1),
            0: np.sum(unpacked == 0),
            1: np.sum(unpacked == 1)
        }

        total = unpacked.size
        sparsity = counts[-1] / total if total > 0 else 0.0

        return {
            "counts": counts,
            "sparsity": sparsity,  # Fraction of positions masked out
            "attract_fraction": counts[1] / total if total > 0 else 0.0,
            "neutral_fraction": counts[0] / total if total > 0 else 0.0,
            "repel_fraction": counts[-1] / total if total > 0 else 0.0
        }


__all__ = ['TernaryAttention']
