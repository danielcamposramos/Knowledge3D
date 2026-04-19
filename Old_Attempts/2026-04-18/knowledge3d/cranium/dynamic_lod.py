"""
Dynamic LOD Saliency Tuning
Implements Grok's semantic-aware resolution adjustment (Step6.txt)
Author: Claude (swarm continuation)

Adjusts Morton LOD levels based on cosine similarity between query and node embeddings.
High similarity → lower LOD (higher resolution)
Low similarity → higher LOD (lower resolution)
"""

from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np


class DynamicLOD:
    """
    GPU-native dynamic LOD adjustment based on semantic saliency.

    Uses warp-cooperative cosine similarity to tune Morton LOD levels
    for each node in the unified buffer.
    """

    def __init__(self, saliency_thresh: float = 0.7):
        """
        Initialize dynamic LOD tuner.

        Args:
            saliency_thresh: Cosine similarity threshold for saliency (default 0.7)
        """
        self.saliency_thresh = saliency_thresh
        self._kernel = None
        self._load_kernel()

    def _load_kernel(self) -> None:
        """Load PTX kernel for dynamic LOD tuning."""
        ptx_path = Path(__file__).parent / "ptx" / "dynamic_lod_tune.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"PTX kernel not found: {ptx_path}")

        module = cp.RawModule(path=str(ptx_path))
        self._kernel = module.get_function("dynamic_lod_tune")

    def tune(
        self,
        unified_buffer: cp.ndarray,
        query_emb: cp.ndarray,
        n_nodes: int,
        saliency_out: Optional[cp.ndarray] = None
    ) -> cp.ndarray:
        """
        Tune LOD levels in-place based on query similarity.

        Args:
            unified_buffer: GPU buffer of UnifiedNode structs (N × 4096 bytes)
            query_emb: Query embedding (512 floats)
            n_nodes: Number of nodes
            saliency_out: Optional output buffer for (cosine, lod) pairs (N × 2 floats)

        Returns:
            Saliency map (N × 2) with [cosine_similarity, tuned_lod] per node
        """
        if saliency_out is None:
            saliency_out = cp.zeros((n_nodes, 2), dtype=cp.float32)

        # Launch kernel: one thread per node
        block_size = 128
        grid_size = (n_nodes + block_size - 1) // block_size

        self._kernel(
            (grid_size,),
            (block_size,),
            (
                unified_buffer,
                query_emb,
                np.int32(n_nodes),
                np.float32(self.saliency_thresh),
                saliency_out
            )
        )

        cp.cuda.runtime.deviceSynchronize()
        return saliency_out

    def tune_and_export(
        self,
        unified_buffer: cp.ndarray,
        query_emb: cp.ndarray,
        n_nodes: int
    ) -> tuple[cp.ndarray, dict]:
        """
        Tune LOD and return saliency map + summary stats.

        Args:
            unified_buffer: GPU buffer of UnifiedNode structs
            query_emb: Query embedding (512 floats)
            n_nodes: Number of nodes

        Returns:
            (saliency_map, stats_dict) where stats contains mean/max cosine, LOD range
        """
        saliency_map = self.tune(unified_buffer, query_emb, n_nodes)

        # Compute summary statistics
        cosine_vals = saliency_map[:, 0]
        lod_vals = saliency_map[:, 1]

        stats = {
            "mean_cosine": float(cp.mean(cosine_vals).get()),
            "max_cosine": float(cp.max(cosine_vals).get()),
            "min_cosine": float(cp.min(cosine_vals).get()),
            "mean_lod": float(cp.mean(lod_vals).get()),
            "lod_range": (int(cp.min(lod_vals).get()), int(cp.max(lod_vals).get())),
            "salient_nodes": int(cp.sum(cosine_vals >= self.saliency_thresh).get())
        }

        return saliency_map, stats


def fsm_state_2_5_saliency_tune(fsm_context: dict) -> dict:
    """
    Integration hook for FSM State 2.5 (between Spatial and Reason).

    Tunes LOD levels based on query saliency, updating the unified buffer in-place.

    Args:
        fsm_context: FSM context dict with keys:
            - unified_buffer: GPU node buffer
            - query_emb: Query embedding
            - n_nodes: Node count

    Returns:
        Updated FSM context with saliency_map added
    """
    lod_tuner = DynamicLOD(saliency_thresh=0.7)

    saliency_map, stats = lod_tuner.tune_and_export(
        unified_buffer=fsm_context["unified_buffer"],
        query_emb=fsm_context["query_emb"],
        n_nodes=fsm_context["n_nodes"]
    )

    # Add to context for downstream states
    fsm_context["saliency_map"] = saliency_map
    fsm_context["saliency_stats"] = stats

    return fsm_context
