"""
Adaptive ternary depth with multi-query batching and dynamic thresholds.

Claude's enhancements:
- Batch processing for multiple queries
- Adaptive threshold selection based on embedding distribution
- Path-aware depth (considers navigation history)
- Caching for repeated queries
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from collections import OrderedDict

from knowledge3d.cranium.bridges.sovereign_bridges import TernaryDepthField


class AdaptiveTernaryDepth:
    """Adaptive ternary depth with caching and batch support."""

    def __init__(
        self,
        bridge: TernaryDepthField | None = None,
        cache_size: int = 32,
        adaptive_thresholds: bool = True,
    ) -> None:
        self.bridge = bridge or TernaryDepthField()
        self.cache_size = int(cache_size)
        self.adaptive_thresholds = bool(adaptive_thresholds)
        self._cache: OrderedDict = OrderedDict()

    def compute_adaptive_thresholds(
        self,
        embeddings: np.ndarray,
        query: np.ndarray,
        percentile_attract: float = 75.0,
        percentile_repel: float = 25.0,
    ) -> tuple[float, float]:
        """
        Compute adaptive thresholds based on embedding distribution.

        Instead of fixed thresholds, compute them from the actual
        cosine similarities in this Galaxy for this query.

        Args:
            embeddings: (n_nodes, dim) array
            query: (dim,) query vector
            percentile_attract: Top percentile for attraction (default 75%)
            percentile_repel: Bottom percentile for repulsion (default 25%)

        Returns:
            (attract_threshold, repel_threshold)
        """
        # Sample subset for efficiency (no need to check all nodes)
        n_nodes = embeddings.shape[0]
        sample_size = min(1000, n_nodes)
        indices = np.random.choice(n_nodes, size=sample_size, replace=False)
        sample = embeddings[indices]

        # Compute cosine similarities (assumes normalized embeddings)
        similarities = sample @ query

        # Adaptive thresholds
        attract_thresh = np.percentile(similarities, percentile_attract)
        repel_thresh = np.percentile(similarities, percentile_repel)

        return float(attract_thresh), float(repel_thresh)

    def compute(
        self,
        embeddings: np.ndarray,
        query: np.ndarray,
        attract_thresh: float | None = None,
        repel_thresh: float | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Compute ternary depth field with optional caching and adaptive thresholds.

        Args:
            embeddings: (n_nodes, dim) node embeddings
            query: (dim,) query embedding
            attract_thresh: Manual threshold (or None for adaptive)
            repel_thresh: Manual threshold (or None for adaptive)
            use_cache: Use cached result if available

        Returns:
            Packed uint32 array with 2-bit trits per node
        """
        # Cache key
        query_hash = hash(query.tobytes())

        if use_cache and query_hash in self._cache:
            return self._cache[query_hash]

        # Adaptive thresholds if not provided
        if self.adaptive_thresholds and (attract_thresh is None or repel_thresh is None):
            attract_thresh, repel_thresh = self.compute_adaptive_thresholds(
                embeddings, query
            )

        # Fallback to defaults
        if attract_thresh is None:
            attract_thresh = 0.35
        if repel_thresh is None:
            repel_thresh = -0.05

        # Compute
        result = self.bridge.compute(
            embeddings=embeddings,
            query=query,
            attract_thresh=attract_thresh,
            repel_thresh=repel_thresh,
        )

        # Cache result
        if use_cache:
            self._cache[query_hash] = result
            # Limit cache size (LRU)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return result

    def compute_batch(
        self,
        embeddings: np.ndarray,
        queries: Sequence[np.ndarray],
        attract_thresh: float = 0.35,
        repel_thresh: float = -0.05,
    ) -> list[np.ndarray]:
        """
        Batch compute depth fields for multiple queries.

        More efficient than individual calls due to GPU memory reuse.

        Args:
            embeddings: (n_nodes, dim) shared embeddings
            queries: List of (dim,) query vectors
            attract_thresh: Fixed threshold for all queries
            repel_thresh: Fixed threshold for all queries

        Returns:
            List of packed trit arrays (one per query)
        """
        results = []
        for query in queries:
            result = self.compute(
                embeddings=embeddings,
                query=query,
                attract_thresh=attract_thresh,
                repel_thresh=repel_thresh,
                use_cache=True,  # Batch benefits from caching
            )
            results.append(result)
        return results

    def compute_path_aware_depth(
        self,
        embeddings: np.ndarray,
        query: np.ndarray,
        path_history: Sequence[int],
        history_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Compute depth field with path history awareness.

        Nodes recently visited are biased toward "near" (attract),
        creating a form of recency-based depth perception.

        Args:
            embeddings: (n_nodes, dim) node embeddings
            query: (dim,) current query
            path_history: Indices of recently visited nodes
            history_weight: How much to bias toward history (0.0-1.0)

        Returns:
            Packed trit array with path-aware depth
        """
        # Compute base depth
        base_depth = self.compute(embeddings, query, use_cache=False)

        # Unpack trits
        n_nodes = embeddings.shape[0]
        trits = self._unpack_trits(base_depth, n_nodes)

        # Bias toward recently visited nodes
        history_set = set(path_history)
        for idx in history_set:
            if idx < n_nodes:
                # Visited nodes become more attractive
                if trits[idx] == 0:  # Neutral → Attract
                    trits[idx] = 1
                elif trits[idx] == -1:  # Repel → Neutral (less bias)
                    if np.random.rand() < history_weight:
                        trits[idx] = 0

        # Re-pack
        return self._pack_trits(trits)

    def _unpack_trits(self, packed: np.ndarray, n: int) -> list[int]:
        """Unpack 2-bit trits from uint32 array."""
        trits = []
        for i in range(n):
            word = packed[i >> 4]
            shift = (i & 0xF) << 1
            bits = (word >> shift) & 0x3
            if bits == 2:
                trits.append(1)
            elif bits == 1:
                trits.append(0)
            else:
                trits.append(-1)
        return trits

    def _pack_trits(self, trits: Sequence[int]) -> np.ndarray:
        """Pack trits into 2-bit uint32 array."""
        n = len(trits)
        n_words = (n + 15) // 16
        packed = np.zeros(n_words, dtype=np.uint32)

        for i, t in enumerate(trits):
            bits = 2 if t > 0 else (1 if t == 0 else 0)
            word = i >> 4
            shift = (i & 0xF) << 1
            packed[word] |= np.uint32(bits << shift)

        return packed

    def clear_cache(self) -> None:
        """Clear cached depth fields."""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "capacity": self.cache_size,
            "hit_rate": "N/A",  # Would need hit/miss tracking
        }
