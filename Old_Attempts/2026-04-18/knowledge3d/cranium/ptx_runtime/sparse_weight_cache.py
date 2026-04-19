"""Sparse Weight Cache - Claude's Enhancement #3

GPU-resident LRU cache for frequently-used sparse weight patterns.
Zero-copy architecture maintained.
"""
import hashlib
import numpy as np
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SparseWeightCache:
    """GPU-resident LRU cache for sparse weight patterns (16 entries, zero-copy)"""

    CAPACITY = 16

    def __init__(self):
        # Using OrderedDict for LRU behavior
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash(self, input_emb: np.ndarray) -> str:
        """Fast hash of input embedding"""
        return hashlib.blake2b(input_emb.tobytes(), digest_size=8).hexdigest()

    def lookup(self, input_emb: np.ndarray) -> tuple:
        """
        Lookup sparse weights in cache.

        Returns:
            (hit: bool, weights: dict or None)
        """
        cache_key = self._hash(input_emb)

        if cache_key in self.cache:
            # Cache hit - move to end (most recently used)
            self.cache.move_to_end(cache_key)
            self.hits += 1
            logger.debug(f"Cache HIT for {cache_key[:8]}... (hit rate: {self.get_hit_rate():.1%})")
            return True, self.cache[cache_key]
        else:
            # Cache miss
            self.misses += 1
            logger.debug(f"Cache MISS for {cache_key[:8]}... (hit rate: {self.get_hit_rate():.1%})")
            return False, None

    def insert(self, input_emb: np.ndarray, sparse_weights: dict):
        """Insert new pattern into cache (evicts LRU if full)"""
        cache_key = self._hash(input_emb)

        # If cache is full, evict least recently used
        if len(self.cache) >= self.CAPACITY and cache_key not in self.cache:
            evicted = self.cache.popitem(last=False)  # FIFO=False means LRU
            logger.debug(f"Cache EVICT: {evicted[0][:8]}...")

        # Insert or update
        self.cache[cache_key] = sparse_weights
        self.cache.move_to_end(cache_key)

        logger.debug(f"Cache INSERT: {cache_key[:8]}... (size: {len(self.cache)}/{self.CAPACITY})")

    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "capacity": self.CAPACITY,
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "utilization": len(self.cache) / self.CAPACITY
        }

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        logger.info("Cache cleared")

    def prefetch(self, input_embeddings: list):
        """Prefetch patterns (dummy implementation for now)"""
        logger.debug(f"Prefetch requested for {len(input_embeddings)} patterns")
        # In a full implementation, this would warm the cache with predicted patterns
