"""
Comprehensive test suite for Step 11 ShapeCache.
Tests semantic-aware caching, intelligent eviction, and predictive prefetching.
"""
import pytest
import numpy as np
import time

from knowledge3d.cranium.ptx_runtime.shape_cache import ShapeCache


class TestShapeCacheBasic:
    """Test basic cache functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache(capacity=5, max_memory_mb=10)

    def test_initialization(self):
        """Test cache initializes with correct parameters."""
        assert self.cache.capacity == 5
        assert self.cache.max_memory_mb == 10
        assert self.cache.hits == 0
        assert self.cache.misses == 0
        assert len(self.cache.cache) == 0

    def test_cache_empty_lookup_misses(self):
        """Test lookup on empty cache results in miss."""
        hit, data = self.cache.lookup('cube', 1.0, (1, 0, 0))

        assert not hit
        assert data is None
        assert self.cache.misses == 1


class TestCacheHitMiss:
    """Test cache hit and miss tracking."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache(capacity=3)
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    def test_cache_miss_then_hit(self):
        """Test miss followed by hit on same parameters."""
        # First lookup - miss
        hit, data = self.cache.lookup('cube', 1.0, (1, 0, 0))
        assert not hit
        assert self.cache.misses == 1

        # Insert
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)

        # Second lookup - hit
        hit, data = self.cache.lookup('cube', 1.0, (1, 0, 0))
        assert hit
        assert data is not None
        assert self.cache.hits == 1

    def test_cache_different_params_miss(self):
        """Test different parameters result in cache miss."""
        # Insert with size 1.0
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)

        # Lookup with size 2.0 - should miss
        hit, data = self.cache.lookup('cube', 2.0, (1, 0, 0))
        assert not hit
        assert self.cache.misses == 1

    def test_cache_hit_rate_calculation(self):
        """Test hit rate calculation."""
        # 2 hits, 3 misses = 40% hit rate
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)

        self.cache.lookup('cube', 1.0, (1, 0, 0))  # Hit
        self.cache.lookup('cube', 1.0, (1, 0, 0))  # Hit
        self.cache.lookup('sphere', 1.0, (1, 0, 0))  # Miss
        self.cache.lookup('cylinder', 1.0, (1, 0, 0))  # Miss
        self.cache.lookup('cone', 1.0, (1, 0, 0))  # Miss

        hit_rate = self.cache.get_hit_rate()
        assert abs(hit_rate - 0.4) < 0.01  # 2/5 = 0.4


class TestIntelligentEviction:
    """Test intelligent cache eviction strategy."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache(capacity=2)  # Small capacity for testing
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2]], dtype=np.uint32)

    def test_eviction_on_capacity_exceeded(self):
        """Test that eviction occurs when capacity is exceeded."""
        # Fill cache to capacity
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)
        self.cache.insert('sphere', 1.0, (0, 1, 0), self.vertices, self.indices)

        assert len(self.cache.cache) == 2
        assert self.cache.evictions == 0

        # Insert third item - should trigger eviction
        self.cache.insert('cylinder', 1.0, (0, 0, 1), self.vertices, self.indices)

        assert len(self.cache.cache) == 2
        assert self.cache.evictions == 1

    def test_lru_eviction_order(self):
        """Test LRU eviction strategy."""
        # Insert two items
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)
        self.cache.insert('sphere', 1.0, (0, 1, 0), self.vertices, self.indices)

        # Access cube to make it more recent
        self.cache.lookup('cube', 1.0, (1, 0, 0))

        # Insert third item - sphere should be evicted (least recently used)
        self.cache.insert('cylinder', 1.0, (0, 0, 1), self.vertices, self.indices)

        # Cube should still be in cache
        hit, _ = self.cache.lookup('cube', 1.0, (1, 0, 0))
        assert hit

    def test_eviction_score_factors(self):
        """Test that eviction considers multiple factors."""
        # Insert item and access it multiple times (high frequency)
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)
        for _ in range(5):
            self.cache.lookup('cube', 1.0, (1, 0, 0))

        # Insert second item with single access
        self.cache.insert('sphere', 1.0, (0, 1, 0), self.vertices, self.indices)

        # Insert third item - sphere should be evicted (lower frequency)
        self.cache.insert('cylinder', 1.0, (0, 0, 1), self.vertices, self.indices)

        # Cube should still be there
        hit, _ = self.cache.lookup('cube', 1.0, (1, 0, 0))
        assert hit


class TestSemanticClustering:
    """Test semantic clustering for intelligent caching."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache()
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2]], dtype=np.uint32)

    def test_semantic_cluster_creation(self):
        """Test semantic clusters are created."""
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices,
                         modal_type='text')

        assert 'cube_text' in self.cache.semantic_clusters
        assert len(self.cache.semantic_clusters['cube_text']) > 0

    def test_semantic_cluster_usage_tracking(self):
        """Test semantic cluster usage is tracked."""
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices,
                         modal_type='text')

        assert 'cube_text' in self.cache.cluster_usage
        assert self.cache.cluster_usage['cube_text'] == 1

        # Insert another cube_text
        self.cache.insert('cube', 2.0, (0, 1, 0), self.vertices, self.indices,
                         modal_type='text')

        assert self.cache.cluster_usage['cube_text'] == 2

    def test_different_modal_types_different_clusters(self):
        """Test different modal types create different clusters."""
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices,
                         modal_type='text')
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices,
                         modal_type='image')

        assert 'cube_text' in self.cache.semantic_clusters
        assert 'cube_image' in self.cache.semantic_clusters


class TestPredictivePrefetching:
    """Test predictive prefetching based on access patterns."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache()
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2]], dtype=np.uint32)

    def test_access_pattern_tracking(self):
        """Test access patterns are tracked."""
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)
        self.cache.insert('sphere', 1.0, (0, 1, 0), self.vertices, self.indices)

        # Access in pattern
        self.cache.lookup('cube', 1.0, (1, 0, 0))
        self.cache.lookup('sphere', 1.0, (0, 1, 0))

        assert len(self.cache.access_history) == 2

    def test_access_history_limited_size(self):
        """Test access history is limited to prevent memory growth."""
        # Fill beyond limit (100)
        for i in range(150):
            self.cache.insert(f'shape_{i}', 1.0, (1, 0, 0), self.vertices, self.indices)
            self.cache.lookup(f'shape_{i}', 1.0, (1, 0, 0))

        # Should be limited to 100
        assert len(self.cache.access_history) == 100


class TestMemoryManagement:
    """Test memory-aware cache management."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache(capacity=100, max_memory_mb=1)  # 1MB limit

    def test_memory_usage_calculation(self):
        """Test memory usage is calculated correctly."""
        vertices = np.random.rand(1000, 3).astype(np.float32)
        indices = np.random.randint(0, 1000, (500, 3), dtype=np.uint32)

        self.cache.insert('large_shape', 1.0, (1, 0, 0), vertices, indices)

        # Memory usage should be > 0
        assert self.cache.memory_usage_mb > 0

    def test_eviction_on_memory_limit(self):
        """Test eviction occurs when memory limit is reached."""
        # Create large vertices to trigger memory limit
        large_vertices = np.random.rand(10000, 3).astype(np.float32)
        large_indices = np.random.randint(0, 10000, (5000, 3), dtype=np.uint32)

        # Insert multiple large shapes
        for i in range(5):
            self.cache.insert(f'shape_{i}', 1.0, (1, 0, 0), large_vertices, large_indices)

        # Should have triggered evictions to stay under memory limit
        assert self.cache.evictions > 0


class TestCacheOptimization:
    """Test cache optimization functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache()
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2]], dtype=np.uint32)

    def test_optimize_cache(self):
        """Test cache optimization runs without errors."""
        # Fill cache with some data
        for i in range(25):
            self.cache.insert(f'shape_{i}', 1.0, (1, 0, 0), self.vertices, self.indices)
            self.cache.lookup(f'shape_{i}', 1.0, (1, 0, 0))

        # Run optimization
        self.cache.optimize_cache()

        # Should complete without errors
        assert True

    def test_cache_report(self):
        """Test cache report generation."""
        # Insert some data
        self.cache.insert('cube', 1.0, (1, 0, 0), self.vertices, self.indices)
        self.cache.lookup('cube', 1.0, (1, 0, 0))

        report = self.cache.get_cache_report()

        assert 'capacity' in report
        assert 'current_size' in report
        assert 'hit_rate' in report
        assert 'memory_usage_mb' in report
        assert 'semantic_clusters' in report


class TestCacheClear:
    """Test cache clearing functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache()
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2]], dtype=np.uint32)

    def test_clear_resets_all_state(self):
        """Test clear() resets all cache state."""
        # Fill cache
        for i in range(5):
            self.cache.insert(f'shape_{i}', 1.0, (1, 0, 0), self.vertices, self.indices)
            self.cache.lookup(f'shape_{i}', 1.0, (1, 0, 0))

        assert len(self.cache.cache) > 0
        assert self.cache.hits > 0

        # Clear
        self.cache.clear()

        # Everything should be reset
        assert len(self.cache.cache) == 0
        assert self.cache.hits == 0
        assert self.cache.misses == 0
        assert self.cache.evictions == 0
        assert self.cache.memory_usage_mb == 0.0
        assert len(self.cache.semantic_clusters) == 0


class TestPerformance:
    """Test cache performance characteristics."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = ShapeCache(capacity=100)
        self.vertices = np.random.rand(8, 3).astype(np.float32)
        self.indices = np.array([[0, 1, 2]], dtype=np.uint32)

    def test_lookup_performance(self):
        """Test cache lookup is fast (<1µs target)."""
        # Fill cache
        for i in range(50):
            self.cache.insert(f'shape_{i}', 1.0, (1, 0, 0), self.vertices, self.indices)

        # Time lookups
        start = time.perf_counter()
        for i in range(1000):
            self.cache.lookup('shape_25', 1.0, (1, 0, 0))
        elapsed_us = (time.perf_counter() - start) * 1e6

        # Average should be < 1µs per lookup
        avg_us = elapsed_us / 1000
        assert avg_us < 1.0, f"Lookup took {avg_us:.2f}µs, expected <1.0µs"

    def test_insert_performance(self):
        """Test cache insert is reasonably fast."""
        start = time.perf_counter()
        for i in range(100):
            self.cache.insert(f'shape_{i}', 1.0, (1, 0, 0), self.vertices, self.indices)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete 100 inserts in <100ms
        assert elapsed_ms < 100, f"100 inserts took {elapsed_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
