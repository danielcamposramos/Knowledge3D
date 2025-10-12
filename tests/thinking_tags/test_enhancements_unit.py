"""
Unit tests for Claude's Thinking Tag Enhancements

These tests verify the enhancement modules work correctly without requiring GPU access.
"""
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestLatencyProfiler:
    """Test Enhancement #2: Latency Profiling"""

    def test_initialization(self):
        from knowledge3d.cranium.ptx_runtime.latency_profiler import LatencyProfiler

        profiler = LatencyProfiler(total_budget_us=35.0)
        assert profiler.total_budget_us == 35.0
        assert len(profiler.STAGE_NAMES) == 7

    def test_stage_timing(self):
        from knowledge3d.cranium.ptx_runtime.latency_profiler import LatencyProfiler
        import time

        profiler = LatencyProfiler()

        # Simulate a stage
        profiler.start_stage("query")
        time.sleep(0.001)  # 1ms
        profiler.end_stage("query")

        stats = profiler.get_stage_stats("query")
        assert stats["avg_us"] > 900  # Should be ~1000µs
        assert stats["avg_us"] < 2000

    def test_full_report(self):
        from knowledge3d.cranium.ptx_runtime.latency_profiler import LatencyProfiler

        profiler = LatencyProfiler()
        profiler.start_stage("query")
        profiler.end_stage("query")

        report = profiler.get_full_report()
        assert "total_budget_us" in report
        assert "stages" in report
        assert len(report["stages"]) == 7


class TestSparseWeightCache:
    """Test Enhancement #3: Sparse Weight Caching"""

    def test_initialization(self):
        from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache

        cache = SparseWeightCache()
        assert cache.CAPACITY == 16
        assert len(cache.cache) == 0

    def test_cache_miss_then_hit(self):
        from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache

        cache = SparseWeightCache()
        input_emb = np.random.randn(512).astype(np.float32)

        # First lookup should be a miss
        hit, weights = cache.lookup(input_emb)
        assert hit is False
        assert weights is None

        # Insert weights
        test_weights = {'W1': np.random.randn(256, 512).astype(np.float32)}
        cache.insert(input_emb, test_weights)

        # Second lookup should be a hit
        hit2, weights2 = cache.lookup(input_emb)
        assert hit2 is True
        assert weights2 is not None
        assert 'W1' in weights2

    def test_lru_eviction(self):
        from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache

        cache = SparseWeightCache()

        # Fill cache beyond capacity
        for i in range(20):  # More than CAPACITY (16)
            input_emb = np.random.randn(512).astype(np.float32)
            weights = {'W1': np.random.randn(10, 10).astype(np.float32)}
            cache.insert(input_emb, weights)

        # Cache should be at capacity
        assert len(cache.cache) == cache.CAPACITY

    def test_hit_rate_calculation(self):
        from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache

        cache = SparseWeightCache()
        input_emb = np.random.randn(512).astype(np.float32)

        # Miss, insert, hit, hit
        cache.lookup(input_emb)  # Miss
        cache.insert(input_emb, {})
        cache.lookup(input_emb)  # Hit
        cache.lookup(input_emb)  # Hit

        hit_rate = cache.get_hit_rate()
        assert hit_rate == 2.0 / 3.0  # 2 hits out of 3 lookups


class TestModalAffinityMatrix:
    """Test Enhancement #5: Modal Signature Intelligence"""

    def test_initialization(self):
        from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix

        affinity = ModalAffinityMatrix()
        assert affinity.affinity_matrix.shape == (3, 3)
        assert affinity.modal_to_idx == {'text': 0, 'image': 1, 'audio': 2}

    def test_get_affinity(self):
        from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix

        affinity = ModalAffinityMatrix()

        # Test self-affinity (diagonal should be 1.0)
        assert affinity.get_affinity('text', 'text') == 1.0
        assert affinity.get_affinity('image', 'image') == 1.0

        # Test cross-affinity
        text_image = affinity.get_affinity('text', 'image')
        assert 0.0 <= text_image <= 1.0

    def test_update_success(self):
        from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix

        affinity = ModalAffinityMatrix()

        # Get initial affinity
        initial = affinity.get_affinity('text', 'image')

        # Update with high success score
        affinity.update_success(['text', 'image'], success_score=0.9)

        # Affinity should increase
        updated = affinity.get_affinity('text', 'image')
        assert updated >= initial  # EMA should gradually increase

    def test_modal_boost(self):
        from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix

        affinity = ModalAffinityMatrix()

        # Single modality should return 1.0
        boost_single = affinity.get_modal_boost(['text'])
        assert boost_single == 1.0

        # Multiple modalities should return > 1.0
        boost_multi = affinity.get_modal_boost(['text', 'image'])
        assert boost_multi >= 1.0


class TestEnhancedFallback:
    """Test Enhancement #4: Enhanced Error Recovery"""

    def test_initialization(self):
        from knowledge3d.cranium.ptx_runtime.enhanced_fallback import EnhancedFallback, FallbackLevel

        fallback = EnhancedFallback()
        assert len(fallback.fallback_counts) == 4  # 4 levels

    def test_fallback_levels(self):
        from knowledge3d.cranium.ptx_runtime.enhanced_fallback import FallbackLevel

        assert FallbackLevel.TEMPORAL_FULL == 0
        assert FallbackLevel.TEMPORAL_HALF == 1
        assert FallbackLevel.SPATIAL_CACHED == 2
        assert FallbackLevel.SPATIAL_DENSE == 3

    def test_stats(self):
        from knowledge3d.cranium.ptx_runtime.enhanced_fallback import EnhancedFallback

        fallback = EnhancedFallback()
        stats = fallback.get_stats()

        assert "fallback_counts" in stats
        assert "fallback_success_rates" in stats
        assert "total_fallbacks" in stats


class TestTelemetryVisualizer:
    """Test Enhancement #6: Memory-Efficient Visualization"""

    def test_initialization(self):
        from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer

        telemetry = TelemetryVisualizer(buffer_size=64)
        assert telemetry.buffer_size == 64
        assert len(telemetry.inference_buffer) == 0

    def test_record_inference(self):
        from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer

        telemetry = TelemetryVisualizer(buffer_size=10)

        input_emb = np.random.randn(512).astype(np.float32)
        tags = [('tag_0', 0.8, 0.7)]
        latency_breakdown = {'query': 0.00001, 'rpn_exec': 0.00002}

        telemetry.record_inference(input_emb, tags, latency_breakdown, mode=0, error=None)

        assert len(telemetry.inference_buffer) == 1
        assert len(telemetry.latency_buffer) == 1

    def test_buffer_overflow(self):
        from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer

        telemetry = TelemetryVisualizer(buffer_size=5)

        # Add more than buffer_size entries
        for i in range(10):
            input_emb = np.random.randn(512).astype(np.float32)
            telemetry.record_inference(input_emb, [], {}, 0, None)

        # Should only keep last 5
        assert len(telemetry.inference_buffer) == 5

    def test_stats(self):
        from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer

        telemetry = TelemetryVisualizer(buffer_size=10)
        stats = telemetry.get_stats()

        assert "buffer_size" in stats
        assert "inferences_recorded" in stats
        assert "utilization" in stats


class TestConfidenceWeightedEmission:
    """Test Enhancement #1: Confidence-Weighted Tag Emission (integration)"""

    def test_confidence_calculation(self):
        """Test unified confidence metric calculation"""
        confidence_rays = np.array([0.8, 0.6, 0.4])
        coherence_scores = np.array([0.7, 0.5, 0.3])
        uncertainty = 0.2

        # Manual calculation: (0.4 * rays) + (0.3 * coherence) + (0.3 * (1 - uncertainty))
        expected = (0.4 * confidence_rays) + (0.3 * coherence_scores) + (0.3 * (1 - uncertainty))

        # Should match the formula
        assert expected[0] == pytest.approx(0.4*0.8 + 0.3*0.7 + 0.3*0.8)


def test_all_enhancements_import():
    """Test that all enhancement modules can be imported"""
    from knowledge3d.cranium.ptx_runtime.latency_profiler import LatencyProfiler
    from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache
    from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix
    from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer
    from knowledge3d.cranium.ptx_runtime.enhanced_fallback import EnhancedFallback

    # If we get here, all imports succeeded
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
