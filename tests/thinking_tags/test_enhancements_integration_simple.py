"""
Simple integration test for Thinking Tag enhancements (GPU optional)
Tests that all enhancements can be initialized and used together.
"""
import pytest
import numpy as np
import os

# Test enhancement modules independently first
from knowledge3d.cranium.ptx_runtime.latency_profiler import LatencyProfiler
from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache
from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix
from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer
from knowledge3d.cranium.ptx_runtime.enhanced_fallback import EnhancedFallback, FallbackLevel


def test_all_enhancements_can_be_initialized():
    """Test that all 6 enhancement modules can be initialized without GPU"""

    # Enhancement #2
    profiler = LatencyProfiler(total_budget_us=35.0)
    assert profiler.total_budget_us == 35.0
    assert len(profiler.STAGE_NAMES) == 7

    # Enhancement #3
    cache = SparseWeightCache()
    assert cache.CAPACITY == 16
    assert len(cache.cache) == 0

    # Enhancement #5
    affinity = ModalAffinityMatrix()
    assert affinity.affinity_matrix.shape == (3, 3)
    assert affinity.get_affinity('text', 'text') == 1.0

    # Enhancement #6
    telemetry = TelemetryVisualizer(buffer_size=64)
    assert telemetry.buffer_size == 64

    # Enhancement #4
    fallback = EnhancedFallback()
    assert len(fallback.fallback_counts) == 4

    print("✓ All 5 enhancement modules initialized successfully")


def test_enhancements_work_together():
    """Test that enhancements can work together in a realistic scenario"""

    # Create all enhancements
    profiler = LatencyProfiler(total_budget_us=35.0)
    cache = SparseWeightCache()
    affinity = ModalAffinityMatrix()
    fallback = EnhancedFallback()

    # Simulate an inference workflow
    input_emb = np.random.randn(512).astype(np.float32)
    modal_sig = ['text', 'image']

    # Stage 1: Profile query stage
    profiler.start_stage("query")
    # ... simulated work ...
    profiler.end_stage("query")

    # Stage 2: Check cache
    cache_hit, weights = cache.lookup(input_emb)
    assert not cache_hit  # First time, should miss

    # Stage 3: "Compute" weights and cache them
    fake_weights = {'W1': np.random.randn(10, 10).astype(np.float32)}
    cache.insert(input_emb, fake_weights)

    # Stage 4: Second lookup should hit
    cache_hit2, weights2 = cache.lookup(input_emb)
    assert cache_hit2  # Second time, should hit
    assert cache.get_hit_rate() == 0.5  # 1 hit out of 2 lookups

    # Stage 5: Get modal boost
    boost = affinity.get_modal_boost(modal_sig)
    assert boost > 1.0  # Multi-modal should boost

    # Stage 6: Update affinity with success
    affinity.update_success(modal_sig, success_score=0.9)

    # Stage 7: Check profiler stats
    report = profiler.get_full_report()
    assert 'stages' in report
    assert 'query' in report['stages']

    # Stage 8: Check fallback stats
    fallback_stats = fallback.get_stats()
    assert fallback_stats['total_fallbacks'] == 0  # No failures yet

    print("✓ All enhancements working together correctly")


def test_confidence_weighted_logic():
    """Test confidence weighting logic without GPU"""

    affinity = ModalAffinityMatrix()

    # Test single modal
    boost_text = affinity.get_modal_boost(['text'])
    assert boost_text == 1.0

    # Test dual modal
    boost_text_image = affinity.get_modal_boost(['text', 'image'])
    assert boost_text_image > 1.0

    # Simulate confidence calculation
    confidence_rays = 0.8
    coherence_scores = 0.7
    uncertainty = 0.2

    # Formula from Enhancement #1
    final_confidence = (0.4 * confidence_rays) + (0.3 * coherence_scores) + (0.3 * (1 - uncertainty))
    assert 0.0 <= final_confidence <= 1.0

    # Apply modal boost
    boosted_confidence = final_confidence * boost_text_image
    assert boosted_confidence > final_confidence

    print(f"✓ Confidence weighting: {final_confidence:.3f} → {boosted_confidence:.3f} (boost: {boost_text_image:.3f}x)")


def test_latency_profiling_workflow():
    """Test complete latency profiling workflow"""

    profiler = LatencyProfiler(total_budget_us=35.0)

    # Simulate all 7 stages
    stages = ["sparsity_calc", "query", "cross_modal", "weight_assembly",
              "rpn_exec", "crystallize", "confidence"]

    for stage in stages:
        profiler.start_stage(stage)
        # Simulate work (very brief)
        _ = np.random.randn(100)
        profiler.end_stage(stage)

    # Get report
    report = profiler.get_full_report()

    assert len(report['stages']) == 7
    assert report['total_budget_us'] == 35.0

    # Verify all stages were profiled
    for stage in stages:
        assert stage in report['stages']
        stage_data = report['stages'][stage]
        assert 'avg_us' in stage_data
        assert stage_data['avg_us'] > 0  # Should have recorded some time

    print("✓ Latency profiling workflow complete")


def test_cache_performance_simulation():
    """Simulate cache performance with realistic access patterns"""

    cache = SparseWeightCache()

    # Create 20 different embeddings
    embeddings = [np.random.randn(512).astype(np.float32) for _ in range(20)]
    weights_list = [{'W': np.random.randn(5, 5)} for _ in range(20)]

    # First pass - all misses
    for emb, weights in zip(embeddings[:16], weights_list[:16]):
        hit, _ = cache.lookup(emb)
        assert not hit
        cache.insert(emb, weights)

    # Second pass - should have hits (cache full)
    hits = 0
    for emb in embeddings[:16]:
        hit, _ = cache.lookup(emb)
        if hit:
            hits += 1

    assert hits > 0  # Should have some hits
    hit_rate = cache.get_hit_rate()
    assert 0.3 < hit_rate < 1.0  # Reasonable hit rate

    print(f"✓ Cache simulation: {hits}/16 hits, hit rate: {hit_rate*100:.1f}%")


@pytest.mark.skipif(
    os.environ.get('SKIP_GPU_TESTS', '0') == '1',
    reason="GPU tests skipped by environment variable"
)
def test_thinking_tag_bridge_integration():
    """Full integration test with ThinkingTagBridge (requires GPU)"""

    try:
        from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge

        # Initialize bridge with all enhancements
        bridge = ThinkingTagBridge()

        # Verify enhancements are integrated
        assert hasattr(bridge, 'latency_profiler')
        assert hasattr(bridge, 'weight_cache')
        assert hasattr(bridge, 'modal_affinity')
        assert hasattr(bridge, 'enhanced_fallback')
        assert hasattr(bridge, 'telemetry')

        # Run a simple inference
        input_emb = np.random.randn(512).astype(np.float32)
        tags = bridge.inference(
            input_embedding=input_emb,
            modal_signature=['text'],
            temporal_anchor=0.5
        )

        # Check we got results
        assert tags is not None

        # Get enhancement statistics
        stats = bridge.get_enhancement_stats()
        assert 'latency_profiler' in stats
        assert 'sparse_cache' in stats
        assert 'modal_affinity' in stats
        assert 'enhanced_fallback' in stats

        print("✓ Full GPU integration test passed!")

    except RuntimeError as e:
        if "CUDA" in str(e) or "GPU" in str(e):
            pytest.skip(f"GPU not available or context busy: {e}")
        else:
            raise


if __name__ == "__main__":
    # Run tests without pytest
    print("Running simple integration tests...")
    print()

    test_all_enhancements_can_be_initialized()
    print()

    test_enhancements_work_together()
    print()

    test_confidence_weighted_logic()
    print()

    test_latency_profiling_workflow()
    print()

    test_cache_performance_simulation()
    print()

    print("=" * 80)
    print("ALL SIMPLE INTEGRATION TESTS PASSED ✓")
    print("=" * 80)
    print()
    print("Note: Full GPU integration test requires pytest and available GPU context")
    print("Run with: pytest tests/thinking_tags/test_enhancements_integration_simple.py -v")
