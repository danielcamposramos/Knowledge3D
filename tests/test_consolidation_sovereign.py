#!/usr/bin/env python3
"""
Test sovereign sleep-time consolidation.
"""


def main() -> int:
    # Keep imports inside main so `pytest` collection doesn't execute GPU work.
    import numpy as np

    from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
    from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator

    print("=" * 70)
    print("Sovereign Sleep-Time Consolidation Test")
    print("=" * 70)

    # Create test engine with embeddings
    print("\n[1/3] Creating RPN engine with test embeddings")
    engine = RPNEmbeddingEngine(embedding_dim=128)
    for i in range(100):
        vec = np.random.randn(128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        engine.embeddings[i] = vec

    print(f"✓ Created engine with {len(engine.embeddings)} embeddings")

    # Create consolidator
    print("\n[2/3] Creating consolidator")
    consolidator = SleepTimeConsolidator(engine, cluster_count=10)
    print("✓ Consolidator initialized")

    # Run consolidation
    print("\n[3/3] Running consolidation...")
    try:
        metrics = consolidator.consolidate()

        print("\n" + "=" * 70)
        print("CONSOLIDATION METRICS")
        print("=" * 70)

        # Cluster refinement
        cluster_metrics = metrics.get("cluster_refinement", {})
        print("\n[Cluster Refinement]")
        print(f"  Clusters: {cluster_metrics.get('clusters', 0)}")
        print(f"  Cohesion before: {cluster_metrics.get('cohesion_before', 0):.4f}")
        print(f"  Cohesion after: {cluster_metrics.get('cohesion_after', 0):.4f}")
        print(f"  Improvement: {cluster_metrics.get('improvement', 0):.4f}")

        # Redundancy pruning
        prune_metrics = metrics.get("redundancy_pruning", {})
        print("\n[Redundancy Pruning]")
        print(f"  Merged pairs: {prune_metrics.get('merged_pairs', 0)}")
        print(f"  Reduction: {prune_metrics.get('reduction', 0):.2f}%")

        # Overall
        print("\n[Overall]")
        print(f"  Elapsed: {metrics.get('elapsed_seconds', 0):.2f}s")
        print(f"  Final vocab size: {metrics.get('vocab_size', 0)}")

        # Verify cohesion is non-zero
        cohesion_before = cluster_metrics.get("cohesion_before", 0)
        cohesion_after = cluster_metrics.get("cohesion_after", 0)

        print("\n" + "=" * 70)
        if cohesion_before > 0 and cohesion_after > 0:
            print("✅ PASS: Consolidation produced non-zero cohesion metrics!")
            print("   → RPN kernel working correctly")
            print("   → Sovereign clustering operational")
            return 0

        print("❌ FAIL: Cohesion metrics are ZERO")
        print("   → RPN kernel may still have issues")
        print("   → Check compute_similarity_matrix_rpn()")
        return 1

    except Exception as e:
        print(f"\n❌ FAIL: Consolidation crashed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
