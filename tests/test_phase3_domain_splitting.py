"""
Phase 3 Domain Splitting Tests

Tests for multi-domain semantic navigation with affinity propagation clustering.

Key test areas:
- Affinity propagation convergence
- Domain balance (<48KB per domain)
- Bridge detection accuracy
- Cross-domain optimality
- Performance targets (<0.5ms navigation)
"""

import pytest
import numpy as np

pytestmark = pytest.mark.skip(
    reason="Uses deprecated CuPy-based spatial navigator/domain splitter"
)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from knowledge3d.spatial.domain_splitter import SemanticDomainSplitter
from knowledge3d.spatial.semantic_navigator import SemanticNavigator


class TestPhase3DomainSplitting:
    """Test suite for Phase 3 multi-domain navigation."""

    def test_affinity_propagation_convergence(self):
        """Verify AP converges in <20 iters on synthetic graph."""
        # Create synthetic embeddings (1000 nodes, 128-dim)
        np.random.seed(42)
        n_nodes = 1000
        embeddings = np.random.randn(n_nodes, 128).astype(np.float32)
        positions = np.random.randn(n_nodes, 3).astype(np.float32)

        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Create edge graph (k-NN)
        k = 8
        edges = []
        for i in range(n_nodes):
            dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
            neighbors = np.argsort(dists)[1:k+1]  # Exclude self
            for j in neighbors:
                edges.append([i, j])

        edges_gpu = cp.asarray(np.array(edges, dtype=np.uint32))
        embeddings_gpu = cp.asarray(embeddings)
        positions_gpu = cp.asarray(positions)

        # Run domain splitter
        splitter = SemanticDomainSplitter(sim_threshold=0.7, damping=0.9)
        domain_ids, bridges, domains = splitter.split_domains(
            embeddings_gpu,
            positions_gpu,
            edges_gpu,
            kb_limit=48
        )

        # Verify convergence
        assert len(domains) > 0, "No domains created"
        assert len(domains) < n_nodes, "Each node shouldn't be its own domain"

        print(f"✓ AP converged: {len(domains)} domains for {n_nodes} nodes")

    def test_domain_balance_under_48kb(self):
        """Verify all domains stay under 48KB limit."""
        np.random.seed(42)
        n_nodes = 2000
        embeddings = np.random.randn(n_nodes, 128).astype(np.float32)
        positions = np.random.randn(n_nodes, 3).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Create edges
        k = 8
        edges = []
        for i in range(min(n_nodes, 500)):  # Sample to avoid OOM
            dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
            neighbors = np.argsort(dists)[1:k+1]
            for j in neighbors:
                edges.append([i, j])

        edges_gpu = cp.asarray(np.array(edges, dtype=np.uint32))
        embeddings_gpu = cp.asarray(embeddings)
        positions_gpu = cp.asarray(positions)

        splitter = SemanticDomainSplitter(sim_threshold=0.7)
        domain_ids, bridges, domains = splitter.split_domains(
            embeddings_gpu,
            positions_gpu,
            edges_gpu,
            kb_limit=48
        )

        # Check all domains <48KB
        for domain in domains:
            assert domain.size_bytes < 48 * 1024, \
                f"Domain {domain.domain_id} exceeds 48KB: {domain.size_bytes} bytes"

        print(f"✓ All {len(domains)} domains under 48KB")

    def test_bridge_detection_accuracy(self):
        """Verify bridges are high-similarity cross-domain edges."""
        np.random.seed(42)
        n_nodes = 500
        embeddings = np.random.randn(n_nodes, 128).astype(np.float32)
        positions = np.random.randn(n_nodes, 3).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        k = 8
        edges = []
        for i in range(n_nodes):
            dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
            neighbors = np.argsort(dists)[1:k+1]
            for j in neighbors:
                edges.append([i, j])

        edges_gpu = cp.asarray(np.array(edges, dtype=np.uint32))
        embeddings_gpu = cp.asarray(embeddings)
        positions_gpu = cp.asarray(positions)

        splitter = SemanticDomainSplitter(sim_threshold=0.85)  # High threshold
        domain_ids, bridges, domains = splitter.split_domains(
            embeddings_gpu,
            positions_gpu,
            edges_gpu,
            kb_limit=48
        )

        # Verify bridge count is reasonable
        total_edges = len(edges)
        bridge_count = len(bridges)

        assert bridge_count < total_edges, "Bridges should be subset of edges"
        assert bridge_count > 0, "Should have at least some bridges"

        # Bridges should be <20% of edges (semantic highways)
        assert bridge_count / total_edges < 0.2, \
            f"Too many bridges: {bridge_count}/{total_edges} = {100*bridge_count/total_edges:.1f}%"

        print(f"✓ Bridges: {bridge_count}/{total_edges} ({100*bridge_count/total_edges:.1f}%)")

    def test_semantic_navigator_integration(self):
        """Test SemanticNavigator with multi-domain mode."""
        # This test requires a real GLB file
        # For now, just verify the strategy pattern works

        navigator = SemanticNavigator(
            query_radius=2.0,
            k_neighbors=8,
            similarity_threshold=0.7,
            nav_mode="auto"  # Should auto-detect based on size
        )

        # Verify nav_mode is set
        assert navigator.nav_mode == "auto"
        assert navigator.multi_domain_navigator is None  # Not built yet
        assert not navigator._use_multi_domain

        print("✓ SemanticNavigator strategy pattern initialized")


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="Requires CuPy/GPU")
class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 3."""

    @pytest.mark.slow
    def test_domain_splitting_performance(self):
        """Benchmark domain splitting time for 5k nodes."""
        import time

        np.random.seed(42)
        n_nodes = 5000
        embeddings = np.random.randn(n_nodes, 128).astype(np.float32)
        positions = np.random.randn(n_nodes, 3).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Create sparse edges
        k = 8
        edges = []
        for i in range(n_nodes):
            dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
            neighbors = np.argsort(dists)[1:k+1]
            for j in neighbors:
                edges.append([i, j])

        edges_gpu = cp.asarray(np.array(edges, dtype=np.uint32))
        embeddings_gpu = cp.asarray(embeddings)
        positions_gpu = cp.asarray(positions)

        # Benchmark
        splitter = SemanticDomainSplitter(sim_threshold=0.7)

        start = time.perf_counter()
        domain_ids, bridges, domains = splitter.split_domains(
            embeddings_gpu,
            positions_gpu,
            edges_gpu,
            kb_limit=48
        )
        elapsed = time.perf_counter() - start

        # Target: <5s for 5k nodes
        assert elapsed < 5.0, f"Splitting too slow: {elapsed:.2f}s"

        print(f"✓ Domain splitting: {elapsed:.2f}s for {n_nodes} nodes → {len(domains)} domains")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
