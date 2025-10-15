#!/usr/bin/env python3
"""Test bridge detection with synthetic data to verify threshold fix."""

import sys
sys.path.insert(0, '/workspace')

import cupy as cp
import numpy as np
from knowledge3d.spatial.domain_splitter import SemanticDomainSplitter
import logging
import pytest

pytestmark = pytest.mark.skip(
    reason="Domain splitter relies on deprecated CuPy-based spatial modules"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_graph(n_nodes=2000, n_domains=4):
    """Create a synthetic graph with multiple semantic domains."""

    # Create clustered embeddings (4 semantic domains)
    embeddings = []
    domain_centers = [
        np.array([1.0, 0.0, 0.0]),  # Domain 0: X-axis
        np.array([0.0, 1.0, 0.0]),  # Domain 1: Y-axis
        np.array([0.0, 0.0, 1.0]),  # Domain 2: Z-axis
        np.array([1.0, 1.0, 0.0]),  # Domain 3: XY diagonal
    ]

    nodes_per_domain = n_nodes // n_domains
    for i, center in enumerate(domain_centers):
        # Add some noise around each center
        domain_embs = center + np.random.normal(0, 0.1, (nodes_per_domain, 3))
        # Normalize
        domain_embs = domain_embs / np.linalg.norm(domain_embs, axis=1, keepdims=True)
        embeddings.append(domain_embs)

    embeddings = np.vstack(embeddings).astype(np.float32)

    # Create positions (spatial clustering matching semantic)
    positions = []
    for i in range(n_domains):
        offset = np.array([i * 10.0, i * 10.0, i * 10.0])
        pos = offset + np.random.uniform(-2, 2, (nodes_per_domain, 3))
        positions.append(pos)

    positions = np.vstack(positions).astype(np.float32)

    # Create edges: dense within domains, sparse between domains
    edges = []
    for i in range(n_nodes):
        domain_i = i // nodes_per_domain
        # Connect to 10 nodes in same domain
        for _ in range(10):
            j = domain_i * nodes_per_domain + np.random.randint(0, nodes_per_domain)
            if i != j:
                edges.append([i, j])

        # Connect to 2 nodes in different domains (these become bridges)
        for _ in range(2):
            other_domain = np.random.randint(0, n_domains)
            if other_domain != domain_i:
                j = other_domain * nodes_per_domain + np.random.randint(0, nodes_per_domain)
                edges.append([i, j])

    edges = np.array(edges, dtype=np.int32)

    return embeddings, positions, edges

def test_bridge_threshold():
    """Test that bridges are detected with 0.7 threshold (not 0.85)."""

    logger.info("=" * 80)
    logger.info("Testing bridge detection threshold fix")
    logger.info("=" * 80)

    # Create synthetic graph
    logger.info("Creating synthetic graph with 2000 nodes, 4 semantic domains...")
    embeddings, positions, edges = create_synthetic_graph(n_nodes=2000, n_domains=4)

    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Positions shape: {positions.shape}")
    logger.info(f"  Edges count: {len(edges)}")

    # Transfer to GPU
    embeddings_gpu = cp.asarray(embeddings)
    positions_gpu = cp.asarray(positions)
    edges_gpu = cp.asarray(edges)

    # Test with OLD threshold (0.85) - should find very few bridges
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Using OLD hardcoded threshold (0.85)")
    logger.info("=" * 80)

    # Manually check cross-domain edges with 0.85 threshold
    src_emb = embeddings_gpu[edges_gpu[:, 0]]
    dst_emb = embeddings_gpu[edges_gpu[:, 1]]
    src_norm = src_emb / (cp.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-8)
    dst_norm = dst_emb / (cp.linalg.norm(dst_emb, axis=1, keepdims=True) + 1e-8)
    cosine_sim = cp.sum(src_norm * dst_norm, axis=1)

    old_threshold_mask = cosine_sim > 0.85
    old_count = int(cp.sum(old_threshold_mask).get())
    logger.info(f"  Edges with similarity > 0.85: {old_count}/{len(edges)} ({100*old_count/len(edges):.1f}%)")

    # Test with NEW threshold (0.7) - should find many more bridges
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Using NEW configurable threshold (0.7)")
    logger.info("=" * 80)

    new_threshold_mask = cosine_sim > 0.7
    new_count = int(cp.sum(new_threshold_mask).get())
    logger.info(f"  Edges with similarity > 0.7: {new_count}/{len(edges)} ({100*new_count/len(edges):.1f}%)")

    # Now test with actual SemanticDomainSplitter
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Running SemanticDomainSplitter with sim_threshold=0.7")
    logger.info("=" * 80)

    splitter = SemanticDomainSplitter(sim_threshold=0.7)
    domain_ids, bridges, domains = splitter.split_domains(
        embeddings_gpu, positions_gpu, edges_gpu, kb_limit=128
    )

    logger.info(f"\nResults:")
    logger.info(f"  Domains created: {len(domains)}")
    logger.info(f"  Bridges found: {len(bridges)}")
    logger.info(f"  Bridge percentage: {100*len(bridges)/len(edges):.1f}%")

    # Verify the fix worked
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    if new_count > old_count * 3:
        logger.info(f"✓ Threshold fix working: 0.7 threshold finds {new_count/old_count:.1f}x more edges than 0.85")
    else:
        logger.warning(f"⚠ Unexpected: 0.7 threshold should find much more than 0.85")

    if len(bridges) > 0:
        logger.info(f"✓ SUCCESS: Domain splitter found {len(bridges)} bridges!")
        logger.info(f"  This means cross-domain navigation is now possible.")
        return True
    else:
        logger.error(f"✗ FAILURE: Domain splitter found 0 bridges.")
        logger.error(f"  Domains are disconnected. The fix may not have worked.")
        return False

if __name__ == "__main__":
    try:
        success = test_bridge_threshold()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
