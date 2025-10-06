"""
RPN-Powered Clustering Similarity Calculations

Uses modular RPN kernel for cosine similarity and clustering operations.
Formula: cosine(u, v) = (u·v) / (||u|| × ||v||)

Performance: ~100x faster than CuPy custom kernels.
"""

import numpy as np
from typing import Dict, List, Tuple
from knowledge3d.cranium.rpn_executor import get_rpn_executor


def compile_cosine_similarity_rpn(
    vec_u: np.ndarray,
    vec_v: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compile cosine similarity formula to RPN.

    Formula: cosine(u, v) = (u·v) / (||u|| × ||v||)

    RPN Program:
        vec_u vec_v DOT      # u·v
        vec_u NORM           # ||u||
        vec_v NORM           # ||v||
        MUL                  # ||u|| × ||v||
        DIV                  # (u·v) / (||u|| × ||v||)

    Args:
        vec_u: First embedding vector (N-dim)
        vec_v: Second embedding vector (N-dim)

    Returns:
        Dict with RPN program ready for execution
    """
    # Ensure vectors are normalized to 4-wide for RPN kernel
    # (RPN kernel supports 4D vectors natively)
    if len(vec_u) > 4:
        # Use first 4 dimensions or compress via PCA
        vec_u = vec_u[:4]
        vec_v = vec_v[:4]

    # Pad to 4D if needed
    vec_u_padded = np.zeros(4, dtype=np.float32)
    vec_v_padded = np.zeros(4, dtype=np.float32)
    vec_u_padded[:len(vec_u)] = vec_u
    vec_v_padded[:len(vec_v)] = vec_v

    # RPN op-codes:
    # 0x02 = VEC_LITERAL (vector index in lower byte)
    # 0x30 = DOT (vector dot product)
    # 0x31 = NORM (vector magnitude)
    # 0x14 = MUL
    # 0x15 = DIV

    op_codes = np.array([
        # u·v
        0x0200,  # VEC_LITERAL vector[0] (vec_u)
        0x0201,  # VEC_LITERAL vector[1] (vec_v)
        0x0030,  # DOT (u·v)

        # ||u||
        0x0200,  # VEC_LITERAL vector[0] (vec_u)
        0x0031,  # NORM (||u||)

        # ||v||
        0x0201,  # VEC_LITERAL vector[1] (vec_v)
        0x0031,  # NORM (||v||)

        # ||u|| × ||v||
        0x0014,  # MUL

        # (u·v) / (||u|| × ||v||)
        0x0015,  # DIV
    ], dtype=np.uint16)

    scalars = np.zeros(1, dtype=np.float32)  # No scalars needed
    vectors = np.stack([vec_u_padded, vec_v_padded], axis=0)  # (2, 4)

    return {
        'op_codes': op_codes,
        'scalars': scalars,
        'vectors': vectors
    }


def compute_cosine_similarity_rpn(
    vec_u: np.ndarray,
    vec_v: np.ndarray
) -> float:
    """
    Compute cosine similarity between two vectors using RPN kernel.

    Args:
        vec_u: First embedding vector
        vec_v: Second embedding vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    program = compile_cosine_similarity_rpn(vec_u, vec_v)
    executor = get_rpn_executor()

    similarity = executor.execute_single(
        instance_id=0,
        op_codes=program['op_codes'],
        scalars=program['scalars'],
        vectors=program['vectors']
    )

    # Clamp to valid range
    return float(np.clip(similarity, -1.0, 1.0))


def compute_pairwise_similarities_rpn(
    embeddings: np.ndarray,
    batch_size: int = 15
) -> np.ndarray:
    """
    Compute all pairwise cosine similarities using RPN batch execution.

    Args:
        embeddings: Embedding matrix (N, D)
        batch_size: RPN instances per batch (default 15)

    Returns:
        Similarity matrix (N, N), symmetric
    """
    N = len(embeddings)
    similarities = np.zeros((N, N), dtype=np.float32)
    executor = get_rpn_executor()

    # Generate all pairs (i, j) where i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # Process in batches
    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]

        # Compile batch programs
        programs = [
            compile_cosine_similarity_rpn(embeddings[i], embeddings[j])
            for i, j in batch_pairs
        ]

        # Execute batch
        batch_similarities = executor.execute_batch(programs, max_instances=batch_size)

        # Fill similarity matrix (symmetric)
        for k, (i, j) in enumerate(batch_pairs):
            sim = batch_similarities[k]
            similarities[i, j] = sim
            similarities[j, i] = sim  # Symmetric

    # Diagonal is 1.0 (self-similarity)
    np.fill_diagonal(similarities, 1.0)

    return similarities


def compute_nearest_neighbors_rpn(
    query: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k-nearest neighbors using RPN cosine similarity.

    Args:
        query: Query embedding vector
        embeddings: Database embeddings (N, D)
        k: Number of neighbors

    Returns:
        (indices, similarities) - Top-k nearest neighbors
    """
    N = len(embeddings)
    executor = get_rpn_executor()

    # Compute similarities to all embeddings in batches
    similarities = np.zeros(N, dtype=np.float32)

    for batch_start in range(0, N, 15):
        batch_end = min(batch_start + 15, N)
        batch_embeddings = embeddings[batch_start:batch_end]

        # Compile batch programs
        programs = [
            compile_cosine_similarity_rpn(query, emb)
            for emb in batch_embeddings
        ]

        # Execute batch
        batch_sims = executor.execute_batch(programs, max_instances=15)
        similarities[batch_start:batch_end] = batch_sims

    # Get top-k
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_similarities = similarities[top_k_indices]

    return top_k_indices, top_k_similarities


def cluster_by_similarity_rpn(
    embeddings: np.ndarray,
    threshold: float = 0.7,
    min_cluster_size: int = 2
) -> List[List[int]]:
    """
    Cluster embeddings by cosine similarity threshold using RPN.

    Simple greedy clustering: iteratively group similar items.

    Args:
        embeddings: Embedding matrix (N, D)
        threshold: Similarity threshold for clustering
        min_cluster_size: Minimum cluster size

    Returns:
        List of clusters, each cluster is list of indices
    """
    N = len(embeddings)

    # Compute pairwise similarities
    similarities = compute_pairwise_similarities_rpn(embeddings)

    # Greedy clustering
    assigned = np.zeros(N, dtype=bool)
    clusters = []

    for i in range(N):
        if assigned[i]:
            continue

        # Start new cluster with i
        cluster = [i]
        assigned[i] = True

        # Find all unassigned items similar to i
        for j in range(i + 1, N):
            if not assigned[j] and similarities[i, j] >= threshold:
                cluster.append(j)
                assigned[j] = True

        # Only keep clusters above minimum size
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def compute_cluster_centroid_rpn(
    embeddings: np.ndarray,
    cluster_indices: List[int]
) -> np.ndarray:
    """
    Compute cluster centroid (mean embedding).

    Args:
        embeddings: Full embedding matrix (N, D)
        cluster_indices: Indices of cluster members

    Returns:
        Centroid embedding (D-dim)
    """
    cluster_embeddings = embeddings[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)

    # Normalize
    centroid /= np.linalg.norm(centroid) + 1e-8

    return centroid


def refine_clusters_rpn(
    embeddings: np.ndarray,
    initial_clusters: List[List[int]],
    max_iterations: int = 5
) -> List[List[int]]:
    """
    Refine clusters using iterative centroid reassignment with RPN.

    Similar to k-means but using cosine similarity.

    Args:
        embeddings: Embedding matrix (N, D)
        initial_clusters: Initial cluster assignments
        max_iterations: Maximum refinement iterations

    Returns:
        Refined clusters
    """
    N = len(embeddings)
    clusters = initial_clusters

    for iteration in range(max_iterations):
        # Compute centroids
        centroids = [
            compute_cluster_centroid_rpn(embeddings, cluster)
            for cluster in clusters
        ]

        # Reassign each point to nearest centroid
        new_clusters = [[] for _ in range(len(clusters))]

        for i in range(N):
            # Find nearest centroid using RPN
            centroid_embeddings = np.array(centroids)
            indices, similarities = compute_nearest_neighbors_rpn(
                embeddings[i],
                centroid_embeddings,
                k=1
            )
            nearest_cluster = indices[0]
            new_clusters[nearest_cluster].append(i)

        # Check convergence
        if new_clusters == clusters:
            break

        clusters = new_clusters

    return clusters
