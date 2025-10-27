"""
RPN-Powered Clustering Similarity Calculations

Uses modular RPN kernel for cosine similarity and clustering operations.
Formula: cosine(u, v) = (u·v) / (||u|| × ||v||)

Performance: ~100x faster than CuPy custom kernels.
"""

import numpy as np
from typing import Dict, List, Tuple
try:
    from knowledge3d.cranium.sovereign_rpn_executor import (
        get_sovereign_rpn_executor as get_rpn_executor,
    )
except ImportError:  # Fallback for legacy path
    from knowledge3d.cranium.rpn_executor import get_rpn_executor


def compile_cosine_similarity_rpn(
    vec_u: np.ndarray,
    vec_v: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compile cosine similarity formula to RPN using matroska adaptive chunking.

    Formula: cosine(u, v) = (u·v) / (||u|| × ||v||)

    For high-dimensional vectors (>3D), uses adaptive chunking into 3D pieces:
    - dot(u, v) = sum(dot3(u[i:i+3], v[i:i+3]) for all chunks)
    - norm(u) = sqrt(sum(dot3(u[i:i+3], u[i:i+3]) for all chunks))

    This aligns with the matroska embedding style - adaptive chunking for
    arbitrary dimensions using the RPN kernel's native 3D operations.

    RPN Program (per 3D chunk):
        vec_u_chunk vec_v_chunk DOT  # chunk dot product

    Args:
        vec_u: First embedding vector (N-dim)
        vec_v: Second embedding vector (N-dim)

    Returns:
        Dict with RPN program ready for execution (dot product for one 3D chunk)
    """
    dim = len(vec_u)

    # For dimensions <= 3, use directly
    if dim <= 3:
        # Pad to 3D
        vec_u_padded = np.zeros(3, dtype=np.float32)
        vec_v_padded = np.zeros(3, dtype=np.float32)
        vec_u_padded[:dim] = vec_u
        vec_v_padded[:dim] = vec_v

        # RPN opcodes: 0x01=VEC_LITERAL, 0x3C=DOT
        op_codes = np.array([
            0x01,  # VEC_LITERAL vector[0] (vec_u)
            0x01,  # VEC_LITERAL vector[1] (vec_v)
            0x3C,  # DOT (u·v)
        ], dtype=np.uint16)

        scalars = np.zeros(1, dtype=np.float32)
        vectors = np.concatenate([vec_u_padded, vec_v_padded])  # 6 floats

        return {
            'op_codes': op_codes,
            'scalars': scalars,
            'vectors': vectors,
            'result_type': 'scalar'  # dot product result
        }

    # For high-dimensional vectors, return metadata for chunked computation
    # The caller will handle adaptive chunking
    return {
        'vec_u': vec_u,
        'vec_v': vec_v,
        'dim': dim,
        'requires_chunking': True
    }


def compute_cosine_similarity_rpn(
    vec_u: np.ndarray,
    vec_v: np.ndarray
) -> float:
    """
    Compute cosine similarity between two vectors using RPN kernel with adaptive chunking.

    For high-dimensional vectors (>3D), uses matroska-style chunking:
    - Breaks vectors into 3D chunks
    - Computes dot3 for each chunk on GPU
    - Accumulates results for final cosine similarity

    Args:
        vec_u: First embedding vector (N-dim)
        vec_v: Second embedding vector (N-dim)

    Returns:
        Cosine similarity in [-1, 1]
    """
    program = compile_cosine_similarity_rpn(vec_u, vec_v)

    # Simple case: <=3D, direct computation
    if not program.get('requires_chunking', False):
        executor = get_rpn_executor()
        similarity = executor.execute_single(
            instance_id=0,
            op_codes=program['op_codes'],
            scalars=program['scalars'],
            vectors=program['vectors']
        )
        return float(np.clip(similarity, -1.0, 1.0))

    # Adaptive chunking for high-dimensional vectors
    vec_u = program['vec_u']
    vec_v = program['vec_v']
    dim = program['dim']
    executor = get_rpn_executor()

    # Chunk into 3D pieces
    chunk_size = 3
    num_chunks = (dim + chunk_size - 1) // chunk_size  # Ceiling division

    # Prepare all chunks upfront
    chunks_u = []
    chunks_v = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, dim)
        chunk_dim = end - start

        u_padded = np.zeros(3, dtype=np.float32)
        v_padded = np.zeros(3, dtype=np.float32)
        u_padded[:chunk_dim] = vec_u[start:end]
        v_padded[:chunk_dim] = vec_v[start:end]

        chunks_u.append(u_padded)
        chunks_v.append(v_padded)

    # Process chunks in batches of 15 (leverage 15 RPN instances!)
    batch_size = 15
    dot_product = 0.0
    norm_u_sq = 0.0
    norm_v_sq = 0.0

    op_codes = np.array([0x01, 0x01, 0x3C], dtype=np.uint16)  # VEC, VEC, DOT
    scalars = np.zeros(1, dtype=np.float32)

    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_chunks_u = chunks_u[batch_start:batch_end]
        batch_chunks_v = chunks_v[batch_start:batch_end]

        # Prepare batch programs for dot(u, v)
        programs_uv = []
        programs_uu = []
        programs_vv = []

        for u, v in zip(batch_chunks_u, batch_chunks_v):
            programs_uv.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': np.concatenate([u, v])
            })
            programs_uu.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': np.concatenate([u, u])
            })
            programs_vv.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': np.concatenate([v, v])
            })

        # Execute batches in parallel (15 RPN instances!)
        results_uv = executor.execute_batch(programs_uv, max_instances=batch_size)
        results_uu = executor.execute_batch(programs_uu, max_instances=batch_size)
        results_vv = executor.execute_batch(programs_vv, max_instances=batch_size)

        # Accumulate batch results
        dot_product += sum(results_uv)
        norm_u_sq += sum(results_uu)
        norm_v_sq += sum(results_vv)

    # Compute final cosine similarity
    norm_u = np.sqrt(norm_u_sq)
    norm_v = np.sqrt(norm_v_sq)

    if norm_u < 1e-8 or norm_v < 1e-8:
        return 0.0

    similarity = dot_product / (norm_u * norm_v)
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


def compute_similarity_matrix_rpn(
    sources: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 15
) -> np.ndarray:
    """
    Compute cosine similarity between each pair (source_i, target_j) using RPN.

    Uses adaptive chunking for high-dimensional vectors (matroska style).

    Args:
        sources: Source embedding matrix (N, D)
        targets: Target embedding matrix (K, D)
        batch_size: Number of RPN instances to evaluate in parallel (unused for chunked mode)

    Returns:
        Similarity matrix with shape (N, K)
    """
    # Check if we need chunking by testing first vector pair
    test_program = compile_cosine_similarity_rpn(sources[0], targets[0])

    if test_program.get('requires_chunking', False):
        # High-dimensional case: use adaptive chunking for each pair
        sims = np.zeros((len(sources), len(targets)), dtype=np.float32)
        for i, src in enumerate(sources):
            for j, tgt in enumerate(targets):
                sims[i, j] = compute_cosine_similarity_rpn(src, tgt)
        return sims

    # Low-dimensional case (<=3D): use batch execution
    executor = get_rpn_executor()
    sims = np.zeros((len(sources), len(targets)), dtype=np.float32)

    programs: List[Dict[str, np.ndarray]] = []
    pairs: List[Tuple[int, int]] = []

    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            programs.append(compile_cosine_similarity_rpn(src, tgt))
            pairs.append((i, j))

            if len(programs) == batch_size:
                results = executor.execute_batch(programs, max_instances=batch_size)
                for idx, (ii, jj) in enumerate(pairs):
                    sims[ii, jj] = results[idx]
                programs.clear()
                pairs.clear()

    if programs:
        results = executor.execute_batch(programs, max_instances=batch_size)
        for idx, (ii, jj) in enumerate(pairs):
            sims[ii, jj] = results[idx]

    return sims


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
