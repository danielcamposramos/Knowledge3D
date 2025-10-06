"""
Semantic Depth Calculation via RPN Kernel

Implements GLM's Suggestion #1: Semantic-aware depth allocation for fractal trees.

Formula: depth = log₂(1 + cluster_size) × information_entropy(cluster)

where information_entropy = -Σ(p_i × log₂(p_i))

Uses modular RPN kernel for all computations (GPU-native, zero-copy).
"""

import numpy as np
from typing import List

from knowledge3d.cranium.rpn_executor import execute_rpn_kernel, execute_rpn_kernel_batch


# RPN Op-Codes (from modular_rpn_kernel.ptx)
OP_LITERAL_SCALAR = 0x00
OP_LITERAL_VECTOR = 0x01
OP_ADD = 0x10
OP_SUB = 0x11
OP_MUL = 0x12
OP_DIV = 0x13
OP_POW = 0x14
OP_NEG = 0x15
OP_SQRT = 0x20
OP_LOG2 = 0x21
OP_EXP = 0x22
OP_DOT = 0x30
OP_CROSS = 0x31
OP_NORM = 0x32
OP_FLOOR = 0x24
OP_DUP = 0x82


def compile_entropy_to_rpn(cluster_embeddings: np.ndarray) -> dict:
    """
    Compile information entropy calculation to RPN.

    Entropy = -Σ(p_i × log₂(p_i)) where p_i is concept distribution.

    Strategy:
    1. Compute pairwise similarities (concept distribution proxy)
    2. Normalize to probabilities (sum = 1)
    3. Compute -Σ(p × log₂(p))

    Args:
        cluster_embeddings: Array of shape (n_concepts, embedding_dim)

    Returns:
        Dict with 'op_codes', 'scalars', 'vectors' for RPN execution
    """
    n_concepts = cluster_embeddings.shape[0]

    if n_concepts < 2:
        # Entropy = 0 for single concept
        return {
            'op_codes': np.array([OP_LITERAL_SCALAR], dtype=np.uint16),
            'scalars': np.array([0.0], dtype=np.float32),
            'vectors': np.array([], dtype=np.float32)
        }

    op_codes = []
    scalars = []
    vectors = []

    # Simplified entropy: Use embedding norms as concept distribution
    # (More sophisticated: compute full pairwise similarities)

    # Step 1: Compute norm for each embedding (as proxy for concept weight)
    for i in range(n_concepts):
        op_codes.extend([OP_LITERAL_VECTOR, OP_NORM])
        vectors.append(cluster_embeddings[i])

    # Step 2: Sum all norms to get total
    # Stack now has: [norm_0, norm_1, ..., norm_n]
    for i in range(n_concepts - 1):
        op_codes.append(OP_ADD)  # Sum all norms

    # Stack now has: [total_norm]
    op_codes.append(OP_DUP)  # Duplicate for repeated use

    # Step 3: Compute probabilities and entropy terms
    # For each concept: p_i = norm_i / total, entropy_term = p_i × log₂(p_i)

    for i in range(n_concepts):
        # Load norm_i again
        op_codes.extend([OP_LITERAL_VECTOR, OP_NORM])
        vectors.append(cluster_embeddings[i])

        # Compute probability: p_i = norm_i / total_norm
        op_codes.append(OP_DIV)  # p_i = norm_i / total

        op_codes.append(OP_DUP)  # Duplicate p_i
        op_codes.append(OP_LOG2)  # log₂(p_i)
        op_codes.append(OP_MUL)  # p_i × log₂(p_i)
        op_codes.append(OP_NEG)  # -(p_i × log₂(p_i))

    # Step 4: Sum all entropy terms
    for i in range(n_concepts - 1):
        op_codes.append(OP_ADD)

    # Final result: information entropy

    return {
        'op_codes': np.array(op_codes, dtype=np.uint16),
        'scalars': np.array(scalars, dtype=np.float32),
        'vectors': np.concatenate(vectors).astype(np.float32) if vectors else np.array([], dtype=np.float32)
    }


def compile_depth_formula_rpn(cluster_size: int) -> dict:
    """
    Compile depth formula: log₂(1 + cluster_size) × entropy

    Args:
        cluster_size: Number of nodes in cluster

    Returns:
        RPN program dict
    """
    op_codes = np.array([
        OP_LITERAL_SCALAR,  # cluster_size
        OP_LITERAL_SCALAR,  # 1.0
        OP_ADD,             # cluster_size + 1
        OP_LOG2,            # log₂(1 + cluster_size)
        # Entropy will be on stack from previous instance
        # Multiply: (result will be computed by loading both results)
    ], dtype=np.uint16)

    scalars = np.array([float(cluster_size), 1.0], dtype=np.float32)
    vectors = np.array([], dtype=np.float32)

    return {
        'op_codes': op_codes,
        'scalars': scalars,
        'vectors': vectors
    }


def compute_semantic_depth_rpn(
    cluster_embeddings: np.ndarray,
    cluster_size: int,
    min_depth: int = 2,
    max_depth: int = 12
) -> int:
    """
    Compute semantic depth for a cluster using RPN kernel.

    Formula: depth = log₂(1 + cluster_size) × information_entropy(cluster)

    Args:
        cluster_embeddings: Embeddings of nodes in cluster (n, embedding_dim)
        cluster_size: Number of nodes in cluster
        min_depth: Minimum allowed depth (default 2)
        max_depth: Maximum allowed depth (default 12)

    Returns:
        Semantic depth (integer) for fractal tree recursion
    """

    # Compile entropy calculation
    entropy_program = compile_entropy_to_rpn(cluster_embeddings)

    # Execute entropy calculation (instance 0)
    entropy = execute_rpn_kernel(
        instance_id=0,
        op_codes=entropy_program['op_codes'],
        scalars=entropy_program['scalars'],
        vectors=entropy_program['vectors']
    )

    # Compile depth formula
    depth_program = compile_depth_formula_rpn(cluster_size)

    # Execute depth calculation (instance 1)
    size_factor = execute_rpn_kernel(
        instance_id=1,
        op_codes=depth_program['op_codes'],
        scalars=depth_program['scalars'],
        vectors=depth_program['vectors']
    )

    # Multiply entropy × size_factor (manual combination)
    raw_depth = entropy * size_factor

    # Clamp to valid range and convert to int
    depth = int(np.clip(np.floor(raw_depth), min_depth, max_depth))

    return depth


def compute_semantic_depths_batch_rpn(
    clusters: List[np.ndarray],
    min_depth: int = 2,
    max_depth: int = 12
) -> np.ndarray:
    """
    Compute semantic depths for multiple clusters in parallel.

    Uses RPN kernel's 15 parallel instances for batch processing.

    Args:
        clusters: List of embedding arrays (one per cluster)
        min_depth: Minimum allowed depth
        max_depth: Maximum allowed depth

    Returns:
        Array of depths (one per cluster)
    """

    depths = []

    # Process in batches of 7 (15 instances / 2 per cluster)
    batch_size = 7

    for batch_start in range(0, len(clusters), batch_size):
        batch = clusters[batch_start:batch_start+batch_size]

        # Compile programs for this batch
        programs = []
        for cluster_embs in batch:
            # Entropy program
            programs.append(compile_entropy_to_rpn(cluster_embs))

            # Depth formula program
            programs.append(compile_depth_formula_rpn(len(cluster_embs)))

        # Execute batch
        results = execute_rpn_kernel_batch(programs, max_instances=15)

        # Combine entropy × size_factor for each cluster
        for i in range(len(batch)):
            entropy = results[i * 2]
            size_factor = results[i * 2 + 1]
            raw_depth = entropy * size_factor
            depth = int(np.clip(np.floor(raw_depth), min_depth, max_depth))
            depths.append(depth)

    return np.array(depths, dtype=np.int32)


def estimate_information_entropy(cluster_embeddings: np.ndarray) -> float:
    """
    Quick entropy estimation (for validation/debugging).

    Uses RPN kernel for computation.

    Args:
        cluster_embeddings: Embeddings array

    Returns:
        Entropy value (float)
    """
    program = compile_entropy_to_rpn(cluster_embeddings)

    entropy = execute_rpn_kernel(
        instance_id=0,
        op_codes=program['op_codes'],
        scalars=program['scalars'],
        vectors=program['vectors']
    )

    return entropy
