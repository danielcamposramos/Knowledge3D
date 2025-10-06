"""
RPN-Powered Honesty Scoring for RLWHF

Uses modular RPN kernel to compute honesty scores from component metrics.
Formula: 0.4×correctness + 0.2×reasoning + 0.2×uncertainty + 0.2×alignment
"""

import numpy as np
from typing import Dict, List
from knowledge3d.cranium.rpn_executor import get_rpn_executor


def compile_honesty_to_rpn(
    correctness: float,
    reasoning: float,
    uncertainty: float,
    alignment: float
) -> Dict[str, np.ndarray]:
    """
    Compile honesty scoring formula to RPN op-codes.

    Formula: 0.4×correctness + 0.2×reasoning + 0.2×uncertainty + 0.2×alignment

    RPN Program:
        correctness 0.4 MUL
        reasoning 0.2 MUL ADD
        uncertainty 0.2 MUL ADD
        alignment 0.2 MUL ADD

    Args:
        correctness: Factual correctness score [0, 1]
        reasoning: Reasoning quality score [0, 1]
        uncertainty: Uncertainty expression score [0, 1]
        alignment: Alignment with honesty score [0, 1]

    Returns:
        Dict with 'op_codes' and 'scalars' arrays ready for RPN executor
    """
    # RPN op-codes:
    # 0x00 = LITERAL (scalar index in lower byte)
    # 0x14 = MUL
    # 0x10 = ADD

    op_codes = np.array([
        # correctness × 0.4
        0x0000,  # LITERAL scalar[0] (correctness)
        0x0001,  # LITERAL scalar[1] (0.4)
        0x0014,  # MUL

        # reasoning × 0.2
        0x0002,  # LITERAL scalar[2] (reasoning)
        0x0003,  # LITERAL scalar[3] (0.2)
        0x0014,  # MUL
        0x0010,  # ADD

        # uncertainty × 0.2
        0x0004,  # LITERAL scalar[4] (uncertainty)
        0x0003,  # LITERAL scalar[3] (0.2, reuse)
        0x0014,  # MUL
        0x0010,  # ADD

        # alignment × 0.2
        0x0005,  # LITERAL scalar[5] (alignment)
        0x0003,  # LITERAL scalar[3] (0.2, reuse)
        0x0014,  # MUL
        0x0010,  # ADD
    ], dtype=np.uint16)

    scalars = np.array([
        correctness,   # scalar[0]
        0.4,          # scalar[1]
        reasoning,    # scalar[2]
        0.2,          # scalar[3]
        uncertainty,  # scalar[4]
        alignment,    # scalar[5]
    ], dtype=np.float32)

    return {
        'op_codes': op_codes,
        'scalars': scalars,
        'vectors': np.zeros((1, 4), dtype=np.float32)  # No vectors needed
    }


def compute_honesty_score_rpn(
    correctness: float,
    reasoning: float,
    uncertainty: float,
    alignment: float
) -> float:
    """
    Compute honesty score using RPN kernel.

    Args:
        correctness: Factual correctness [0, 1]
        reasoning: Reasoning quality [0, 1]
        uncertainty: Uncertainty expression [0, 1]
        alignment: Honesty alignment [0, 1]

    Returns:
        Honesty score in [0, 1]
    """
    # Compile to RPN
    program = compile_honesty_to_rpn(correctness, reasoning, uncertainty, alignment)

    # Execute on GPU (instance 0)
    executor = get_rpn_executor()
    score = executor.execute_single(
        instance_id=0,
        op_codes=program['op_codes'],
        scalars=program['scalars'],
        vectors=program['vectors']
    )

    # Clamp to [0, 1]
    return float(np.clip(score, 0.0, 1.0))


def compute_honesty_batch_rpn(
    components: List[Dict[str, float]],
    max_batch_size: int = 15
) -> np.ndarray:
    """
    Compute honesty scores for batch of component sets using RPN.

    Args:
        components: List of dicts with keys:
            - 'correctness': float in [0, 1]
            - 'reasoning': float in [0, 1]
            - 'uncertainty': float in [0, 1]
            - 'alignment': float in [0, 1]
        max_batch_size: Maximum batch size (15 RPN instances available)

    Returns:
        Array of honesty scores, one per input
    """
    executor = get_rpn_executor()

    # Compile all programs
    programs = []
    for comp in components:
        prog = compile_honesty_to_rpn(
            comp['correctness'],
            comp['reasoning'],
            comp['uncertainty'],
            comp['alignment']
        )
        programs.append(prog)

    # Execute in batches
    scores = executor.execute_batch(programs, max_instances=max_batch_size)

    # Clamp to [0, 1]
    return np.clip(scores, 0.0, 1.0)


def compute_honesty_weighted_rpn(
    correctness: float,
    reasoning: float,
    uncertainty: float,
    alignment: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute honesty score with custom weights via RPN.

    Args:
        correctness: Factual correctness [0, 1]
        reasoning: Reasoning quality [0, 1]
        uncertainty: Uncertainty expression [0, 1]
        alignment: Honesty alignment [0, 1]
        weights: Optional custom weights (default: 0.4, 0.2, 0.2, 0.2)

    Returns:
        Weighted honesty score
    """
    if weights is None:
        weights = {
            'correctness': 0.4,
            'reasoning': 0.2,
            'uncertainty': 0.2,
            'alignment': 0.2
        }

    # Custom RPN program with provided weights
    op_codes = np.array([
        0x0000, 0x0001, 0x0014,  # correctness × w1
        0x0002, 0x0003, 0x0014, 0x0010,  # reasoning × w2 + ADD
        0x0004, 0x0005, 0x0014, 0x0010,  # uncertainty × w3 + ADD
        0x0006, 0x0007, 0x0014, 0x0010,  # alignment × w4 + ADD
    ], dtype=np.uint16)

    scalars = np.array([
        correctness,
        weights['correctness'],
        reasoning,
        weights['reasoning'],
        uncertainty,
        weights['uncertainty'],
        alignment,
        weights['alignment'],
    ], dtype=np.float32)

    vectors = np.zeros((1, 4), dtype=np.float32)

    executor = get_rpn_executor()
    score = executor.execute_single(0, op_codes, scalars, vectors)

    return float(np.clip(score, 0.0, 1.0))
