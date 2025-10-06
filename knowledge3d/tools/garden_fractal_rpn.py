"""
RPN-Powered Golden Ratio (φ) Calculations for Fractal Gardens

Uses modular RPN kernel for all φ-based fractal constraints:
- Branch angle: θ = 2π/φ ≈ 137.5° (golden angle)
- Max depth: d = int(φ × honesty × 10)
- Branch thickness: t = base / φ^depth
"""

import numpy as np
from typing import Dict, List
from knowledge3d.cranium.rpn_executor import get_rpn_executor


# Golden ratio constant
PHI = 1.618033988749895


def compile_golden_angle_rpn() -> Dict[str, np.ndarray]:
    """
    Compile golden angle formula to RPN: θ = 2π/φ

    RPN Program:
        PI 2.0 MUL PHI DIV

    Returns:
        Dict with RPN program (op_codes, scalars, vectors)
    """
    # RPN op-codes:
    # 0x00 = LITERAL (scalar)
    # 0x25 = PI (pushes π constant)
    # 0x26 = PHI (pushes φ constant)
    # 0x14 = MUL
    # 0x15 = DIV

    op_codes = np.array([
        0x0025,  # PI (π)
        0x0000,  # LITERAL scalar[0] (2.0)
        0x0014,  # MUL (2π)
        0x0026,  # PHI (φ)
        0x0015,  # DIV (2π/φ)
    ], dtype=np.uint16)

    scalars = np.array([2.0], dtype=np.float32)
    vectors = np.zeros((1, 4), dtype=np.float32)

    return {'op_codes': op_codes, 'scalars': scalars, 'vectors': vectors}


def compute_golden_angle_rpn() -> float:
    """
    Compute golden angle θ = 2π/φ ≈ 137.5° using RPN kernel.

    Returns:
        Golden angle in radians (~2.39996)
    """
    program = compile_golden_angle_rpn()
    executor = get_rpn_executor()

    angle = executor.execute_single(
        instance_id=0,
        op_codes=program['op_codes'],
        scalars=program['scalars'],
        vectors=program['vectors']
    )

    return float(angle)


def compile_max_depth_rpn(honesty: float) -> Dict[str, np.ndarray]:
    """
    Compile max depth formula to RPN: d = φ × honesty × 10

    RPN Program:
        PHI honesty MUL 10.0 MUL FLOOR

    Args:
        honesty: Honesty score in [0, 1]

    Returns:
        Dict with RPN program
    """
    op_codes = np.array([
        0x0026,  # PHI (φ)
        0x0000,  # LITERAL scalar[0] (honesty)
        0x0014,  # MUL (φ × honesty)
        0x0001,  # LITERAL scalar[1] (10.0)
        0x0014,  # MUL (× 10)
        0x0023,  # FLOOR (convert to int)
    ], dtype=np.uint16)

    scalars = np.array([honesty, 10.0], dtype=np.float32)
    vectors = np.zeros((1, 4), dtype=np.float32)

    return {'op_codes': op_codes, 'scalars': scalars, 'vectors': vectors}


def compute_max_depth_rpn(honesty: float) -> int:
    """
    Compute max fractal depth from honesty score using RPN.

    Formula: d = int(φ × honesty × 10)

    Args:
        honesty: Honesty score in [0, 1]

    Returns:
        Max depth (int), typically in range [0, 16]
    """
    program = compile_max_depth_rpn(honesty)
    executor = get_rpn_executor()

    depth = executor.execute_single(
        instance_id=0,
        op_codes=program['op_codes'],
        scalars=program['scalars'],
        vectors=program['vectors']
    )

    return int(depth)


def compile_thickness_rpn(base: float, depth: int) -> Dict[str, np.ndarray]:
    """
    Compile branch thickness formula to RPN: t = base / φ^depth

    RPN Program:
        PHI depth POW    # φ^depth
        base SWAP DIV    # base / φ^depth

    Args:
        base: Base trunk thickness (e.g., 1.0)
        depth: Current depth level

    Returns:
        Dict with RPN program
    """
    op_codes = np.array([
        0x0000,  # LITERAL scalar[0] (base)
        0x0026,  # PHI (φ)
        0x0001,  # LITERAL scalar[1] (depth as float)
        0x0024,  # POW (φ^depth)
        0x0015,  # DIV (base / φ^depth)
    ], dtype=np.uint16)

    scalars = np.array([base, float(depth)], dtype=np.float32)
    vectors = np.zeros((1, 4), dtype=np.float32)

    return {'op_codes': op_codes, 'scalars': scalars, 'vectors': vectors}


def compute_thickness_rpn(base: float, depth: int) -> float:
    """
    Compute branch thickness at given depth using RPN.

    Formula: t = base / φ^depth

    Args:
        base: Base trunk thickness
        depth: Current depth level

    Returns:
        Branch thickness at depth
    """
    program = compile_thickness_rpn(base, depth)
    executor = get_rpn_executor()

    thickness = executor.execute_single(
        instance_id=0,
        op_codes=program['op_codes'],
        scalars=program['scalars'],
        vectors=program['vectors']
    )

    return float(thickness)


def compute_fractal_constraints_batch_rpn(
    honesty_scores: List[float],
    base_thickness: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute all fractal constraints for batch of honesty scores using RPN.

    For each honesty score, computes:
    - Max depth: d = int(φ × honesty × 10)
    - Thickness at each depth: t = base / φ^d

    Args:
        honesty_scores: List of honesty scores in [0, 1]
        base_thickness: Base trunk thickness (default 1.0)

    Returns:
        Dict with:
            'golden_angle': float (same for all, ~2.4 rad)
            'max_depths': array of int depths
            'thickness_curves': list of arrays (thickness at each depth)
    """
    executor = get_rpn_executor()

    # Compute golden angle once (constant)
    golden_angle = compute_golden_angle_rpn()

    # Compute max depths in batch (up to 15 at once)
    depth_programs = [compile_max_depth_rpn(h) for h in honesty_scores]
    max_depths = executor.execute_batch(depth_programs, max_instances=15)
    max_depths = max_depths.astype(int)

    # Compute thickness curves for each tree
    thickness_curves = []
    for max_depth in max_depths:
        # Generate thickness at each depth level
        depth_range = np.arange(0, max_depth + 1)
        thickness_programs = [
            compile_thickness_rpn(base_thickness, d)
            for d in depth_range
        ]

        # Execute in batch
        thicknesses = executor.execute_batch(thickness_programs, max_instances=15)
        thickness_curves.append(thicknesses)

    return {
        'golden_angle': golden_angle,
        'max_depths': max_depths,
        'thickness_curves': thickness_curves
    }


def compute_branching_density_rpn(depth: int) -> int:
    """
    Compute number of branches at given depth using φ ratio.

    Formula: branches = φ^depth (rounded)

    Args:
        depth: Tree depth level

    Returns:
        Number of branches at depth
    """
    # Simple formula: φ^depth
    density = PHI ** depth
    return int(np.round(density))


def compute_golden_spiral_position_rpn(
    theta: float,
    radius_base: float
) -> tuple:
    """
    Compute position on golden spiral using RPN.

    Formula:
        r = radius_base × φ^(θ / (2π))
        x = r × cos(θ)
        y = r × sin(θ)

    Args:
        theta: Angle in radians
        radius_base: Base radius

    Returns:
        (x, y) position on spiral
    """
    # Compute radius scaling
    exponent = theta / (2 * np.pi)
    radius = radius_base * (PHI ** exponent)

    # Convert to Cartesian
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return (float(x), float(y))
