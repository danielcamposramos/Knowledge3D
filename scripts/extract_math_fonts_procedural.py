#!/usr/bin/env python3
"""
Procedural Math Font Extractor - Dual-Modal Symbol Generation

Extracts math symbols from font files as procedural RPN drawing commands,
mapping each symbol to both:
1. visual_rpn: How to DRAW the symbol (MOVE, LINE, QUAD, CUBIC, etc.)
2. math_rpn: How to EXECUTE it (OP_GRADIENT, OP_SQRT, OP_ADD, etc.)

This is sovereignty-first: Parse font outlines directly to RPN, no bitmap rendering.
95% math symbol coverage per MATH_GALAXY_MULTIVARIATE_DESIGN.md.

Usage:
    python scripts/extract_math_fonts_procedural.py \
        --font-dir /K3D/Knowledge3D.local/fonts/math/ \
        --output /K3D/Knowledge3D.local/datasets/math_symbols_procedural.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Font parsing (fontTools is pure Python, no external C deps)
try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import RecordingPen
except ImportError:
    print("ERROR: fontTools not installed. Install with:")
    print("  pip install fonttools")
    sys.exit(1)

# Import RPN opcodes
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# UNICODE MATH SYMBOL → RPN MATH OPCODE MAPPING
# =============================================================================

MATH_SYMBOL_MAPPING = {
    # Basic arithmetic
    '+': ('ADD', OP_ADD, 'Addition: pop b, pop a, push a+b'),
    '−': ('SUBTRACT', OP_SUB, 'Subtraction: pop b, pop a, push a-b'),  # U+2212 MINUS SIGN
    '-': ('SUBTRACT', OP_SUB, 'Subtraction: pop b, pop a, push a-b'),  # ASCII hyphen-minus
    '×': ('MULTIPLY', OP_MUL, 'Multiplication: pop b, pop a, push a*b'),  # U+00D7
    '·': ('MULTIPLY', OP_MUL, 'Multiplication: pop b, pop a, push a*b'),  # U+00B7 MIDDLE DOT
    '÷': ('DIVIDE', OP_DIV, 'Division: pop b, pop a, push a/b'),
    '/': ('DIVIDE', OP_DIV, 'Division: pop b, pop a, push a/b'),

    # Powers and roots
    '√': ('SQRT', OP_SQRT, 'Square root: pop x, push sqrt(x)'),
    '∛': ('CBRT', [OP_DUP, OP_DUP, OP_MUL, OP_MUL], 'Cube root: x^(1/3)'),
    '∜': ('FOURTHROOT', [0xE4, 0.25, OP_VAR_X, OP_SWAP], 'Fourth root: x^(1/4)'),  # Placeholder
    '^': ('POWER', OP_MUL, 'Exponentiation (placeholder for multi-token)'),  # Context-dependent
    '²': ('SQUARE', OP_DUP, 'Square: pop x, push x*x'),  # Superscript 2
    '³': ('CUBE', [OP_DUP, OP_DUP, OP_MUL, OP_MUL], 'Cube: pop x, push x*x*x'),  # Superscript 3

    # Exponential/logarithmic
    'e': ('E_CONST', [0xE4, 2.71828182845905], 'Euler\'s number: push e'),
    'exp': ('EXP', OP_EXP, 'Exponential: pop x, push e^x'),
    'ln': ('LOG', OP_LOG, 'Natural log: pop x, push ln(x)'),
    'log': ('LOG10', OP_LOG10, 'Base-10 log: pop x, push log10(x)'),
    'lg': ('LOG2', OP_LOG2, 'Base-2 log: pop x, push log2(x)'),

    # Trigonometric
    'sin': ('SIN', OP_SIN, 'Sine: pop x, push sin(x)'),
    'cos': ('COS', OP_COS, 'Cosine: pop x, push cos(x)'),
    'tan': ('TAN', OP_TAN, 'Tangent: pop x, push tan(x)'),
    'arcsin': ('ASIN', OP_ASIN, 'Arc sine: pop x, push asin(x)'),
    'arccos': ('ACOS', OP_ACOS, 'Arc cosine: pop x, push acos(x)'),
    'arctan': ('ATAN', OP_ATAN, 'Arc tangent: pop x, push atan(x)'),
    'sinh': ('SINH', OP_SINH, 'Hyperbolic sine: pop x, push sinh(x)'),
    'cosh': ('COSH', OP_COSH, 'Hyperbolic cosine: pop x, push cosh(x)'),
    'tanh': ('TANH', OP_TANH, 'Hyperbolic tangent: pop x, push tanh(x)'),

    # Constants
    'π': ('PI_CONST', [0xE4, 3.14159265358979], 'Pi constant: push π'),
    '∞': ('INF_CONST', [0xE4, float('inf')], 'Infinity: push ∞'),
    '∅': ('EMPTY_SET', [0xE4, 0.0], 'Empty set: push 0'),

    # Calculus operators (multivariate)
    '∇': ('GRADIENT', OP_GRADIENT, 'Gradient: ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]'),
    '∂': ('PARTIAL_DERIVATIVE', OP_SYMBOLIC_DIFF, 'Partial derivative: ∂f/∂x'),
    '∫': ('INTEGRATE', OP_SYMBOLIC_INTEGRATE, 'Integration: ∫f dx'),
    '∑': ('SUM', OP_SERIES_SUM, 'Summation: Σ'),
    '∏': ('PRODUCT', OP_SERIES_PRODUCT, 'Product: Π'),
    '∆': ('LAPLACIAN', OP_LAPLACIAN, 'Laplacian: ∇²f'),
    '∇·': ('DIVERGENCE', OP_DIVERGENCE, 'Divergence: ∇·F'),
    '∇×': ('CURL', OP_CURL, 'Curl: ∇×F'),

    # Comparisons
    '=': ('EQUAL', OP_EQ, 'Equality: pop b, pop a, push (a==b)'),
    '≠': ('NOT_EQUAL', [OP_EQ, OP_NOT], 'Inequality: a ≠ b'),
    '<': ('LESS_THAN', OP_LT, 'Less than: pop b, pop a, push (a<b)'),
    '>': ('GREATER_THAN', OP_GT, 'Greater than: pop b, pop a, push (a>b)'),
    '≤': ('LESS_EQUAL', [OP_GT, OP_NOT], 'Less or equal: a ≤ b'),
    '≥': ('GREATER_EQUAL', [OP_LT, OP_NOT], 'Greater or equal: a ≥ b'),
    '≈': ('APPROX', [OP_SUB, OP_ABS, 0xE4, 1e-6, OP_LT], 'Approximately equal'),

    # Set theory
    '∈': ('ELEMENT_OF', [OP_SET_INTERSECTION, OP_ABS, 0xE4, 0.0, OP_GT], 'Element of set'),
    '∉': ('NOT_ELEMENT', [OP_SET_INTERSECTION, OP_ABS, 0xE4, 0.0, OP_EQ], 'Not element of'),
    '∪': ('UNION', OP_SET_UNION, 'Set union: A ∪ B'),
    '∩': ('INTERSECTION', OP_SET_INTERSECTION, 'Set intersection: A ∩ B'),
    '∖': ('SET_DIFFERENCE', OP_SET_DIFFERENCE, 'Set difference: A \\ B'),
    '⊂': ('SUBSET', [OP_SET_DIFFERENCE, OP_ABS, 0xE4, 0.0, OP_EQ], 'Proper subset'),
    '⊆': ('SUBSET_EQUAL', [OP_SET_DIFFERENCE, OP_ABS, 0xE4, 0.0, OP_EQ], 'Subset or equal'),

    # Logic
    '∧': ('AND', OP_AND, 'Logical AND'),
    '∨': ('OR', OP_OR, 'Logical OR'),
    '¬': ('NOT', OP_NOT, 'Logical NOT'),
    '⊕': ('XOR', OP_XOR, 'Logical XOR'),
    '→': ('IMPLIES', [OP_NOT, OP_SWAP, OP_OR], 'Logical implication: A → B'),
    '↔': ('IFF', [OP_XOR, OP_NOT], 'If and only if: A ↔ B'),
    '∀': ('FOR_ALL', OP_LOOP, 'Universal quantifier'),
    '∃': ('EXISTS', OP_BRANCH, 'Existential quantifier'),

    # Linear algebra
    '⋅': ('DOT_PRODUCT', OP_DOT_PRODUCT, 'Dot product: a·b'),
    '×': ('CROSS_PRODUCT', OP_CROSS_PRODUCT, 'Cross product: a×b (vector)'),  # Context-dependent
    '⊗': ('TENSOR_PRODUCT', OP_OUTER_PRODUCT, 'Tensor/outer product'),
    'det': ('DETERMINANT', OP_MATRIX_DET, 'Matrix determinant'),
    'tr': ('TRACE', OP_TRACE_TENSOR, 'Matrix trace'),
    '⊤': ('TRANSPOSE', OP_MATRIX_TRANSPOSE, 'Matrix transpose'),
    '⁻¹': ('INVERSE', OP_MATRIX_INV, 'Matrix inverse'),

    # Special functions
    'Γ': ('GAMMA', OP_GAMMA, 'Gamma function: Γ(x)'),
    'β': ('BETA', OP_BETA, 'Beta function: B(x,y)'),
    '!': ('FACTORIAL', OP_FACTORIAL, 'Factorial: n!'),
    'C': ('BINOMIAL', OP_BINOMIAL, 'Binomial coefficient: nCr'),

    # Multivariate variables (reference opcodes)
    'x': ('VAR_X', OP_VAR_X, 'Variable x: push x'),
    'y': ('VAR_Y', OP_VAR_Y, 'Variable y: push y'),
    'z': ('VAR_Z', OP_VAR_Z, 'Variable z: push z'),
    'w': ('VAR_W', OP_VAR_W, 'Variable w: push w'),

    # Stack operations (for completeness)
    'dup': ('DUP', OP_DUP, 'Duplicate top of stack'),
    'swap': ('SWAP', OP_SWAP, 'Swap top two stack items'),
    'drop': ('DROP', OP_DROP, 'Drop top of stack'),

    # Absolute value and norms
    '|': ('ABS', OP_ABS, 'Absolute value: |x|'),
    '‖': ('NORM', OP_VEC_L2_NORM, 'Vector L2 norm: ‖v‖'),

    # Rounding
    '⌊': ('FLOOR', OP_FLOOR, 'Floor: ⌊x⌋'),
    '⌈': ('CEIL', OP_CEIL, 'Ceiling: ⌈x⌉'),

    # Limits and special
    'lim': ('LIMIT', OP_LIMIT, 'Limit: lim f(x)'),
    'max': ('MAX', OP_MAX, 'Maximum: max(a, b)'),
    'min': ('MIN', OP_MIN, 'Minimum: min(a, b)'),
    'mod': ('MODULO', OP_MOD, 'Modulo: a mod b'),

    # Greek letters (commonly used in math, some already covered)
    'α': ('ALPHA_VAR', OP_VAR_X, 'Variable alpha (mapped to x)'),
    'θ': ('THETA_VAR', OP_VAR_X, 'Variable theta (angle, mapped to x)'),
    'λ': ('LAMBDA_VAR', OP_VAR_X, 'Variable lambda (eigenvalue, mapped to x)'),
    'μ': ('MU_VAR', OP_MEAN, 'Mean (mu): E[X]'),
    'σ': ('SIGMA_VAR', [OP_VARIANCE, OP_SQRT], 'Standard deviation: σ = √Var'),
    'Σ': ('SIGMA_SUM', OP_SERIES_SUM, 'Summation: Σ'),
    'Π': ('PI_PRODUCT', OP_SERIES_PRODUCT, 'Product: Π'),
    'Δ': ('DELTA', OP_SUB, 'Delta (difference): Δ = b - a'),
    'ε': ('EPSILON', [0xE4, 1e-6], 'Epsilon (small constant)'),
    'φ': ('PHI', [0xE4, 1.618033988749895], 'Golden ratio: φ'),
    'ψ': ('PSI_VAR', OP_VAR_X, 'Variable psi (wavefunction)'),
    'ω': ('OMEGA', [0xE4, 2.0 * 3.14159265358979], 'Angular frequency: 2π'),

    # =========================================================================
    # COMPOSITIONAL OPERATIONS (built from atomic operations)
    # =========================================================================

    # Advanced roots (compositional)
    '∜': ('FOURTHROOT', [OP_SQRT, OP_SQRT], 'Fourth root: ∜x = √√x'),
    'ⁿ√': ('NTHROOT', [0xE4, 1.0, OP_SWAP, OP_DIV, OP_MUL], 'Nth root: x^(1/n)'),

    # Hyperbolic inverses (compositional)
    'arcsinh': ('ARCSINH', [OP_DUP, OP_MUL, 0xE4, 1.0, OP_ADD, OP_SQRT, OP_ADD, OP_LOG],
                'Arc hyperbolic sine: arcsinh(x) = ln(x + √(x²+1))'),
    'arccosh': ('ARCCOSH', [OP_DUP, OP_MUL, 0xE4, 1.0, OP_SUB, OP_SQRT, OP_ADD, OP_LOG],
                'Arc hyperbolic cosine: arccosh(x) = ln(x + √(x²-1))'),
    'arctanh': ('ARCTANH', [OP_DUP, 0xE4, 1.0, OP_ADD, OP_SWAP, 0xE4, 1.0, OP_SWAP, OP_SUB, OP_DIV, OP_LOG, 0xE4, 0.5, OP_MUL],
                'Arc hyperbolic tangent: arctanh(x) = 0.5·ln((1+x)/(1-x))'),

    # Trigonometric variants (compositional)
    'sec': ('SECANT', [OP_COS, 0xE4, 1.0, OP_SWAP, OP_DIV], 'Secant: sec(x) = 1/cos(x)'),
    'csc': ('COSECANT', [OP_SIN, 0xE4, 1.0, OP_SWAP, OP_DIV], 'Cosecant: csc(x) = 1/sin(x)'),
    'cot': ('COTANGENT', [OP_TAN, 0xE4, 1.0, OP_SWAP, OP_DIV], 'Cotangent: cot(x) = 1/tan(x)'),
    'arcsec': ('ARCSECANT', [0xE4, 1.0, OP_SWAP, OP_DIV, OP_ACOS], 'Arc secant: arcsec(x) = acos(1/x)'),
    'arccsc': ('ARCCOSECANT', [0xE4, 1.0, OP_SWAP, OP_DIV, OP_ASIN], 'Arc cosecant: arccsc(x) = asin(1/x)'),
    'arccot': ('ARCCOTANGENT', [0xE4, 1.0, OP_SWAP, OP_DIV, OP_ATAN], 'Arc cotangent: arccot(x) = atan(1/x)'),

    # ML activation functions (compositional)
    'sigmoid': ('SIGMOID', [OP_SUB, OP_EXP, 0xE4, 1.0, OP_ADD, 0xE4, 1.0, OP_SWAP, OP_DIV],
                'Sigmoid function: σ(x) = 1/(1+e^(-x))'),
    'softplus': ('SOFTPLUS', [OP_EXP, 0xE4, 1.0, OP_ADD, OP_LOG],
                 'Softplus function: softplus(x) = ln(1+e^x)'),
    'relu': ('RELU', [0xE4, 0.0, OP_MAX], 'ReLU activation: relu(x) = max(0,x)'),
    'leaky_relu': ('LEAKY_RELU', [OP_DUP, 0xE4, 0.01, OP_MUL, OP_SWAP, OP_MAX],
                   'Leaky ReLU: leaky_relu(x) = max(0.01x, x)'),
    'gelu': ('GELU', [OP_DUP, 0xE4, 0.5, OP_MUL, OP_SWAP, 0xE4, 1.702, OP_MUL, OP_TANH, 0xE4, 1.0, OP_ADD, OP_MUL],
             'GELU activation: gelu(x) ≈ 0.5x(1+tanh(1.702x))'),

    # Statistical functions (compositional)
    'variance': ('VARIANCE', OP_VARIANCE, 'Variance: Var(X) = E[(X-μ)²]'),
    'std_dev': ('STD_DEV', [OP_VARIANCE, OP_SQRT], 'Standard deviation: σ = √Var'),
    'covariance': ('COVARIANCE', OP_DOT_PRODUCT, 'Covariance: Cov(X,Y) = E[(X-μx)(Y-μy)]'),

    # Number theory (algorithmic/compositional)
    'gcd': ('GCD', OP_LOOP, 'Greatest common divisor: gcd(a,b) via Euclidean algorithm'),
    'lcm': ('LCM', [OP_DUP, OP_MUL, OP_ABS, OP_SWAP, OP_LOOP, OP_DIV],
            'Least common multiple: lcm(a,b) = |a·b|/gcd(a,b)'),

    # Combinatorics (compositional)
    'permutation': ('PERMUTATION', [OP_SWAP, OP_DUP, OP_SWAP, OP_SUB, OP_FACTORIAL, OP_SWAP, OP_FACTORIAL, OP_DIV],
                    'Permutations: P(n,r) = n!/(n-r)!'),

    # Matrix operations (compositional)
    'frobenius': ('FROBENIUS_NORM', [OP_DUP, OP_MATRIX_TRANSPOSE, OP_MATRIX_MULT, OP_TRACE_TENSOR, OP_SQRT],
                  'Frobenius norm: ||A||_F = √tr(A^T A)'),
    'matrix_exp': ('MATRIX_EXPONENTIAL', OP_SERIES_SUM, 'Matrix exponential: exp(A) = Σ A^k/k!'),

    # Differential operations (compositional)
    'second_deriv': ('SECOND_DERIVATIVE', [OP_SYMBOLIC_DIFF, OP_SYMBOLIC_DIFF],
                     'Second derivative: f\'\'(x) = d²f/dx²'),
    'third_deriv': ('THIRD_DERIVATIVE', [OP_SYMBOLIC_DIFF, OP_SYMBOLIC_DIFF, OP_SYMBOLIC_DIFF],
                    'Third derivative: f\'\'\'(x) = d³f/dx³'),

    # Ternary operations (NEW - semantic-rich)
    'sign_ternary': ('SIGN_TERNARY', 0xF5, 'Ternary sign function: sgn₃(x) ∈ {-1, 0, +1}'),
    'compare_ternary': ('COMPARE_TERNARY', 0xF6, 'Ternary comparison: cmp₃(a,b) ∈ {<, =, >}'),

    # Logarithms (different bases, compositional)
    'log_n': ('LOG_BASE_N', [OP_LOG, OP_SWAP, OP_LOG, OP_DIV],
              'Logarithm base n: log_n(x) = ln(x)/ln(n)'),
}


# =============================================================================
# PROCEDURAL OUTLINE EXTRACTION
# =============================================================================

def extract_glyph_outline_rpn(font: TTFont, char: str) -> Optional[str]:
    """
    Extract glyph outline as RPN drawing commands.

    Traces font outline procedurally using quadratic/cubic Bézier curves.

    Args:
        font: Loaded TTFont object
        char: Character to extract

    Returns:
        RPN drawing command string (e.g., "0.1 0.2 MOVE 0.5 0.8 LINE STROKE")
        or None if glyph not found
    """
    try:
        # Get character map
        cmap = font.getBestCmap()
        if cmap is None:
            return None

        # Get glyph name for character
        char_code = ord(char)
        if char_code not in cmap:
            logger.debug(f"Character '{char}' (U+{char_code:04X}) not in font")
            return None

        glyph_name = cmap[char_code]
        glyph_set = font.getGlyphSet()

        if glyph_name not in glyph_set:
            return None

        # Use RecordingPen to capture drawing operations
        pen = RecordingPen()
        glyph_set[glyph_name].draw(pen)

        # Get font units per em for normalization
        units_per_em = font['head'].unitsPerEm

        # Convert pen operations to RPN
        rpn_commands = []

        for op, args in pen.value:
            if op == 'moveTo':
                # Normalize coordinates to 0-1 range
                x, y = args[0]
                nx = x / units_per_em
                ny = y / units_per_em
                rpn_commands.append(f"{nx:.4f} {ny:.4f} MOVE")

            elif op == 'lineTo':
                x, y = args[0]
                nx = x / units_per_em
                ny = y / units_per_em
                rpn_commands.append(f"{nx:.4f} {ny:.4f} LINE")

            elif op == 'qCurveTo':
                # Quadratic Bézier (TrueType native)
                # args is list of control points + endpoint
                if len(args) == 2:
                    # Simple quadratic: control point, endpoint
                    cx, cy = args[0]
                    ex, ey = args[1]
                    ncx = cx / units_per_em
                    ncy = cy / units_per_em
                    nex = ex / units_per_em
                    ney = ey / units_per_em
                    rpn_commands.append(f"{ncx:.4f} {ncy:.4f} {nex:.4f} {ney:.4f} QUAD")
                else:
                    # Multiple control points (convert to multiple quads)
                    for i in range(0, len(args) - 1, 2):
                        if i + 1 < len(args):
                            cx, cy = args[i]
                            ex, ey = args[i + 1]
                            ncx = cx / units_per_em
                            ncy = cy / units_per_em
                            nex = ex / units_per_em
                            ney = ey / units_per_em
                            rpn_commands.append(f"{ncx:.4f} {ncy:.4f} {nex:.4f} {ney:.4f} QUAD")

            elif op == 'curveTo':
                # Cubic Bézier (PostScript/OpenType)
                if len(args) == 3:
                    c1x, c1y = args[0]
                    c2x, c2y = args[1]
                    ex, ey = args[2]
                    nc1x = c1x / units_per_em
                    nc1y = c1y / units_per_em
                    nc2x = c2x / units_per_em
                    nc2y = c2y / units_per_em
                    nex = ex / units_per_em
                    ney = ey / units_per_em
                    rpn_commands.append(
                        f"{nc1x:.4f} {nc1y:.4f} {nc2x:.4f} {nc2y:.4f} "
                        f"{nex:.4f} {ney:.4f} CUBIC"
                    )

            elif op == 'closePath':
                rpn_commands.append("CLOSE")

        # Add stroke command if we have any paths
        if rpn_commands:
            rpn_commands.append("STROKE")
            return ' '.join(rpn_commands)

        return None

    except Exception as e:
        logger.debug(f"Error extracting '{char}': {e}")
        return None


def opcode_to_rpn_string(opcode_data) -> str:
    """Convert opcode data to RPN string representation."""
    if isinstance(opcode_data, int):
        # Single opcode - use hex representation
        return f"0x{opcode_data:02X}"
    elif isinstance(opcode_data, list):
        # Multiple opcodes
        tokens = []
        for op in opcode_data:
            if isinstance(op, int):
                tokens.append(f"0x{op:02X}")
            elif isinstance(op, float):
                tokens.append(f"{op}")
            else:
                tokens.append(str(op))
        return ' '.join(tokens)
    else:
        return str(opcode_data)


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_math_symbols_from_font(
    font_path: Path,
    output_records: List[Dict]
) -> int:
    """
    Extract all math symbols from a single font file.

    Returns:
        Number of symbols extracted
    """
    logger.info(f"Processing font: {font_path.name}")

    try:
        font = TTFont(font_path)
    except Exception as e:
        logger.error(f"Failed to load font {font_path}: {e}")
        return 0

    extracted_count = 0

    for symbol, (name, opcode, semantic) in MATH_SYMBOL_MAPPING.items():
        # Extract visual RPN (procedural outline)
        visual_rpn = extract_glyph_outline_rpn(font, symbol)

        if visual_rpn is None:
            # Symbol not in this font
            continue

        # Convert opcode to RPN string
        math_rpn = opcode_to_rpn_string(opcode)

        # Create dual-modal record
        record = {
            'symbol': symbol,
            'unicode': f"U+{ord(symbol):04X}",
            'name': name,
            'visual_rpn': visual_rpn,
            'math_rpn': math_rpn,
            'semantic': semantic,
            'font_name': font_path.stem,
            'category': categorize_symbol(symbol),
            'multivariate': is_multivariate(opcode),
        }

        output_records.append(record)
        extracted_count += 1

    font.close()
    logger.info(f"  Extracted {extracted_count} symbols from {font_path.name}")

    return extracted_count


def categorize_symbol(symbol: str) -> str:
    """Categorize math symbol for organization."""
    if symbol in '+-−×·÷/':
        return 'arithmetic'
    elif symbol in 'sincostanexp√^²³':
        return 'functions'
    elif symbol in '∇∂∫∑∏∆':
        return 'calculus'
    elif symbol in '=≠<>≤≥≈':
        return 'relations'
    elif symbol in '∈∉∪∩∖⊂⊆':
        return 'set_theory'
    elif symbol in '∧∨¬⊕→↔∀∃':
        return 'logic'
    elif symbol in 'xyz wαθλμσ':
        return 'variables'
    elif symbol in 'πeφω':
        return 'constants'
    elif symbol in '⋅×⊗⊤⁻¹':
        return 'linear_algebra'
    else:
        return 'other'


def is_multivariate(opcode) -> bool:
    """Check if opcode supports multivariate operations."""
    if isinstance(opcode, int):
        # Multivariate opcodes
        return opcode in [
            OP_GRADIENT, OP_DIVERGENCE, OP_CURL, OP_LAPLACIAN,
            OP_VAR_X, OP_VAR_Y, OP_VAR_Z, OP_VAR_W
        ]
    elif isinstance(opcode, list):
        # Check if any element is multivariate
        return any(is_multivariate(op) for op in opcode if isinstance(op, int))
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract math symbols from fonts as dual-modal RPN representations'
    )
    parser.add_argument(
        '--font-dir',
        type=Path,
        default=Path('/K3D/Knowledge3D.local/fonts/math/'),
        help='Directory containing math fonts (.otf, .ttf)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/K3D/Knowledge3D.local/datasets/math_symbols_procedural.jsonl'),
        help='Output JSONL file'
    )
    parser.add_argument(
        '--max-fonts',
        type=int,
        default=None,
        help='Limit number of fonts to process (for testing)'
    )

    args = parser.parse_args()

    # Find all font files
    font_files = []
    for ext in ['*.otf', '*.ttf', '*.OTF', '*.TTF']:
        font_files.extend(args.font_dir.glob(ext))

    if not font_files:
        logger.error(f"No font files found in {args.font_dir}")
        return 1

    font_files = sorted(font_files)
    if args.max_fonts:
        font_files = font_files[:args.max_fonts]

    logger.info(f"Found {len(font_files)} font files")
    logger.info(f"Math symbol mapping: {len(MATH_SYMBOL_MAPPING)} symbols")

    # Extract symbols from all fonts
    all_records = []
    total_extracted = 0

    for font_path in font_files:
        count = extract_math_symbols_from_font(font_path, all_records)
        total_extracted += count

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction complete!")
    logger.info(f"  Total symbols extracted: {total_extracted}")
    logger.info(f"  Unique symbols: {len(set(r['symbol'] for r in all_records))}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"{'='*60}")

    # Print category statistics
    from collections import Counter
    category_counts = Counter(r['category'] for r in all_records)
    logger.info("\nSymbols by category:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat:20s}: {count:4d}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
