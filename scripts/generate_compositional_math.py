#!/usr/bin/env python3
"""
Generate Compositional Math Operations Dataset

Creates a comprehensive dataset of compositional math operations that:
1. Uses extracted font symbols where available (e.g., ∜, σ)
2. Generates synthetic entries for text-only operations (e.g., "arcsinh", "sigmoid")
3. Includes full semantic descriptions for text-to-math grounding

This enables the model to understand both:
- Visual symbols (drawing)
- Text names (semantic)
- Execution (RPN bytecode)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import *


# Compositional math operations (text-based, not Unicode symbols)
COMPOSITIONAL_TEXT_OPERATIONS = {
    # Hyperbolic inverses
    'arcsinh': {
        'name': 'ARCSINH',
        'math_rpn': [OP_DUP, OP_MUL, 0xE4, 1.0, OP_ADD, OP_SQRT, OP_ADD, OP_LOG],
        'semantic': 'Arc hyperbolic sine: arcsinh(x) = ln(x + √(x²+1))',
        'category': 'hyperbolic_inverse',
        'compositional_from': ['DUP', 'MUL', 'CONST', 'ADD', 'SQRT', 'LOG'],
    },
    'arccosh': {
        'name': 'ARCCOSH',
        'math_rpn': [OP_DUP, OP_MUL, 0xE4, 1.0, OP_SUB, OP_SQRT, OP_ADD, OP_LOG],
        'semantic': 'Arc hyperbolic cosine: arccosh(x) = ln(x + √(x²-1))',
        'category': 'hyperbolic_inverse',
        'compositional_from': ['DUP', 'MUL', 'CONST', 'SUB', 'SQRT', 'ADD', 'LOG'],
    },
    'arctanh': {
        'name': 'ARCTANH',
        'math_rpn': [OP_DUP, 0xE4, 1.0, OP_ADD, OP_SWAP, 0xE4, 1.0, OP_SWAP, OP_SUB, OP_DIV, OP_LOG, 0xE4, 0.5, OP_MUL],
        'semantic': 'Arc hyperbolic tangent: arctanh(x) = 0.5·ln((1+x)/(1-x))',
        'category': 'hyperbolic_inverse',
        'compositional_from': ['DUP', 'CONST', 'ADD', 'SWAP', 'SUB', 'DIV', 'LOG', 'MUL'],
    },

    # Trigonometric variants
    'sec': {
        'name': 'SECANT',
        'math_rpn': [OP_COS, 0xE4, 1.0, OP_SWAP, OP_DIV],
        'semantic': 'Secant: sec(x) = 1/cos(x)',
        'category': 'trigonometric_variant',
        'compositional_from': ['COS', 'CONST', 'SWAP', 'DIV'],
    },
    'csc': {
        'name': 'COSECANT',
        'math_rpn': [OP_SIN, 0xE4, 1.0, OP_SWAP, OP_DIV],
        'semantic': 'Cosecant: csc(x) = 1/sin(x)',
        'category': 'trigonometric_variant',
        'compositional_from': ['SIN', 'CONST', 'SWAP', 'DIV'],
    },
    'cot': {
        'name': 'COTANGENT',
        'math_rpn': [OP_TAN, 0xE4, 1.0, OP_SWAP, OP_DIV],
        'semantic': 'Cotangent: cot(x) = 1/tan(x)',
        'category': 'trigonometric_variant',
        'compositional_from': ['TAN', 'CONST', 'SWAP', 'DIV'],
    },
    'arcsec': {
        'name': 'ARCSECANT',
        'math_rpn': [0xE4, 1.0, OP_SWAP, OP_DIV, OP_ACOS],
        'semantic': 'Arc secant: arcsec(x) = acos(1/x)',
        'category': 'trigonometric_variant',
        'compositional_from': ['CONST', 'SWAP', 'DIV', 'ACOS'],
    },
    'arccsc': {
        'name': 'ARCCOSECANT',
        'math_rpn': [0xE4, 1.0, OP_SWAP, OP_DIV, OP_ASIN],
        'semantic': 'Arc cosecant: arccsc(x) = asin(1/x)',
        'category': 'trigonometric_variant',
        'compositional_from': ['CONST', 'SWAP', 'DIV', 'ASIN'],
    },
    'arccot': {
        'name': 'ARCCOTANGENT',
        'math_rpn': [0xE4, 1.0, OP_SWAP, OP_DIV, OP_ATAN],
        'semantic': 'Arc cotangent: arccot(x) = atan(1/x)',
        'category': 'trigonometric_variant',
        'compositional_from': ['CONST', 'SWAP', 'DIV', 'ATAN'],
    },

    # ML activation functions
    'sigmoid': {
        'name': 'SIGMOID',
        'math_rpn': [OP_SUB, OP_EXP, 0xE4, 1.0, OP_ADD, 0xE4, 1.0, OP_SWAP, OP_DIV],
        'semantic': 'Sigmoid function: σ(x) = 1/(1+e^(-x))',
        'category': 'ml_activation',
        'compositional_from': ['SUB', 'EXP', 'CONST', 'ADD', 'SWAP', 'DIV'],
        'text_aliases': ['σ', 'logistic'],
    },
    'softplus': {
        'name': 'SOFTPLUS',
        'math_rpn': [OP_EXP, 0xE4, 1.0, OP_ADD, OP_LOG],
        'semantic': 'Softplus function: softplus(x) = ln(1+e^x)',
        'category': 'ml_activation',
        'compositional_from': ['EXP', 'CONST', 'ADD', 'LOG'],
    },
    'relu': {
        'name': 'RELU',
        'math_rpn': [0xE4, 0.0, OP_MAX],
        'semantic': 'ReLU activation: relu(x) = max(0,x)',
        'category': 'ml_activation',
        'compositional_from': ['CONST', 'MAX'],
    },
    'leaky_relu': {
        'name': 'LEAKY_RELU',
        'math_rpn': [OP_DUP, 0xE4, 0.01, OP_MUL, OP_SWAP, OP_MAX],
        'semantic': 'Leaky ReLU: leaky_relu(x) = max(0.01x, x)',
        'category': 'ml_activation',
        'compositional_from': ['DUP', 'CONST', 'MUL', 'SWAP', 'MAX'],
    },
    'gelu': {
        'name': 'GELU',
        'math_rpn': [OP_DUP, 0xE4, 0.5, OP_MUL, OP_SWAP, 0xE4, 1.702, OP_MUL, OP_TANH, 0xE4, 1.0, OP_ADD, OP_MUL],
        'semantic': 'GELU activation: gelu(x) ≈ 0.5x(1+tanh(1.702x))',
        'category': 'ml_activation',
        'compositional_from': ['DUP', 'CONST', 'MUL', 'SWAP', 'TANH', 'ADD'],
    },

    # Statistical functions
    'std_dev': {
        'name': 'STD_DEV',
        'math_rpn': [OP_VARIANCE, OP_SQRT],
        'semantic': 'Standard deviation: σ = √Var',
        'category': 'statistics',
        'compositional_from': ['VARIANCE', 'SQRT'],
        'text_aliases': ['standard deviation', 'std'],
    },

    # Number theory (algorithmic)
    'gcd': {
        'name': 'GCD',
        'math_rpn': [OP_LOOP],  # Simplified (full algorithm requires control flow)
        'semantic': 'Greatest common divisor: gcd(a,b) via Euclidean algorithm',
        'category': 'number_theory',
        'algorithmic': True,
        'text_aliases': ['greatest common divisor'],
    },
    'lcm': {
        'name': 'LCM',
        'math_rpn': [OP_DUP, OP_MUL, OP_ABS, OP_SWAP, OP_LOOP, OP_DIV],
        'semantic': 'Least common multiple: lcm(a,b) = |a·b|/gcd(a,b)',
        'category': 'number_theory',
        'compositional_from': ['DUP', 'MUL', 'ABS', 'SWAP', 'LOOP', 'DIV'],
        'text_aliases': ['least common multiple'],
    },

    # Combinatorics
    'permutation': {
        'name': 'PERMUTATION',
        'math_rpn': [OP_SWAP, OP_DUP, OP_SWAP, OP_SUB, OP_FACTORIAL, OP_SWAP, OP_FACTORIAL, OP_DIV],
        'semantic': 'Permutations: P(n,r) = n!/(n-r)!',
        'category': 'combinatorics',
        'compositional_from': ['SWAP', 'DUP', 'SUB', 'FACTORIAL', 'DIV'],
        'text_aliases': ['P', 'nPr'],
    },

    # Matrix operations
    'frobenius_norm': {
        'name': 'FROBENIUS_NORM',
        'math_rpn': [OP_DUP, OP_MATRIX_TRANSPOSE, OP_MATRIX_MULT, OP_TRACE_TENSOR, OP_SQRT],
        'semantic': 'Frobenius norm: ||A||_F = √tr(A^T A)',
        'category': 'linear_algebra',
        'compositional_from': ['DUP', 'TRANSPOSE', 'MATRIX_MULT', 'TRACE', 'SQRT'],
        'text_aliases': ['||A||_F', 'matrix norm'],
    },

    # Differential operations
    'second_derivative': {
        'name': 'SECOND_DERIVATIVE',
        'math_rpn': [OP_SYMBOLIC_DIFF, OP_SYMBOLIC_DIFF],
        'semantic': 'Second derivative: f\'\'(x) = d²f/dx²',
        'category': 'calculus',
        'compositional_from': ['SYMBOLIC_DIFF', 'SYMBOLIC_DIFF'],
        'text_aliases': ['f\'\'', 'd2f/dx2', 'second derivative'],
    },
    'third_derivative': {
        'name': 'THIRD_DERIVATIVE',
        'math_rpn': [OP_SYMBOLIC_DIFF, OP_SYMBOLIC_DIFF, OP_SYMBOLIC_DIFF],
        'semantic': 'Third derivative: f\'\'\'(x) = d³f/dx³',
        'category': 'calculus',
        'compositional_from': ['SYMBOLIC_DIFF', 'SYMBOLIC_DIFF', 'SYMBOLIC_DIFF'],
        'text_aliases': ['f\'\'\'', 'd3f/dx3', 'third derivative'],
    },

    # Logarithms (different bases)
    'log_base_n': {
        'name': 'LOG_BASE_N',
        'math_rpn': [OP_LOG, OP_SWAP, OP_LOG, OP_DIV],
        'semantic': 'Logarithm base n: log_n(x) = ln(x)/ln(n)',
        'category': 'functions',
        'compositional_from': ['LOG', 'SWAP', 'DIV'],
        'text_aliases': ['log_n', 'logarithm base n'],
    },
}


def opcode_list_to_string(opcode_list):
    """Convert list of opcodes to hex string representation."""
    tokens = []
    for op in opcode_list:
        if isinstance(op, int):
            tokens.append(f"0x{op:02X}")
        elif isinstance(op, float):
            tokens.append(str(op))
        else:
            tokens.append(str(op))
    return ' '.join(tokens)


def generate_compositional_dataset(output_path: Path):
    """Generate compositional math operations dataset."""
    print(f"\n{'='*70}")
    print("COMPOSITIONAL MATH OPERATIONS GENERATION")
    print(f"{'='*70}\n")

    dataset = []

    for text_op, details in COMPOSITIONAL_TEXT_OPERATIONS.items():
        # Convert math_rpn list to string
        math_rpn_str = opcode_list_to_string(details['math_rpn'])

        # Create record
        record = {
            'operation': text_op,
            'name': details['name'],
            'math_rpn': math_rpn_str,
            'semantic': details['semantic'],
            'category': details['category'],
            'type': 'compositional_math',
            'compositional': True,
            'text_aliases': details.get('text_aliases', [text_op]),
            'compositional_from': details.get('compositional_from', []),
            'algorithmic': details.get('algorithmic', False),
        }

        dataset.append(record)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Statistics
    from collections import Counter
    category_counts = Counter(r['category'] for r in dataset)

    print(f"✅ Generated {len(dataset)} compositional operations")
    print(f"\nCategories:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:25s}: {count:3d}")
    print(f"\nOutput: {output_path}")
    print(f"\n{'='*70}\n")

    return len(dataset)


def main():
    """Generate compositional math dataset."""
    output_path = Path("/K3D/Knowledge3D.local/datasets/compositional_math_operations.jsonl")
    count = generate_compositional_dataset(output_path)

    print(f"Total compositional operations: {count}")
    print("These operations are built from atomic RPN opcodes")
    print("Each has semantic text description for grounding\n")


if __name__ == '__main__':
    main()
