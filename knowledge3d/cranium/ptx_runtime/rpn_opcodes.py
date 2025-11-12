"""Shared opcode constants for the modular RPN executors."""

# Literals
OP_LITERAL_SCALAR = 0x00
OP_LITERAL_VECTOR = 0x01
OP_POINTER_LITERAL = 0x03

# Basic arithmetic (Tier-1)
OP_ADD = 0x0A
OP_SUB = 0x0B
OP_MUL = 0x0C
OP_DIV = 0x0D

# Math functions (Tier-1)
OP_SQRT = 0x14
OP_EXP = 0x15
OP_LOG = 0x16
OP_SIN = 0x18
OP_COS = 0x19
OP_TAN = 0x1A

# Phase 2: Inverse & Hyperbolic Trig (Tier-1)
OP_ASIN = 0x1B
OP_ACOS = 0x1C
OP_ATAN = 0x1D
OP_ATAN2 = 0x1E
OP_SINH = 0x1F
OP_COSH = 0x25
OP_TANH = 0x26

# Phase 2: Rounding & Absolute (Tier-1)
OP_ABS = 0x27
OP_CEIL = 0x29
OP_FLOOR = 0x2B
OP_ROUND = 0x2D

# Comparisons (Tier-1)
OP_GT = 0x28
OP_LT = 0x2A
OP_EQ = 0x2C
OP_MAX = 0x2E
OP_MIN = 0x2F

# Stack operations (Tier-1)
OP_DUP = 0x32
OP_SWAP = 0x33
OP_DROP = 0x34

# Phase 2: Modulo & Logarithms (Tier-1)
OP_MOD = 0x38
OP_LOG2 = 0x39
OP_LOG10 = 0x3A
OP_COMPLEX_REAL = 0x3B
OP_COMPLEX_IMAG = 0x3C
OP_COMPLEX_CONJ = 0x3D
OP_COMPLEX_ARG = 0x3E

OP_SPARSE_LOAD = 0x40
OP_SMAV = 0x41
OP_ENTROPY_SUM = 0x42
OP_SIGMOID_APPROX = 0x43

# Phase 1A – TRM integration opcodes (Tier-3 execution surface)
OP_TRM_MATVEC_512x1024 = 0x60
OP_TRM_MATVEC_1024x512 = 0x61
OP_TRM_VEC_ADD3_512 = 0x62
OP_TRM_SWIGLU_512 = 0x63
OP_TRM_SWIGLU_1024 = 0x64

# Phase 2: Bitwise Logic (Tier-1)
OP_AND = 0x80
OP_OR = 0x81
OP_XOR = 0x82
OP_NOT = 0x83

# Tier-2 cooperative ops
OP_MEMCPY_F32 = 0x90
OP_FILL_F32 = 0x91
OP_REDUCE_SUM_F32 = 0x92
OP_REDUCE_MAX_F32 = 0x93
OP_REDUCE_MIN_F32 = 0x94
OP_MEAN = 0x95
OP_MEDIAN = 0x96
OP_VARIANCE = 0x97
OP_MATVEC_F32 = 0xA0
OP_VECTOR_RELU = 0xA1
OP_VECTOR_MUL_F32 = 0xA2
OP_VECTOR_SIGMOID = 0xA3

# Matrix ops for swarm coordination
OP_MATMUL_SMALL = 0xA4
OP_DOT_BATCH = 0xA5
OP_TRACE_TENSOR = 0xA6
OP_MATRIX_DET = 0xA7
OP_MATRIX_INV = 0xA8
OP_MATRIX_TRANSPOSE = 0xA9
OP_MATRIX_MULT = 0xAA
OP_GAMMA = 0xAB
OP_FACTORIAL = 0xAC
OP_BINOMIAL = 0xAD
OP_BETA = 0xAE

# Programmability opcodes (Step 13-E foundations)
OP_BRANCH = 0xB0
OP_LOOP = 0xB1
OP_NEXT = 0xB2
OP_STORE = 0xB3
OP_RECALL = 0xB4
OP_SYMBOLIC_DIFF = 0xB5
OP_GRADIENT = 0xB6
OP_SYMBOLIC_INTEGRATE = 0xB7
OP_LIMIT = 0xB9
OP_SERIES_SUM = 0xBA
OP_SERIES_PRODUCT = 0xBB
OP_DIVERGENCE = 0xBC
OP_CURL = 0xBD
OP_LAPLACIAN = 0xBE

# Sovereign clustering opcodes (Tier 1-2)
OP_VEC_L2_NORM = 0xC0
OP_VEC_NORMALIZE = 0xC1
OP_VEC_ARGMAX = 0xC2
OP_VEC_BLEND = 0xC3
OP_COSINE_SIM_BATCH = 0xC4
OP_CLUSTER_ASSIGN = 0xC5
OP_SET_UNION = 0xC6
OP_SET_INTERSECTION = 0xC7
OP_SET_DIFFERENCE = 0xC8
OP_SET_CARTESIAN = 0xC9
OP_DOT_PRODUCT = 0xCA
OP_CROSS_PRODUCT = 0xCB
OP_OUTER_PRODUCT = 0xCC
OP_EIGENVALUES = 0xCD
OP_SVD_SMALL = 0xCE
OP_QR_DECOMP = 0xCF
OP_CHOLESKY = 0xD0
OP_LU_DECOMP = 0xD1
OP_QUANTUM_SUPERPOSE = 0xD2
OP_QUANTUM_MEASURE = 0xD3
OP_QUANTUM_ENTANGLE = 0xD4
OP_QUANTUM_PHASE = 0xD5
OP_QUANTUM_HADAMARD = 0xD6
OP_QUANTUM_CNOT = 0xD7

# Multivariate variable reference opcodes (Tier 0: 1 cycle)
OP_VAR_X = 0xE0
OP_VAR_Y = 0xE1
OP_VAR_Z = 0xE2
OP_VAR_W = 0xE3
OP_CONST = 0xE4

# Temporal reasoning opcodes (Phase 1C)
OP_TEMPORAL_COHERENCE = 0xF0
OP_TEMPORAL_MASK = 0xF1
OP_TEMPORAL_AGGREGATE = 0xF2

__all__ = [
    # Literals
    "OP_LITERAL_SCALAR",
    "OP_LITERAL_VECTOR",
    "OP_POINTER_LITERAL",
    # Basic arithmetic
    "OP_ADD",
    "OP_SUB",
    "OP_MUL",
    "OP_DIV",
    # Math functions
    "OP_SQRT",
    "OP_EXP",
    "OP_LOG",
    "OP_SIN",
    "OP_COS",
    "OP_TAN",
    # Phase 2: Inverse & Hyperbolic Trig
    "OP_ASIN",
    "OP_ACOS",
    "OP_ATAN",
    "OP_ATAN2",
    "OP_SINH",
    "OP_COSH",
    "OP_TANH",
    # Phase 2: Rounding & Absolute
    "OP_ABS",
    "OP_CEIL",
    "OP_FLOOR",
    "OP_ROUND",
    # Comparisons
    "OP_GT",
    "OP_LT",
    "OP_EQ",
    "OP_MAX",
    "OP_MIN",
    # Stack operations
    "OP_DUP",
    "OP_SWAP",
    "OP_DROP",
    # Phase 2: Modulo & Logarithms
    "OP_MOD",
    "OP_LOG2",
    "OP_LOG10",
    # Phase 4: Complex helpers
    "OP_COMPLEX_REAL",
    "OP_COMPLEX_IMAG",
    "OP_COMPLEX_CONJ",
    "OP_COMPLEX_ARG",
    # Special ops
    "OP_SPARSE_LOAD",
    "OP_SMAV",
    "OP_ENTROPY_SUM",
    "OP_SIGMOID_APPROX",
    # TRM integration
    "OP_TRM_MATVEC_512x1024",
    "OP_TRM_MATVEC_1024x512",
    "OP_TRM_VEC_ADD3_512",
    "OP_TRM_SWIGLU_512",
    "OP_TRM_SWIGLU_1024",
    # Phase 2: Bitwise Logic
    "OP_AND",
    "OP_OR",
    "OP_XOR",
    "OP_NOT",
    # Tier-2 ops
    "OP_MEMCPY_F32",
    "OP_FILL_F32",
    "OP_REDUCE_SUM_F32",
    "OP_REDUCE_MAX_F32",
    "OP_REDUCE_MIN_F32",
    "OP_MEAN",
    "OP_MEDIAN",
    "OP_VARIANCE",
    "OP_MATVEC_F32",
    "OP_VECTOR_RELU",
    "OP_VECTOR_MUL_F32",
    "OP_VECTOR_SIGMOID",
    # Matrix ops
    "OP_MATMUL_SMALL",
    "OP_DOT_BATCH",
    "OP_TRACE_TENSOR",
    "OP_MATRIX_DET",
    "OP_MATRIX_INV",
    "OP_MATRIX_TRANSPOSE",
    "OP_MATRIX_MULT",
    "OP_GAMMA",
    "OP_FACTORIAL",
    "OP_BINOMIAL",
    "OP_BETA",
    # Programmability
    "OP_BRANCH",
    "OP_LOOP",
    "OP_NEXT",
    "OP_STORE",
    "OP_RECALL",
    "OP_SYMBOLIC_DIFF",
    "OP_GRADIENT",
    "OP_SYMBOLIC_INTEGRATE",
    "OP_LIMIT",
    "OP_SERIES_SUM",
    "OP_SERIES_PRODUCT",
    "OP_DIVERGENCE",
    "OP_CURL",
    "OP_LAPLACIAN",
    # Clustering
    "OP_VEC_L2_NORM",
    "OP_VEC_NORMALIZE",
    "OP_VEC_ARGMAX",
    "OP_VEC_BLEND",
    "OP_COSINE_SIM_BATCH",
    "OP_CLUSTER_ASSIGN",
    "OP_SET_UNION",
    "OP_SET_INTERSECTION",
    "OP_SET_DIFFERENCE",
    "OP_SET_CARTESIAN",
    "OP_DOT_PRODUCT",
    "OP_CROSS_PRODUCT",
    "OP_OUTER_PRODUCT",
    "OP_EIGENVALUES",
    "OP_SVD_SMALL",
    "OP_QR_DECOMP",
    "OP_CHOLESKY",
    "OP_LU_DECOMP",
    # Quantum operations
    "OP_QUANTUM_SUPERPOSE",
    "OP_QUANTUM_MEASURE",
    "OP_QUANTUM_ENTANGLE",
    "OP_QUANTUM_PHASE",
    "OP_QUANTUM_HADAMARD",
    "OP_QUANTUM_CNOT",
    # Multivariate variable references
    "OP_VAR_X",
    "OP_VAR_Y",
    "OP_VAR_Z",
    "OP_VAR_W",
    "OP_CONST",
    # Temporal reasoning
    "OP_TEMPORAL_COHERENCE",
    "OP_TEMPORAL_MASK",
    "OP_TEMPORAL_AGGREGATE",
]
