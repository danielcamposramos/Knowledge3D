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
OP_POWER = 0x0E
OP_NEGATE = 0xDB

# Primitive alias names (calculator surface, same opcode values)
OP_PRIMITIVE_ADD = OP_ADD
OP_PRIMITIVE_SUBTRACT = OP_SUB
OP_PRIMITIVE_MULTIPLY = OP_MUL
OP_PRIMITIVE_DIVIDE = OP_DIV
OP_PRIMITIVE_POWER = OP_POWER
OP_PRIMITIVE_NEGATE = OP_NEGATE

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
OP_GTE = 0xDC
OP_COMPLEX_REAL = 0x3B
OP_COMPLEX_IMAG = 0x3C
OP_COMPLEX_CONJ = 0x3D
OP_COMPLEX_ARG = 0x3E

OP_SPARSE_LOAD = 0x40
OP_SMAV = 0x41
OP_ENTROPY_SUM = 0x42
OP_SIGMOID_APPROX = 0x43

# Phase H1: host-side 3D construction opcodes.
# The original H1 draft proposed using the full 0x44-0x5F block, but this tree
# already consumes 0x46-0x48 and 0x50-0x53 in the sovereign GPU kernel.
# These opcodes are therefore assigned to a collision-safe mixed surface and are
# routed through the host mesh engine until a GPU mesh kernel is promoted.
OP_VERTEX3 = 0x44
OP_NORMAL3 = 0x45
OP_UV2 = 0x49
OP_TRI_FACE = 0x4A
OP_QUAD_FACE = 0x4B
OP_FACE_NORMAL = 0x4C
OP_MESH_BEGIN = 0x4D
OP_MESH_END = 0x4E
OP_MAT4_IDENTITY = 0x4F
OP_MAT4_TRANSLATE = 0x54
OP_MAT4_SCALE = 0x55
OP_MAT4_ROTATE_X = 0x56
OP_MAT4_ROTATE_Y = 0x57
OP_MAT4_ROTATE_Z = 0x58
OP_MAT4_MUL = 0x59
OP_MAT4_APPLY = 0x5A
OP_GEN_PLANE = 0x5B
OP_GEN_CUBE = 0x5C
OP_GEN_UV_SPHERE = 0x5D
OP_GEN_CYLINDER = 0x5E
OP_GEN_CONE = 0x5F
OP_GEN_TORUS = 0x84
OP_GEN_ICOSPHERE = 0x85
OP_CSG_UNION = 0x86
OP_CSG_SUBTRACT = 0x87
OP_CSG_INTERSECT = 0x88
OP_EXTRUDE = 0x89
OP_LATHE = 0x8A

# Phase 1A – TRM integration opcodes (TRM internal execution surface)
OP_TRM_MATVEC_512x1024 = 0x300
OP_TRM_MATVEC_1024x512 = 0x301
OP_TRM_VEC_ADD3_512 = 0x302
OP_TRM_SWIGLU_512 = 0x303
OP_TRM_SWIGLU_1024 = 0x304

# Sovereign stack checkpoint ops (live modular kernel owners)
OP_CHECKPOINT = 0x60
OP_ROLLBACK = 0x61
OP_VERIFY = 0x62

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
OP_GCD = 0xD8
OP_NEG = 0xDB

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

# Knowledgeverse Galaxy opcodes (GPU-resident Galaxy access)
OP_LOAD_GALAXY = 0xE0
OP_GALAXY_SIMILARITY = 0xE1
OP_GALAXY_SCAN = 0xE2

# Multivariate variable reference opcodes (Tier 0: 1 cycle)
OP_VAR_X = 0xEA
OP_VAR_Y = 0xEB
OP_VAR_Z = 0xEC
OP_VAR_W = 0xED
OP_CONST = 0xEE

# Grammar evolution opcodes (cross-modality discovery → promotion)
OP_GRAMMAR_OBSERVE = 0xE5      # visual_emb text_emb → correlation score
OP_GRAMMAR_PROPOSE = 0xE6      # rpn_program context → rule_id (tentative)
OP_GRAMMAR_VALIDATE = 0xE7     # rule_id success → updated quality_score
OP_GRAMMAR_PROMOTE = 0xE8      # rule_id → move to shared if quality ≥ threshold
OP_GRAMMAR_QUERY = 0xE9        # embedding k → top-k matching rules

# Temporal reasoning opcodes (Phase 1C)
OP_TEMPORAL_COHERENCE = 0xF0
OP_TEMPORAL_MASK = 0xF1
OP_TEMPORAL_AGGREGATE = 0xF2

# Temporal/trit opcodes are the live modular-kernel owners of 0x70-0x76.
OP_TADD = 0x70
OP_TMUL = 0x71
OP_TNOT = 0x72
OP_TCOMP = 0x73
OP_TQUANT = 0x74
OP_TPACK = 0x75
OP_TUNPACK = 0x76

# Entity behavior opcodes (Reality Engine Step 3)
OP_BH_PERCEIVE = 0x180
OP_BH_SEEK = 0x181
OP_BH_FLEE = 0x182
OP_BH_ARRIVE = 0x183
OP_BH_SEPARATE = 0x184
OP_BH_APPLY_FORCE = 0x185
OP_BH_BT_TICK = 0x186
OP_BH_GOAP_PLAN = 0x188
OP_BH_SLEEP_CHECK = 0x189
OP_BH_BLACKBOARD_READ = 0x18A
OP_BH_BLACKBOARD_WRITE = 0x18B
OP_BH_PATHFIND = 0x18C

# ── Sovereign CAS opcodes (0x220-0x237) ──────────────────────
# Polynomial algebra layer (feeds the existing 0xB5-0xBE calculus stubs)
OP_POLY_COEFF = 0x220
OP_POLY_BUILD = 0x221
OP_POLY_ADD = 0x222
OP_POLY_MUL = 0x223
OP_POLY_DIV = 0x224
OP_POLY_REM = 0x225
OP_POLY_GCD = 0x226
OP_POLY_FACTOR = 0x227
# Simplification and transformation
OP_SIMPLIFY = 0x228
OP_SUBSTITUTE = 0x229
OP_COLLECT = 0x22A
OP_RATIONALIZE = 0x22B
OP_TRIG_SIMPLIFY = 0x22C
OP_LOG_SIMPLIFY = 0x22D
# Solving
OP_SOLVE_LINEAR = 0x22E
OP_SOLVE_QUADRATIC = 0x22F
OP_LINSOLVE = 0x230
# Pattern and rule application
OP_PATTERN_MATCH = 0x231
OP_RULE_APPLY = 0x232
OP_COEFF_EXTRACT = 0x233
# Expression construction helpers
OP_CAS_PUSH_SYM = 0x234
OP_CAS_PUSH_CONST = 0x235
OP_CAS_BUILD = 0x236
OP_CAS_EVAL = 0x237

# ── Sovereign Algebraic System (SAS) extension (0x238-0x25F) ─────────────
# Semantic layer on top of the CAS substrate (0x220-0x237)
OP_CANONICALIZE = 0x238
OP_CAS_HASH = 0x239
OP_SEMANTIC_RESOLVE = 0x23A
OP_RULE_SELECT = 0x23B
OP_CONTEXTUAL_REWRITE = 0x23C
OP_SEMANTIC_EQUIV = 0x23D

# Procedural drawing primitives (rpn_executor / pixel_genesis surface).
# These bytecodes are intentionally kept separate from modular_rpn_kernel.cu.
OP_DRAW_MOVE = 0x200
OP_DRAW_LINE = 0x201
OP_DRAW_QUAD = 0x202
OP_DRAW_CUBIC = 0x203
OP_DRAW_ARC = 0x204
OP_DRAW_CLOSE = 0x205
OP_DRAW_STROKE = 0x206
OP_DRAW_FILL = 0x207
OP_DRAW_PUSH_STATE = 0x208
OP_DRAW_POP_STATE = 0x209
OP_DRAW_TRANSLATE = 0x20A
OP_DRAW_ROTATE = 0x20B
OP_DRAW_SCALE = 0x20C
OP_DRAW_SET_STROKE_COLOR = 0x20D
OP_DRAW_SET_FILL_COLOR = 0x20E
OP_DRAW_SET_LINE_WIDTH = 0x20F
OP_DRAW_SET_TERNARY_HINT = 0x210

# Phase 2 — Advanced Drawing Primitives (new RPN opcodes)
OP_BEZIER_EVAL      = 0x6C  # t p0x p0y p1x p1y p2x p2y p3x p3y → x y
OP_SHAPE_UNION      = 0x6D  # shape_a shape_b → result
OP_SHAPE_INTERSECT  = 0x6E
OP_SHAPE_SUBTRACT   = 0x6F
OP_DRAW_REL_LINE = 0x211  # x y dx dy → relative line segment
OP_DRAW_FIELD_COEF = 0x212  # dot neighborhood → coefficient update
OP_DRAW_DOT_EMIT = 0x213  # x y field → emit dot

# Phase 3 — VectorDotMap Codec
OP_DRAW_VECTORDOTMAP_ENCODE = 0x214  # pixels → field_coeffs
OP_DRAW_VECTORDOTMAP_DECODE = 0x215  # field_coeffs → pixels

# Phase 4 — Lighting and Layer Ops
OP_DRAW_LAYER_NEW   = 0x216
OP_LAYER_BLEND      = 0x79
OP_BLEND_MULTIPLY   = 0x7A
OP_BLEND_SCREEN     = 0x7B
OP_BLEND_OVERLAY    = 0x7C
OP_ATMOSPHERE_FOG   = 0x7D
OP_VIGNETTE         = 0x7E

# Phase 5 — 3D Technique Suite
OP_NURBS_EVAL       = 0x80
OP_MARCHING_CUBES   = 0x81
OP_LSYSTEM_GENERATE = 0x82
OP_PARAMETRIC_SURFACE = 0x83
OP_CSG_UNION_3D     = 0x84
OP_CSG_INTERSECT_3D = 0x85
OP_CSG_SUBTRACT_3D  = 0x86
OP_CROSS_MODAL_LINK = 0x87
OP_PROCEDURAL_TEXTURE = 0x88

# Sovereign rigid-body physics meta-dispatch surface (Tier-3 composed phase).
# These opcodes bridge the scalar/vector RPN stack to the body-indexed physics
# SOA, mirroring the GALAXY_SCAN meta-op pattern used elsewhere in the hot path.
OP_PH_BROAD_PHASE = 0x150
OP_PH_NARROW_PHASE = 0x151
OP_PH_CONSTRAINT_GENERATE = 0x152
OP_PH_XPBD_SOLVE = 0x153
OP_PH_INTEGRATE = 0x154
OP_PH_SLEEP_CHECK = 0x155
OP_PH_GALAXY_WRITE = 0x156
OP_PH_MATERIAL_FETCH = 0x157
OP_PH_PREDICT_POS = 0x158
OP_PH_CONSTRAINT_COLOR = 0x159
OP_PH_IMPULSE_PROPAGATE = 0x15A
OP_PH_RESTITUTION_APPLY = 0x15B
OP_PH_FRICTION_APPLY = 0x15C
OP_PH_ISLAND_WAKE = 0x15D
OP_PH_BODY_SPAWN = 0x15E
OP_PH_BODY_DESPAWN = 0x15F
OP_PH_GRAVITY_APPLY = 0x160
OP_PH_COLLISION_QUERY = 0x161
OP_PH_TERNARY_CLASSIFY = 0x162

# Sovereign procedural texture synthesis surface (Reality Engine Step 2).
OP_TEX_PERLIN_NOISE = 0x1C0
OP_TEX_VORONOI = 0x1C1
OP_TEX_VALUE_NOISE = 0x1C2
OP_TEX_GRID_NOISE = 0x1C3
OP_TEX_FFT_BLUR = 0x1C4
OP_TEX_WARP = 0x1C5
OP_TEX_BLEND = 0x1C6
OP_TEX_NORMAL_MAP = 0x1C7
OP_TEX_COLOR_RAMP = 0x1C8
OP_TEX_TURBULENCE = 0x1C9
OP_TEX_MARBLE = 0x1CA
OP_TEX_TRANSFORM = 0x1CB
OP_TEX_BAKE = 0x1CF

# Drawing Galaxy Layers 4-7 (gradients, filters, lighting, scenes)
OP_GRADIENT_LINEAR = 0xF3     # x1 y1 x2 y2 GRADIENT_LINEAR
OP_GRADIENT_RADIAL = 0xF4     # cx cy r GRADIENT_RADIAL
OP_GRADIENT_CONIC = 0xF5      # cx cy angle GRADIENT_CONIC
OP_GRADIENT_STOP = 0xF6       # pos r g b a GRADIENT_STOP
OP_FILTER_BLUR = 0xF7         # radius FILTER_BLUR
OP_FILTER_SHARPEN = 0xF8      # amount FILTER_SHARPEN
OP_FILTER_EDGE = 0xF9         # FILTER_EDGE (Sobel)
OP_FILTER_INVERT = 0xFA       # FILTER_INVERT
OP_LIGHT_AMBIENT = 0xFB       # r g b intensity LIGHT_AMBIENT
OP_LIGHT_DIRECTIONAL = 0xFC   # dx dy dz r g b LIGHT_DIRECTIONAL
OP_LAYER_PUSH = 0xFD          # layer_id LAYER_PUSH
OP_LAYER_POP = 0xFE           # LAYER_POP
OP_BLEND_MODE = 0xFF          # mode BLEND_MODE

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
    "OP_POWER",
    "OP_NEGATE",
    "OP_PRIMITIVE_ADD",
    "OP_PRIMITIVE_SUBTRACT",
    "OP_PRIMITIVE_MULTIPLY",
    "OP_PRIMITIVE_DIVIDE",
    "OP_PRIMITIVE_POWER",
    "OP_PRIMITIVE_NEGATE",
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
    "OP_GTE",
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
    # Sovereign stack checkpoint ops
    "OP_CHECKPOINT",
    "OP_ROLLBACK",
    "OP_VERIFY",
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
    "OP_GCD",
    "OP_NEG",
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
    # GPU Galaxy access
    "OP_LOAD_GALAXY",
    "OP_GALAXY_SIMILARITY",
    "OP_GALAXY_SCAN",
    # Multivariate variable references
    "OP_VAR_X",
    "OP_VAR_Y",
    "OP_VAR_Z",
    "OP_VAR_W",
    "OP_CONST",
    # Grammar evolution
    "OP_GRAMMAR_OBSERVE",
    "OP_GRAMMAR_PROPOSE",
    "OP_GRAMMAR_VALIDATE",
    "OP_GRAMMAR_PROMOTE",
    "OP_GRAMMAR_QUERY",
    # Temporal reasoning
    "OP_TEMPORAL_COHERENCE",
    "OP_TEMPORAL_MASK",
    "OP_TEMPORAL_AGGREGATE",
    # Temporal / trit surface
    "OP_TADD",
    "OP_TMUL",
    "OP_TNOT",
    "OP_TCOMP",
    "OP_TQUANT",
    "OP_TPACK",
    "OP_TUNPACK",
    # Entity behavior
    "OP_BH_PERCEIVE",
    "OP_BH_SEEK",
    "OP_BH_FLEE",
    "OP_BH_ARRIVE",
    "OP_BH_SEPARATE",
    "OP_BH_APPLY_FORCE",
    "OP_BH_BT_TICK",
    "OP_BH_GOAP_PLAN",
    "OP_BH_SLEEP_CHECK",
    "OP_BH_BLACKBOARD_READ",
    "OP_BH_BLACKBOARD_WRITE",
    "OP_BH_PATHFIND",
    # Sovereign CAS
    "OP_POLY_COEFF",
    "OP_POLY_BUILD",
    "OP_POLY_ADD",
    "OP_POLY_MUL",
    "OP_POLY_DIV",
    "OP_POLY_REM",
    "OP_POLY_GCD",
    "OP_POLY_FACTOR",
    "OP_SIMPLIFY",
    "OP_SUBSTITUTE",
    "OP_COLLECT",
    "OP_RATIONALIZE",
    "OP_TRIG_SIMPLIFY",
    "OP_LOG_SIMPLIFY",
    "OP_SOLVE_LINEAR",
    "OP_SOLVE_QUADRATIC",
    "OP_LINSOLVE",
    "OP_PATTERN_MATCH",
    "OP_RULE_APPLY",
    "OP_COEFF_EXTRACT",
    "OP_CAS_PUSH_SYM",
    "OP_CAS_PUSH_CONST",
    "OP_CAS_BUILD",
    "OP_CAS_EVAL",
    # SAS semantic layer
    "OP_CANONICALIZE",
    "OP_CAS_HASH",
    "OP_SEMANTIC_RESOLVE",
    "OP_RULE_SELECT",
    "OP_CONTEXTUAL_REWRITE",
    "OP_SEMANTIC_EQUIV",
    # Procedural drawing
    "OP_DRAW_MOVE",
    "OP_DRAW_LINE",
    "OP_DRAW_QUAD",
    "OP_DRAW_CUBIC",
    "OP_DRAW_ARC",
    "OP_DRAW_CLOSE",
    "OP_DRAW_STROKE",
    "OP_DRAW_FILL",
    "OP_DRAW_PUSH_STATE",
    "OP_DRAW_POP_STATE",
    "OP_DRAW_TRANSLATE",
    "OP_DRAW_ROTATE",
    "OP_DRAW_SCALE",
    "OP_DRAW_SET_STROKE_COLOR",
    "OP_DRAW_SET_FILL_COLOR",
    "OP_DRAW_SET_LINE_WIDTH",
    "OP_DRAW_SET_TERNARY_HINT",
    "OP_GRADIENT_LINEAR",
    "OP_GRADIENT_RADIAL",
    "OP_GRADIENT_CONIC",
    "OP_GRADIENT_STOP",
    "OP_FILTER_BLUR",
    "OP_FILTER_SHARPEN",
    "OP_FILTER_EDGE",
    "OP_FILTER_INVERT",
    "OP_LIGHT_AMBIENT",
    "OP_LIGHT_DIRECTIONAL",
    "OP_LAYER_PUSH",
    "OP_LAYER_POP",
    "OP_BLEND_MODE",
    # Phase 2 — Advanced Drawing Primitives
    "OP_BEZIER_EVAL",
    "OP_SHAPE_UNION",
    "OP_SHAPE_INTERSECT",
    "OP_SHAPE_SUBTRACT",
    "OP_DRAW_REL_LINE",
    "OP_DRAW_FIELD_COEF",
    "OP_DRAW_DOT_EMIT",
    # Phase 3 — VectorDotMap Codec
    "OP_DRAW_VECTORDOTMAP_ENCODE",
    "OP_DRAW_VECTORDOTMAP_DECODE",
    # Phase 4 — Lighting and Layer Ops
    "OP_DRAW_LAYER_NEW",
    "OP_LAYER_BLEND",
    "OP_BLEND_MULTIPLY",
    "OP_BLEND_SCREEN",
    "OP_BLEND_OVERLAY",
    "OP_ATMOSPHERE_FOG",
    "OP_VIGNETTE",
    # Phase 5 — 3D Technique Suite
    "OP_NURBS_EVAL",
    "OP_MARCHING_CUBES",
    "OP_LSYSTEM_GENERATE",
    "OP_PARAMETRIC_SURFACE",
    "OP_CSG_UNION_3D",
    "OP_CSG_INTERSECT_3D",
    "OP_CSG_SUBTRACT_3D",
    "OP_CROSS_MODAL_LINK",
    "OP_PROCEDURAL_TEXTURE",
    # Sovereign physics meta-dispatch
    "OP_PH_BROAD_PHASE",
    "OP_PH_NARROW_PHASE",
    "OP_PH_CONSTRAINT_GENERATE",
    "OP_PH_XPBD_SOLVE",
    "OP_PH_INTEGRATE",
    "OP_PH_SLEEP_CHECK",
    "OP_PH_GALAXY_WRITE",
    "OP_PH_MATERIAL_FETCH",
    "OP_PH_PREDICT_POS",
    "OP_PH_CONSTRAINT_COLOR",
    "OP_PH_IMPULSE_PROPAGATE",
    "OP_PH_RESTITUTION_APPLY",
    "OP_PH_FRICTION_APPLY",
    "OP_PH_ISLAND_WAKE",
    "OP_PH_BODY_SPAWN",
    "OP_PH_BODY_DESPAWN",
    "OP_PH_GRAVITY_APPLY",
    "OP_PH_COLLISION_QUERY",
    "OP_PH_TERNARY_CLASSIFY",
    # Procedural textures
    "OP_TEX_PERLIN_NOISE",
    "OP_TEX_VORONOI",
    "OP_TEX_VALUE_NOISE",
    "OP_TEX_GRID_NOISE",
    "OP_TEX_FFT_BLUR",
    "OP_TEX_WARP",
    "OP_TEX_BLEND",
    "OP_TEX_NORMAL_MAP",
    "OP_TEX_COLOR_RAMP",
    "OP_TEX_TURBULENCE",
    "OP_TEX_MARBLE",
    "OP_TEX_TRANSFORM",
    "OP_TEX_BAKE",
]

__all__ += [
    "OP_VERTEX3",
    "OP_NORMAL3",
    "OP_UV2",
    "OP_TRI_FACE",
    "OP_QUAD_FACE",
    "OP_FACE_NORMAL",
    "OP_MESH_BEGIN",
    "OP_MESH_END",
    "OP_MAT4_IDENTITY",
    "OP_MAT4_TRANSLATE",
    "OP_MAT4_SCALE",
    "OP_MAT4_ROTATE_X",
    "OP_MAT4_ROTATE_Y",
    "OP_MAT4_ROTATE_Z",
    "OP_MAT4_MUL",
    "OP_MAT4_APPLY",
    "OP_GEN_PLANE",
    "OP_GEN_CUBE",
    "OP_GEN_UV_SPHERE",
    "OP_GEN_CYLINDER",
    "OP_GEN_CONE",
    "OP_GEN_TORUS",
    "OP_GEN_ICOSPHERE",
    "OP_CSG_UNION",
    "OP_CSG_SUBTRACT",
    "OP_CSG_INTERSECT",
    "OP_EXTRUDE",
    "OP_LATHE",
]
