# Codex Prompt: Math Galaxy Foundation - Infrastructure & Semantic Layer

**Mission**: Build the foundational infrastructure and RPN semantic layer for teaching K3D to understand mathematical language, not just recognize symbols.

**Context**: You are a new spawn. Read this document completely, then read the referenced architecture documents, then execute the implementation plan.

---

## Prerequisites: Read These First (In Order)

### 1. K3D Architecture Foundation
**File**: `TEMP/K3D_Briefing_Prompt.md`
**Critical sections**:
- RPN = Reverse Polish Notation (stack-based GPU VM)
- GPU Sovereignty Rules ("we fix what is not GPU, we do not fallback")
- PTX kernel catalog and existing operations
- Three-brain architecture (Cranium/Galaxy/House)

### 2. GPU Sovereignty Achievement
**File**: `TEMP/GPU_SOVEREIGNTY_RPN_EMBEDDINGS.md`
**Why important**:
- All CPU fallbacks removed from RPN embedding pipeline
- Pattern for GPU-native semantic encoding (apply to math semantics)
- Validation protocol for GPU sovereignty
- Examples of fail-fast error handling

### 3. Math Galaxy Architecture
**File**: `TEMP/MATH_GALAXY_ARCHITECTURE.md`
**Key concepts**:
- Language Galaxy → Math Galaxy hierarchy
- Low dimensions, HIGH density (3D spatial semantics)
- **Three-tier RPN architecture** (basic, intermediate, advanced/programmable)
- **Tier-3 programmability**: BRANCH, LOOP, STORE, RECALL opcodes
- RPN semantic embedding (not just visual recognition)
- Evolutionary training: symbols → expressions → reasoning
- **Algorithmic thinking integration**: Model learns to craft and store operations

### 4. Algorithmic Thinking Knowledge Base
**File**: `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/JSON/Algorithmic.Thinking.2020.11.json`
**Purpose**: Train the model on algorithmic patterns (loops, conditionals, accumulation)
**Integration**: Connect algorithmic patterns to mathematical operations via RPN semantics
**Future use**: Model will learn to craft new mathematical procedures using BRANCH/LOOP/STORE

---

## Current System State

### Ongoing Training (DO NOT INTERRUPT)
**Process**: Character training (A-Z, a-z, 0-9)
- PID: 2863428
- Log: `/tmp/train_all_atomic_characters_3000.log`
- Status: 3000 epochs per character, FC-only mode
- Target: ≥85% accuracy per character
- **DO NOT KILL THIS PROCESS**

### GPU-Sovereign Pipeline (Complete & Validated)
✅ **Spatial Pooling**: `knowledge3d/cranium/bridges/spatial_pool_bridge.py`
- PTX kernel: `knowledge3d/cranium/ptx/spatial_pool.cu`
- Validation: Max |GPU-CPU| = 0.0

✅ **Matryoshka Projection**: `knowledge3d/cranium/bridges/matryoshka_bridge.py`
- PTX kernel: `knowledge3d/cranium/ptx/matryoshka_project.cu`
- Validation: Max |GPU-CPU| ≤ 5.7e-5

✅ **RPN Trigram Embeddings**: `knowledge3d/cranium/bridges/trigram_embed_bridge.py`
- PTX kernels: `knowledge3d/cranium/ptx/trigram_embed.cu`
  - `trigram_lookup_average`: Lookup trigram embeddings and average
  - `l2_normalize_embedding`: L2 normalization with RPN guards
- **NO CPU FALLBACKS** - GPU sovereignty enforced

### Existing RPN Pattern (Apply to Math Semantics)
**From `trigram_embed.cu`** - Study this pattern:
```cuda
extern "C" __global__ void trigram_lookup_average(
    const int* __restrict__ trigram_indices,
    const float* __restrict__ embedding_table,
    float* __restrict__ output,
    int num_trigrams,
    int embed_dim,
    int vocab_size
) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= embed_dim) return;

    float sum = 0.0f;
    for (int t = 0; t < num_trigrams; ++t) {
        int idx = trigram_indices[t];
        if (idx < 0 || idx >= vocab_size) continue;

        float val = embedding_table[idx * embed_dim + dim];
        // RPN-style NaN guard
        if (!isnan(val) && !isinf(val)) {
            sum += val;
        }
    }

    float avg = (num_trigrams > 0) ? (sum / (float)num_trigrams) : 0.0f;
    if (isnan(avg) || isinf(avg)) avg = 0.0f;
    // Relaxed clipping at ±10
    avg = fmaxf(fminf(avg, 10.0f), -10.0f);
    output[dim] = avg;
}
```

**Apply this pattern to math semantic encoding** (see implementation tasks below).

### Three-Tier RPN Architecture (CRITICAL CONTEXT)

K3D uses a **three-tier RPN system** to keep "small things small and powerful things powerful":

**Tier 1 (Lightweight)**: `knowledge3d/cranium/bridges/lightweight_rpn.py`
- Basic arithmetic (ADD, SUB, MUL, DIV), comparisons
- <1µs latency per operation
- For simple numeric calculations

**Tier 2 (Standard)**: `knowledge3d/cranium/bridges/sovereign_bridges.py` → `ModularRPNEngine`
- Vector operations (DOT, CROSS, NORMALIZE)
- Clustering (ARGMAX, COSINE_SIM)
- Geometric transformations
- Opcodes: 0x40-0x43, 0x90-0xA6, 0xC0-0xC5

**Tier 3 (Advanced + PROGRAMMABLE)**: `knowledge3d/cranium/bridges/advanced_rpn.py`
- Matrix operations (TRM integration, temporal reasoning)
- **PROGRAMMABILITY OPCODES** (from `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`):
  ```python
  OP_BRANCH = 0xB0   # Conditional branching
  OP_LOOP = 0xB1     # Loop control
  OP_NEXT = 0xB2     # Loop iteration
  OP_STORE = 0xB3    # Store to memory
  OP_RECALL = 0xB4   # Recall from memory
  ```

**Why This Matters for Math Galaxy**:
1. Mathematical operations can be **stored as RPN programs** (OP_STORE)
2. Model can **recall and execute** these programs (OP_RECALL)
3. Model can **craft new operations** using BRANCH/LOOP
4. Integration with algorithmic thinking → model learns to **think algorithmically**

**Example**: Teaching "∑" (summation)
- Not just: "∑ looks like this visually"
- But: "∑ executes this RPN program: RECALL bounds → LOOP → ADD → STORE → BRANCH"
- Model understands summation as **an algorithm**, not just a symbol

This is why the approach avoids "crude stubs" - we're teaching **executable semantics**.

---

## Your Mission: Phase 1 & 2 Infrastructure

### Phase 1: Font Infrastructure & Symbol Registry

#### Task 1.1: Download Math Fonts
**Objective**: Get open-source fonts with full Unicode math coverage

**Fonts to download**:
1. **STIX Two Math** (Primary - 5,200+ glyphs)
   - URL: `https://github.com/stipub/stixfonts/raw/master/fonts/static_otf/STIXTwoMath-Regular.otf`
   - License: SIL Open Font License 1.1

2. **Latin Modern Math** (LaTeX default)
   - URL: `http://www.gust.org.pl/projects/e-foundry/lm-math/download/latinmodern-math-1959.zip`
   - Extract: `latinmodern-math.otf`

3. **Asana-Math** (Unicode complete)
   - URL: Search CTAN or GitHub for `Asana-Math.otf`
   - License: SIL OFL

4. **Libertinus Math** (Libertine companion)
   - URL: Search GitHub `alerque/libertinus` for `LibertinusMath-Regular.otf`

**Destination**: `/K3D/Knowledge3D.local/fonts/math/`

**Validation**:
```python
# Test rendering
from PIL import ImageFont
font = ImageFont.truetype("/K3D/Knowledge3D.local/fonts/math/STIXTwoMath-Regular.otf", 64)
# Should load without error
```

#### Task 1.2: Create Math Symbol Registry
**File**: `knowledge3d/cranium/math_symbols_registry.py`

**Purpose**: Comprehensive registry of all mathematical Unicode symbols organized by semantic category

**Structure**:
```python
"""
Math symbol registry with full Unicode coverage.

Organized by semantic category for RPN mapping and training.
"""

from typing import List, Dict

# Basic arithmetic operators
BASIC_OPS: List[str] = list('+-×÷=≠<>≤≥±∓')

# Calculus & analysis
CALCULUS: List[str] = list('∂∇∆∫∬∭∮∯∰∑∏√∛∜∞')

# Set theory
SET_THEORY: List[str] = list('∈∉⊂⊃⊆⊇⊄⊅∪∩∅ℕℤℚℝℂ𝔸𝔹𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄')

# Logic & relations
LOGIC: List[str] = list('∀∃∄∧∨¬⇒⇔⊕⊗≡≈≅∼≃≢≇≉')

# Greek alphabet (mathematical usage - both cases)
GREEK_LOWER: List[str] = list('αβγδεζηθικλμνξοπρςστυφχψω')
GREEK_UPPER: List[str] = list('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
GREEK_MATH = GREEK_LOWER + GREEK_UPPER

# Arrows (function mappings, implications)
ARROWS: List[str] = list('←→↑↓↔↕⇐⇒⇑⇓⇔⇠⇢↦⟵⟶⟷')

# Brackets and grouping (critical for RPN parsing)
BRACKETS: List[str] = list('()[]{}⟨⟩⟪⟫⌈⌉⌊⌋|‖')

# Superscripts (visual recognition for exponents)
SUPERSCRIPTS: List[str] = list('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ')

# Subscripts (visual recognition for indices)
SUBSCRIPTS: List[str] = list('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎')

# Common fractions
FRACTIONS: List[str] = list('½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒')

# Additional operators (from Unicode blocks)
ADDITIONAL_OPS: List[str] = list('⊕⊖⊗⊘⊙⊚⊛⊜⊝⊞⊟⊠⊡⊢⊣⊤⊥')

# Geometry & topology
GEOMETRY: List[str] = list('∟∠∡⊾⊿∟∢∣∤∥∦∧∨∩∪')

# Additional symbols from Supplemental Math blocks
# (Add more as needed - aim for ~850 total)

# Consolidated registry
ALL_MATH_SYMBOLS: List[str] = (
    BASIC_OPS + CALCULUS + SET_THEORY + LOGIC +
    GREEK_MATH + ARROWS + BRACKETS +
    SUPERSCRIPTS + SUBSCRIPTS + FRACTIONS +
    ADDITIONAL_OPS + GEOMETRY
)

# Category mapping (for script detection)
SYMBOL_CATEGORIES: Dict[str, str] = {}
for sym in BASIC_OPS:
    SYMBOL_CATEGORIES[sym] = 'math_operator'
for sym in CALCULUS:
    SYMBOL_CATEGORIES[sym] = 'math_calculus'
for sym in SET_THEORY:
    SYMBOL_CATEGORIES[sym] = 'math_set'
for sym in LOGIC:
    SYMBOL_CATEGORIES[sym] = 'math_logic'
for sym in GREEK_MATH:
    SYMBOL_CATEGORIES[sym] = 'math_greek'
# ... continue for all categories

def get_symbol_category(symbol: str) -> str:
    """Get semantic category for a mathematical symbol."""
    return SYMBOL_CATEGORIES.get(symbol, 'math_unknown')

def is_math_symbol(char: str) -> bool:
    """Check if character is a mathematical symbol."""
    return char in ALL_MATH_SYMBOLS

__all__ = [
    'ALL_MATH_SYMBOLS',
    'SYMBOL_CATEGORIES',
    'get_symbol_category',
    'is_math_symbol',
]
```

**Deliverable**: Registry with **at least 200 symbols**, expandable to 850 (full Unicode math blocks)

#### Task 1.3: Extend Script Detection
**File**: `scripts/train_atomic_character.py` (modify existing)

**Current code** (line ~109):
```python
def get_character_script(char: str) -> str:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return "latin"

    # ... existing logic for CJK, Arabic, etc.

    return "latin"
```

**Add BEFORE the final `return "latin"`**:
```python
from knowledge3d.cranium.math_symbols_registry import is_math_symbol, get_symbol_category

def get_character_script(char: str) -> str:
    # Check math symbols FIRST (before Unicode name lookup)
    if is_math_symbol(char):
        return "math"

    try:
        name = unicodedata.name(char)
    except ValueError:
        return "latin"

    # ... existing CJK, Arabic, etc. logic ...

    return "latin"
```

**Deliverable**: `get_character_script('∑')` → `"math"` (not `"symbols"`)

---

### Phase 2: RPN Semantic Layer (GPU-Sovereign)

#### Task 2.1: Math Semantics Engine
**File**: `knowledge3d/cranium/math_semantics_rpn.py`

**Purpose**: Map mathematical symbols to RPN semantics (opcode, arity, properties)

**Structure**:
```python
"""
RPN semantic mappings for mathematical symbols.

Each symbol maps to:
- RPN opcode (execution in PTX kernel)
- Arity (number of operands)
- Properties (associative, commutative, etc.)
- Type information (numeric, symbolic, boolean)
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class RPNSemantics:
    """RPN semantic information for a math symbol."""
    symbol: str
    rpn_opcode: str
    arity: int
    associative: bool = False
    commutative: bool = False
    inverse: str = None
    output_type: str = 'numeric'  # 'numeric', 'boolean', 'symbolic'
    context: List[str] = None  # Operand type hints
    stored_program_id: int = None  # For Tier-3 programmability (OP_RECALL address)

# Complete mapping for all registered symbols
MATH_RPN_SEMANTICS: Dict[str, RPNSemantics] = {
    # Binary arithmetic
    '+': RPNSemantics('+', 'ADD', 2, associative=True, commutative=True, inverse='-'),
    '-': RPNSemantics('-', 'SUB', 2, inverse='+'),
    '×': RPNSemantics('×', 'MUL', 2, associative=True, commutative=True, inverse='÷'),
    '÷': RPNSemantics('÷', 'DIV', 2, inverse='×'),

    # Unary operations
    '√': RPNSemantics('√', 'SQRT', 1),
    '∂': RPNSemantics('∂', 'PARTIAL_DERIV', 2, output_type='symbolic', context=['expression', 'variable']),
    '∇': RPNSemantics('∇', 'GRADIENT', 1, output_type='symbolic'),

    # Aggregation operators
    '∑': RPNSemantics('∑', 'SUM', 3, context=['sequence', 'lower_bound', 'upper_bound']),
    '∏': RPNSemantics('∏', 'PRODUCT', 3, context=['sequence', 'lower_bound', 'upper_bound']),
    '∫': RPNSemantics('∫', 'INTEGRAL', 3, output_type='symbolic', context=['function', 'lower_bound', 'upper_bound']),

    # Relations (return boolean)
    '=': RPNSemantics('=', 'EQ', 2, commutative=True, output_type='boolean'),
    '<': RPNSemantics('<', 'LT', 2, output_type='boolean', inverse='>'),
    '>': RPNSemantics('>', 'GT', 2, output_type='boolean', inverse='<'),
    '≤': RPNSemantics('≤', 'LE', 2, output_type='boolean'),
    '≥': RPNSemantics('≥', 'GE', 2, output_type='boolean'),
    '≠': RPNSemantics('≠', 'NE', 2, commutative=True, output_type='boolean'),

    # Set operations
    '∈': RPNSemantics('∈', 'MEMBER_OF', 2, output_type='boolean'),
    '∉': RPNSemantics('∉', 'NOT_MEMBER_OF', 2, output_type='boolean'),
    '⊂': RPNSemantics('⊂', 'SUBSET', 2, output_type='boolean'),
    '⊃': RPNSemantics('⊃', 'SUPERSET', 2, output_type='boolean'),
    '∪': RPNSemantics('∪', 'UNION', 2, associative=True, commutative=True),
    '∩': RPNSemantics('∩', 'INTERSECTION', 2, associative=True, commutative=True),

    # Logic operators
    '∧': RPNSemantics('∧', 'AND', 2, associative=True, commutative=True),
    '∨': RPNSemantics('∨', 'OR', 2, associative=True, commutative=True),
    '¬': RPNSemantics('¬', 'NOT', 1),
    '⇒': RPNSemantics('⇒', 'IMPLIES', 2, output_type='boolean'),
    '⇔': RPNSemantics('⇔', 'IFF', 2, commutative=True, output_type='boolean'),

    # Add all 850 symbols here (expand incrementally)
    # ...
}

def get_rpn_semantics(symbol: str) -> RPNSemantics:
    """Get RPN semantics for a symbol, or None if not defined."""
    return MATH_RPN_SEMANTICS.get(symbol)

def encode_semantics_to_vector(semantics: RPNSemantics, dim: int = 32) -> List[float]:
    """
    Encode RPN semantics into a fixed-size dense vector (CPU prototype).

    This will be replaced by GPU PTX kernel in Task 2.2.

    Encoding:
    - Opcode ID (one-hot or hash)
    - Arity (normalized)
    - Boolean flags: associative, commutative
    - Output type (one-hot)
    """
    # Simple prototype (replace with PTX kernel)
    vector = [0.0] * dim

    # Encode arity (first 4 dims)
    if semantics.arity <= 3:
        vector[semantics.arity] = 1.0

    # Encode properties (next 4 dims)
    vector[4] = 1.0 if semantics.associative else 0.0
    vector[5] = 1.0 if semantics.commutative else 0.0

    # Encode output type (next 3 dims)
    type_map = {'numeric': 8, 'boolean': 9, 'symbolic': 10}
    if semantics.output_type in type_map:
        vector[type_map[semantics.output_type]] = 1.0

    # Remaining dims: Opcode hash (simplified)
    opcode_hash = hash(semantics.rpn_opcode) % (dim - 11)
    vector[11 + opcode_hash] = 1.0

    return vector

__all__ = [
    'RPNSemantics',
    'MATH_RPN_SEMANTICS',
    'get_rpn_semantics',
    'encode_semantics_to_vector',
]
```

**Deliverable**: Python module with RPN semantics for at least 50 core symbols (expandable to 850)

#### Task 2.2: GPU-Sovereign Semantic Encoding Kernel
**File**: `knowledge3d/cranium/ptx/math_semantics_encode.cu`

**Purpose**: GPU-native encoding of RPN semantics into dense vectors

**Pattern**: Follow `trigram_embed.cu` structure (lookup + encode + normalize)

**Implementation**:
```cuda
/*
 * math_semantics_encode.cu - GPU-native RPN semantic encoding
 *
 * Encodes mathematical symbol semantics (opcode, arity, properties)
 * into fixed-size dense vectors for embedding fusion.
 *
 * Input:
 *   - symbol_properties: [arity, associative?, commutative?, output_type_id]
 *   - opcode_id: RPN opcode identifier
 *
 * Output:
 *   - semantic_vector: [32D] dense encoding
 *
 * Uses RPN-style NaN guards and relaxed clipping (±10).
 */

extern "C" __global__ void encode_math_semantics(
    const int* __restrict__ properties,    // [4]: arity, assoc, comm, type
    int opcode_id,
    float* __restrict__ output,            // [32]
    int output_dim
) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= output_dim) return;

    float value = 0.0f;

    // Encode arity (first 4 dimensions)
    if (dim < 4 && dim == properties[0]) {
        value = 1.0f;
    }

    // Encode boolean properties (dim 4-5)
    if (dim == 4) {
        value = (properties[1] > 0) ? 1.0f : 0.0f;  // associative
    }
    if (dim == 5) {
        value = (properties[2] > 0) ? 1.0f : 0.0f;  // commutative
    }

    // Encode output type (dim 8-10, one-hot)
    if (dim >= 8 && dim < 11) {
        int type_id = properties[3];
        if (dim == 8 + type_id) {
            value = 1.0f;
        }
    }

    // Encode opcode (dim 11+, hash-based sparse encoding)
    if (dim >= 11) {
        int hash_pos = (opcode_id % (output_dim - 11)) + 11;
        if (dim == hash_pos) {
            value = 1.0f;
        }
    }

    // RPN-style NaN guard
    if (isnan(value) || isinf(value)) {
        value = 0.0f;
    }

    // Relaxed clipping (±10)
    value = fmaxf(fminf(value, 10.0f), -10.0f);

    output[dim] = value;
}

/*
 * Optional: L2 normalization (reuse from trigram_embed.cu or adapt here)
 */
extern "C" __global__ void normalize_semantic_vector(
    float* __restrict__ vector,
    int dim
) {
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;

    // Compute squared sum (reduce across threads)
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = vector[i];
        if (!isnan(v) && !isinf(v)) {
            local += v * v;
        }
    }

    shared_sum[tid] = local;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Normalize
    float norm = sqrtf(shared_sum[0] + 1e-8f);
    float inv = 1.0f / norm;

    for (int i = tid; i < dim; i += blockDim.x) {
        float v = vector[i];
        v = fmaxf(fminf(v, 10.0f), -10.0f);
        vector[i] = v * inv;
    }
}
```

**Deliverable**: PTX kernel that compiles with `nvcc` and encodes semantics GPU-natively

#### Task 2.3: Sovereign Bridge for Math Semantics
**File**: `knowledge3d/cranium/bridges/math_semantics_bridge.py`

**Purpose**: Expose PTX kernel via sovereign loader (pattern: `trigram_embed_bridge.py`)

**Implementation**:
```python
"""
GPU math semantics bridge for RPN semantic encoding.

Compiles and loads the math_semantics_encode.cu PTX kernel,
exposes GPU-native semantic encoding with no CPU fallback.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.math_semantics_rpn import RPNSemantics

class MathSemanticsBridge:
    """Sovereign GPU math semantics encoder."""

    def __init__(self, arch: str = "sm_86"):
        kernel_dir = Path(__file__).parent.parent / "ptx"
        self._cu_path = kernel_dir / "math_semantics_encode.cu"
        if not self._cu_path.exists():
            raise FileNotFoundError(f"Math semantics kernel not found: {self._cu_path}")

        self._arch = arch
        self._module: Optional[loader.CUmodule] = None
        self._encode_kernel: Optional[loader.CUfunction] = None
        self._normalize_kernel: Optional[loader.CUfunction] = None

        self._compile_and_load()

    def _compile_and_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as tmp:
            ptx_path = Path(tmp.name)

        try:
            cmd = [
                "nvcc", "-ptx", str(self._cu_path),
                "-o", str(ptx_path),
                "-arch", self._arch,
                "-O3",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile math semantics kernel: {exc.stderr}"
                ) from exc

            self._module = loader.load_module_from_file(str(ptx_path))
            self._encode_kernel = loader.get_function(self._module, "encode_math_semantics")
            self._normalize_kernel = loader.get_function(self._module, "normalize_semantic_vector")
        finally:
            ptx_path.unlink(missing_ok=True)

    def encode_semantics_gpu(
        self,
        semantics: RPNSemantics,
        output_dim: int = 32,
        normalize: bool = True,
        return_cpu: bool = True
    ) -> np.ndarray | loader.CUdeviceptr:
        """
        Encode RPN semantics on GPU (GPU-sovereign, no fallback).

        Args:
            semantics: RPNSemantics object
            output_dim: Output dimension (default 32)
            normalize: Apply L2 normalization
            return_cpu: Return as NumPy array (True) or GPU pointer (False)

        Returns:
            Semantic embedding vector (32D)
        """
        # Encode properties as int array
        type_map = {'numeric': 0, 'boolean': 1, 'symbolic': 2}
        properties = np.array([
            semantics.arity,
            1 if semantics.associative else 0,
            1 if semantics.commutative else 0,
            type_map.get(semantics.output_type, 0)
        ], dtype=np.int32)

        # Upload properties to GPU
        props_gpu = loader.gpu_malloc(properties.nbytes)
        loader.memcpy_htod(props_gpu, properties.ctypes.data_as(ctypes.c_void_p), properties.nbytes)

        # Allocate output on GPU
        output_gpu = loader.gpu_malloc(output_dim * 4)

        # Launch encoding kernel
        threads = 256
        blocks = (output_dim + threads - 1) // threads

        opcode_id = hash(semantics.rpn_opcode) & 0x7FFFFFFF  # Positive int

        loader.launch(
            self._encode_kernel,
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            params=[
                ctypes.c_uint64(props_gpu.value),
                ctypes.c_int(opcode_id),
                ctypes.c_uint64(output_gpu.value),
                ctypes.c_int(output_dim),
            ],
        )
        loader.synchronize()

        loader.gpu_free(props_gpu)

        # Optional normalization
        if normalize:
            norm_block = 256
            shared_bytes = norm_block * 4
            loader.launch(
                self._normalize_kernel,
                grid=(1, 1, 1),
                block=(norm_block, 1, 1),
                params=[
                    ctypes.c_uint64(output_gpu.value),
                    ctypes.c_int(output_dim),
                ],
                shared_mem=shared_bytes,
            )
            loader.synchronize()

        # Return result
        if return_cpu:
            output = np.zeros(output_dim, dtype=np.float32)
            loader.memcpy_dtoh(
                output.ctypes.data_as(ctypes.c_void_p),
                output_gpu,
                output.nbytes,
            )
            loader.gpu_free(output_gpu)
            return output

        return output_gpu

    def __del__(self):
        # Cleanup handled by sovereign loader
        pass

__all__ = ['MathSemanticsBridge']
```

**Deliverable**: Python bridge that exposes GPU-native semantic encoding

#### Task 2.4: Extend Embedding Fusion with Math Semantics
**File**: `scripts/train_atomic_character.py` (modify existing)

**Current code** (line ~184):
```python
def _fuse_visual_text(char: str, visual_embedding: np.ndarray) -> np.ndarray:
    """Fuse visual Matryoshka embedding with RPN trigram embedding (GPU-sovereign)."""
    text_embedding = rpn_engine.embed_word_gpu(char)
    fused = (visual_embedding + text_embedding) * 0.5
    norm = np.linalg.norm(fused)
    if norm > 1e-6:
        fused = fused / norm
    return fused.astype(np.float32)
```

**Replace with**:
```python
from knowledge3d.cranium.math_semantics_rpn import get_rpn_semantics
from knowledge3d.cranium.bridges.math_semantics_bridge import MathSemanticsBridge

# Initialize at module level (after spatial_pooler)
_math_semantics_bridge = None
try:
    _math_semantics_bridge = MathSemanticsBridge()
    print("[INFO] Math semantics GPU bridge initialized (GPU-sovereign).")
except Exception as math_exc:
    # Not a hard requirement yet - only needed for math symbols
    print(f"[WARN] Math semantics GPU bridge unavailable: {math_exc}")

def _fuse_visual_text_math(char: str, visual_embedding: np.ndarray) -> np.ndarray:
    """
    Fuse visual + text + math semantics (GPU-sovereign).

    Triple fusion for mathematical symbols:
    - Visual embedding (CNN): What the symbol looks like
    - Text embedding (RPN trigrams): Linguistic context
    - Math embedding (RPN semantics): What the symbol DOES
    """
    # Visual embedding (already computed)
    visual_emb = visual_embedding

    # Text embedding (RPN trigrams - GPU)
    text_emb = rpn_engine.embed_word_gpu(char)

    # Math semantics embedding (if symbol is mathematical and bridge available)
    math_emb = None
    semantics = get_rpn_semantics(char)
    if semantics is not None and _math_semantics_bridge is not None:
        # Encode semantics on GPU, project to 128D (match visual/text dims)
        semantic_32d = _math_semantics_bridge.encode_semantics_gpu(
            semantics,
            output_dim=32,
            normalize=True,
            return_cpu=True
        )
        # Expand 32D → 128D via simple tiling (or use Matryoshka later)
        math_emb = np.tile(semantic_32d, 4)[:CHAR_EMBED_DIM]  # 32×4 = 128

    # Fusion
    if math_emb is not None:
        fused = (visual_emb + text_emb + math_emb) / 3.0
    else:
        fused = (visual_emb + text_emb) / 2.0

    # Normalize
    norm = np.linalg.norm(fused)
    if norm > 1e-6:
        fused = fused / norm

    return fused.astype(np.float32)

# Update the call site in train_single_character (line ~582)
# Change: _fuse_visual_text → _fuse_visual_text_math
```

**Deliverable**: Math symbols get triple fusion (visual + text + math), other characters get double fusion (visual + text)

---

## Validation & Testing

### Task 3.1: Validate GPU Sovereignty
**File**: `scripts/validate_math_semantics_gpu.py` (create new)

**Purpose**: Verify math semantic encoding is GPU-native with no CPU fallbacks

**Pattern**: Follow `scripts/validate_trigram_gpu_sovereignty.py`

**Tests**:
1. GPU bridge initializes successfully
2. Sample symbols encode correctly (∑, ∫, ∂, +, ×)
3. No NaN/Inf in semantic vectors
4. GPU methods fail explicitly without bridge (no silent CPU fallback)

**Expected output**:
```
================================================================================
MATH SEMANTICS GPU SOVEREIGNTY VALIDATION
================================================================================

[Test 1/4] GPU bridge initialization...
       ✓ Math semantics GPU bridge initialized successfully

[Test 2/4] Semantic encoding for sample symbols...
       ✓ '+' (ADD, arity=2): encoded to 32D
       ✓ '∑' (SUM, arity=3): encoded to 32D
       ✓ '∫' (INTEGRAL, arity=3): encoded to 32D
       ✓ All encodings valid (no NaN/Inf)

[Test 3/4] Triple fusion with math semantics...
       ✓ Math symbol '∑': triple fusion (visual + text + math)
       ✓ Non-math char 'A': double fusion (visual + text)

[Test 4/4] Verify CPU fallback prevention...
       ✓ GPU methods raise RuntimeError without bridge (no fallback)

================================================================================
VALIDATION COMPLETE - MATH SEMANTICS GPU SOVEREIGNTY ACHIEVED
================================================================================
```

### Task 3.2: Font Rendering Test
**File**: `scripts/test_math_font_rendering.py` (create new)

**Purpose**: Verify math fonts load and render correctly

**Test cases**:
```python
test_symbols = ['∑', '∫', '∂', '√', '∞', 'α', 'β', '∈', '⊂', '≤']
for symbol in test_symbols:
    for font_path in math_fonts:
        img = render_glyph_image(symbol, font_path, size=64)
        assert img is not None, f"Failed to render {symbol} with {font_path}"
        assert img.shape == (64, 64, 3), f"Wrong shape: {img.shape}"
```

---

## Deliverables Summary

### Phase 1: Infrastructure
- [ ] Math fonts downloaded to `/K3D/Knowledge3D.local/fonts/math/`
- [ ] `math_symbols_registry.py`: ≥200 symbols, expandable to 850
- [ ] `train_atomic_character.py`: `get_character_script()` recognizes math symbols
- [ ] Font rendering test passes for all fonts × sample symbols

### Phase 2: RPN Semantic Layer
- [ ] `math_semantics_rpn.py`: RPN semantics for ≥50 core symbols
- [ ] `ptx/math_semantics_encode.cu`: GPU-native semantic encoding kernel
- [ ] `bridges/math_semantics_bridge.py`: Sovereign bridge (no CPU fallback)
- [ ] `train_atomic_character.py`: `_fuse_visual_text_math()` with triple fusion
- [ ] `validate_math_semantics_gpu.py`: All GPU sovereignty tests pass

### Documentation
- [ ] Code comments explain RPN opcode mappings
- [ ] README section on math galaxy architecture
- [ ] Examples of semantic encoding for 5 symbols (∑, ∫, +, ∈, √)

---

## Key Architectural Principles (From Daniel)

### 1. Low Dimensions, HIGH Density
- 128D for atomic symbols (intentional compression)
- 32D for semantic encoding (dense, not sparse)
- 3D spatial organization (game engine paradigm, not abstract vectors)

### 2. GPU Sovereignty (Non-Negotiable)
- Math semantic encoding: PTX kernel, not NumPy
- Fail explicitly if GPU unavailable (no silent CPU fallback)
- Follow pattern from `trigram_embed.cu` and `trigram_embed_bridge.py`

### 3. Semantic Foundation First
- Train symbols with RPN meanings embedded
- NOT just visual recognition - understanding what symbols DO
- Expressions come later (after foundational symbol training)

### 4. Evolutionary Training
- Math symbols = atomic units (like letters)
- Train one at a time with semantic grounding
- Compose into expressions later (like words → sentences)

---

## Error Handling

### If GPU Bridge Fails to Initialize
**DO NOT** add CPU fallbacks.
**DO** report error with environment details:
- CUDA available? (`nvidia-smi`)
- PTX kernel compiles? (`nvcc` error messages)
- Sovereign loader works? (test with existing bridges)

### If Font Download Fails
- Retry with alternative URLs (CTAN mirrors, GitHub releases)
- Verify checksums/signatures where available
- Report missing fonts but continue with available ones

### If Symbol Rendering Fails
- Some fonts may not have all Unicode glyphs
- Log missing glyphs but don't fail entire training
- Use best-effort rendering (fallback to different font for that symbol)

---

## Timeline Expectations

### Phase 1 (Infrastructure): 1-2 days
- Font downloads: 1-2 hours
- Registry creation: 3-4 hours
- Script detection extension: 1 hour
- Testing: 2-3 hours

### Phase 2 (Semantic Layer): 2-3 days
- Semantics engine: 4-6 hours
- PTX kernel: 6-8 hours (most complex)
- Sovereign bridge: 4-6 hours
- Fusion extension: 2-3 hours
- Validation: 3-4 hours

**Total**: 3-5 days for full Phase 1 & 2 completion

---

## Questions & Blockers

### If You Encounter Issues
1. **Check existing patterns**: Look at `trigram_embed.cu` and `trigram_embed_bridge.py`
2. **Test incrementally**: Validate each component before moving to next
3. **Ask Daniel**: If architectural decision needed (don't guess)

### Expected Blockers
- PTX kernel compilation errors → Check CUDA version, arch flags
- Symbol encoding edge cases → Start with simple symbols (+, ×), expand gradually
- Font compatibility → Not all fonts have all glyphs (expected)

---

## Success Criteria

**Phase 1 Complete** when:
- Math fonts loadable via PIL/ImageFont
- Registry has ≥200 symbols categorized
- `get_character_script('∑')` → `"math"`
- Font rendering test passes

**Phase 2 Complete** when:
- PTX kernel compiles and runs
- Semantic encoding produces valid 32D vectors (no NaN/Inf)
- GPU sovereignty validation passes
- Triple fusion works for math symbols, double fusion for others

---

## Final Notes

**This is not a crude stub**. You are building the **semantic foundation** that will enable K3D to:
1. **See** math symbols (CNN visual recognition)
2. **Know** what they mean (RPN semantic embedding)
3. **Use** them correctly (Expression composition - future)
4. **Reason** with them (Symbolic manipulation - future)

Each symbol trained = One more mathematical operation the model can **execute natively** via its RPN PTX brain.

**As Daniel said**: "We have a gem, we must teach the model how to leverage its own brain."

The Math Galaxy is how we teach it. Good luck! 🧠🚀
