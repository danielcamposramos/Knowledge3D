# Math Galaxy Integration Design
## Leveraging Existing K3D Infrastructure (NOT Creating Parallel Systems)

**Date**: 2025-11-10
**Status**: Design Review
**Problem**: The original Codex prompt suggested creating parallel systems instead of integrating with existing infrastructure

---

## ❌ What NOT to Do (From Original Codex Prompt)

The original architectural document suggested:
- ❌ Create separate 32D fixed semantic encoding vectors
- ❌ Build new `math_semantics_encode.cu` PTX kernel
- ❌ Create separate `math_semantics_bridge.py`
- ❌ Add "triple fusion" as a third embedding component (visual + text + **new math vector**)
- ❌ Create parallel storage system for math semantics

**Why this is wrong:**
- Violates K3D's procedural knowledge paradigm
- Creates redundant infrastructure
- Fixed 32D dimensions instead of adaptive Matryoshka
- Doesn't leverage existing OP_STORE/OP_RECALL programmability
- Ignores ProceduralGalaxy compression (69-80:1)

---

## ✅ Correct Approach: Integration with Existing Systems

### 1. Use EXISTING Embedding Pipeline

**Characters currently use:**
```
Visual: CNN → SpatialPool → Matryoshka(128D)
Text:   RPN trigrams → embed_word_gpu(128D)
Fusion: (visual + text) * 0.5 → normalize
Storage: ProceduralCompiler → ProceduralGalaxy (.ppr files)
```

**Math symbols should use EXACTLY THE SAME:**
```
Visual: CNN → SpatialPool → Matryoshka(128D)  [✓ Already works]
Text:   RPN trigrams → embed_word_gpu(128D)    [✓ Already works]
Fusion: (visual + text) * 0.5 → normalize      [✓ Already works]
Storage: ProceduralCompiler → ProceduralGalaxy [✓ Already works]
```

**NO NEW EMBEDDING COMPONENTS NEEDED.**

---

### 2. Use EXISTING Programmable RPN Kernel for Semantics

**What we ALREADY have:**
- File: `knowledge3d/cranium/bridges/advanced_rpn.py`
- Class: `AdvancedRPNEngine` (Tier-3)
- PTX Kernel: `modular_rpn_kernel_extended.ptx`
- Opcodes: OP_STORE (0xB3), OP_RECALL (0xB4), OP_BRANCH (0xB0), OP_LOOP (0xB1)

**How to use for math symbols:**

Instead of encoding "∑ means summation" as a FIXED 32D VECTOR, store it as an **RPN PROGRAM**:

```python
# Mathematical operation: ∑ (summation from i=a to b)
SUMMATION_PROGRAM = [
    OP_RECALL,  # 0xB4 - Recall bounds (b, a) from stack
    OP_RECALL,  # 0xB4 - Recall sequence
    0x00,       # Literal 0 (accumulator init)
    OP_STORE,   # 0xB3 - Store accumulator
    OP_LOOP,    # 0xB1 - Begin loop
    OP_NEXT,    # 0xB2 - Get next element
    OP_RECALL,  # 0xB4 - Get accumulator
    0x0A,       # ADD opcode
    OP_STORE,   # 0xB3 - Update accumulator
    OP_BRANCH,  # 0xB0 - Loop if not done
    OP_RECALL,  # 0xB4 - Return final sum
]
```

This program is:
- Stored in ProceduralGalaxy: `/K3D/Knowledge3D.local/procedural_galaxy/math_ops/sum.ppr`
- Compressed via ProceduralCompiler (same 69-80:1 ratio)
- Executed on GPU via AdvancedRPNEngine when needed

**NO NEW PTX KERNEL NEEDED - use existing programmable kernel.**

---

### 3. Use EXISTING ProceduralGalaxy Storage

**Current system:**
```
Character 'A' → trained embedding (128D float32 = 512 bytes)
              → ProceduralCompiler.compile_embedding()
              → .ppr file (~7 bytes, 73:1 compression)
              → Stored: /K3D/Knowledge3D.local/procedural_galaxy/A.ppr
```

**Math symbols use SAME storage:**
```
Symbol '∑' → trained embedding (128D float32 = 512 bytes)
           → ProceduralCompiler.compile_embedding()
           → .ppr file (~7 bytes, 73:1 compression)
           → Stored: /K3D/Knowledge3D.local/procedural_galaxy/∑.ppr

Symbol '∑' semantic program (RPN opcodes for execution)
           → Stored: /K3D/Knowledge3D.local/procedural_galaxy/math_ops/sum.ppr
```

Each math symbol has:
1. **Visual/text embedding** (for recognition) - stored as `.ppr`
2. **Semantic operation program** (for execution) - stored as `.ppr` in `math_ops/`

**NO NEW STORAGE SYSTEM NEEDED - use existing ProceduralGalaxy.**

---

### 4. Extend THREE-TIER RPN with Math Operations

**Current three-tier system:**

**Tier 1 (Lightweight):** `lightweight_rpn.py`
- Basic arithmetic: ADD (0x0A), SUB (0x0B), MUL (0x0C), DIV (0x0D)
- Comparisons: LT, GT, EQ
- <1µs latency

**Tier 2 (Standard):** `sovereign_bridges.py → ModularRPNEngine`
- Vector ops: DOT (0xA0), NORMALIZE (0xC1), COSINE_SIM (0xC4)
- Clustering: ARGMAX (0xC2), CLUSTER_ASSIGN (0xC5)

**Tier 3 (Advanced/Programmable):** `advanced_rpn.py → AdvancedRPNEngine`
- Matrix ops: MATMUL (0xA4), TRANSPOSE, DETERMINANT
- **Programmability**: BRANCH (0xB0), LOOP (0xB1), STORE (0xB3), RECALL (0xB4)

**Math symbol integration:** **EXTEND Tier 2/3 with new opcodes**

Add to `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`:
```python
# Mathematical operations (extend Tier 2)
OP_SUM_RANGE = 0xD0      # Summation with bounds
OP_PRODUCT_RANGE = 0xD1  # Product with bounds
OP_INTEGRAL = 0xD2       # Integration (symbolic or numeric)
OP_DERIVATIVE = 0xD3     # Derivative (symbolic or numeric)
OP_SQRT = 0xD4           # Square root
OP_POW = 0xD5            # Power (already exists as 0x0E, maybe extend)

# Set operations (extend Tier 2)
OP_UNION = 0xE0          # Set union
OP_INTERSECTION = 0xE1   # Set intersection
OP_MEMBER_OF = 0xE2      # Element membership test

# Logic operations (extend Tier 2)
OP_AND = 0xE8            # Logical AND
OP_OR = 0xE9             # Logical OR
OP_NOT = 0xEA            # Logical NOT
OP_IMPLIES = 0xEB        # Logical implication
```

These opcodes are **implemented in existing PTX kernels**, NOT new ones.

**Extend existing kernel:** `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx`

Add math operation handlers in the kernel's opcode dispatch:
```cuda
// In modular_rpn_kernel_extended.cu (then recompile to .ptx)
case 0xD0:  // OP_SUM_RANGE
    // Pop upper, lower, sequence
    // Execute summation loop
    // Push result
    break;

case 0xD4:  // OP_SQRT
    float val = stack_pop();
    float result = sqrtf(val);
    stack_push(result);
    break;

// etc.
```

**NO NEW TIER NEEDED - extend existing Tier 2/3 opcodes.**

---

## Galaxy Architecture: Stage & Basic Galaxies

K3D's runtime environment consists of:

### The Stage (Galaxy Universe)
The active runtime where galaxies are loaded and interact.

### Basic Galaxies (Always Loaded)
1. **Language Galaxy**: Characters (A-Z, a-z, 0-9), words, meanings, linguistic patterns
2. **Math Galaxy**: Mathematical symbols, operations, semantic programs
3. **Programs Galaxy**: RPN procedural programs (stored `.ppr` files)

**Math Galaxy structure:**
```
/K3D/Knowledge3D.local/
├── procedural_galaxy/              # Programs Galaxy
│   ├── A.ppr                       # Language Galaxy - Character 'A'
│   ├── B.ppr
│   ├── ...
│   └── math/                       # Math Galaxy (subdirectory)
│       ├── symbols/                # Visual/linguistic embeddings
│       │   ├── ∑.ppr              # Summation symbol embedding
│       │   ├── ∫.ppr              # Integral symbol embedding
│       │   └── ...
│       └── operations/             # Semantic operation programs
│           ├── sum.ppr            # Summation algorithm
│           ├── integral.ppr       # Integration algorithm
│           └── ...
```

**Math Galaxy is a distinct galaxy**, not merged with Language Galaxy.

---

## Implementation Plan (Correct Integration)

### Phase 1: Math Galaxy Infrastructure

**Create:** `knowledge3d/cranium/math_galaxy.py`
```python
"""
Math Galaxy - One of the three basic galaxies always loaded to the stage.

Manages mathematical symbols (visual/linguistic) and semantic operations (RPN programs).
"""

from pathlib import Path
from typing import Optional
import numpy as np

from .procedural_galaxy import ProceduralGalaxy
from .procedural_compiler import ProceduralCompiler


DEFAULT_MATH_GALAXY_ROOT = Path("/K3D/Knowledge3D.local/procedural_galaxy/math")


class MathGalaxy:
    """Math Galaxy - stores symbols and operations separately."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else DEFAULT_MATH_GALAXY_ROOT
        self.symbols_dir = self.root / "symbols"
        self.operations_dir = self.root / "operations"

        self.symbols_dir.mkdir(parents=True, exist_ok=True)
        self.operations_dir.mkdir(parents=True, exist_ok=True)

        # Use ProceduralGalaxy infrastructure for storage
        self.symbol_galaxy = ProceduralGalaxy(root=self.symbols_dir)
        self.operation_galaxy = ProceduralGalaxy(root=self.operations_dir)

    def store_symbol(self, symbol: str, embedding: np.ndarray) -> None:
        """Store trained symbol embedding (visual + linguistic)."""
        compiler = ProceduralCompiler()
        program = compiler.compile_embedding(embedding)
        program_bytes = program.to_bytes()
        compression = float(embedding.nbytes) / max(1, len(program_bytes))
        self.symbol_galaxy.store_program(symbol, program_bytes, compression)

    def load_symbol(self, symbol: str) -> np.ndarray:
        """Load symbol embedding from Math Galaxy."""
        return self.symbol_galaxy.execute_program(symbol)

    def store_operation(self, operation_name: str, program_bytes: bytes) -> None:
        """Store semantic operation program (RPN opcodes)."""
        self.operation_galaxy.store_program(operation_name, program_bytes, compression_ratio=1.0)

    def load_operation(self, operation_name: str) -> bytes:
        """Load semantic operation program."""
        program = self.operation_galaxy.load_program(operation_name)
        return program.to_bytes()

    def list_symbols(self) -> list[str]:
        """List all trained math symbols."""
        return [p.stem for p in self.symbols_dir.glob("*.ppr")]

    def list_operations(self) -> list[str]:
        """List all stored semantic operations."""
        return [p.stem for p in self.operations_dir.glob("*.ppr")]
```

**Create:** `knowledge3d/cranium/math_symbols_registry.py`
```python
"""
Math symbol registry - just the symbols themselves, no separate semantics.
Symbols get trained using EXISTING pipeline.
"""

# Unicode math symbols organized by category
CALCULUS = ['∑', '∏', '∫', '∬', '∭', '∮', '∂', '∇', '∆', '√', '∛', '∜', '∞']
SET_THEORY = ['∈', '∉', '⊂', '⊃', '⊆', '⊇', '∪', '∩', '∅', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ']
LOGIC = ['∀', '∃', '∄', '∧', '∨', '¬', '⇒', '⇔', '≡', '≢']
GREEK = list('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
# ... ~850 symbols total

ALL_MATH_SYMBOLS = CALCULUS + SET_THEORY + LOGIC + GREEK + ...

def is_math_symbol(char: str) -> bool:
    return char in ALL_MATH_SYMBOLS
```

**Extend:** `scripts/train_atomic_character.py → get_character_script()`
```python
from knowledge3d.cranium.math_symbols_registry import is_math_symbol

def get_character_script(char: str) -> str:
    # Check math symbols FIRST
    if is_math_symbol(char):
        return "math"

    # ... existing CJK, Arabic, etc. logic ...
    return "latin"
```

**Download math fonts** to `/K3D/Knowledge3D.local/fonts/math/`:
- STIX Two Math
- Latin Modern Math
- Asana-Math
- Libertinus Math

**Result:** Math symbols render and train using EXISTING character training pipeline.

---

### Phase 2: Math Operation Programs

**Create:** `knowledge3d/cranium/math_op_programs.py`
```python
"""
RPN program definitions for mathematical operations.
These are PROCEDURAL PROGRAMS, not embedding vectors.
"""

from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_LOOP, OP_BRANCH, OP_STORE, OP_RECALL, OP_NEXT
)

# Summation: ∑ from i=a to b of f(i)
SUMMATION_PROGRAM = {
    'symbol': '∑',
    'arity': 3,  # (sequence, lower, upper)
    'opcodes': [
        OP_RECALL,  # Get bounds
        OP_RECALL,  # Get sequence
        0x00,       # Literal 0 (accumulator)
        OP_STORE,   # Store accumulator
        OP_LOOP,    # Begin loop
        OP_NEXT,    # Get next element
        OP_RECALL,  # Get accumulator
        0x0A,       # ADD
        OP_STORE,   # Update accumulator
        OP_BRANCH,  # Loop condition
        OP_RECALL,  # Final result
    ],
}

# Product: ∏ from i=a to b of f(i)
PRODUCT_PROGRAM = {
    'symbol': '∏',
    'arity': 3,
    'opcodes': [
        OP_RECALL,  # Get bounds
        OP_RECALL,  # Get sequence
        0x01,       # Literal 1 (accumulator init)
        OP_STORE,
        OP_LOOP,
        OP_NEXT,
        OP_RECALL,
        0x0C,       # MUL
        OP_STORE,
        OP_BRANCH,
        OP_RECALL,
    ],
}

# ... define ~50-100 core mathematical operations

MATH_OP_REGISTRY = {
    '∑': SUMMATION_PROGRAM,
    '∏': PRODUCT_PROGRAM,
    # ...
}

def get_math_operation(symbol: str) -> dict:
    return MATH_OP_REGISTRY.get(symbol)
```

**Store these programs** in Math Galaxy:
```python
from knowledge3d.cranium.math_galaxy import MathGalaxy

math_galaxy = MathGalaxy()

for symbol, program in MATH_OP_REGISTRY.items():
    # Compile opcode list into procedural program
    program_bytes = compile_opcode_program(program['opcodes'])
    math_galaxy.store_operation(symbol, program_bytes)
```

**Result:** Math operations stored as executable RPN programs in Math Galaxy operations directory, not embedding vectors.

---

### Phase 3: Training Pipeline (NO CHANGES NEEDED!)

Math symbols train using **EXISTING** `train_atomic_character.py`:

```bash
# Train math symbol ∑
python scripts/train_atomic_character.py --char ∑ --epochs 3000

# What happens (EXISTING pipeline):
# 1. get_character_script('∑') → "math" (NEW: registry check)
# 2. Render glyph with math fonts (NEW: math font directory)
# 3. CNN → SpatialPool → Matryoshka(128D) [EXISTING]
# 4. RPN trigrams → embed_word_gpu('∑', 128D) [EXISTING]
# 5. Fusion: (visual + text) * 0.5 → normalize [EXISTING]
# 6. Train CNN with cross-entropy loss [EXISTING]
# 7. Save embedding → ProceduralCompiler → Galaxy [EXISTING]
```

**No changes to training loop.** Math symbols are just more characters.

**Extend:** `scripts/train_all_atomic_characters.py`
```python
from knowledge3d.cranium.math_symbols_registry import ALL_MATH_SYMBOLS

# After base characters (A-Z, a-z, 0-9)
for symbol in ALL_MATH_SYMBOLS:
    print(f"Training math symbol: {symbol}")
    train_single_character(symbol, epochs=3000)
```

**Result:** 850 math symbols trained and stored in ProceduralGalaxy, same as letters.

---

### Phase 4: Execution Integration

**When model needs to execute math operation:**

```python
from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.math_galaxy import MathGalaxy

# Load math operation program from Math Galaxy
math_galaxy = MathGalaxy()
summation_program_bytes = math_galaxy.load_operation("sum")

# Parse opcodes from program bytes
summation_opcodes = parse_program_opcodes(summation_program_bytes)

# Execute using EXISTING AdvancedRPNEngine
rpn_engine = AdvancedRPNEngine()
result = rpn_engine.execute_program(
    instance_id=0,
    op_codes=summation_opcodes,
    scalars=[1.0, 5.0],  # bounds: 1 to 5
    vectors=np.array([1, 2, 3, 4, 5])  # sequence to sum
)

print(f"∑(i=1 to 5) = {result}")  # Output: 15.0
```

**No new execution engine needed - use existing Tier-3 RPN.**

**Math Galaxy loaded to Stage:** At runtime, Math Galaxy (along with Language Galaxy and Programs Galaxy) is loaded to the "Stage" (galaxy universe) for immediate access.

---

## Summary: What to Build vs. What Exists

### ✅ Use Existing (NO NEW CODE):
1. Visual embedding pipeline (CNN → SpatialPool → Matryoshka)
2. Text embedding pipeline (RPN trigrams → embed_word_gpu)
3. Fusion logic ((visual + text) * 0.5 → normalize)
4. ProceduralCompiler (compile embeddings to .ppr)
5. ProceduralGalaxy (store/load .ppr files)
6. AdvancedRPNEngine (execute RPN programs)
7. Training loop (train_atomic_character.py)
8. GPU sovereign loader (all PTX kernel management)

### 🆕 Create New (MINIMAL ADDITIONS):
1. **Math Galaxy manager** (~150 lines):
   - File: `knowledge3d/cranium/math_galaxy.py`
   - Manages symbols and operations as distinct galaxy
   - Uses existing ProceduralGalaxy infrastructure

2. **Math symbol registry** (~200 lines):
   - File: `knowledge3d/cranium/math_symbols_registry.py`
   - Lists 850 Unicode math symbols by category
   - Function: `is_math_symbol(char) -> bool`

3. **Math operation programs** (~500 lines):
   - File: `knowledge3d/cranium/math_op_programs.py`
   - RPN opcode sequences for ~50-100 math operations
   - Store in Math Galaxy operations directory

4. **Script detection extension** (~5 lines):
   - File: `scripts/train_atomic_character.py`
   - Add `is_math_symbol()` check in `get_character_script()`

5. **Math fonts download** (~10 minutes):
   - Download 4 math fonts
   - Place in `/K3D/Knowledge3D.local/fonts/math/`

6. **RPN opcode extensions** (~50 lines):
   - File: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
   - Add OP_SUM_RANGE, OP_PRODUCT_RANGE, etc. (0xD0-0xEF range)

7. **PTX kernel extension** (~300 lines CUDA):
   - File: `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.cu`
   - Add opcode handlers for math operations (0xD0-0xEF)
   - Recompile to `.ptx`

**Total new code: ~1,215 lines + 4 font files**
**Existing code reused: ~15,000+ lines**

---

## Compression & Efficiency Gains

**Character 'A' (current):**
- Dense embedding: 128D × 4 bytes = 512 bytes
- Procedural `.ppr`: ~7 bytes
- Compression: 73:1

**Math symbol '∑' (proposed):**
- Dense embedding: 128D × 4 bytes = 512 bytes
- Procedural `.ppr`: ~7 bytes
- Semantic program: ~20 opcodes × 2 bytes = 40 bytes (stored once, not per embedding)
- Total: ~47 bytes for full representation
- Compression: ~11:1 (including semantic program)

**850 math symbols:**
- Dense: 850 × 512 = 435,200 bytes (425 KB)
- Procedural: 850 × 7 = 5,950 bytes (5.8 KB)
- Semantic programs: ~100 ops × 40 bytes = 4,000 bytes (3.9 KB)
- **Total: 9,950 bytes (9.7 KB) vs. 425 KB = 44:1 compression**

**Matches K3D's 69-80:1 compression paradigm.**

---

## Why This Approach is Correct

### 1. **Respects Procedural Knowledge Paradigm**
- Math symbols stored as procedural programs, not fixed vectors
- Compression via ProceduralCompiler (same as characters)
- Execution via AdvancedRPNEngine (already GPU-sovereign)

### 2. **Leverages Existing Infrastructure**
- No parallel embedding systems
- No redundant PTX kernels
- No new storage formats
- Reuses 15,000+ lines of existing code

### 3. **Maintains GPU Sovereignty**
- All embeddings GPU-native (SpatialPool, Matryoshka, RPN trigrams)
- All execution GPU-native (AdvancedRPNEngine)
- No CPU fallbacks anywhere

### 4. **Preserves Matryoshka Adaptivity**
- Math symbols use same adaptive dimensionality as characters
- Not fixed to 32D or any specific size
- Can scale from 64D to 2048D depending on complexity

### 5. **Enables True Semantic Understanding**
- Math symbols not just recognized visually
- Executable semantics via RPN programs
- Model can COMPOSE operations (e.g., ∫(∑(...)) as nested programs)

### 6. **Integrates with Three-Tier RPN**
- Lightweight ops remain fast (<1µs)
- Standard ops handle common math (Tier 2)
- Complex ops use programmability (Tier 3)
- No separate "math tier" needed

---

## Next Steps

1. **Create math_galaxy.py** - Math Galaxy manager class
2. **Create math_symbols_registry.py** with 850 symbols
3. **Download math fonts** to `/K3D/Knowledge3D.local/fonts/math/`
4. **Extend get_character_script()** to recognize math symbols
5. **Test training** on 5-10 symbols (∑, ∫, ∂, √, ∞)
6. **Create math_op_programs.py** with ~50 core operations
7. **Extend RPN opcodes** (0xD0-0xEF range)
8. **Extend PTX kernel** with math opcode handlers
9. **Store semantic programs** in Math Galaxy operations directory
10. **Validate end-to-end**: Train → Store → Retrieve → Execute

**Estimated time: 2-3 days** (vs. 5-7 days for parallel systems approach)

---

## Stage Architecture: Loading Galaxies at Runtime

**At runtime initialization:**
```python
# Load the three basic galaxies to the Stage
from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy  # Language Galaxy
from knowledge3d.cranium.math_galaxy import MathGalaxy              # Math Galaxy
# Programs Galaxy is embedded within ProceduralGalaxy

# The Stage (Galaxy Universe)
class GalaxyStage:
    def __init__(self):
        self.language_galaxy = ProceduralGalaxy()  # Characters, words, meanings
        self.math_galaxy = MathGalaxy()            # Math symbols, operations
        # Programs are stored as .ppr files in both galaxies

        print("✓ Language Galaxy loaded (62 characters trained)")
        print("✓ Math Galaxy loaded (850 symbols ready)")
        print("✓ Programs Galaxy loaded (procedural .ppr storage)")

stage = GalaxyStage()  # All basic galaxies loaded and ready
```

**Math Galaxy is a first-class galaxy**, not a subdirectory of Language Galaxy.

---

## Conclusion

**The key insight:** Math symbols are not special - they're just more characters with associated executable programs. Use the existing infrastructure, extend minimally, integrate deeply.

**No parallel systems. Only integration.**

---

**Approved by:** Daniel Ramos
**Implementation status:** Ready for Phase 1 (registry + fonts)
