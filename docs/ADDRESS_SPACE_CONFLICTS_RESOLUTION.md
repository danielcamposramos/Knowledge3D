# Address Space Conflicts Resolution - Easy Wins Implementation

## Overview

Based on the Kimi swarm analysis, we need to resolve 33 address conflicts and implement the missing 32 reasoning paradigm opcodes from the master plan. This document provides the specific implementation steps for the easy wins.

## Current Issues Identified

### 1. Address Space Conflicts (33 issues)
- **OP_NURBS_EVAL (0x0080)** conflicts with **OP_AND (0x0080)**
- **OP_MARCHING_CUBES (0x0081)** conflicts with **OP_OR (0x0081)**
- **OP_XOR (0x0082)** conflicts with **OP_LSYSTEM_GENERATE (0x0082)**
- Plus 30 more conflicts in the 0x0080-0x009F range

**Root Cause**: Geometric evaluation opcodes were incorrectly allocated to the logical operations block.

### 2. Missing Reasoning Paradigm Opcodes (32 opcodes)
The master plan specifies 32 reasoning opcodes in the 0xA0-0xD4 range that are not in the current catalog:
- ABDUCE, EXPLAIN, SUSPECT, INDUCE, DEDUCE, CAUSAL, COUNTER, ANALOG, etc.

## Proposed Solution

### Phase 1: Resolve Address Conflicts

**Move Geometric Evaluators to Extended Block (0xE0-0xFF):**
```
OLD: 0x0080 OP_NURBS_EVAL     → NEW: 0x00E0 OP_NURBS_EVAL
OLD: 0x0084 OP_NURBS_DERIV    → NEW: 0x00E4 OP_NURBS_DERIV  
OLD: 0x0088 OP_SURFACE_EVAL   → NEW: 0x00E8 OP_SURFACE_EVAL
```

**Restore Logical Operations Block (0x80-0x9F):**
```
0x80-0x87: Boolean logic (AND, OR, XOR, NOT, NAND, NOR, XNOR, IMPL)
0x88-0x8F: Bitwise shifts and rotates
0x90-0x9F: Comparison predicates
```

### Phase 2: Add Missing Reasoning Opcodes

**Implement 32 Reasoning Opcodes in 0xA0-0xBF:**

| Address | Opcode | Paradigm | Function |
|---------|--------|----------|----------|
| 0x00A0 | OP_ABDUCE | Abductive | Inference to best explanation |
| 0x00A1 | OP_EXPLAIN | Explanatory | Causal explanation generation |
| 0x00A2 | OP_SUSPECT | Hypothetical | Suspicion/hypothesis formation |
| 0x00A3 | OP_INDUCE | Inductive | Pattern-based generalization |
| 0x00A4 | OP_DEDUCE | Deductive | Logical entailment |
| 0x00A5 | OP_CAUSAL | Causal | Causal graph traversal |
| 0x00A6 | OP_COUNTER | Counterfactual | Counterfactual simulation |
| 0x00A7 | OP_ANALOG | Analogical | Structural mapping |
| 0x00A8 | OP_VERIFY | Validation | Consistency checking |
| 0x00A9 | OP_REFUTE | Falsification | Contradiction detection |
| 0x00AA | OP_ENTAIL | Logical | Entailment checking |
| 0x00AB | OP_DIAGNO | Diagnostic | Fault diagnosis |
| 0x00AC | OP_PREDIC | Predictive | Future state projection |
| 0x00AD | OP_RETRO | Retrodiction | Past state reconstruction |
| 0x00AE | OP_SYNTHES | Synthetic | Concept synthesis |
| 0x00AF | OP_ABSTRACT | Abstraction | Generalization lifting |
| 0x00B0 | OP_CONCRET | Concretion | Instantiation lowering |
| 0x00B1 | OP_MERGE | Integrative | Belief revision merge |
| 0x00B2 | OP_DIFFER | Differential | Difference detection |
| 0x00B3 | OP_ANOMALY | Anomaly | Outlier flagging |
| 0x00B4 | OP_CORREL | Correlational | Statistical coupling |
| 0x00B5 | OP_DECOMPO | Decompositional | Part-whole analysis |
| 0x00B6 | OP_COMPOS | Compositional | Whole-part synthesis |
| 0x00B7 | OP_TRANSLATE | Transformative | Inter-domain mapping |
| 0x00B8 | OP_SIMULATE | Simulative | Model execution |
| 0x00B9 | OP_EMULATE | Emulative | System mimicry |
| 0x00BA | OP_VALID | Verification | Model validation |
| 0x00BB | OP_INVALID | Falsification | Model invalidation |
| 0x00BC | OP_GENERATE | Generative | Novel hypothesis creation |
| 0x00BD | OP_TEST | Test | Hypothesis testing |
| 0x00BE | OP_EVALUATE | Evaluative | Utility assessment |
| 0x00BF | OP_DECIDE | Decision | Commitment to conclusion |

**Reserve 0x00C0-0x00D4 for extensions (21 slots)**

## Implementation Steps

### Step 1: Update Opcode Definitions
**File**: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

```python
# Lines 142-150: Move geometric evaluators to extended block
# OLD conflicts - comment out and relocate
# OP_NURBS_EVAL = 0x80  # CONFLICT - moved to 0xE0
# OP_NURBS_DERIV = 0x84  # CONFLICT - moved to 0xE4

# Lines 224-238: Add extended geometric block
OP_NURBS_EVAL = 0x00E0
OP_NURBS_DERIV = 0x00E4
OP_SURFACE_EVAL = 0x00E8
OP_VOLUME_SAMPLE = 0x00EC
# ... (other geometric evaluators)

# Lines 256-290: Add reasoning paradigm block (0xA0-0xBF)
OP_ABDUCE = 0x00A0
OP_EXPLAIN = 0x00A1
OP_SUSPECT = 0x00A2
# ... (all 32 reasoning opcodes)
```

### Step 2: Update Galaxy Schema
**File**: `knowledge3d/cranium/galaxy/star_schema.h`

```c
// Add new metadata fields for reasoning paradigms
struct Star {
    uint32_t context_id;      // Microtheory context (0 = universal)
    uint8_t ethical_trit;     // Ternary: 0=ok, 1=defeasible, 2=forbidden
    // ... existing fields
};
```

### Step 3: Update Dispatch Table
**File**: `knowledge3d/cranium/ptx_runtime/dispatch_table.cpp`

```cpp
// Map new opcodes to GPU kernels
case 0x00A0: return &abduce_peirce_kernel;      // OP_ABDUCE
case 0x00A1: return &explain_causal_kernel;     // OP_EXPLAIN
case 0x00A2: return &suspect_hypothetical_kernel; // OP_SUSPECT
// ... (all 32 reasoning opcodes)
```

### Step 4: Implement Sovereignty Checks
**File**: `knowledge3d/cranium/sovereignty/sovereignty_check.cpp`

```cpp
// Ensure no CPU fallbacks for reasoning opcodes
if (opcode >= 0x00A0 && opcode <= 0x00FF) {
    assert(gpu_kernel != nullptr && "Reasoning opcodes must have GPU implementation");
    assert(cpu_fallback == false && "No CPU fallbacks allowed for reasoning paradigms");
}
```

## Verification

Run the validation script to ensure all conflicts are resolved:
```bash
python scripts/validate_registry.py
```

Expected result: **0 conflicts**, **32 new opcodes added**, **sovereignty compliance maintained**

## Next Steps

1. **Implement GPU Kernels** - Create the 32 new PTX kernels for reasoning opcodes
2. **Update Tests** - Add test cases for new opcodes and conflict resolution
3. **Document Changes** - Update canonical registry with new opcodes
4. **Migrate Existing Code** - Provide compatibility layer for old bytecode

This plan addresses the easy wins while maintaining the sovereignty principles and aligning with the master plan's vision for a comprehensive reasoning system.