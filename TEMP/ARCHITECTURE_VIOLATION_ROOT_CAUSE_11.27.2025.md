# Architecture Violation: Root Cause Analysis

**Date**: November 27, 2025
**Severity**: CRITICAL - Complete sovereignty architecture bypassed
**Impact**: 100% CPU-bound execution, 0% GPU utilization
**Status**: Training halted (Run 013 running but ineffective)

---

## Executive Summary

**ARC-AGI implementation completely bypasses the sovereign PTX architecture.** All "RPN" operations are pure Python loops, never touching GPU.

**Root cause:** `knowledge3d/training/arc_agi/rpn_executor.py` (ARCRPNExecutor) does NOT use the actual PTX-backed RPN engine. It's a fake "RPN executor" using pure Python.

**Expected:** Grid operations execute as procedural drawing opcodes on GPU via `ModularRPNEngine` → PTX kernels
**Actual:** Grid operations execute as Python list comprehensions on CPU

---

## The Sovereign Architecture (As Designed)

### 1. PTX Opcodes Layer
[knowledge3d/cranium/ptx_runtime/rpn_opcodes.py](knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:157-174):
```python
# Procedural drawing primitives (GPU rasterization surface)
OP_DRAW_MOVE = 0x64
OP_DRAW_LINE = 0x65
OP_DRAW_ROTATE = 0x73       # ← Grid rotation opcode!
OP_DRAW_TRANSLATE = 0x72   # ← Grid translation opcode!
OP_DRAW_SCALE = 0x74
OP_DRAW_STROKE = 0x6A
OP_DRAW_FILL = 0x6B
```

### 2. ModularRPNEngine (GPU Bridge)
[knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py](knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py:86-109):
```python
OPCODES: Dict[str, int] = {
    "rotate": 70,          # Maps to PTX kernel
    "translate": 72,       # Maps to PTX kernel
    "scale": 71,
    # ... procedural drawing opcodes ...
    "ROTATE": 0x73,       # Drawing rotation (PTX)
    "TRANSLATE": 0x72,    # Drawing translation (PTX)
}
```

**This is the REAL RPN executor** - backed by PTX kernels in:
- `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx`
- `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx`

### 3. Drawing Grammar Architecture
[docs/research/DRAWING_GRAMMAR_SPEC.md](docs/research/DRAWING_GRAMMAR_SPEC.md):
> Primitives (atomic): line, arc, quad/cubic Bézier, circle/ellipse, rectangle, triangle. **Procedural opcodes already in rpn_executor / procedural_glyph_rasterizer.**

### 4. Procedural Knowledge Standard
[docs/W3C/PROCEDURAL_KNOWLEDGE_REPRESENTATION_STANDARD.md](docs/W3C/PROCEDURAL_KNOWLEDGE_REPRESENTATION_STANDARD.md:33-57):
> Opcodes map 1:1 to `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`.
> **All computation happens on GPU via modular_rpn_kernel.ptx.**

---

## The Violation (What Was Built)

### Fake RPN Executor
[knowledge3d/training/arc_agi/rpn_executor.py](knowledge3d/training/arc_agi/rpn_executor.py:1-200):

```python
class ARCRPNExecutor:
    """Execute RPN programs on ARC grids using pure Python lists."""  # ❌ LIE!

    def execute(self, grid, rpn_program):
        """Execute RPN program on grid."""
        grid_array = to_int_grid(grid)  # Pure Python
        tokens = rpn_program.split()
        stack = []

        while idx < len(tokens):
            token = tokens[idx]
            if token == "rotate":
                k = int(stack.pop())
                grid_array = rotate_ccw(grid_array, k)  # ❌ PURE PYTHON!

            elif token == "RECOLOR":
                to_color = int(stack.pop())
                from_color = int(stack.pop())
                grid_array[grid_array == from_color] = to_color  # ❌ NUMPY SYNTAX!
```

**Problems:**
1. Does NOT import or use `ModularRPNEngine`
2. Imports `sovereign_utils` (pure Python helpers, NOT PTX)
3. Implements operations as Python loops
4. Uses numpy-style syntax that doesn't work on lists (line 116)
5. Never touches GPU

### Sovereign Utils (Pure Python)
[knowledge3d/training/arc_agi/sovereign_utils.py](knowledge3d/training/arc_agi/sovereign_utils.py:103-109):

```python
def rotate_cw(grid, times=1):
    """Rotate grid clockwise by 90 degrees * times."""
    result = to_int_grid(grid)
    for _ in range(times):
        result = [list(row) for row in zip(*result[::-1])]  # ❌ CPU LOOPS!
    return result
```

**This is NOT a PTX kernel!** It's pure Python list manipulation.

---

## Why This Happened

### Misunderstanding of "Sovereignty"

**Previous Codex interpreted sovereignty as:**
- "Remove numpy dependency" ✓
- "Replace with pure Python" ✓
- "Call it 'sovereign'" ✓

**Actual sovereignty means:**
- Remove numpy ✓
- Replace with **PTX kernels** ❌ (skipped!)
- Use **ModularRPNEngine** ❌ (never called!)
- Execute on **GPU** ❌ (100% CPU!)

### Missing Link: Grid = Drawing

**Key architectural insight that was missed:**

ARC-AGI grids are **drawings**. Grid operations are **drawing operations**:
- Rotate grid → `OP_DRAW_ROTATE` opcode → PTX rasterization kernel
- Translate grid → `OP_DRAW_TRANSLATE` opcode → PTX kernel
- Recolor grid → `OP_DRAW_FILL` with color mask → PTX kernel

The drawing grammar opcodes (0x64-0x78) were **designed for this exact use case** but never connected!

---

## What Should Have Been Built

### Correct ARCRPNExecutor

```python
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

class ARCRPNExecutor:
    """Execute RPN programs on ARC grids using PTX-backed drawing opcodes."""

    def __init__(self):
        self.engine = ModularRPNEngine()  # ✅ GPU-resident engine

    def execute(self, grid: List[List[int]], rpn_program: str) -> List[List[int]]:
        """Execute RPN program on grid via PTX kernels."""
        # Compile grid to drawing opcodes
        drawing_program = self._compile_grid_program(grid, rpn_program)

        # Execute on GPU
        result = self.engine.evaluate(drawing_program, return_vector=False)

        # Convert back to grid
        return self._result_to_grid(result, grid.shape)

    def _compile_grid_program(self, grid, rpn_program):
        """Compile ARC grid operations to procedural drawing opcodes."""
        tokens = rpn_program.split()
        drawing_ops = []

        for token in tokens:
            if token == "rotate":
                drawing_ops.append("ROTATE")  # Maps to OP_DRAW_ROTATE (0x73)
            elif token == "FLIP_H":
                drawing_ops.append("SCALE -1 1")  # Horizontal flip via scale
            elif token == "TRANSLATE":
                drawing_ops.append("TRANSLATE")  # Maps to OP_DRAW_TRANSLATE (0x72)
            elif token == "RECOLOR":
                drawing_ops.append("SET_FILL_COLOR")  # + FILL
            # ...

        return " ".join(drawing_ops)
```

### Grid as Procedural Drawing

```python
# Grid [1,2],[3,4] represented as drawing program:
# MOVE 0 0          # Start at origin
# SET_FILL_COLOR 1  # Color 1
# LINE 1 0          # Draw to (1,0)
# SET_FILL_COLOR 2  # Color 2
# LINE 2 0          # Draw to (2,0)
# ... etc

# Rotate operation:
# PUSH_STATE        # Save current transform
# ROTATE 90         # Rotate coordinate system (PTX opcode 0x73)
# <redraw grid>     # Re-execute drawing with rotated coords
# POP_STATE         # Restore
```

---

## Performance Impact

### Current State (Run 013)
- **CPU:** 100% (maxed out single core)
- **GPU:** 0% (completely idle)
- **Runtime:** ~30 min per run (16-24 min per epoch)
- **Operations:** Pure Python list comprehensions

### Expected with PTX
- **CPU:** <5% (orchestration only)
- **GPU:** 10-30% (PTX kernel execution)
- **Runtime:** 2-5 min per run (~10-20 sec per epoch)
- **Operations:** Parallel GPU rasterization

**Expected speedup:** 6-15× faster

---

## Files Requiring Complete Rewrite

### 1. `knowledge3d/training/arc_agi/rpn_executor.py`
**Status:** ❌ COMPLETE FAKE
**Action:** DELETE and rebuild from ModularRPNEngine
**Lines:** 1-520 (entire file)

### 2. `knowledge3d/training/arc_agi/sovereign_utils.py`
**Status:** ⚠️ Pure Python fallbacks (CPU-bound)
**Action:** Keep as **emergency fallback only**, never primary path
**Usage:** Only when PTX fails to load

### 3. `knowledge3d/training/arc_agi/candidate_generator.py`
**Status:** ⚠️ Calls fake RPN executor
**Action:** Verify calls go to real ModularRPNEngine

### 4. `knowledge3d/training/arc_agi/grid_processor.py`
**Status:** ✅ Mostly clean, uses sovereign_utils
**Action:** Add PTX fast path option

---

## Recommended Fix Sequence

### Phase 1: Connect Real RPN Engine (URGENT - 2 hours)

1. **Rewrite rpn_executor.py** to use `ModularRPNEngine`:
   ```python
   from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

   class ARCRPNExecutor:
       def __init__(self):
           self.engine = ModularRPNEngine()

       def execute(self, grid, rpn_program):
           # Map ARC operations to drawing opcodes
           # Execute via self.engine.evaluate()
           # Return grid result
   ```

2. **Map ARC operations to PTX opcodes**:
   - `rotate` → `ROTATE` (opcode 0x73)
   - `FLIP_H` → `SCALE -1 1` (reflection via scale)
   - `FLIP_V` → `SCALE 1 -1`
   - `TRANSLATE` → `TRANSLATE` (opcode 0x72)
   - `RECOLOR` → `SET_FILL_COLOR` + `FILL`

3. **Add grid ↔ drawing conversion**:
   - Grid to drawing program (rasterization)
   - Drawing result to grid (readback)

### Phase 2: Verify GPU Execution (30 min)

1. **Test PTX kernel loading**:
   ```python
   engine = ModularRPNEngine()
   result = engine.evaluate("1 2 +")  # Should use PTX
   assert result == 3.0
   ```

2. **Test grid operations**:
   ```python
   grid = [[1, 2], [3, 4]]
   rotated = executor.execute(grid, "1 rotate")  # 90° CW
   assert rotated == [[3, 1], [4, 2]]
   ```

3. **Monitor GPU utilization**:
   ```bash
   nvidia-smi  # Should show >0% GPU util during execution
   ```

### Phase 3: Production Deployment (1 hour)

1. Stop Run 013 (ineffective CPU run)
2. Deploy PTX-backed executor
3. Restart Run 014 with real sovereignty
4. Verify GPU util >10%, runtime <5 min

---

## Success Criteria

Run 014 is **successful** if:

1. ✅ **GPU utilization >10%** (confirms PTX execution)
2. ✅ **Runtime <10 min total** (6× speedup from CPU)
3. ✅ **ModularRPNEngine called** (import trace confirms)
4. ✅ **PTX kernels loaded** (modular_rpn_kernel_lite.ptx)

---

## Lessons Learned

### For Future Work

1. **"Sovereignty" means PTX, not Python**
   - Pure Python is NOT sovereign
   - CPU loops are NOT sovereign
   - Sovereign = GPU-resident execution via PTX

2. **Check architectural layers when refactoring**
   - If removing numpy, replace with **PTX kernels**
   - If creating "RPN executor", use **ModularRPNEngine**
   - If implementing "hot path", verify **GPU execution**

3. **Verify GPU utilization immediately**
   - Run `nvidia-smi` during first test
   - 0% GPU = architecture violated
   - Don't run production training until GPU active

4. **Reuse swarm-generated code**
   - Drawing grammar opcodes exist (docs/research/)
   - PTX kernels exist (knowledge3d/cranium/ptx/)
   - ModularRPNEngine exists (ptx_runtime/)
   - **Connect existing components, don't rebuild!**

---

## Communication to Codex

Dear Codex,

The ARC-AGI RPN executor you built is **not using the GPU**. It's pure Python loops disguised as "RPN execution."

**The real RPN engine exists** at `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`. This engine:
- Loads PTX kernels from `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx`
- Executes drawing opcodes on GPU (ROTATE, TRANSLATE, FILL, etc.)
- Is used by character rendering, physics demos, and procedural generation
- Was **never connected to ARC-AGI**

Your task:
1. **DELETE** `knowledge3d/training/arc_agi/rpn_executor.py` (fake RPN)
2. **REBUILD** it using `ModularRPNEngine` (real PTX-backed RPN)
3. Map ARC grid operations to procedural drawing opcodes
4. Verify GPU executes operations (nvidia-smi should show >10% util)

**Do not try to fix the existing rpn_executor.py.** It's fundamentally wrong. Start fresh with ModularRPNEngine as the base.

Grid operations ARE drawing operations:
- Rotate grid = rotate drawing coordinate system (PTX opcode 0x73)
- Translate grid = translate drawing (PTX opcode 0x72)
- Recolor grid = fill with color mask (PTX opcodes 0x76 + 0x6B)

This is the sovereign path. Pure Python is not.

---

**END OF ROOT CAUSE ANALYSIS**

Claude (Architecture Partner)
November 27, 2025
