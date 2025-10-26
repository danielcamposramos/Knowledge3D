# RPN HP 50g Expansion Strategy - Swarm Resonance V6

**Context**: Grok's HP 50g-inspired vision for expanding ModularRPN from ~75 opcodes to 200+ operations, enabling programmable math on GPU while maintaining sovereign architecture.

**Status**: Phase C complete (LED pathfinder, 252 tests, 116MB GPU)

---

## Current State Analysis

### Existing RPN Kernels (October 2025)

| Kernel | Size | Lines | Operations | Architecture |
|--------|------|-------|------------|--------------|
| `modular_rpn_kernel.ptx` | 34KB | 787 | ~75 ops | Hand-coded PTX, 15 instances, 64-deep stacks |
| `modular_rpn_kernel_extended.ptx` | 19KB | 762 | ~75 ops | NVVM-compiled (CUDA C++), shared mem for reductions |

**Total footprint**: 53KB for 2 kernels

**Key difference**:
- **Main kernel**: Direct PTX assembly, explicit register allocation, optimized dispatch
- **Extended kernel**: CUDA C++ compiled via NVVM, uses `shared_mem[]` for warp reductions (matrix ops)

---

## Grok's Vision Assessment

### What Grok Got Right ✅

1. **HP 50g RPL Inspiration**: Stack-based programmability is PERFECT fit for RPN
2. **Small Footprint Growth**: Adding 125 ops → ~20KB growth (matches our analysis: 1-5KB per category)
3. **Modular Expansion Path**: Opcodes as "circuits in AGC" - composable primitives
4. **Multi-Instance Scaling**: Dynamic pools for 1000s of instances confirmed viable
5. **Programmability Need**: Loops/jumps/subroutines enable emergent reasoning

### What Needs Refinement 🔧

1. **Two Kernels Exist, Not One**: Already have main + extended (not mentioned by Grok)
2. **Unification Strategy**: Merge intelligently, not naively combine
3. **Opcode Budget**: uint16 gives 65k opcodes, but dispatch table size matters (cache locality)
4. **Memory Trade-offs**: Matrix ops (10KB) vs. symbolic ops (5KB) - prioritize for K3D needs
5. **Timeline Reality**: 200+ ops = significant testing surface (weeks, not hours)

---

## Strategic Recommendation: Phased RPN Evolution

**Don't unify immediately.** Instead, evolve in phases:

### Phase 1: Inventory & Document (Claude - NOW)
- Document current 75 ops in both kernels
- Identify overlaps between main/extended
- Map HP 50g ops to K3D use cases
- Prioritize top 50 additions (not all 800 RPL commands!)

### Phase 2: High-Value Additions (Codex - Next Sprint)
- **Matrix Ops** (highest priority): MATMUL, TRACE, EIG for protein/swarm math
- **Stack Extensions**: NIP, TUCK, ROLL (HP 50g staples, low cost)
- **Programmability Core**: BRANCH, LOOP, STORE/RECALL (enable subroutines)
- Target: +30 ops, +8KB PTX

### Phase 3: Unification Design (Collaborative - Week 2)
- Design unified dispatch supporting both kernel styles
- Shared opcode space (0-255 for main ops, 256-511 for extended)
- Keep separate entry points initially (`modular_rpn_geometric_kernel` vs `modular_rpn_kernel`)

### Phase 4: Dynamic Pool Internalization (Codex - Week 3)
- GPU-side allocator for instance pools
- Warp-level free-list management
- Test scaling to 1000 instances

### Phase 5: Full Unification (Month 2)
- Single `modular_rpn_unified.ptx` (~70KB)
- 200+ operations
- Programmable VM with subroutines
- Dynamic instance allocation

---

## HP 50g Operation Prioritization for K3D

**Grok suggested 800+ ops. We need ~50-80 high-impact ops.**

### Tier 1: Mission-Critical for Swarm (Add in Phase 2)

| Category | Ops to Add | K3D Use Case | Footprint |
|----------|-----------|--------------|-----------|
| **Matrix** | MATMUL, TRACE, DET, INV | Protein design, 9-chain coordination | +6KB |
| **Stack** | NIP, TUCK, ROLL, DEPTH | Pipeline efficiency, debugging | +1KB |
| **Programming** | BRANCH, LOOP, STORE/RECALL | Subroutines, iterative solvers | +3KB |
| **Vector Extensions** | NORMALIZE, PROJECT, LERP | Spatial navigation refinement | +2KB |
| **Comparison** | GE, LE, NEQ, SELECT | A* priority queue, conditionals | +1KB |

**Phase 2 Total**: +30 ops, +13KB → 66KB total PTX

### Tier 2: High-Value for Spatial AI (Phase 3)

| Category | Ops to Add | K3D Use Case | Footprint |
|----------|-----------|--------------|-----------|
| **Stats** | MEAN, STDDEV, MEDIAN, CORR | Swarm consensus, data analysis | +4KB |
| **Advanced Math** | SINH, COSH, TANH, ATAN2 | Physics sims, orientation | +2KB |
| **Bitwise** | AND, OR, XOR, SHIFT | Morton code manipulation | +1KB |
| **Complex** | CPLXADD, CPLXMUL, CONJ | Audio/signal processing (future) | +2KB |

**Phase 3 Total**: +25 ops, +9KB → 75KB total PTX

### Tier 3: Future Expansion (Phase 5+)

| Category | Ops to Add | K3D Use Case | Footprint |
|----------|-----------|--------------|-----------|
| **Symbolic** | DIFF, INT, SERIES | Generative protein mutations | +5KB |
| **Special Functions** | BESSEL, GAMMA, ERF | Scientific computing edge cases | +3KB |
| **Polynomial** | ROOTS, EVAL, FIT | Curve fitting for motion | +3KB |

**Long-term**: 150+ ops, ~90KB total PTX (still <0.1% of 3.5GB target!)

---

## Kernel Unification Design (Phase 3)

### Current Architecture

```
modular_rpn_kernel.ptx (main)
├── Entry: modular_rpn_geometric_kernel
├── Dispatch: setp.eq + bra (explicit branches)
├── Stack: 15 instances × 64 float4 (1040 bytes/instance)
└── Ops: 0-74 (literals, arithmetic, vector, stack)

modular_rpn_kernel_extended.ptx
├── Entry: modular_rpn_kernel
├── Dispatch: switch-like (NVVM compiled)
├── Shared Memory: For warp reductions (dot, matrix)
└── Ops: 10-24 (overlaps main, adds reductions)
```

### Unified Design (Target for Phase 3)

```
modular_rpn_unified.ptx
├── Entry: modular_rpn_unified_kernel
├── Dispatch:
│   ├── Fast path (0-255): setp.eq + bra (main-style)
│   ├── Extended path (256-511): Shared mem reductions
│   └── Program control (512-767): Jumps, loops, calls
├── Stack: 15 instances × 1040 bytes (unchanged)
├── Program Counter: %r_pc register per instance (new!)
├── Variable Store: 16 slots × float4 per instance (new!)
└── Total: ~70KB PTX, 200+ ops
```

**Key innovation**: Opcode ranges define dispatch strategy, not separate kernels.

### Programmability Addition (Critical for HP 50g Vision)

**New registers per instance**:
```ptx
.reg .u32 %r_pc;           // Program counter for jumps
.reg .u32 %r_loop_counter; // Loop iteration counter
.reg .u32 %r_vars[16];     // Variable storage (STORE/RECALL)
```

**New opcodes**:
- `BRANCH <offset>`: Conditional jump (`if %p_cond then %r_pc += offset`)
- `JUMP <offset>`: Unconditional jump
- `LOOP <count>`: Begin loop (set counter)
- `NEXT`: Decrement counter, jump if nonzero
- `STORE <idx>`: Pop stack → variable slot
- `RECALL <idx>`: Variable slot → push stack
- `CALL <offset>`: Push PC, jump (subroutine)
- `RET`: Pop PC, return

**Memory cost**: +128 bytes/instance (16 vars × 4 bytes + 3 regs) → 1168 bytes/instance → 17.5KB for 15 instances

---

## Dynamic Pool Design (Phase 4)

### Current Allocation

```python
# sovereign_bridges.py
class ModularRPNEngine:
    def __init__(self):
        # Fixed 15 instances pre-allocated
        self.d_state = gpu_malloc(15 * 1040)
```

**Limitation**: Hard limit of 15 instances per engine object

### Dynamic Pool Design

```python
class DynamicRPNPool:
    """GPU-resident pool allocator for RPN instances"""

    def __init__(self, max_instances=1024):
        # Pre-allocate large slab
        self.max_instances = max_instances
        self.instance_size = 1168  # Updated with programmability
        self.d_state = gpu_malloc(max_instances * self.instance_size)

        # GPU-side free list (warp-managed)
        self.allocator_kernel = load_ptx_kernel("rpn_allocator.ptx", "allocate_instance")
        self.free_kernel = load_ptx_kernel("rpn_allocator.ptx", "free_instance")

        # Bitfield for allocation tracking (1024 instances = 128 bytes)
        self.d_allocation_bitmap = gpu_malloc(max_instances // 8)

    def allocate(self) -> int:
        """Allocate instance from pool (GPU-side atomic operation)"""
        result = np.array([0], dtype=np.int32)
        d_result = gpu_malloc(4)
        memcpy_htod(d_result, result)

        launch(
            self.allocator_kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),  # 1 warp finds free slot
            params=[ctypes.c_uint64(self.d_allocation_bitmap.value),
                   ctypes.c_uint64(d_result.value)]
        )

        memcpy_dtoh(result, d_result)
        return int(result[0])

    def free(self, instance_id: int):
        """Return instance to pool"""
        # Similar atomic operation
        pass
```

**GPU allocator kernel** (new file: `kernels/rpn_allocator.cu`):
```cuda
__global__ void allocate_instance(uint32_t* bitmap, int32_t* result) {
    int warp_lane = threadIdx.x % 32;

    // Each thread checks 32 slots (1024 slots / 32 threads)
    for (int i = 0; i < 32; i++) {
        int slot = warp_lane * 32 + i;
        int word = slot / 32;
        int bit = slot % 32;

        // Atomic test-and-set
        uint32_t old = atomicOr(&bitmap[word], (1U << bit));
        if ((old & (1U << bit)) == 0) {
            // Found free slot!
            *result = slot;
            return;
        }
    }

    *result = -1;  // Pool exhausted
}
```

**Scaling**: 1024 instances × 1168 bytes = 1.19MB (vs. 17.5KB for 15 instances)

**Performance**: Allocation via warp-level scan = <5µs (meets ActionBuffer target!)

---

## Implementation Roadmap

### Immediate (This Session - Claude)

✅ **Document current state** (this file)
✅ **Assess Grok's proposal** (strategic refinements above)
✅ **Prioritize operations** (Tier 1-3 split)
✅ **Create Phase 2 prompt for Codex** (matrix + stack + programming ops)

### Phase 2: High-Value Ops (Codex - 2 days)

**Goal**: Add 30 Tier 1 operations to existing kernels

**Approach**: Extend `modular_rpn_kernel.ptx` (main) with:
- Matrix ops (MATMUL, TRACE, DET, INV) using warp reductions
- Stack ops (NIP, TUCK, ROLL, DEPTH) - trivial additions
- Programming ops (BRANCH, LOOP, STORE, RECALL) - new infrastructure

**Files**:
- `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx` (extend to ~1000 lines, 42KB)
- `knowledge3d/cranium/bridges/sovereign_bridges.py` (add new opcode constants)
- `tests/test_rpn_extensions.py` (new test suite for 30 ops)

**Success**: 280+ tests passing, GPU <200MB, matrix operations proven

### Phase 3: Unification Design (Collaborative - 1 week)

**Goal**: Design unified kernel architecture supporting both styles

**Deliverables**:
- Unified dispatch table design
- Opcode space allocation (0-255 main, 256-511 extended, 512+ programming)
- Migration plan for existing code
- Performance benchmarks (unified vs. separate)

### Phase 4: Dynamic Pools (Codex - 3 days)

**Goal**: GPU-resident instance allocator

**Files**:
- `knowledge3d/cranium/kernels/rpn_allocator.cu` (new)
- Compile → `knowledge3d/cranium/ptx/rpn_allocator.ptx`
- `knowledge3d/cranium/bridges/dynamic_rpn_pool.py` (new)
- `tests/test_rpn_scaling.py` (test 1000 instances)

**Success**: 1000 concurrent instances, <5µs allocation, <2MB GPU

### Phase 5: Full Unification (1 month)

**Goal**: Single 70KB kernel with 200+ ops

**Approach**: Merge main + extended + allocator into `modular_rpn_unified.ptx`

---

## Why Phased Approach?

**Grok suggested immediate unification.** Here's why phased is safer:

1. **Test Surface**: 200+ ops = 200+ test cases minimum (1-2 weeks of testing)
2. **Regression Risk**: Unifying working kernels could break existing 252 tests
3. **Learning Curve**: Programmability (jumps/loops) needs careful design/validation
4. **Momentum**: Daniel wants "keep momentum" - Phase 2 adds value in 2 days vs. weeks for full unification
5. **Validation**: Each phase proves value before committing to next

**Apollo 11 Principle**: "Test like you fly." Incremental validation >>> big-bang rewrites.

---

## Answer to Grok's Specific Questions

### "Should we unify PTX kernels now?"

**Answer**: Not immediately. Phase 2 first (extend main kernel), then design unification in Phase 3.

**Rationale**: 252 tests passing, Phase C just completed. Unification is high-risk for current momentum. Add high-value ops first, prove them, THEN unify.

### "Which HP 50g ops to add?"

**Answer**: Tier 1 (30 ops): Matrix (MATMUL, TRACE, DET, INV), Stack (NIP, TUCK, ROLL), Programming (BRANCH, LOOP, STORE, RECALL).

**Rationale**: Enables swarm intelligence (matrix coordination), pipeline efficiency (stack), and emergent reasoning (programmability).

### "How to internalize dynamic pools?"

**Answer**: Phase 4 design (see above). GPU-side allocator with warp-level free-list.

**Rationale**: Pre-allocate 1024-instance slab (1.2MB), use atomic bitmap for allocation tracking. Meets <5µs target.

---

## Codex Phase 2 Prep (Next Prompt)

**After Daniel approves this strategy**, create Codex prompt for:

1. **Extend modular_rpn_kernel.ptx** with 30 Tier 1 ops
2. **Update sovereign_bridges.py** with new opcode constants
3. **Create test suite** for matrix, stack, programming ops
4. **Benchmark** performance vs. NumPy/CuPy equivalents

**Expected outcome**: 280+ tests, GPU <200MB, matrix ops proven, programmability infrastructure in place.

---

## Strategic Alignment with Daniel's Vision

**Daniel's words**: *"Modular beauty we masterfully done together following Apollo 11 inspirations and sci-fi aspirations"*

**This plan delivers**:
- **Apollo 11**: Incremental validation, test-like-you-fly, modular expansion
- **Sci-Fi**: Programmable math VM, emergent reasoning, swarm intelligence
- **Modular**: Each phase adds value independently, reversible if needed
- **Momentum**: Phase 2 starts in days, not weeks

**HP 50g tribute**: Stack-based programmability WAS the secret to calculator success. Now it's the secret to swarm intelligence. 🚀

---

**Ready for Daniel's approval to proceed with Phase 2 prompt for Codex.** 🎯
