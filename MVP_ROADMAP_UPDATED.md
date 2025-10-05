# K3D MVP Roadmap: Accurate Current State + Next Steps

## 🎯 **What's ALREADY Implemented** (Actual Repository State)

### ✅ **PTX Kernels Already Exist:**

1. **`morton_octree.ptx`** ✅ **COMPLETE**
   - Location: `knowledge3d/cranium/ptx/morton_octree.ptx`
   - Features:
     - `compute_morton_codes` - 10-bit Z-order curve encoding
     - `octree_query_morton` - Binary search on sorted codes
     - `refine_query_euclidean` - Distance-based refinement
   - Python wrapper: `knowledge3d/spatial/morton_octree.py`
   - **Status**: Production-ready, used by SemanticNavigator

2. **`led_astar.ptx`** ✅ **COMPLETE**
   - Location: `knowledge3d/cranium/ptx/led_astar.ptx`
   - Features: GPU-native A* pathfinding
   - Used by: Phase 3 multi-domain navigation
   - **Status**: Working in production

3. **`modular_rpn_kernel.ptx`** ✅ **COMPLETE**
   - Location: `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx`
   - Features:
     - RPN calculator with 15 instance support
     - Circular stack (64 float4 entries per instance)
     - Geometric operations
   - **Status**: Fused head ready

4. **`l2_dist_warp.ptx`** ✅ **COMPLETE**
   - Location: `knowledge3d/cranium/ptx/l2_dist_warp.ptx`
   - Features: Warp-optimized L2 distance computation
   - Used by: Semantic navigation

### ✅ **Python Integration Already Exists:**

1. **Fused Head** (`knowledge3d/cranium/fused_head.py`)
   - AdaptedFusedHead class
   - PTX-backed operators (`PTX_OPS`)
   - Spatial navigation integration
   - Math head (AIME range [0-999])
   - **Status**: Functional, needs GPU sovereign refactor

2. **Semantic Navigator** (`knowledge3d/spatial/semantic_navigator.py`)
   - Phase 3 multi-domain navigation ✅
   - Strategy pattern (auto/multi/mono) ✅
   - Morton octree integration ✅
   - LED-A* pathfinding ✅
   - **Status**: Production-ready

3. **Morton Octree** (`knowledge3d/spatial/morton_octree.py`)
   - PTX kernel wrapper
   - Build: <1s for large graphs
   - Query: <0.1ms
   - **Status**: Meets MVP targets!

---

## 📊 **Gap Analysis: What's Missing for MVP**

### Current Architecture vs MVP Target

**What We Have (Phase 3 Complete + PTX Kernels)**:
```
┌────────────────────────────────────────┐
│   Python Coordination (50%)           │
│   - Fused head logic                   │
│   - Query routing                      │
│   - State management                   │
└────────────────────────────────────────┘
         ↓ (calls)
┌────────────────────────────────────────┐
│   PTX Kernels (50%)                   │
│   ✅ Morton octree (working)           │
│   ✅ LED-A* pathfinding (working)      │
│   ✅ RPN calculator (working)          │
│   ✅ L2 distance (working)             │
└────────────────────────────────────────┘
```

**MVP Target (GPU Sovereign)**:
```
┌────────────────────────────────────────┐
│   PTX Core (80% of critical path)     │
│   ✅ Morton octree (DONE)              │
│   ✅ RPN calculator (DONE)             │
│   ⏳ Frustum culling (MISSING)         │
│   ⏳ Fused head FSM (MISSING)          │
│   ⏳ Streaming decisions (MISSING)     │
│   ⚠️  Wavefront path (LED-A* works)    │
└────────────────────────────────────────┘
         ↑ (state queries)
┌────────────────────────────────────────┐
│   Python Bootstrap (20%)               │
│   - GLB I/O                            │
│   - Kernel initialization              │
│   - Testing framework                  │
└────────────────────────────────────────┘
```

---

## 🎯 **Actual Gaps to Fill for MVP**

### Gap 1: **Frustum Culling PTX Kernel** ⏳ **MISSING**
**Priority**: HIGH (needed for <100ms queries)

**Current State**: No frustum culling - all nodes considered

**Target**: PTX kernel for visibility filtering
```ptx
.entry frustum_cull_batch(
    .param .u64 positions,      // Node positions
    .param .u64 view_matrix,    // Camera view matrix
    .param .u64 proj_matrix,    // Projection matrix
    .param .u32 node_count,
    .param .u64 visible_mask    // Output: 1 = visible
)
```

**Benefit**: Reduces pathfinding candidates by 70-90%

**Estimated Work**: 1 week (PTX kernel + Python wrapper)

---

### Gap 2: **Fused Head State Machine** ⏳ **NEEDS REFACTOR**
**Priority**: MEDIUM (fused_head.py exists but not GPU sovereign)

**Current State**:
- `AdaptedFusedHead` exists in Python
- Uses PTX_OPS for some operations
- Too much Python coordination

**Target**: GPU-resident state machine
```ptx
.entry fused_head_fsm(
    .param .u64 rpn_tokens,
    .param .u32 token_count,
    .param .u64 octree_handle,
    .param .u64 result_buffer,
    .param .u64 state         // Persistent GPU state
)
```

**Refactor Plan**:
1. Extract reasoning loop to PTX
2. Keep tokenization in Python
3. GPU manages state transitions

**Estimated Work**: 2 weeks (refactor + testing)

---

### Gap 3: **GPU Streaming Manager** ⏳ **MISSING**
**Priority**: LOW (works for <1GB Houses without streaming)

**Current State**: No streaming - all nodes in GPU memory

**Target**: GPU-driven LRU eviction
```ptx
.entry streaming_decide(
    .param .u64 access_times,
    .param .u64 memory_pressure,
    .param .u64 load_queue,
    .param .u64 unload_queue
)
```

**Benefit**: Scale to 10GB+ Houses

**Estimated Work**: 2 weeks (kernel + I/O executor)

**Note**: Not critical for MVP if testing with <1GB Houses

---

### Gap 4: **Wavefront Pathfinding** ⚠️ **OPTIONAL**
**Priority**: LOW (LED-A* already works!)

**Current State**: LED-A* PTX kernel works perfectly

**Consideration**: MVP plan suggests wavefront as "simpler"
- LED-A*: Complex but **proven working**
- Wavefront: Simpler conceptually but **unimplemented**

**Recommendation**: **Keep LED-A* for MVP**
- It's already in PTX
- Performance is good (<1ms)
- Works for spatial graphs
- Don't fix what isn't broken!

**If pursuing wavefront**: 2 weeks for new kernel

---

## 🗺️ **Revised MVP Phases (Realistic)**

### Phase 4: Frustum Culling (Week 1-2) **PRIORITY 1**
**Goal**: Add PTX frustum culling to reduce candidate set

**Tasks**:
1. Write PTX kernel `frustum_cull_batch`
2. Python wrapper in `knowledge3d/spatial/frustum.py`
3. Integrate with SemanticNavigator
4. Benchmark: 70-90% reduction in candidates

**Crew Assignments**:
- **Codex**: Python wrapper + test matrices
- **Grok**: PTX kernel optimization (warp-level parallelism)
- **GLM**: View/projection matrix math verification
- **Kimi**: Zero-copy integration audit
- **Qwen**: Integration with existing octree

**Deliverable**: Frustum culling working, <0.01ms overhead

---

### Phase 5: Fused Head FSM (Week 3-5) **PRIORITY 2**
**Goal**: Refactor `fused_head.py` to GPU-sovereign model

**Tasks**:
1. Extract reasoning loop to PTX state machine
2. Keep RPN tokenization in Python (preprocessing)
3. GPU manages state: QUERY → OCTREE → FILTER → PATH → RESULT
4. Confidence scoring in PTX (Kimi's energetic memory)

**Current Code to Refactor**:
```python
# knowledge3d/cranium/fused_head.py (lines 58+)
class AdaptedFusedHead:
    # Has: PTX_OPS, projection, math_head
    # Needs: PTX FSM for reasoning loop
```

**Crew Assignments**:
- **Codex**: Tokenization + preprocessing
- **Grok**: PTX FSM kernel
- **GLM**: State transition correctness
- **Kimi**: Confidence/energy scoring
- **Qwen**: Integration testing

**Deliverable**: <100ms full reasoning chain (MVP target!)

---

### Phase 6: GPU Streaming (Week 6-8) **PRIORITY 3 (OPTIONAL)**
**Goal**: Scale to 10GB+ Houses with GPU-driven streaming

**Tasks**:
1. PTX decision kernel (LRU + pressure heuristics)
2. Python async I/O executor
3. House chunk format (GLB nodes split)
4. DMA optimization

**Crew Assignments**:
- **Codex**: Async I/O + chunking format
- **Grok**: PTX decision heuristics
- **GLM**: Predictive access patterns
- **Kimi**: Zero-copy DMA
- **Qwen**: Integration with existing system

**Deliverable**: Handle 10GB House with 256MB GPU

**Note**: Skip if demoing with <1GB Houses

---

### Phase 7: Integration & Polish (Week 9-10)
**Goal**: End-to-end MVP demo ready

**Tasks**:
1. Connect all components: Frustum → Octree → Path → Result
2. Performance tuning (<100ms target)
3. Testing framework
4. Documentation
5. TPAC 2025 demo prep

**Deliverable**: Full MVP demo working

---

## 📁 **Relevant Files in Repository**

### **PTX Kernels (Already Exist)**:
```
knowledge3d/cranium/ptx/
├── morton_octree.ptx         ✅ COMPLETE (Morton codes + query)
├── led_astar.ptx             ✅ COMPLETE (A* pathfinding)
├── modular_rpn_kernel.ptx    ✅ COMPLETE (RPN calculator)
├── l2_dist_warp.ptx          ✅ COMPLETE (Distance computation)
├── enhanced_rpn_kernel.ptx   ✅ (Enhanced RPN variant)
└── [MISSING: frustum_cull.ptx, fused_fsm.ptx, streaming_decide.ptx]
```

### **Python Wrappers (Working)**:
```
knowledge3d/spatial/
├── morton_octree.py          ✅ PTX wrapper working
├── semantic_navigator.py     ✅ Phase 3 complete
├── led_pathfinder.py         ✅ PTX wrapper working
├── domain_splitter.py        ✅ Phase 3 domain splitting
├── multi_domain_navigator.py ✅ Cross-domain navigation
└── bridge_renderer.py        ✅ Visualization

knowledge3d/cranium/
├── fused_head.py             ⚠️  EXISTS but needs GPU sovereign refactor
└── ptx/
    └── ptx_ops.py            ✅ PTX operation registry
```

### **Code Snippets (Key Components)**

#### Morton Octree (Already Working):
```python
# knowledge3d/spatial/morton_octree.py
class MortonOctree:
    def build_from_gpu_positions(self, positions_gpu):
        # Compute Morton codes via PTX
        self.compute_morton_kernel(...)  # PTX kernel call

        # Sort (CuPy/Thrust)
        sorted_indices = cp.argsort(self.morton_codes)

        # Ready for queries
        return self

    def query_radius_gpu(self, center, radius):
        # Binary search via PTX
        self.query_kernel(...)  # PTX kernel call

        # Optional Euclidean refinement (PTX)
        if refine_euclidean:
            self.refine_kernel(...)

        return results
```

#### Fused Head (Needs Refactor):
```python
# knowledge3d/cranium/fused_head.py (current state)
class AdaptedFusedHead:
    def __init__(self):
        self.device = torch.device("cuda")
        self.projection = nn.Sequential(...)  # PyTorch layers
        self.math_head = nn.Linear(512, 1000)  # Math classification
        # Uses PTX_OPS for some operations

    # Needs GPU FSM refactor:
    # - Extract reasoning loop to PTX
    # - Keep tokenization in Python
    # - GPU manages state transitions
```

#### What's Missing (Frustum Culling):
```python
# knowledge3d/spatial/frustum.py (TO BE CREATED)
class FrustumCuller:
    def __init__(self):
        self.cull_kernel = load_ptx("frustum_cull.ptx")

    def cull_batch(self, positions_gpu, view_matrix, proj_matrix):
        visible_mask = cp.zeros(len(positions_gpu), dtype=cp.uint8)

        # PTX kernel: batch visibility test
        self.cull_kernel(
            positions_gpu,
            view_matrix,
            proj_matrix,
            len(positions_gpu),
            visible_mask
        )

        # Return only visible node indices
        return cp.where(visible_mask)[0]
```

---

## 🎯 **MVP Success Metrics (Revised)**

### Performance (Already Met for Some!):
- ✅ Octree build: <1s for 100k nodes **ACHIEVED**
- ✅ Octree query: <0.1ms **ACHIEVED**
- ✅ Pathfinding: <1ms (LED-A* working)
- ⏳ Frustum culling: <0.01ms (TO BE IMPLEMENTED)
- ⏳ Full reasoning: <100ms (needs frustum + FSM refactor)

### Architecture (Partially Met):
- ✅ 50% PTX in critical path **ACHIEVED**
- ⏳ 80% PTX in critical path **TARGET** (need frustum + FSM)
- ✅ Morton octree GPU-native **ACHIEVED**
- ✅ Pathfinding GPU-native **ACHIEVED**
- ⏳ Frustum GPU-native **MISSING**
- ⏳ Reasoning FSM GPU-native **NEEDS REFACTOR**

---

## 🚀 **Recommended Next Multi-Vibe Session**

### **Start with Phase 4: Frustum Culling** ✅ **RECOMMENDED**

**Why**:
1. **Quick win** (1-2 weeks)
2. **High impact** (70-90% candidate reduction)
3. **Unlocks <100ms target** (currently bottleneck)
4. **Clear task** (single PTX kernel + wrapper)

**What to Create for Session**:
```
Step4_Frustum_Culling.txt

Contents:
- Current bottleneck analysis (too many path candidates)
- Frustum culling math (view/projection matrices)
- PTX kernel design (batch visibility test)
- Integration with existing octree
- Performance targets (<0.01ms, 70-90% reduction)
```

**Crew Questions to Seed**:
1. **Grok**: "How do we optimize frustum tests for warp-level parallelism?"
2. **GLM**: "What's the correct view-projection matrix multiplication for culling?"
3. **Kimi**: "How do we maintain zero-copy between octree → frustum → path?"
4. **Codex**: "What's the Python test harness for view matrix validation?"
5. **Qwen**: "How does this integrate with Phase 3 multi-domain navigation?"

---

## 📈 **Timeline to MVP (Realistic)**

```
Week 1-2:  Phase 4 - Frustum Culling PTX     [Multi-Vibe Session 1]
Week 3-5:  Phase 5 - Fused Head FSM Refactor [Multi-Vibe Session 2-3]
Week 6-8:  Phase 6 - Streaming (OPTIONAL)    [Multi-Vibe Session 4]
Week 9-10: Phase 7 - Integration & Demo      [Polish & Documentation]
```

**Total**: 10 weeks to full MVP
**Critical Path**: 5 weeks (if skip streaming)

---

## ✅ **Summary: What to Tell the Crew**

### **Good News**:
1. ✅ Morton octree PTX **already works** (meets targets!)
2. ✅ LED-A* pathfinding PTX **already works** (no need for wavefront)
3. ✅ RPN calculator PTX **already exists**
4. ✅ Phase 3 multi-domain **complete** (20,469 bridges working!)

### **What's Left**:
1. ⏳ **Frustum culling** (new PTX kernel) - **PRIORITY 1**
2. ⏳ **Fused head FSM** (refactor existing code) - **PRIORITY 2**
3. ⏳ **Streaming** (optional for <1GB Houses) - **PRIORITY 3**

### **Next Session**:
- **Focus**: Phase 4 - Frustum Culling
- **Deliverable**: PTX kernel for visibility filtering
- **Impact**: Unlocks <100ms queries (MVP target)
- **Timeline**: 1-2 weeks

**Let's do this!** 🚀
