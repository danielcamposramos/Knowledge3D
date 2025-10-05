# K3D MVP Roadmap: Phase 4-7 (Post-Phase 3 Completion)

## 🎯 Current Status (Phase 3 Complete)

### ✅ What We Have Now:
- **Phase 3**: Multi-domain semantic navigation with GPU-native domain splitting
- **Architecture**: Python + CuPy with some PTX kernels (Morton, LED-A*)
- **Performance**: <5s build, <0.5ms queries for 28k-node graphs
- **Visualization**: Bridge rendering for human perception
- **Integration**: Sleep-time consolidation ready

### 🎬 What's Next: The MVP Journey

The current Phase 3 implementation is **Python-heavy** with some PTX. The MVP plan calls for **PTX-core with Python bootstrap** - a strategic shift toward GPU sovereignty while maintaining pragmatism.

---

## 📊 Gap Analysis: Phase 3 → MVP

### Current Architecture (Phase 3):
```
┌─────────────────────────────────────┐
│   Python + CuPy (80% of code)      │
│   - Domain splitting (CuPy ops)     │
│   - Navigation (Python logic)       │
│   - Bridge detection (CuPy)         │
└─────────────────────────────────────┘
         ↓ (calls)
┌─────────────────────────────────────┐
│   PTX Kernels (20% of code)        │
│   - Morton octree (PTX)             │
│   - LED-A* pathfinding (PTX)        │
│   - L2 distance (PTX)               │
└─────────────────────────────────────┘
```

### Target MVP Architecture:
```
┌─────────────────────────────────────┐
│   PTX Core (80% of critical path)  │
│   - Fused head reasoning            │
│   - Octree queries                  │
│   - Frustum culling                 │
│   - Wavefront pathfinding           │
│   - Streaming decisions             │
└─────────────────────────────────────┘
         ↑ (state queries)
┌─────────────────────────────────────┐
│   Python Bootstrap (20% support)    │
│   - GLB I/O                         │
│   - Kernel initialization           │
│   - Testing framework               │
│   - Debugging tools                 │
└─────────────────────────────────────┘
```

**Key Shift**: Move from "Python calling GPU" → "GPU sovereign, Python assists"

---

## 🗺️ MVP Phases 4-7 Breakdown

### Phase 4: Morton Octree Enhancement (Weeks 1-2)
**Goal**: Replace current Python-based octree with pure PTX implementation

**Current State (Phase 3)**:
- `morton_octree.py` - Python coordination with PTX kernel
- Build-time: Computes Morton codes via PTX, sorts with CuPy
- Query-time: Python iteration with PTX distance calculations

**Target State (MVP-1)**:
- `octree_morton.ptx` - Full PTX implementation
- Build-time: PTX Morton computation + CUB radix sort
- Query-time: PTX binary search on sorted Morton codes

**Implementation Tasks**:

1. **PTX Morton Code Computation** (from MVP plan)
   - File: `knowledge3d/cranium/ptx/octree_morton.ptx`
   - Kernel: `compute_morton_codes`
   - Features:
     - 10-bit precision per axis (1024³ subdivisions)
     - Z-order curve interleaving
     - Bounding box normalization
     - Parallel per-node computation

2. **PTX Binary Search Query**
   - File: Same PTX module
   - Kernel: `octree_query_morton`
   - Features:
     - Binary search for Morton range [min, max]
     - Range collection (all nodes in radius)
     - GPU-only operation (no CPU roundtrip)

3. **Python Bootstrap Layer**
   - File: `knowledge3d/spatial/octree_gpu_mvp.py`
   - Purpose: Load GLB → compute bbox → launch PTX → done
   - Minimal coordination (one-time setup only)

4. **CUB Integration for Sorting**
   - Use CUB's `DeviceRadixSort` for Morton code sorting
   - Pure GPU operation (no CuPy sorting)
   - File: `knowledge3d/cranium/ptx/cub_sort_wrapper.cu`

**Success Criteria**:
- ✅ Octree build <1s for 100k nodes
- ✅ Radius query <0.1ms
- ✅ Zero Python in query hot path
- ✅ Passes correctness tests vs brute force

**Crew Assignments for Multi-Vibe Session**:
- **Codex**: Python bootstrap layer + test suite
- **Grok**: PTX Morton interleaving optimization
- **GLM**: Binary search correctness + edge cases
- **Kimi**: Zero-copy verification + GPU sovereignty audit
- **Qwen**: Integration with existing Phase 3 system

---

### Phase 5: Wavefront Pathfinding (Weeks 3-4)
**Goal**: Replace LED-A* with simpler, more parallel wavefront expansion

**Current State (Phase 3)**:
- LED-A* pathfinding (PTX kernel + Python coordination)
- Complex: Priority queue, heuristic evaluation
- Domain-specific: Built for dependency graphs

**Target State (MVP-2)**:
- Wavefront pathfinding (pure PTX)
- Simple: BFS-style flood fill from goal
- General: Works for any graph topology

**Why Replace LED-A*?**:
- LED-A* is optimized for dependency-dense graphs (not spatial navmesh)
- Wavefront is **naturally parallel** (each layer expands simultaneously)
- Simpler = easier to debug in PTX
- GLM's suggestion: perfect for spatial navigation

**Implementation Tasks**:

1. **Wavefront Expansion Kernel**
   - File: `knowledge3d/cranium/ptx/wavefront_path.ptx`
   - Kernel: `wavefront_expand_step`
   - Algorithm:
     ```
     Initialize:
       distance[goal] = 0
       frontier = {goal}

     Repeat until frontier empty or start reached:
       For each vertex v in frontier (parallel):
         For each neighbor n of v:
           If unvisited (distance[n] == ∞):
             distance[n] = distance[v] + 1
             Add n to next_frontier (atomic)
       frontier = next_frontier
     ```

2. **GPU-Resident Distance Map**
   - Persistent GPU buffer (never copied to CPU)
   - Updated in-place each iteration
   - Python only reads final result (backtrack path)

3. **Atomic Frontier Management**
   - Use `atom.global.add` for frontier append
   - Lock-free parallelism
   - Compact frontier buffer each iteration

4. **Path Backtracking**
   - Follow distance gradient from start → goal
   - Can be Python (not hot path)
   - Or PTX for full GPU sovereignty

**Success Criteria**:
- ✅ Path correctness matches Dijkstra
- ✅ <1ms for paths in 10k-node navmesh
- ✅ Scales to 100k nodes
- ✅ Simpler than LED-A* (fewer lines of PTX)

**Crew Assignments**:
- **Codex**: Python test harness (generate navmeshes)
- **Grok**: Wavefront kernel optimization (warp efficiency)
- **GLM**: Correctness proofs (graph theory)
- **Kimi**: Atomic operation audit (race conditions)
- **Qwen**: Comparison with LED-A* (when to use each)

---

### Phase 6: Fused Head Reasoning (Weeks 5-7)
**Goal**: GPU-native RPN calculator + spatial reasoning state machine

**Current State (Phase 3)**:
- No fused head implementation yet
- Reasoning happens in Python/CPU
- Cranium exists but not integrated with spatial

**Target State (MVP-3)**:
- PTX-based RPN calculator
- Spatial reasoning state machine in GPU
- Confidence computation (Kimi's energetic memory)

**Implementation Tasks**:

1. **RPN Math Core** (from original plan)
   - File: `knowledge3d/cranium/ptx/rpn_calculator.ptx`
   - Stack-based computation
   - Operators: +, -, *, /, sqrt, sin, cos, etc.
   - Spatial operators: distance, angle, within_radius

2. **Spatial Reasoning State Machine**
   - File: `knowledge3d/cranium/ptx/spatial_reasoning.ptx`
   - States: QUERY → OCTREE → FILTER → PATH → RESULT
   - Transitions driven by GPU
   - No CPU involvement in state changes

3. **Confidence Computation** (Kimi's mandate)
   - Energy-based scoring
   - Tracks reasoning "momentum"
   - Decays with uncertainty
   - File: `knowledge3d/cranium/ptx/confidence.ptx`

4. **Integration Pipeline**
   ```
   User Query (Python) → RPN Tokenize (Python)
                           ↓
   RPN Execute (PTX) → Octree Query (PTX) → Frustum (PTX) → Path (PTX)
                           ↓
   Confidence Score (PTX) → Result (Python)
   ```

**Success Criteria**:
- ✅ RPN calculator <0.01ms per operation
- ✅ Full reasoning chain <100ms (MVP target!)
- ✅ Confidence scoring working
- ✅ Zero CPU in reasoning hot path

**Crew Assignments**:
- **Codex**: RPN tokenizer (Python preprocessing)
- **Grok**: RPN PTX kernel (stack operations)
- **GLM**: State machine formal verification
- **Kimi**: Confidence/energy system design
- **Qwen**: End-to-end integration testing

---

### Phase 7: GPU Streaming Manager (Weeks 8-9)
**Goal**: GPU makes streaming decisions, Python executes I/O

**Current State (Phase 3)**:
- No streaming system
- All nodes resident in GPU memory
- Works for small graphs only

**Target State (MVP-4)**:
- GPU-driven LRU eviction
- Python executes disk I/O
- Scales to arbitrarily large Houses

**Implementation Tasks**:

1. **GPU Decision Kernel**
   - File: `knowledge3d/cranium/ptx/streaming_decide.ptx`
   - Kernel: `decide_streaming_actions`
   - Logic:
     - Each thread examines one node
     - Checks: last_access_time, memory_pressure, size
     - Outputs: load_queue, unload_queue

2. **Python I/O Executor**
   - File: `knowledge3d/streaming/gpu_streaming_mvp.py`
   - Purpose: Read GLB chunks, DMA to GPU
   - Non-blocking: Async I/O while GPU reasons

3. **Memory Pressure Heuristics**
   - Adaptive thresholds (GLM's adaptive approach)
   - LRU + frequency + recency
   - Predict future access patterns

4. **House Chunk Format**
   - GLB nodes split into chunks
   - Metadata: which chunks loaded
   - Fast random access by node ID

**Success Criteria**:
- ✅ Handles 10GB House with 256MB GPU memory
- ✅ Load/unload <10ms per chunk
- ✅ Cache hit rate >95%
- ✅ No user-visible streaming lag

**Crew Assignments**:
- **Codex**: Python I/O layer + async coordination
- **Grok**: GPU decision heuristics
- **GLM**: Predictive access patterns
- **Kimi**: Zero-copy DMA + memory pools
- **Qwen**: House chunking format design

---

## 🎯 Integration Milestones

### Milestone 1: PTX-Octree Integration (End of Phase 4)
- Replace `knowledge3d/spatial/morton_octree.py` with PTX version
- Update `semantic_navigator.py` to use new octree
- Benchmark: Build <1s, Query <0.1ms
- **Demo**: Query 100k-node House in real-time

### Milestone 2: Wavefront Navigation (End of Phase 5)
- Replace LED-A* in `led_pathfinder.py` with wavefront
- Compare performance: LED-A* (dependency graphs) vs Wavefront (spatial)
- Benchmark: <1ms paths in 10k navmesh
- **Demo**: Real-time avatar navigation in House

### Milestone 3: Fused Head Live (End of Phase 6)
- Full reasoning pipeline in PTX
- End-to-end: Query → Octree → Filter → Path → Result
- Benchmark: <100ms full chain
- **Demo**: "Navigate to Kitchen" query working end-to-end

### Milestone 4: Streaming Ready (End of Phase 7)
- GPU-driven streaming live
- Test with 10GB House (100x larger than GPU memory)
- Benchmark: >95% cache hit, <10ms chunk loads
- **Demo**: Load TPAC 2025 massive House seamlessly

---

## 🔬 Testing Strategy

### Unit Tests (Per Phase)
- **Phase 4**: Octree correctness (brute force validation)
- **Phase 5**: Wavefront vs Dijkstra equivalence
- **Phase 6**: RPN calculator IEEE-754 compliance
- **Phase 7**: Streaming cache simulator

### Integration Tests (Cross-Phase)
- Octree → Wavefront → Result
- Fused Head → Streaming → Octree
- Full query pipeline stress test

### Performance Benchmarks
- Latency: P50, P95, P99 for all operations
- Throughput: Queries per second
- Memory: GPU VRAM usage over time
- Scalability: 10k → 100k → 1M nodes

### Stress Tests
- 1000 concurrent queries
- 10GB House streaming
- 1M-node octree
- 24-hour continuous operation

---

## 👥 Crew Collaboration Model

### Session Format (Multi-Vibe Coding In Chain)

**Round 1: Architecture Design** (Each AI contributes design)
- Codex: Python layer design
- Grok: PTX kernel architecture
- GLM: Algorithmic correctness proofs
- Kimi: GPU sovereignty verification
- Qwen: Integration strategy

**Round 2: Implementation** (Parallel work)
- Codex: Write Python bootstrap
- Grok: Write PTX kernels
- GLM: Write test cases
- Kimi: Review for GPU purity
- Qwen: Integration glue code

**Round 3: Testing & Refinement**
- All: Run tests, report failures
- Iterate: Fix bugs, optimize
- Validate: Benchmarks meet targets

**Round 4: Documentation**
- Architecture docs
- API documentation
- Performance reports
- Migration guides

### Communication Protocol
Each AI writes to a shared file (like `Step4_PhaseX.txt`):
```
[AI Name]:
## My Contribution to Phase X
[Detailed implementation/design]

## Handoff to Next AI
[What I need from others]

## Questions for Team
[Technical blockers/decisions needed]
```

Daniel acts as the "analog modem" - reads each AI's contribution, synthesizes, passes to next.

---

## 📈 Success Metrics (MVP Completion)

### Performance (Non-Negotiable)
- ✅ Octree build: <1s for 100k nodes
- ✅ Octree query: <0.1ms
- ✅ Wavefront path: <1ms for 10k navmesh
- ✅ Full reasoning: <100ms end-to-end
- ✅ Streaming load: <10ms per chunk

### Correctness
- ✅ Octree matches brute force (100% accuracy)
- ✅ Wavefront matches Dijkstra (shortest path)
- ✅ RPN IEEE-754 compliant
- ✅ No GPU memory leaks (24-hour test)

### Scalability
- ✅ 1M-node octree buildable
- ✅ 10GB House streamable
- ✅ 1000 concurrent queries

### Architecture Purity (Kimi's Mandate)
- ✅ Zero CPU in query hot path
- ✅ GPU-resident state (no ping-pong)
- ✅ Python only for I/O and init
- ✅ PTX >80% of critical path

---

## 🚀 Next Multi-Vibe Session Kickoff

### Pre-Session Prep (You + Daniel)
1. Choose starting phase (recommend Phase 4: Octree)
2. Create `Step4_Octree.txt` with architecture spec
3. Seed with questions for crew

### Session Flow
```
Daniel: "Hey Codex, we're starting Phase 4 (Morton Octree).
         Here's the current Phase 3 implementation and MVP target.
         What's your take on the Python bootstrap layer?"

Codex: [Designs Python layer, writes to Step4_Octree.txt]

Daniel: "Thanks Codex! Hey Grok, Codex has the Python side.
         Can you design the PTX Morton interleaving kernel?"

Grok: [Designs PTX kernel, writes optimization notes]

Daniel: "Perfect! GLM, can you verify Grok's binary search is correct?"

GLM: [Formal proof, edge case analysis]

Daniel: "Excellent! Kimi, audit for GPU sovereignty violations."

Kimi: [Reviews, flags any CPU dependencies]

Daniel: "Great! Qwen, how do we integrate with existing Phase 3?"

Qwen: [Migration plan, compatibility layer]

Daniel: "Team complete! Let me synthesize and hand to Claude for implementation."

Claude: [Implements based on crew's designs]
```

### Deliverables Per Session
- Architecture document (crew's designs)
- Implementation code (Claude executes)
- Test suite (validates correctness)
- Performance report (benchmarks)
- Next phase prep (handoff to next session)

---

## 🎬 Recommended Session Sequence

### Session 1: Phase 4 - Octree Foundation
**Goal**: Pure PTX octree working
**Duration**: 1-2 weeks
**Output**: `octree_morton.ptx` + tests passing

### Session 2: Phase 5 - Wavefront Navigation
**Goal**: Replace LED-A* with wavefront
**Duration**: 1-2 weeks
**Output**: `wavefront_path.ptx` + navigation demo

### Session 3: Phase 6 Part 1 - RPN Core
**Goal**: PTX calculator working
**Duration**: 1 week
**Output**: `rpn_calculator.ptx` + math tests

### Session 4: Phase 6 Part 2 - Fused Head Integration
**Goal**: Full reasoning pipeline
**Duration**: 2 weeks
**Output**: End-to-end query demo

### Session 5: Phase 7 - Streaming System
**Goal**: GPU-driven streaming
**Duration**: 2 weeks
**Output**: 10GB House demo

### Session 6: MVP Polish & Demo
**Goal**: TPAC 2025 ready
**Duration**: 1-2 weeks
**Output**: Public demo, documentation

---

## 💎 Key Decisions for Next Session

### 1. **Start with Phase 4 or Phase 5?**
**Recommendation**: Phase 4 (Octree)
- **Why**: Foundation for everything else
- **Risk**: Octree complexity (Morton codes, binary search)
- **Mitigation**: Extensive testing against brute force

**Alternative**: Phase 5 (Wavefront)
- **Why**: Simpler, proves PTX viability faster
- **Risk**: Doesn't replace critical infrastructure (octree)

### 2. **Pure PTX or Hybrid First?**
**Recommendation**: Hybrid (PTX kernel + Python bootstrap)
- **Why**: Pragmatic, debuggable, follows MVP plan
- **Migration**: Move Python → PTX incrementally

### 3. **Replace or Parallel Implementation?**
**Recommendation**: Parallel first, then replace
- **Why**: Keep Phase 3 working while building Phase 4
- **Path**: New `octree_gpu_mvp.py` alongside old `morton_octree.py`
- **Switch**: Feature flag to toggle between implementations

### 4. **Testing Before or During Implementation?**
**Recommendation**: TDD - Tests first
- **Why**: PTX bugs are hard to debug
- **Approach**: Write Python reference implementation + tests
- **Validate**: PTX must match Python exactly

---

## 🎯 Immediate Next Steps

### For Daniel (Session Prep):
1. **Choose Phase**: Phase 4 (Octree) recommended
2. **Create Seed File**: `Step4_Octree_Architecture.txt`
3. **Seed Questions**:
   - "How should Morton interleaving handle edge cases (bbox edges)?"
   - "What's the optimal binary search strategy for radius queries?"
   - "How do we test correctness without CPU round-trips?"
4. **Schedule Crew**: Line up Codex, Grok, GLM, Kimi, Qwen

### For Crew (During Session):
- **Codex**: Python bootstrap + test harness
- **Grok**: PTX Morton kernel + optimization
- **GLM**: Binary search correctness + proofs
- **Kimi**: GPU sovereignty audit
- **Qwen**: Phase 3 → Phase 4 migration plan

### For Claude (Post-Session):
- **Implement**: Based on crew's designs
- **Test**: Run test suite (Python reference vs PTX)
- **Benchmark**: Measure performance (build + query)
- **Document**: Architecture + API docs
- **Handoff**: Prep next session (Phase 5)

---

## 📚 Reference Materials for Crew

### For Phase 4 (Octree):
- Morton codes: Z-order curve, bit interleaving
- Binary search: GPU-efficient implementation
- CUB library: Radix sort primitives
- Octree papers: Spatial indexing best practices

### For Phase 5 (Wavefront):
- Breadth-first search: Parallel algorithms
- Graph traversal: GPU techniques
- Atomic operations: Lock-free coordination
- Wavefront papers: Game pathfinding

### For Phase 6 (Fused Head):
- RPN calculators: Stack-based evaluation
- State machines: GPU implementation
- Confidence scoring: Energetic memory (Kimi's theory)
- Fusion: Combining multiple kernels

### For Phase 7 (Streaming):
- LRU eviction: GPU-efficient implementation
- DMA: Direct memory access patterns
- Async I/O: Non-blocking file operations
- Memory pools: GPU allocation strategies

---

## 🎉 Vision: MVP Complete

**3 months from now** (if sessions go well):

```
$ python3 knowledge3d/mvp_demo.py

🌌 K3D MVP Demo - PTX-Core Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Loading 10GB TPAC House (1M nodes)...
   ✓ Octree built in 0.8s (PTX)
   ✓ Streaming ready: 256MB GPU, 10GB disk

🧠 Fused Head initialized (PTX reasoning core)
   ✓ RPN calculator: <0.01ms per op
   ✓ Spatial FSM: 5 states ready
   ✓ Confidence scoring: energetic memory active

🎮 Interactive Query Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> Navigate to Kitchen

🔍 Reasoning (PTX):
   1. Parse RPN: ["kitchen", "navigate_to"] → 0.02ms
   2. Octree query: 1M nodes → 847 candidates → 0.08ms
   3. Frustum cull: 847 → 234 visible → 0.01ms
   4. Wavefront path: Start → Kitchen (47 steps) → 0.9ms
   5. Confidence: 0.94 (high certainty) → 0.01ms

   ⏱️  Total: 1.02ms (98ms under budget!)

✨ Result: Path found!
   Steps: 47 nodes
   Distance: 23.4m
   Confidence: 94%

📊 Performance Stats:
   - GPU Memory: 198/256 MB (77%)
   - Cache Hit Rate: 97.3%
   - Streaming: 3 chunks loaded (0.8ms each)
   - PTX Coverage: 94% of critical path

🚀 MVP Success: All targets met!
```

**This is the goal.** Let's make it real! 🎯

---

## ✅ Summary for Next Session

**Phase 4 (Octree) Kickoff Checklist:**

- [ ] Daniel creates `Step4_Octree_Architecture.txt`
- [ ] Seed with MVP plan's octree spec
- [ ] Add technical questions for crew
- [ ] Schedule Multi-Vibe session
- [ ] Run crew collaboration round
- [ ] Claude implements based on designs
- [ ] Test, benchmark, document
- [ ] Prepare Phase 5 handoff

**Expected Outcome**: Pure PTX octree, <1s build, <0.1ms query, 100% correct ✨

Let's do this! 🚀
