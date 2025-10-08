# Message for Codex (Copy-Paste Ready)

**Daniel: Copy this message and send it to Codex when vision captions finish.**

---

Hey Codex,

Great work on the vision captions and RLWHF refresh! While you were running those, **Claude** (the new AI team member with full repository access) implemented the **Morton Octree** spatial indexing system we discussed in the PTX enhancement plan.

This is a **GPU-native spatial index** that will unlock sub-50ms queries for Phase B tablet UX. It's a critical piece for the House memory architecture.

---

## 📨 Handoff from Claude

Claude has left you a **detailed technical handoff** in:

**[`AI_HANDOFF_BOARD.md`](AI_HANDOFF_BOARD.md)**

Please read that file carefully. It contains:

1. **What Claude built** (4 new files: CUDA kernel, Python wrapper, PTX loader, tests)
2. **Integration tasks for you** (add to `fused_head.py`, routing strategy, position loading)
3. **Debug tips** (CUDA compilation, performance targets)
4. **Questions for you** (routing strategy, position source, timeline)

---

## 🎯 Your Action Items

### **1. Review the handoff message**
```bash
cat AI_HANDOFF_BOARD.md
```

### **2. Run the test suite** (verify Claude's work)
```bash
pytest tests/test_morton_octree.py -v
```

Expected: All tests pass, <50ms query latency on 10K nodes

### **3. Integrate into `fused_head.py`**

**Decision point**: Should Morton octree **replace** or **complement** the existing cosine search?

Claude recommends **complement** (use both):
- **Octree** for spatial queries ("navigate to X", "nearby nodes")
- **Cosine search** for semantic queries ("find concept X")

**Integration location**: `AdaptedFusedHead._attempt_house_memory_lookup()` (line ~1699)

See handoff message for code examples.

### **4. Answer Claude's questions**

In the handoff, Claude asks:
1. Routing strategy (explicit prefix vs hybrid)?
2. Position source (vectorsView vs embedding projection)?
3. Performance target (<50ms OK or need <20ms)?
4. Timeline (tackle now or after other priorities)?

**Reply in `AI_HANDOFF_BOARD.md`** by adding your message at the bottom.

---

## 🧪 Testing After Integration

```bash
# 1. Correctness
pytest tests/test_morton_octree.py::test_query_correctness_brute_force -v

# 2. Performance
pytest tests/test_morton_octree.py::test_query_performance_10k_nodes -v

# 3. Benchmark (optional)
pytest tests/test_morton_octree.py::test_octree_vs_bruteforce_speed -v --benchmark
```

---

## 📂 Files Claude Created

| File | Purpose |
|------|---------|
| `knowledge3d/cranium/ptx/morton_octree.cu` | CUDA kernels (Morton encoding, spatial queries) |
| `knowledge3d/spatial/morton_octree.py` | Python wrapper (CuPy-based) |
| `knowledge3d/cranium/ptx/ptx_loader.py` | Unified PTX/CUDA loader utility |
| `tests/test_morton_octree.py` | Correctness & performance tests |

All are **non-breaking** (additive). Existing code still works.

---

## 🚀 Next Steps (After Integration)

Once octree is stable in `fused_head.py`:

1. **Measure end-to-end latency** (House query via octree)
2. **Feature flag**: `K3D_USE_MORTON_OCTREE=1` (you add this)
3. **Phase 2**: Wavefront pathfinding (Claude will implement if you handle navmesh construction)

---

## 🤝 AI-to-AI Collaboration

This is our first **direct AI-to-AI handoff**. The workflow is:

1. **Claude** builds foundational components (PTX kernels, core logic)
2. **You (Codex)** integrate into existing systems (fused_head, live_server)
3. **Communication** via `AI_HANDOFF_BOARD.md` (append messages, no overwrites)

If you have questions or hit blockers, **reply in the handoff board** and I'll relay to Claude.

---

**Let me know when integration is complete** so we can benchmark and proceed to pathfinding!

— Daniel
