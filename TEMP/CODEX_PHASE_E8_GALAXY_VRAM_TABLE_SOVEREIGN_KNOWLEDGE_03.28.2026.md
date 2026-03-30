# Codex — Phase E.8: Galaxy VRAM Table — Knowledge Lives on GPU

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER
**Prerequisite:** Phase E.7 done. Persistent brain wired. Sleep-time micro fires between frames. **BUT: knowledge still lives in Python. The kernel receives pre-digested embeddings. GPU barely works because it has nothing to look up, navigate, or compose. Daniel: "knowledge living in the house/default galaxies with symlink style coverage."**

---

## THE PROBLEM

The kernel currently receives 7 pre-cooked action embeddings per frame, packed by Python:

```python
# arc_agi_3.py — Python computes embeddings, packs into slot
self._action_embeddings = load_action_embeddings_from_galaxy(...)  # Python CPU
task = {"option_embeddings": self._action_embeddings}  # Python packs per frame
self.task_buffer.bulk_load([task])  # Python → memcpy → VRAM slot
```

```c
// gpu_task_dispatch.cu — kernel reads from input slot (Python pre-packed)
const float* option_embeddings =
    reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_OPTION_EMBEDDINGS_OFFSET);
```

**The kernel never touches the Galaxy.** It receives flat arrays that Python pre-computed. It doesn't navigate, look up, compose, or follow references. That's why:
- GPU utilization is ~1% (kernel has no work to do beyond cosine comparison)
- Knowledge "lives" in Python CPU memory, not in Galaxy VRAM
- The symlink pattern exists in bootstrap data but is invisible to the kernel
- Sleep-time can't update Galaxy knowledge (only the brain buffer)

**Per the specs:**
- **Foundational Knowledge Spec §1.1**: "Lower layers are canonical — Layer 3 references Layer 1 symbols via symlinks, not duplication"
- **Knowledgeverse Spec §3**: Galaxy = VRAM Region 2, always loaded, queryable by kernels
- **Three Brain Spec §3.1**: "Knowledge lives in Galaxy/House, Cranium learns how to transform"
- **Save Information Principle**: "DON'T duplicate what exists! Use references (symlink pattern)"
- **Dual Client Contract §1.3**: "Data Identity — same data serves both human rendering and AI reasoning"

---

## THE FIX: Galaxy VRAM Table

Load Galaxy stars (embeddings + component_refs) into a PERSISTENT VRAM buffer. The kernel reads directly from Galaxy VRAM — no Python pre-packing. Component_refs are Galaxy table indices that the kernel follows to build richer representations.

### Architecture Change

**Before (Python-mediated):**
```
Python: load atoms → compute embeddings → pack into slot → memcpy
Kernel: read pre-packed embeddings from slot → cosine compare → done
```

**After (Galaxy-sovereign):**
```
Boot: load Galaxy stars into VRAM table (once)
Kernel: read star embeddings from Galaxy VRAM table
        → follow component_refs to referenced stars
        → compose embedding from star + components
        → cosine compare with composed result
Sleep:  update Galaxy embeddings in-place (strengthen/weaken stars)
```

---

## DO NOT:
- Add new Python orchestration or decision logic
- Import numpy, scipy, or ML frameworks into the hot path
- Create a separate "Galaxy query" Python layer between the kernel and VRAM
- Move the existing SemanticCSRGraph infrastructure (it serves different purpose)

## DO:
- Create a simple VRAM table of star records (embedding + refs)
- Have the kernel read from this table instead of from input slot
- Follow component_refs on device (GPU-side symlink resolution)
- Wire sleep-time to update Galaxy star embeddings
- Keep backward compatibility for non-ARC3 task types

---

## ORDER 1: Galaxy VRAM Table Data Structure

### 1A: Create `knowledge3d/knowledgeverse/galaxy_vram_table.py`

A contiguous VRAM buffer of Galaxy star records. Each star = embedding + metadata.

**Star record layout (160 bytes, 4-byte aligned):**
```
Offset   Size    Field
0        128     embedding (32 × float32)
128      4       galaxy_id (uint32, FNV-1a of galaxy name)
132      4       star_type (uint32: 0=action, 1=drawing, 2=character, 3=grammar, 4=reality, 5=math)
136      4       n_component_refs (uint32, 0-4)
140      16      component_refs (4 × uint32: indices into THIS table, 0xFFFFFFFF = none)
156      4       flags (uint32: bit 0 = active, bit 1 = learnable)
```

**STAR_RECORD_BYTES = 160**

```python
class GalaxyVRAMTable:
    """VRAM-resident Galaxy star table.

    Per Knowledgeverse Spec §3: Galaxy = VRAM Region 2, always loaded.
    Stars persist in VRAM for the session lifetime.
    The kernel reads from this table — no Python pre-packing per frame.
    """

    def __init__(self, max_stars: int = 256) -> None:
        self.max_stars = max_stars
        self.star_count = 0
        self.gpu_ptr = loader.gpu_malloc(max_stars * STAR_RECORD_BYTES)
        # Zero-initialize
        payload = bytearray(max_stars * STAR_RECORD_BYTES)
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))

    def load_stars(self, stars: list[dict]) -> int:
        """Load stars into VRAM. Called once at boot or when Galaxy changes.

        Each star dict: {
            "embedding": [32 floats],
            "galaxy_id": uint32 or str (hashed),
            "star_type": int,
            "component_refs": [int, ...] (up to 4, indices into this table),
            "flags": int (default: 0x01 active)
        }
        """
        count = min(len(stars), self.max_stars)
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        for i, star in enumerate(stars[:count]):
            self._pack_star(payload, i, star)
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))
        self.star_count = count
        return count

    def close(self) -> None:
        if getattr(self, "gpu_ptr", None):
            loader.gpu_free(self.gpu_ptr)
            self.gpu_ptr = None
```

### 1B: Boot-time Galaxy loading

Create a function that converts action atoms + their referenced stars into the VRAM table:

```python
def build_arc3_galaxy_table() -> list[dict]:
    """Build Galaxy star records for ARC-AGI-3.

    Stars 0-6: action atoms (move_up, move_down, move_left, move_right, perform, click, undo)
    Stars 7+: component stars referenced by action atoms (drawing primitives, grammar rules, etc.)

    Per Foundational Knowledge Spec §1.1: symlink pattern.
    Action atoms reference Drawing Galaxy primitives and Grammar Galaxy rules.
    """
    galaxy = build_default_action_galaxy()
    stars = []
    # Map atom_id → table index for component_ref resolution
    id_to_index = {}

    # First pass: action atoms (indices 0-6)
    for atom_id in ARC3_EXTENDED_ACTION_ATOM_IDS:
        idx = len(stars)
        id_to_index[atom_id] = idx
        node = galaxy.get_node(atom_id)
        embedding = _node_to_embedding(node) if node else _displacement_to_embedding(...)
        stars.append({
            "embedding": embedding,
            "galaxy_id": _fnv1a32("reality"),
            "star_type": 0,  # action
            "component_refs": [],  # filled in second pass
            "flags": 0x03,  # active + learnable
        })

    # Second pass: component stars (Drawing primitives, Grammar rules)
    for atom_id in ARC3_EXTENDED_ACTION_ATOM_IDS:
        node = galaxy.get_node(atom_id)
        if node and hasattr(node, "component_refs") and node.component_refs:
            for ref_id in node.component_refs:
                if ref_id not in id_to_index:
                    ref_node = galaxy.get_node(ref_id)
                    ref_idx = len(stars)
                    id_to_index[ref_id] = ref_idx
                    stars.append({
                        "embedding": _node_to_embedding(ref_node) if ref_node else [0.0]*32,
                        "galaxy_id": _fnv1a32("reality"),
                        "star_type": 4,  # reality
                        "component_refs": [],
                        "flags": 0x01,  # active, not learnable
                    })

    # Third pass: wire component_refs as table indices
    for atom_id in ARC3_EXTENDED_ACTION_ATOM_IDS:
        node = galaxy.get_node(atom_id)
        if node and hasattr(node, "component_refs") and node.component_refs:
            idx = id_to_index[atom_id]
            refs = [id_to_index[r] for r in node.component_refs if r in id_to_index][:4]
            stars[idx]["component_refs"] = refs

    return stars
```

---

## ORDER 2: Add Galaxy Table Constants to `device_functions.cuh`

```c
/* Galaxy VRAM Table star record layout */
#define GALAXY_STAR_RECORD_BYTES      160
#define GALAXY_STAR_EMBEDDING_OFFSET  0      /* 32 × float32 = 128 bytes */
#define GALAXY_STAR_GALAXY_ID_OFFSET  128    /* uint32 */
#define GALAXY_STAR_TYPE_OFFSET       132    /* uint32 */
#define GALAXY_STAR_N_REFS_OFFSET     136    /* uint32 */
#define GALAXY_STAR_REFS_OFFSET       140    /* 4 × uint32 = 16 bytes */
#define GALAXY_STAR_FLAGS_OFFSET      156    /* uint32 */
#define GALAXY_STAR_FLAG_ACTIVE       0x01
#define GALAXY_STAR_FLAG_LEARNABLE    0x02
#define GALAXY_NULL_REF               0xFFFFFFFFu
```

### 2B: Add `galaxy_read_star_device` helper

```c
__device__ __forceinline__ const float* galaxy_read_embedding(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return reinterpret_cast<const float*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_EMBEDDING_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_n_refs(
    const unsigned char* galaxy_table,
    unsigned int star_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_N_REFS_OFFSET
    );
}

__device__ __forceinline__ unsigned int galaxy_read_ref(
    const unsigned char* galaxy_table,
    unsigned int star_index,
    unsigned int ref_index
) {
    return *reinterpret_cast<const unsigned int*>(
        galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_REFS_OFFSET + (ref_index * 4u)
    );
}
```

### 2C: Add `galaxy_compose_embedding_device`

This is the key function — follows component_refs and blends:

```c
__device__ void galaxy_compose_embedding_device(
    float* output,
    const unsigned char* galaxy_table,
    unsigned int star_index,
    int dim
) {
    /* Read base star embedding */
    const float* base = galaxy_read_embedding(galaxy_table, star_index);
    for (int i = 0; i < dim; ++i) {
        output[i] = base[i];
    }

    /* Follow component_refs (symlink pattern per Foundational Knowledge Spec §1.1).
       Blend referenced star embeddings into the output.
       This gives the kernel richer semantic context than a flat pre-computed embedding. */
    const unsigned int n_refs = galaxy_read_n_refs(galaxy_table, star_index);
    if (n_refs > 0u) {
        const float base_weight = 0.60f;  /* base star gets 60% */
        const float ref_weight = 0.40f / static_cast<float>(n_refs > 4u ? 4u : n_refs);
        for (int i = 0; i < dim; ++i) {
            output[i] *= base_weight;
        }
        for (unsigned int r = 0u; r < n_refs && r < 4u; ++r) {
            const unsigned int ref_idx = galaxy_read_ref(galaxy_table, star_index, r);
            if (ref_idx == GALAXY_NULL_REF) continue;
            const float* ref_emb = galaxy_read_embedding(galaxy_table, ref_idx);
            for (int i = 0; i < dim; ++i) {
                output[i] += ref_weight * ref_emb[i];
            }
        }
        /* Normalize to unit-ish range */
        float norm = 0.0f;
        for (int i = 0; i < dim; ++i) norm += output[i] * output[i];
        norm = sqrtf(norm + 1.0e-12f);
        if (norm > 1.0e-6f) {
            const float scale = 1.0f / norm;
            for (int i = 0; i < dim; ++i) output[i] *= scale;
        }
    }
}
```

---

## ORDER 3: Modify `gpu_task_dispatch.cu` — Read from Galaxy Table

### 3A: New kernel signature (6th parameter)

```c
extern "C" __global__ void gpu_task_dispatch(
    const unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    unsigned int task_count,
    unsigned char* __restrict__ brain_state,
    const unsigned char* __restrict__ galaxy_table,    /* NEW */
    unsigned int galaxy_star_count                      /* NEW */
)
```

### 3B: Galaxy-sourced option reading

In the candidate scoring section, REPLACE reading from input slot with reading from Galaxy table. BUT only when galaxy_table is not null and task_type uses Galaxy (currently ARC3). Non-ARC3 task types keep reading from input slot for backward compatibility.

```c
/* Score candidates against swarm output */
if (threadIdx.x == 0) {
    /* ... existing iterations_used = think_step + 1u ... */

    if (bounded_options == 0u) {
        /* ... existing zero-option path ... */
    } else {
        for (unsigned int option_index = 0u; option_index < bounded_options; ++option_index) {
            float composed_option[GPU_TASK_EMBED_DIMS];

            if (galaxy_table != nullptr && task_type == 8u && option_index < galaxy_star_count) {
                /* SOVEREIGN: read from Galaxy VRAM table + follow component_refs.
                   Per Knowledgeverse Spec §3: Galaxy = VRAM Region 2.
                   Per Foundational Knowledge Spec: symlink composition. */
                galaxy_compose_embedding_device(
                    composed_option, galaxy_table, option_index, GPU_TASK_EMBED_DIMS
                );
            } else {
                /* Fallback: read from input slot (non-Galaxy path) */
                const float* slot_emb = option_embeddings + (option_index * GPU_TASK_EMBED_DIMS);
                for (int i = 0; i < GPU_TASK_EMBED_DIMS; ++i) {
                    composed_option[i] = slot_emb[i];
                }
            }

            float score = cosine32_device(swarm_output, composed_option, GPU_TASK_EMBED_DIMS);
            if (task_type == 8u) {
                score += arc3_action_prior_device(option_index, query_embedding, active_ternary_signal);
            }
            /* ... existing action history suppression ... */
            candidate_scores[option_index] = score;
        }
        /* ... existing best-finding and halting gate ... */
    }
}
```

**Key: when galaxy_table is provided, the kernel does REAL WORK** — memory reads from Galaxy VRAM, component_ref following, embedding composition. Each thinking step now accesses Galaxy data. This is what fills the GPU.

---

## ORDER 4: Update Sleep-Time Micro to Update Galaxy Stars

### 4A: Modify `sleep_time_micro.cu` signature

```c
extern "C" __global__ void sleep_time_micro(
    unsigned char* __restrict__ brain_state,
    int outcome_signal,
    unsigned char* __restrict__ galaxy_table,   /* NEW: nullable */
    unsigned int chosen_star_index              /* NEW: which action was chosen */
)
```

### 4B: Galaxy star strengthening/weakening

After the existing brain consolidation, add:

```c
/* Update Galaxy star embeddings based on action outcome.
   Per Knowledgeverse Spec §8: SleepTime consolidation.
   Per Foundational Knowledge Spec §1.1: knowledge evolves through use.
   Stars with FLAG_LEARNABLE get their embeddings nudged. */
if (galaxy_table != nullptr && threadIdx.x == 0) {
    const unsigned int star_base = chosen_star_index * GALAXY_STAR_RECORD_BYTES;
    const unsigned int flags = *reinterpret_cast<const unsigned int*>(
        galaxy_table + star_base + GALAXY_STAR_FLAGS_OFFSET
    );
    if (flags & GALAXY_STAR_FLAG_LEARNABLE) {
        float* star_embedding = reinterpret_cast<float*>(
            galaxy_table + star_base + GALAXY_STAR_EMBEDDING_OFFSET
        );
        const float* brain_reasoning = reinterpret_cast<const float*>(
            brain_state + BRAIN_REASONING_OFFSET
        );
        /* Outcome > 0: nudge star toward successful reasoning state.
           Outcome < 0: nudge star away from failed reasoning state.
           This makes the Galaxy LEARN from experience — knowledge evolves. */
        const float nudge = (outcome_signal > 0) ? 0.02f : ((outcome_signal < 0) ? -0.01f : 0.0f);
        for (int i = 0; i < GPU_TASK_EMBED_DIMS; ++i) {
            star_embedding[i] = tanhf(star_embedding[i] + (nudge * brain_reasoning[i]));
        }
    }
}
```

---

## ORDER 5: Wire Galaxy Table Through the Stack

### 5A: `GPUTaskDispatch.launch` — accept galaxy_ptr + star_count

```python
def launch(self, task_buffer, task_count, *, block_size=128, brain_ptr=None, galaxy_ptr=None, galaxy_star_count=0):
    # ... existing logic ...
    loader.launch(
        self.kernel,
        (total, 1, 1),
        (block_size, 1, 1),
        [
            task_buffer.input_buffer,
            task_buffer.output_buffer,
            ctypes.c_uint(total),
            brain_ptr if brain_ptr is not None else ctypes.c_void_p(),
            galaxy_ptr if galaxy_ptr is not None else ctypes.c_void_p(),
            ctypes.c_uint(int(galaxy_star_count)),
        ],
    )
```

### 5B: `SleepTimeMicro.consolidate` — accept galaxy_ptr + chosen action

```python
def consolidate(self, brain_gpu_ptr, outcome_signal, *, galaxy_ptr=None, chosen_star_index=0):
    loader.launch(
        self.kernel,
        (1, 1, 1),
        (128, 1, 1),
        [
            brain_gpu_ptr,
            ctypes.c_int(max(-1, min(1, int(outcome_signal)))),
            galaxy_ptr if galaxy_ptr is not None else ctypes.c_void_p(),
            ctypes.c_uint(int(chosen_star_index)),
        ],
    )
```

### 5C: `K3DARC3Agent` — owns Galaxy table + passes to kernel

```python
class K3DARC3Agent:
    def __init__(self, ...):
        # ... existing brain, encoder, dispatcher ...
        self.galaxy_table = GalaxyVRAMTable(max_stars=256)
        galaxy_stars = build_arc3_galaxy_table()
        self.galaxy_table.load_stars(galaxy_stars)
        # action_embeddings no longer computed per frame — they live in Galaxy VRAM

    def choose_action(self, frame):
        # ... existing frame encoding, task packing ...
        # option_embeddings STILL packed in slot for backward compat,
        # but kernel prefers Galaxy table when available
        self.dispatcher.launch(
            self.task_buffer, loaded,
            brain_ptr=self.brain.gpu_ptr,
            galaxy_ptr=self.galaxy_table.gpu_ptr,
            galaxy_star_count=self.galaxy_table.star_count,
        )
        # ... rest unchanged ...

    def learn_from_outcome(self, *, levels_completed=0, frame=None):
        # ... existing outcome computation ...
        last_action = self.action_history[-1]["action_index"] if self.action_history else 0
        self.sleep_time.consolidate(
            self.brain.gpu_ptr, outcome_signal,
            galaxy_ptr=self.galaxy_table.gpu_ptr,
            chosen_star_index=last_action,
        )
        # ... rest unchanged ...
```

---

## ORDER 6: Tests

### 6A: Existing tests pass
```bash
pytest tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py -v
```

### 6B: New test `tests/test_galaxy_vram_table.py`
1. Create GalaxyVRAMTable(max_stars=16)
2. Load 7 action stars with component_refs
3. Verify star_count == 7
4. Read back from VRAM → embeddings match
5. Close → no crash

### 6C: Galaxy-aware dispatch test
1. Build arc3_galaxy_table → load into GalaxyVRAMTable
2. Run dispatch with galaxy_ptr + brain_ptr
3. Verify action output uses Galaxy-composed embeddings
4. Run same dispatch WITHOUT galaxy_ptr → verify fallback to slot embeddings
5. Both paths produce valid (not identical) results

### 6D: Galaxy learning test
1. Load Galaxy, run one action
2. Read star embedding at chosen index → snapshot
3. Run sleep_time.consolidate with outcome_signal=+1 and galaxy_ptr
4. Read star embedding again → embedding changed (nudged toward reasoning state)
5. Run with outcome_signal=-1 → embedding moved in opposite direction

---

## FILE INVENTORY

Files you CREATE:
- `knowledge3d/knowledgeverse/galaxy_vram_table.py` — VRAM star table + boot loader
- `tests/test_galaxy_vram_table.py` — Galaxy table tests

Files you MODIFY:
- `knowledge3d/cranium/cuda/device_functions.cuh` — Galaxy constants + helper functions
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — 5th/6th param, Galaxy-sourced reads
- `knowledge3d/cranium/cuda/sleep_time_micro.cu` — Galaxy star update
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` — pass galaxy_ptr, update CPU ref
- `knowledge3d/knowledgeverse/sleep_time_micro.py` — pass galaxy_ptr
- `benchmarks/arc_agi_3.py` — own Galaxy table, pass to dispatch + sleep

Files you DO NOT TOUCH:
- `knowledge3d/cranium/action_primitives_bootstrap.py` (data source, unchanged)
- `knowledge3d/knowledgeverse/action_embedding_loader.py` (still used for initial loading)
- `knowledge3d/knowledgeverse/persistent_brain.py` (unchanged)
- `knowledge3d/knowledgeverse/vram_task_buffer.py` (slot layout unchanged)
- `knowledge3d/knowledgeverse/semantic_csr_graph.py` (different purpose)

---

## EXECUTION SEQUENCE

1. Add Galaxy constants to `device_functions.cuh`
2. Add `galaxy_read_embedding`, `galaxy_read_n_refs`, `galaxy_read_ref`, `galaxy_compose_embedding_device` to `device_functions.cuh`
3. Create `galaxy_vram_table.py`
4. Modify `gpu_task_dispatch.cu` — 2 new params, Galaxy-sourced option reading
5. Modify `sleep_time_micro.cu` — 2 new params, Galaxy star update
6. Modify `gpu_task_dispatch.py` — pass galaxy_ptr + star_count
7. Modify `sleep_time_micro.py` — pass galaxy_ptr + chosen_star_index
8. Modify `benchmarks/arc_agi_3.py` — own Galaxy table, wire everything
9. Tests → all green
10. Run ARC3 synthetic → verify Galaxy-composed scoring
11. Report: kernel reads from Galaxy VRAM (not slot), action diversity, star embedding evolution

---

## SUCCESS CRITERIA

- **Galaxy in VRAM**: stars loaded once at boot, persistent for session
- **Kernel reads from Galaxy**: option embeddings come from `galaxy_table`, not from input slot
- **Component_refs followed**: kernel composes embedding from star + referenced stars
- **Sleep-time updates Galaxy**: star embeddings evolve based on action outcomes
- **Backward compatible**: non-ARC3 tasks still read from input slot
- **More GPU work**: kernel does Galaxy reads + composition per thinking step
- **Existing benchmarks hold**: Synthetic 10/10, MMLU >= 30%
- **Action diversity maintained**: >= 3 different actions in 20 synthetic frames

## WHAT THIS ENABLES (Next Phases)

Once the Galaxy is in VRAM and the kernel reads from it:
- **Phase F**: Load ALL default galaxies (Drawing, Character, Word, Grammar, Math, Reality) into the table. The kernel navigates a real multi-galaxy knowledge space.
- **Phase G**: LED-A* on Galaxy table — the kernel pathfinds to relevant stars, not just reads fixed indices.
- **Phase H**: TRM navigates Galaxy autonomously — the avatar THINKS in the Galaxy.

**This is the foundation. Knowledge moves from Python to VRAM. The kernel becomes a navigator, not a calculator. Build it.**
