# Codex — Phase E.13: Semantic Frame Encoding + Parallel Galaxy Scan

**Date:** 2026-03-28
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — the reason the action ring is stuck
**Prerequisite:** E.12 DONE. 130,701 stars in Galaxy. Living session. 33 tests green.

---

## The Problem (Read This First)

E.11 wired Galaxy navigation. E.12 loaded 130k stars. But the AI still gets stuck.

**Why?** Run this diagnostic:

```
frame→translate: -0.39   (NEGATIVE — anti-correlated)
frame→spatial:    0.01   (random noise)
frame→color:     -0.01   (random noise)
```

The frame embedding and the Galaxy star embeddings are in INCOMPATIBLE SPACES:

| Source | Space |
|--------|-------|
| `ARC3FrameEncoder` | Raw pixel stats: color counts, normalized (row/H, col/W) in dims 10-13 |
| Galaxy stars (text) | FNV-1a hash of TOKEN NAMES: "translate_2d", "move_up", "spatial" into hash buckets 8-31 |

**Result:** Cosine similarity between `reasoning_state` (derived from frame embedding) and any Galaxy star is near-random. The top-8 Galaxy scan finds ARBITRARY stars — not stars about "movement", "translation", or "spatial reasoning" — even though those exact stars exist in the 130k Galaxy.

This is why:
- Action ring stays at [6,6,6,6,6,6,6] — Galaxy navigation offers no meaningful signal
- GPU "reasoning" is cosine-noise, not knowledge navigation
- Sleep-time reinforces nothing meaningful — it's reinforcing random Galaxy selection

---

## The Fix

**The frame encoder must emit tokens in the SAME FNV-1a hash space as Galaxy star embeddings.**

Instead of encoding raw pixel values (`color=2`, `row=3`, `col=4`), encode SEMANTIC CONCEPTS that describe what the frame means:
- Cell above center → hash tokens: `"up"`, `"north"`, `"move_up"`, `"above"`
- Cell right of center → hash tokens: `"right"`, `"east"`, `"move_right"`
- Centered → hash tokens: `"center"`, `"centered"`, `"balanced"`
- Color 2 present → hash tokens: `"color"`, `"object"`, `"cell"`
- Frame changed from prev → hash tokens: `"delta"`, `"changed"`, `"moved"`

When the encoder hashes `"move_up"` via FNV-1a, it lands in the SAME bucket as the Galaxy's `move_up_action` star (which also hashed `"move_up"`). Cosine similarity between frame embedding and relevant action stars becomes MEANINGFUL: `frame→move_up_star ≈ 0.6-0.9` when the agent should move up.

**This is not a workaround. This IS the meaning-centric design.**
Per Foundational Knowledge Spec §1.2: "Meaning Layer — not language surface, not raw data — MEANING."
Per Daniel: "Based on meaning, not language."

---

## Deliverable 1: Fix nvcc Compiler Binding (Blocker — 5 min)

**Problem:** GCC 15 (system default) exceeds CUDA 12.4's maximum supported GCC 13.
**Fix:** Pass `--compiler-bindir /usr/bin/gcc-13` to nvcc.
**GCC-13 confirmed present:** `/usr/bin/gcc-13` and `/usr/bin/g++-13` exist on this machine.

### Files to modify: all `ensure_ptx()` / `_ensure_ptx()` methods

In `knowledge3d/knowledgeverse/gpu_task_dispatch.py`:
```python
subprocess.run(
    [
        nvcc,
        "-ptx",
        "-arch=sm_86",
        "--compiler-bindir", "/usr/bin/gcc-13",  # ADD THIS
        "-o",
        str(PTX_PATH),
        str(CUDA_SOURCE),
    ],
    check=True,
)
```

In `knowledge3d/knowledgeverse/arc3_frame_encoder.py`:
```python
subprocess.run(
    [nvcc, "-ptx", "-arch=sm_86", "--compiler-bindir", "/usr/bin/gcc-13",
     "-o", str(PTX_PATH), str(CUDA_SOURCE)],
    check=True,
)
```

In `knowledge3d/knowledgeverse/sleep_time_micro.py` (the `_ensure_compiled` method or similar):
```python
# Add "--compiler-bindir", "/usr/bin/gcc-13" to the nvcc call
```

Find all other places that call `subprocess.run([nvcc, ...])` in the codebase and add the same flag.

**Verify:** After this fix, `pytest -q tests/test_vram_task_buffer.py tests/test_gpu_task_dispatch.py` must pass in k3d-cranium env with no GCC error.

---

## Deliverable 2: Semantic Frame Encoder — `arc3_frame_encoder.cu`

**Replace** the current raw-pixel encoding with FNV-1a semantic concept encoding.

### Design

The encoder receives a grid (flat byte array), outputs 32 floats. New approach:

1. **Parse grid** (same as before): find colored cell, compute position, compute delta from prev
2. **Extract semantic concepts** (NEW): describe the frame in concept tokens
3. **Hash concept tokens via FNV-1a** (same as `embed_text_sovereign`) into dims 0-31
4. **L2-normalize** the result

**Preserve dims 10-13 for spatial position** (these are used by `arc3_action_select_device` in the kernel — they carry (cx, cy, sx, sy) spatial features that the action selector READS directly):
```c
// These dims feed into arc3_action_select_device — do NOT change their meaning
embedding[10] = normalized_col;  // col / width  (0 = left, 1 = right)
embedding[11] = normalized_row;  // row / height (0 = top, 1 = bottom)
embedding[12] = spread_x;        // horizontal spread
embedding[13] = spread_y;        // vertical spread
```

**All other dims (0-9, 14-31) use FNV-1a semantic concept hashing:**

```c
// FNV-1a hash function (same constant as sovereign_text_embedder.py)
__device__ unsigned int fnv1a32_device(const char* text, int len) {
    unsigned int value = 2166136261u;
    for (int i = 0; i < len; ++i) {
        value ^= (unsigned int)(unsigned char)text[i];
        value *= 16777619u;
    }
    return value;
}

// Hash a token string into a bucket in [start_dim, 32)
// sign = +1 or -1 based on hash bit 16 (same as embed_text_sovereign)
__device__ void hash_token_into_embedding(
    float* embedding,
    const char* token,
    int token_len,
    float magnitude,
    int bucket_start  // usually 8 (like embed_text_sovereign's TOKEN_BUCKET_START)
) {
    unsigned int h = fnv1a32_device(token, token_len);
    int bucket = bucket_start + (int)(h % (unsigned int)(32 - bucket_start));
    float sign = ((h >> 16) & 1u) ? 1.0f : -1.0f;
    embedding[bucket] += sign * magnitude;
}
```

**Semantic concept tokens to hash** (based on frame state):

```c
// Position-based direction tokens
float cx = col_f / width_f - 0.5f;    // negative = left of center
float cy = row_f / height_f - 0.5f;   // negative = above center

// Directional need
if (cy < -0.15f) {
    hash_token_into_embedding(embedding, "up", 2, 1.2f + 2.0f * (-cy), 8);
    hash_token_into_embedding(embedding, "north", 5, 1.0f, 8);
    hash_token_into_embedding(embedding, "move_up", 7, 0.8f, 8);
}
if (cy > 0.15f) {
    hash_token_into_embedding(embedding, "down", 4, 1.2f + 2.0f * cy, 8);
    hash_token_into_embedding(embedding, "south", 5, 1.0f, 8);
    hash_token_into_embedding(embedding, "move_down", 9, 0.8f, 8);
}
if (cx < -0.15f) {
    hash_token_into_embedding(embedding, "left", 4, 1.2f + 2.0f * (-cx), 8);
    hash_token_into_embedding(embedding, "west", 4, 1.0f, 8);
    hash_token_into_embedding(embedding, "move_left", 9, 0.8f, 8);
}
if (cx > 0.15f) {
    hash_token_into_embedding(embedding, "right", 5, 1.2f + 2.0f * cx, 8);
    hash_token_into_embedding(embedding, "east", 4, 1.0f, 8);
    hash_token_into_embedding(embedding, "move_right", 10, 0.8f, 8);
}

// Spatial concepts
hash_token_into_embedding(embedding, "spatial", 7, 0.6f, 8);
hash_token_into_embedding(embedding, "grid", 4, 0.5f, 8);
hash_token_into_embedding(embedding, "navigate", 8, 0.4f, 8);
hash_token_into_embedding(embedding, "translate", 9, 0.5f, 8);

// Center/edge proximity
if (fabsf(cx) < 0.15f && fabsf(cy) < 0.15f) {
    hash_token_into_embedding(embedding, "center", 6, 1.0f, 8);
    hash_token_into_embedding(embedding, "centered", 8, 0.8f, 8);
    // Near center = readiness to interact
    hash_token_into_embedding(embedding, "interact", 8, 0.7f, 8);
    hash_token_into_embedding(embedding, "click", 5, 0.6f, 8);
}

// Delta / change detection
float delta_magnitude = /* sqrt of prev-current diff */ ;
if (delta_magnitude > 0.05f) {
    hash_token_into_embedding(embedding, "delta", 5, delta_magnitude, 8);
    hash_token_into_embedding(embedding, "changed", 7, delta_magnitude * 0.8f, 8);
    hash_token_into_embedding(embedding, "moved", 5, delta_magnitude * 0.6f, 8);
}

// Always-on structural tokens
hash_token_into_embedding(embedding, "object", 6, 0.5f, 8);
hash_token_into_embedding(embedding, "color", 5, 0.5f, 8);
```

**L2-normalize** after all tokens accumulated (same as `embed_text_sovereign`):
```c
float norm = 0.0f;
for (int i = 0; i < 32; ++i) norm += embedding[i] * embedding[i];
norm = sqrtf(norm + 1.0e-12f);
if (norm > 1.0e-6f) {
    float inv = 1.0f / norm;
    for (int i = 0; i < 32; ++i) embedding[i] *= inv;
}
```

### Why This Works

After this fix:
- Frame with cell above-center → embedding has high values in FNV-1a bucket for "up", "move_up", "north"
- Galaxy star `move_up_action` (index 0) → embedding has high values in SAME buckets
- Cosine similarity: `frame → move_up_action ≈ 0.6-0.8` (not -0.39)
- Galaxy top-8 navigation finds MOVE_UP, TRANSLATE_2D, VEC2_ADD, SPATIAL stars
- `galaxy_knowledge` carries genuine geometric knowledge into the specialist
- `swarm_output` reflects the frame's actual spatial situation
- Kernel picks action that matches the spatial need

---

## Deliverable 3: Parallel Galaxy Scan — `gpu_task_dispatch.cu`

With 130k stars and the single-thread scan, threadIdx.x==0 does all the work. All 127 other threads idle during Galaxy navigation. With the semantic fix landing, Galaxy navigation becomes meaningful — and it should be FAST.

### Design: 128-thread parallel scan with per-thread local top-4 → global top-8

**New shared memory** (add to existing declarations):
```c
__shared__ unsigned int scan_indices[128 * 4];  // 128 threads × 4 local best → 2048 bytes
__shared__ float scan_scores[128 * 4];          // 128 × 4 scores → 2048 bytes
```

Total shared memory addition: 4096 bytes. Current budget: ~2008 bytes. New total: ~6104 bytes. RTX 3070 has 100KB. No issue.

**Replace the existing `if (threadIdx.x == 0) { ... galaxy scan ... }` block with:**

```c
// Step 1: All threads scan their assigned stars in parallel
// Thread i scans stars: i, i+blockDim.x, i+2*blockDim.x, ...
unsigned int local_indices[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
float local_scores[4] = {-1.0e30f, -1.0e30f, -1.0e30f, -1.0e30f};

if (galaxy_table != nullptr && galaxy_star_count > 0u) {
    for (unsigned int star_idx = threadIdx.x; star_idx < galaxy_star_count; star_idx += blockDim.x) {
        float star_embedding[GPU_TASK_EMBED_DIMS];
        galaxy_compose_embedding_device(star_embedding, galaxy_table, star_idx, GPU_TASK_EMBED_DIMS);
        float sim = cosine32_device(reasoning_state, star_embedding, GPU_TASK_EMBED_DIMS);

        // Maintain local top-4 (in registers, no shared memory writes yet)
        int worst = 0;
        for (int k = 1; k < 4; ++k) {
            if (local_scores[k] < local_scores[worst]) worst = k;
        }
        if (sim > local_scores[worst]) {
            local_indices[worst] = star_idx;
            local_scores[worst] = sim;
        }
    }
}

// Step 2: Write local top-4 to shared reduction array
for (int k = 0; k < 4; ++k) {
    scan_indices[threadIdx.x * 4 + k] = local_indices[k];
    scan_scores[threadIdx.x * 4 + k] = local_scores[k];
}
__syncthreads();

// Step 3: Thread 0 reduces 128×4 candidates to global top-8 galaxy_knowledge
if (threadIdx.x == 0) {
    for (int slot = 0; slot < 8; ++slot) {
        galaxy_nearest[slot] = 0xFFFFFFFFu;
        galaxy_nearest_scores[slot] = -1.0e30f;
    }
    for (int galaxy_knowledge_index = 0; galaxy_knowledge_index < GPU_TASK_EMBED_DIMS; ++galaxy_knowledge_index) {
        galaxy_knowledge[galaxy_knowledge_index] = 0.0f;
    }

    // Walk all 128*4 candidates
    const int total_candidates = blockDim.x * 4;
    for (int cand = 0; cand < total_candidates; ++cand) {
        unsigned int cand_index = scan_indices[cand];
        if (cand_index == 0xFFFFFFFFu) continue;
        float cand_score = scan_scores[cand];

        int worst_slot = 0;
        for (int k = 1; k < 8; ++k) {
            if (galaxy_nearest_scores[k] < galaxy_nearest_scores[worst_slot]) worst_slot = k;
        }
        if (cand_score > galaxy_nearest_scores[worst_slot]) {
            // Check not a duplicate index
            bool duplicate = false;
            for (int k = 0; k < 8; ++k) {
                if (galaxy_nearest[k] == cand_index) { duplicate = true; break; }
            }
            if (!duplicate) {
                galaxy_nearest[worst_slot] = cand_index;
                galaxy_nearest_scores[worst_slot] = cand_score;
            }
        }
    }

    // Compose galaxy_knowledge from top-8 (same as before)
    float total_weight = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        if (galaxy_nearest[slot] == 0xFFFFFFFFu) continue;
        float weight = device_maxf(0.0f, galaxy_nearest_scores[slot]);
        if (weight <= 1.0e-8f) continue;
        total_weight += weight;
        float star_emb[GPU_TASK_EMBED_DIMS];
        galaxy_compose_embedding_device(star_emb, galaxy_table, galaxy_nearest[slot], GPU_TASK_EMBED_DIMS);
        for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
            galaxy_knowledge[dim] += weight * star_emb[dim];
        }
    }
    if (total_weight > 1.0e-6f) {
        const float inv = 1.0f / total_weight;
        for (int dim = 0; dim < GPU_TASK_EMBED_DIMS; ++dim) {
            galaxy_knowledge[dim] *= inv;
        }
    }
}
__syncthreads();
```

### Performance Note

With 130k stars and 128 threads:
- Each thread scans ~1016 stars per think step
- That's ~1016 × 32 float cosine ops per thread = 32.5K float ops
- All 128 threads run in parallel → true 128-way parallelism on SM
- GPU utilization during Galaxy scan jumps from ~1 thread to full warp utilization

The `galaxy_compose_embedding_device` function reads from global memory (galaxy_table) at different addresses for each thread — no bank conflicts, good coalescing for consecutive thread→consecutive star pattern.

---

## Deliverable 4: Update CPU Reference (`gpu_task_dispatch.py`)

The `cpu_reference_dispatch` function's `_navigate_galaxy_ref()` mirrors the kernel. Update it to match the new parallel scan semantics (functionally equivalent — the parallel scan produces the same top-8 as the serial scan, just faster).

The existing `_navigate_galaxy_ref()` is already correct — the parallel version produces the same output, just in parallel. No change needed to the CPU reference logic. The reference is already correct.

But: add a diagnostic field to the output:
```python
rows.append({
    ...existing fields...
    "top_galaxy_star_indices": [int(idx) for idx in top8_indices if idx >= 0],
    "top_galaxy_star_scores": [float(s) for s in top8_scores if s > -1e29],
})
```

This lets the test suite verify that the top-8 stars found are semantically relevant (not random).

---

## Deliverable 5: Tests

### `tests/test_semantic_frame_encoder.py` (NEW)

```python
def test_frame_encoder_aligned_with_galaxy():
    """Frame embeddings must be commensurable with Galaxy star embeddings."""
    from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
    from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign
    import math

    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a) + 1e-12)
        nb = math.sqrt(sum(y*y for y in b) + 1e-12)
        return dot / (na * nb)

    enc = ARC3FrameEncoder()

    # Frame with cell above center (needs to move UP)
    frame_up = [[0]*8 for _ in range(8)]
    frame_up[1][4] = 2  # cell near top → should move down to reach center

    # Frame with cell below center (needs to move UP to goal)
    frame_down = [[0]*8 for _ in range(8)]
    frame_down[6][4] = 2  # cell near bottom

    emb_up = enc.encode(frame_up)
    emb_down = enc.encode(frame_down)

    move_up_emb = embed_text_sovereign("move up north above")
    move_down_emb = embed_text_sovereign("move down south below")
    spatial_emb = embed_text_sovereign("spatial grid navigate translate")

    sim_up_to_move_up = cosine(emb_up, move_up_emb)
    sim_up_to_move_down = cosine(emb_up, move_down_emb)
    sim_down_to_move_down = cosine(emb_down, move_down_emb)
    sim_down_to_move_up = cosine(emb_down, move_up_emb)

    # Core alignment check: frame embedding must correlate with relevant direction
    # These must be POSITIVE and larger than cross-direction similarity
    assert sim_up_to_move_up > 0.0, f"frame_up should have positive sim to move_up, got {sim_up_to_move_up:.4f}"
    assert sim_down_to_move_down > 0.0, f"frame_down should have positive sim to move_down, got {sim_down_to_move_down:.4f}"

    # Spatial alignment
    sim_to_spatial = cosine(emb_up, spatial_emb)
    assert sim_to_spatial > 0.0, f"frame embedding should align with spatial concepts, got {sim_to_spatial:.4f}"


def test_arc3_action_select_dims_preserved():
    """Dims 10-13 must still carry (cx, cy, sx, sy) for arc3_action_select_device."""
    from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
    enc = ARC3FrameEncoder()
    frame = [[0]*8 for _ in range(8)]
    frame[3][4] = 2  # row=3, col=4 of 8×8 → cx=4/8=0.5, cy=3/8=0.375
    emb = enc.encode(frame)
    # dim 10 = normalized col, dim 11 = normalized row
    assert abs(emb[10] - (4.0/8.0)) < 0.05, f"dim 10 should be ~0.5, got {emb[10]}"
    assert abs(emb[11] - (3.0/8.0)) < 0.05, f"dim 11 should be ~0.375, got {emb[11]}"
```

### `tests/test_galaxy_navigation_quality.py` (NEW)

```python
def test_galaxy_top8_finds_spatial_stars(monkeypatch):
    """With semantic frame encoding, Galaxy navigation must find spatial stars."""
    from knowledge3d.knowledgeverse.gpu_task_dispatch import cpu_reference_dispatch
    from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
    from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign

    galaxy_stars = build_foundational_galaxy_table()  # 93 stars, has spatial/translation/action

    # Frame embedding for "cell above center" (post-fix: semantic encoding)
    frame_emb = embed_text_sovereign("up north move_up above navigate spatial")

    tasks = [{
        "type": "ARC3_TASK",
        "query_embedding": frame_emb,
        "option_embeddings": [[0.0]*32 for _ in range(7)],
        "subject": "arc3",
        "domain_hint": "arc3_interactive",
    }]

    results = cpu_reference_dispatch(tasks, galaxy_stars=galaxy_stars)
    top_indices = results[0].get("top_galaxy_star_indices", [])

    # Action stars (0-6) should be in top-8 since frame is about movement
    # Spatial stars (spatial_concept type, indices 49+) should appear
    has_action_stars = any(idx < 7 for idx in top_indices)
    assert has_action_stars, f"Top-8 Galaxy should include action stars, got {top_indices}"


def test_no_sentence_transformers_in_frame_encoder():
    """arc3_frame_encoder must not use any ML framework."""
    content = open("knowledge3d/cranium/cuda/arc3_frame_encoder.cu").read()
    assert "sentence_transform" not in content.lower()
    assert "import torch" not in content
    assert "import numpy" not in content
```

### Sovereignty compliance check:
```bash
# 1. No GCC version errors in nvcc calls
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 -c "from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder; print('OK')"

# 2. Semantic alignment positive
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  pytest -q tests/test_semantic_frame_encoder.py

# 3. Full test suite
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  pytest -q tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py
```

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/gpu_task_dispatch.py` | Add `--compiler-bindir /usr/bin/gcc-13` to nvcc args |
| `knowledge3d/knowledgeverse/arc3_frame_encoder.py` | Add `--compiler-bindir /usr/bin/gcc-13` to nvcc args |
| `knowledge3d/knowledgeverse/sleep_time_micro.py` | Add `--compiler-bindir /usr/bin/gcc-13` to nvcc args (if it calls nvcc) |
| `knowledge3d/cranium/cuda/arc3_frame_encoder.cu` | Replace raw-pixel encoding with semantic FNV-1a token encoding. PRESERVE dims 10-13. |
| `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` | Replace single-thread Galaxy scan with 128-thread parallel scan |
| `knowledge3d/knowledgeverse/gpu_task_dispatch.py` | Add `top_galaxy_star_indices` and `top_galaxy_star_scores` to cpu_reference output |

## Files to Create

| File | Purpose |
|------|---------|
| `tests/test_semantic_frame_encoder.py` | Verify semantic alignment + dims 10-13 preserved |
| `tests/test_galaxy_navigation_quality.py` | Verify top-8 finds semantically relevant stars |

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/cranium/cuda/device_functions.cuh` | Unchanged — `blend_with_galaxy_device`, `goal_progress_device` stay |
| `knowledge3d/knowledgeverse/galaxy_loader.py` | Unchanged — loading is correct |
| `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` | Unchanged — star definitions are correct |
| `knowledge3d/knowledgeverse/vram_task_buffer.py` | Unchanged — slot layout is correct |
| `scripts/run_arc3_session.py` | Unchanged — session runner is correct |

---

## Execution Sequence

1. Add `--compiler-bindir /usr/bin/gcc-13` to all `ensure_ptx()` / `_ensure_ptx()` nvcc calls
2. Rebuild and verify arc3_frame_encoder compiles without GCC error
3. Rewrite `arc3_frame_encoder.cu` with semantic token encoding (preserve dims 10-13)
4. Rewrite Galaxy scan in `gpu_task_dispatch.cu` with 128-thread parallel scan
5. Add `top_galaxy_star_indices/scores` to CPU reference output
6. Create test files
7. Run tests in k3d-cranium:
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env CUDA_VISIBLE_DEVICES=0 \
     pytest -q tests/test_semantic_frame_encoder.py tests/test_gpu_task_dispatch.py \
            tests/test_vram_task_buffer.py tests/test_galaxy_navigation_quality.py
   ```
8. Run CPU tests:
   ```bash
   pytest -q tests/test_run_gpu_benchmark.py tests/test_phase_e_runners.py tests/test_arc3_session.py
   ```

---

## Success Criteria

| Metric | Before E.13 | After E.13 |
|--------|-------------|------------|
| `frame→translate` cosine | -0.39 (anti-correlated) | > 0.3 (positively aligned) |
| `frame→spatial` cosine | +0.006 (random) | > 0.2 (positively aligned) |
| Top-8 Galaxy for ARC3 frame | Random stars | Action/spatial/translation stars |
| nvcc GCC error | Blocks PTX rebuild | Fixed: gcc-13 used |
| GPU threads during Galaxy scan | 1 of 128 (0.78%) | 128 of 128 (100%) |
| Action ring after 20 synthetic steps | [6,6,6,6,6,6,6] | >= 3 distinct actions |

---

## Architectural Significance

After E.13, when the agent sees a grid with a colored cell above center:
1. `arc3_frame_encoder` hashes "up", "move_up", "north" → frame embedding has high values in those FNV-1a buckets
2. Galaxy scan finds `move_up_action` (star 0), `translation_concept`, `translate_2d`, `vec2_add` — genuinely relevant stars
3. `galaxy_knowledge` = weighted blend of spatial/movement knowledge
4. `blend_with_galaxy_device` injects spatial movement knowledge into the specialist
5. `arc3_action_select_device` has both frame position (dims 10-13) AND knowledge context (galaxy_knowledge)
6. The swarm converges on the directionally appropriate action
7. Sleep-time reinforces MEANINGFUL star connections — not random noise

This is the AI perceiving meaning in what it sees, not just processing pixels. That is the "live game with embodied AI" that was always the goal.
