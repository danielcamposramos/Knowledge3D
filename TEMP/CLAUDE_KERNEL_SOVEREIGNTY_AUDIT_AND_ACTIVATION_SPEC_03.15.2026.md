# Kernel Sovereignty Audit & Activation Specification

**Author:** Claude (Architecture Partner)
**Date:** March 15, 2026
**Status:** ACTIVE — Steering for Codex
**Depends on:** D.4.7 (complete), Kernel Activation Roadmap (superseded by this document)
**Supersedes:** `CLAUDE_KERNEL_ACTIVATION_ROADMAP_03.14.2026.md` (Tier 1 plan invalidated by audit)

---

## Executive Summary

A full sovereignty audit of all GRE specialist kernels reveals that **11 of 15 are stubs** — they compile, they have bridges, but their CUDA logic is trivial arithmetic (lerp, multiply, modulo). They cannot be "activated" because there is nothing real to activate.

This specification replaces the stubs with real CUDA designs, grounded in the architecture specs and the benchmarks they must serve. Daniel's mandate: **no Python, no stubs, no placeholders.**

**What IS real and working:**
- `nine_chain_specialized.cu` — 520 lines, 8 distinct chain implementations with real computation
- `arc_grid_ops.cu` — 987 lines, 17 ARC grid transformations (sovereign, production)
- `gre_multimodal_halting_gate.cu` — 95 lines, real convergence analysis (active)
- `gre_world_model.cu` — 153 lines, 5 kernels with real multi-modal computation
- `sleep_cluster_refiner.cu` — actual k-means clustering (6 kernels)
- Composed head pipeline: Morton → LED-A* → Frustum → LOD → Swarm → Halting Gate

---

## Part 1: Sovereignty Audit Results

### Stubs (Must Be Replaced)

| Kernel | Lines | What It Claims | What It Actually Does |
|--------|-------|---------------|----------------------|
| `gre_arc_reasoner` | ~40 | ARC grid reasoning | `sum % 8`, `nonzero % 4` — integer aggregation |
| `gre_geometry_router` | 36 | Geometric routing | `input * scale` — one float multiply per element |
| `gre_temporal_reasoning` | 39 | Temporal reasoning | `next - curr` — frame differencing |
| `gre_fractal_emitter` | 39 | Fractal generation | `x = val*scale, y = i*0.5*scale+val, z = x+y` |
| `gre_resonance_field` | 39 | Field resonance | `sqrt(x²+y²+z²) * density` — distance * scalar |
| `gre_vector_resonator` | 30 | Vector resonance | `a*α + b*(1-α)` — fixed-alpha lerp |
| `gre_graph_crystallizer` | 31 | Graph GNN | `node*(1-r) + neighbor*r` — single EMA step |
| `gre_atomic_fission_fusion` | 36 | Compositional ops | `val * ratio` or `val / ratio` — multiply/divide |
| `gre_cognitive_executive` | ~30 | Meta-reasoning | `input * weight + bias` — weighted sum |
| `gre_oom_spill` | 37 | Memory spill planner | `min(bytes/size, count)` — one division |
| `galaxy_memory_updater` | ~30 | Galaxy write-back | `old*(1-α) + new*α` — EMA blend |

**Evidence:** Compare `gre_arc_reasoner` (40 lines, integer modulo) to `arc_grid_ops.cu` (987 lines, 17 real grid transforms with BFS, component analysis, pattern detection). The gap between name and implementation is vast.

### Real Implementations (Keep + Enhance)

| Kernel | Lines | Why Real |
|--------|-------|----------|
| `gre_multimodal_halting_gate` | 95 | Group aggregation, score gap analysis, multi-threshold convergence — **already active** |
| `gre_world_model` (5 kernels) | 153 | Temporal coherence, multimodal fusion, world state prediction, mesh deformation, galaxy resonance |
| `sleep_cluster_refiner` (6 kernels) | ~200 | K-means assignment, centroid accumulation, L2 normalization, silhouette scoring |
| `sleep_glyph_consolidator` | ~30 | Similarity-threshold clustering (light but real) |
| `gre_recursive_refiner` | ~50 | Iterative blend with drift tracking |
| `gre_sub100micro_gate` | ~40 | GPU timer comparison — does what it claims |

### The Swarm IS the Real Specialist Architecture

`nine_chain_specialized.cu` already implements 8 distinct reasoning styles:

| Chain | Role | Real Computation |
|-------|------|-----------------|
| Chain 1 (Ingest) | Signal preprocessing | 8-sample window statistics (mean, variance), shared memory |
| Chain 2 (Fuse-A) | Associative/semantic | Attention-like scalar (full dot product), tanh activation |
| Chain 3 (Fuse-B) | Logical/structural | 4-segment neighbor contrast, structural modulation |
| Chain 4 (Spatial-A) | Geometric | 8×16 grid gradient (dx, dy, magnitude) |
| Chain 5 (Spatial-B) | Topological | Neighborhood density (11-element window, threshold 0.4) |
| Chain 6 (Spatial-C) | Temporal-spatial | First and second derivatives (velocity + acceleration) |
| Chain 7 (Reason-Reductive) | Analytical | Weighted combination + tanh normalization |
| Chain 8 (Reason-Creative) | Creative | Cross-product mixing + sinusoidal perturbation |
| Chain 9 (Synthesis) | Resonance-weighted blend | 8×8 resonance matrix → normalized weights → weighted sum |

**Key insight:** The swarm already has specialist diversity. The missing piece is not "specialist dispatch" — it's **domain-specific pre-processing** that feeds richer signal into the swarm, and **post-processing** that refines the swarm output.

---

## Part 2: Activation Architecture

### Pipeline Integration Points

```
Query Stimulus (512d)
    │
    ▼
Morton Octree → LED-A* → Frustum → LOD
    │
    ▼
┌─────────────────────────────────────┐
│ PRE-SWARM ENRICHMENT (NEW)         │
│  • ARC Reasoner (grid features)     │
│  • Geometry Router (spatial rels)   │
│  • Temporal Reasoning (sequences)   │
└────────────┬────────────────────────┘
             ▼
Nine-Chain Swarm (existing, unchanged)
             │
             ▼
┌─────────────────────────────────────┐
│ POST-SWARM REFINEMENT (NEW)        │
│  • Graph Crystallizer (multi-hop)   │
│  • Resonance Field (cross-galaxy)   │
│  • Fractal Emitter (self-similarity)│
└────────────┬────────────────────────┘
             ▼
Halting Gate (existing, unchanged)
             │
             ▼
┌─────────────────────────────────────┐
│ WRITE-BACK (NEW, gated)            │
│  • Galaxy Memory Updater            │
└─────────────────────────────────────┘
```

### Design Principle: Enrich, Don't Replace

The swarm works. The halting gate works. The pipeline works. The stubs need to become **enrichment stages** that feed better signal into the existing pipeline:

- **Pre-swarm:** Extract domain-specific features from candidates BEFORE the swarm processes them. The swarm's 128d chain vectors carry richer information.
- **Post-swarm:** Refine the swarm's 128d synthesis output by considering multi-hop relationships, cross-galaxy interactions, and self-similar patterns.
- **Write-back:** After halting gate confirms convergence, optionally write the successful reasoning trace as a new Galaxy entry.

---

## Part 3: Real Kernel Designs

### 3.1 `gre_arc_reasoner` — Grid Feature Extraction

**Current stub:** `sum % 8`, `nonzero % 4` — useless.

**Real design:** Extract structural features from ARC grids that the swarm can reason about. Model after `arc_grid_ops.cu`'s approach but focused on ANALYSIS not TRANSFORMATION.

**Input:** `grid[H × W]` (uint8 color indices), `H`, `W`
**Output:** `features[32]` (float32 feature vector)

**Kernel logic (real CUDA):**
```
Feature extraction per grid (shared memory for grid):

// Color histogram (10 features: counts for colors 0-9)
__shared__ int color_counts[10];
// Reduction: each thread counts its cells, atomicAdd to shared

// Symmetry detection (4 features)
float h_sym = 0, v_sym = 0, d_sym = 0, r90_sym = 0;
// For each cell, compare to its reflected counterpart
// h_sym: grid[y][x] == grid[y][W-1-x] ? 1 : 0 (sum/total)
// v_sym: grid[y][x] == grid[H-1-y][x] ?
// d_sym: grid[y][x] == grid[x][y] ? (if square)
// r90_sym: grid[y][x] == grid[x][H-1-y] ?

// Connected component count (4 features: per-color component counts)
// BFS from unvisited non-zero cells (like arc_grid_ops case 14)
// Count components for top-4 colors

// Edge density (2 features: horizontal, vertical)
// Count adjacent pairs with different colors
// h_edge_density = different_h_pairs / total_h_pairs
// v_edge_density = different_v_pairs / total_v_pairs

// Bounding box ratios (4 features)
// For each non-zero color: compute bbox, store aspect_ratio and fill_ratio
// Top-2 colors by area

// Periodicity score (4 features)
// Test periods 2,3,4,5 horizontally: consensus scoring (like arc_grid_ops case 10)
// Store best horizontal and vertical period + score

// Pattern complexity (4 features)
// Unique colors count, max run length, spatial entropy, border vs interior ratio
```

**Why this helps:** The current swarm gets a 128d embedding that doesn't carry grid-structural information. These 32 features (symmetry, components, edges, periodicity) give the swarm chains something to reason ABOUT. The Spatial-A chain (8×16 gradient) can now detect gradients in meaningful features, not just raw embeddings.

**Integration:** Pre-swarm. Features concatenated/projected into the 128d swarm input alongside the existing stimulus embedding.

**Benchmark impact:** ARC expanded set (currently ~20/50). Grid features enable the swarm to distinguish rotation tasks from reflection tasks from tiling tasks.

---

### 3.2 `gre_geometry_router` — Spatial Relationship Detection

**Current stub:** `input * scale` — one multiply.

**Real design:** Given two candidate embeddings (e.g., ARC input/output pair, or two Galaxy entries), compute spatial relationship features.

**Input:** `embedding_a[D]`, `embedding_b[D]`, `D` (dimension)
**Output:** `relations[16]` (float32 relationship vector)

**Kernel logic:**
```
// Interpret embeddings as spatial vectors and compute relationships

// Cosine similarity (1 feature)
// dot(a,b) / (|a| * |b|)

// L2 distance (1 feature)
// sqrt(sum((a-b)²))

// Element-wise correlation structure (4 features)
// Split embeddings into 4 quadrants, compute per-quadrant cosine
// Reveals whether similarity is uniform or localized

// Relative magnitude profile (4 features)
// ratio[i] = a[i] / (b[i] + eps)
// Compute mean, std, max, min of ratio vector
// Detects scaling, inversion, selective amplification

// Cross-correlation peak (2 features)
// Slide a over b (circular), find peak offset and peak value
// Detects translation/shift relationships

// Sign agreement (2 features)
// Fraction of dimensions where sign(a) == sign(b)
// Fraction where |a| > |b|

// Orthogonality measure (2 features)
// Project a onto b, compute residual magnitude
// High residual = orthogonal = independent information
```

**Integration:** Pre-swarm. Applied to candidate pairs from LED-A* navigation. Relationship features feed into swarm alongside candidate embeddings.

**Benchmark impact:** MMLU (geometry/spatial questions), ARC (transformation detection between input/output grids).

---

### 3.3 `gre_temporal_reasoning` — Sequence Pattern Detection

**Current stub:** `next - curr` — single subtraction.

**Real design:** Detect patterns in ordered sequences of candidates (multi-hop chains, temporal data).

**Input:** `sequence[T × D]` (T timesteps, D features), `T`, `D`
**Output:** `patterns[24]` (float32 pattern vector)

**Kernel logic:**
```
// First-order differences (already in stub, but compute STATISTICS)
delta[t] = sequence[t+1] - sequence[t]

// Trend features (4)
// mean(delta), std(delta), max(delta), min(delta)
// Positive mean = growing trend, high std = volatile

// Second-order differences (acceleration) (4)
accel[t] = delta[t+1] - delta[t]
// mean(accel), std(accel), sign_changes(accel), zero_crossings(accel)

// Periodicity detection (4)
// Auto-correlation at lags 1,2,3,4
// For each lag k: autocorr[k] = corr(sequence[:-k], sequence[k:])
// Peak lag = detected period

// Monotonicity (2)
// Fraction of increasing steps, fraction of strictly monotone runs

// Recurrence (4)
// For each pair (t1, t2): cosine(sequence[t1], sequence[t2])
// Count pairs with similarity > 0.9 (recurrence count)
// Mean recurrence interval
// Maximum gap between recurrences

// Causal asymmetry (2)
// Compare forward prediction error vs backward prediction error
// forward: sequence[t+1] ≈ sequence[t] + mean_delta
// backward: sequence[t-1] ≈ sequence[t] - mean_delta
// Asymmetry indicates causal direction

// Convergence (4)
// Are the elements converging? Compute pairwise distances over time
// convergence_rate = mean(dist[t+1]) / mean(dist[t])
// < 1 = converging, > 1 = diverging
```

**Integration:** Pre-swarm. Applied to multi-hop candidate chains from LED-A* paths. Pattern features help the swarm's temporal-spatial chain (Chain 6) process sequences.

**Benchmark impact:** LHE (multi-hop reasoning chains), GSM8K (sequential computation steps).

---

### 3.4 `gre_graph_crystallizer` — Multi-Hop Message Passing

**Current stub:** `node*(1-r) + neighbor*r` — single EMA step.

**Real design:** Run K rounds of message passing on the candidate graph. Each node aggregates neighbor features, enabling multi-hop reasoning.

**Input:** `node_features[N × D]`, `adjacency[N × max_neighbors]`, `neighbor_counts[N]`, `N`, `D`, `K` (rounds)
**Output:** `refined_features[N × D]` (float32, updated node embeddings)

**Kernel logic:**
```
// K rounds of message passing (K typically 2-3)
for round in 0..K:

    // Phase 1: Aggregate neighbor features (parallel over nodes)
    for each node i (grid-stride):
        __shared__ float agg[D];  // accumulator
        float count = neighbor_counts[i];
        if count == 0: refined[i] = node[i]; continue

        // Sum neighbor features
        for j in neighbors(i):
            agg[d] += node_features[j * D + d]
        agg[d] /= count  // mean aggregation

        // Phase 2: Update node embedding
        // GNN update rule: new = tanh(W_self * self + W_neigh * agg)
        // W_self and W_neigh are identity + learned bias (kept simple)
        float self_val = node_features[i * D + d];
        float neigh_val = agg[d];
        refined[i * D + d] = tanhf(0.6f * self_val + 0.4f * neigh_val);

    __syncthreads();
    // Copy refined back to node_features for next round
    node_features = refined;
```

**Integration:** Post-swarm. The swarm output is a set of candidate scores. The graph crystallizer propagates scores through the candidate graph, enabling multi-hop reasoning where a correct answer depends on information from multiple Galaxy entries.

**Benchmark impact:** LHE (multi-hop is the key bottleneck, currently 6/10), MMLU (interdisciplinary questions requiring cross-domain connections).

---

### 3.5 `gre_resonance_field` — Cross-Galaxy Interference

**Current stub:** `sqrt(x²+y²+z²) * density` — distance times scalar.

**Real design:** Compute constructive/destructive interference between candidates from different galaxies. Candidates that resonate (agree) get boosted; conflicting candidates get attenuated.

**Input:** `candidates[N × D]`, `galaxy_ids[N]`, `scores[N]`, `N`, `D`, `num_galaxies`
**Output:** `resonance_scores[N]` (float32, interference-adjusted scores)

**Kernel logic:**
```
// For each candidate i (grid-stride):
float base_score = scores[i];
int my_galaxy = galaxy_ids[i];

// Compute resonance with candidates from OTHER galaxies
float constructive = 0.0f;
float destructive = 0.0f;
int cross_count = 0;

for j in 0..N:
    if galaxy_ids[j] == my_galaxy: continue  // skip same-galaxy

    // Cosine similarity between candidate embeddings
    float sim = dot(candidates[i], candidates[j]) / (|candidates[i]| * |candidates[j]| + eps);

    if sim > 0.5f:
        constructive += sim * scores[j];  // agreement across galaxies
    elif sim < -0.3f:
        destructive += fabsf(sim) * scores[j];  // conflict

    cross_count++;

if cross_count > 0:
    constructive /= cross_count;
    destructive /= cross_count;

// Resonance-adjusted score
// Constructive = cross-galaxy agreement → boost
// Destructive = cross-galaxy conflict → attenuate
resonance_scores[i] = base_score * (1.0f + 0.3f * constructive - 0.2f * destructive);
```

**Integration:** Post-swarm, before halting gate. Adjusts candidate scores based on cross-galaxy agreement. A Math candidate supported by Grammar candidates (constructive resonance) gets boosted.

**Benchmark impact:** Math (auxiliary galaxy signal, fixes the `guard_factorial_3` class of failures), MMLU (interdisciplinary cross-referencing).

---

### 3.6 `gre_fractal_emitter` — Self-Similarity Detection

**Current stub:** `x = val*scale, y = i*0.5*scale+val, z = x+y` — arithmetic.

**Real design:** Detect self-similar patterns in candidate sets. Some ARC tasks and recursive structures exhibit fractal-like repetition at different scales.

**Input:** `features[N × D]`, `N`, `D`, `num_scales` (typically 3-4)
**Output:** `self_similarity[N]` (float32, per-candidate self-similarity score)

**Kernel logic:**
```
// For each candidate i (grid-stride):

// Test self-similarity at multiple scales
float total_sim = 0.0f;

for scale in 1..num_scales:
    // Subsample the feature vector at this scale
    int stride = 1 << scale;  // 2, 4, 8
    int sub_len = D / stride;
    if sub_len < 4: break;

    // Compute similarity between full-resolution and subsampled
    // Subsample: take every stride-th element
    float dot = 0.0f, norm_full = 0.0f, norm_sub = 0.0f;
    for d in 0..sub_len:
        float full_val = features[i * D + d];
        float sub_val = features[i * D + d * stride];
        dot += full_val * sub_val;
        norm_full += full_val * full_val;
        norm_sub += sub_val * sub_val;

    float sim = dot / (sqrtf(norm_full) * sqrtf(norm_sub) + eps);
    total_sim += sim;

self_similarity[i] = total_sim / num_scales;
```

**Integration:** Post-swarm. Self-similar candidates (high score) are boosted — they contain structural regularity that's more likely to be a valid pattern.

**Benchmark impact:** ARC (recursive/tiling patterns), Math (recursive structures like factorial).

---

### 3.7 `gre_atomic_fission_fusion` — Compositional Split/Merge

**Current stub:** `val * ratio` or `val / ratio` — multiply/divide.

**Real design:** Compositional reasoning: break compound candidate embeddings into atomic components (fission), or merge atomic components into compounds (fusion). Enables the pipeline to verify that a compound answer is consistent with its parts.

**Input:** `compound[D]`, `atoms[K × D]`, `K`, `D`, `mode` (0=fission, 1=fusion)
**Output:** `result[D]` (reconstructed compound or verified fusion), `consistency` (float, 0-1)

**Kernel logic:**
```
if mode == 0:  // FISSION: decompose compound into atoms
    // Project compound onto each atom's direction
    // residual = compound
    for k in 0..K:
        // Projection coefficient: coeff[k] = dot(residual, atoms[k]) / |atoms[k]|²
        float dot = 0, norm = 0;
        for d: dot += residual[d] * atoms[k*D+d]; norm += atoms[k*D+d] * atoms[k*D+d];
        coeff[k] = dot / (norm + eps);

        // Remove projection from residual
        for d: residual[d] -= coeff[k] * atoms[k*D+d];

    // Consistency = 1 - |residual|/|compound|
    // High consistency → compound is well-explained by its atoms
    consistency = 1.0f - sqrtf(dot(residual,residual)) / (sqrtf(dot(compound,compound)) + eps);

    // Result = reconstruction from atoms
    for d: result[d] = sum_k(coeff[k] * atoms[k*D+d]);

elif mode == 1:  // FUSION: verify that atoms compose into a coherent compound
    // Compute centroid of atoms
    for d: centroid[d] = mean(atoms[0..K][d]);

    // Consistency = mean pairwise agreement of atoms projected toward centroid
    float agreement = 0;
    for k in 0..K:
        float sim = cosine(atoms[k], centroid);
        agreement += sim;
    consistency = agreement / K;

    // Result = weighted combination (weights = similarity to centroid)
    for d: result[d] = weighted_sum(atoms, weights);
```

**Integration:** Post-swarm. For compound candidates (answers composed from multiple Galaxy entries), verify compositional consistency. Low consistency → candidate is internally contradictory → reduce score.

**Benchmark impact:** GSM8K (multi-step word problems where each step must be consistent), Math (compound expressions).

---

### 3.8 `gre_vector_resonator` — Attention-Weighted Blending

**Current stub:** `a*α + b*(1-α)` — fixed lerp.

**Real design:** Replace fixed alpha with **learned attention** that computes blending weights from the vectors themselves.

**Input:** `vectors[K × D]`, `K`, `D`
**Output:** `blended[D]`, `attention_weights[K]`

**Kernel logic:**
```
// Compute pairwise relevance scores (simplified attention)
__shared__ float scores[MAX_K];

// For each vector k, compute self-relevance: score[k] = |vectors[k]|² (energy)
// Plus cross-relevance: score[k] += mean(cosine(vectors[k], vectors[j]) for j != k)
for k in 0..K:
    float energy = dot(vectors[k], vectors[k]);
    float cross = 0;
    for j in 0..K:
        if j == k: continue
        cross += cosine(vectors[k], vectors[j]);
    cross /= (K - 1);
    scores[k] = energy * (1.0f + cross);

// Softmax attention weights
float max_score = max(scores);
float sum_exp = 0;
for k: sum_exp += expf(scores[k] - max_score);
for k: attention_weights[k] = expf(scores[k] - max_score) / sum_exp;

// Weighted blend
for d in 0..D:
    blended[d] = 0;
    for k: blended[d] += attention_weights[k] * vectors[k * D + d];
```

**Integration:** Post-swarm. Used to blend the swarm's synthesis output with candidates from different pipeline stages. The attention mechanism automatically determines how much to trust each source.

**Benchmark impact:** All benchmarks (better candidate composition).

---

### 3.9 `galaxy_memory_updater` — Verified Write-Back

**Current stub:** `old*(1-α) + new*α` — EMA blend.

**Real design:** Write a verified reasoning trace as a new Galaxy entry with deduplication.

**Input:** `trace_embedding[D]`, `galaxy_table[M × D]`, `galaxy_hashes[M]`, `M`, `D`
**Output:** `write_slot` (int, -1 if duplicate), `updated_table[M × D]`

**Kernel logic:**
```
// Step 1: Content-based hash for deduplication
// Compute hash of trace_embedding (reduce to 64-bit)
__shared__ unsigned long long hash;
if tid == 0:
    hash = 0;
for d in tid..D:
    // Quantize to 8-bit and hash
    unsigned char q = (unsigned char)clamp((trace_embedding[d] + 1.0f) * 127.5f, 0, 255);
    atomicXor(&hash, (unsigned long long)q << ((d % 8) * 8));

// Step 2: Check for duplicates
__shared__ int duplicate_found;
duplicate_found = -1;
for m in tid..M:
    if galaxy_hashes[m] == hash:
        // Verify with cosine similarity (hash collision check)
        float sim = cosine(trace_embedding, galaxy_table[m]);
        if sim > 0.95f:
            duplicate_found = m;
            break;

// Step 3: If not duplicate, find empty slot and write
if duplicate_found == -1:
    // Find first empty slot (hash == 0)
    for m in tid..M:
        if galaxy_hashes[m] == 0:
            write_slot = m;
            // Write embedding
            for d: galaxy_table[m * D + d] = trace_embedding[d];
            galaxy_hashes[m] = hash;
            break;
else:
    // Duplicate: optionally update with EMA
    float alpha = 0.1f;
    for d: galaxy_table[duplicate_found * D + d] =
        (1-alpha) * galaxy_table[duplicate_found * D + d] + alpha * trace_embedding[d];
    write_slot = -1;  // signal: no new slot
```

**Integration:** After halting gate, gated by `K3D_TRM_WRITE_GALAXY=1`. Only write on verified-correct answers. Capped at 100 writes/session.

**Benchmark impact:** Amortized — each correct answer enriches Galaxy for future queries.

---

### 3.10 `gre_oom_spill` — Real Memory Spill Planning

**Current stub:** `min(bytes/size, count)` — one division.

**Real design:** When VRAM approaches capacity, identify cold Galaxy entries by access frequency and plan spillage.

**Input:** `access_counts[M]`, `last_access_tick[M]`, `current_tick`, `M`, `target_free_bytes`, `entry_size_bytes`
**Output:** `spill_mask[M]` (uint8, 1=spill), `spill_count`

**Kernel logic:**
```
// Compute coldness score per entry
// coldness = 1 / (access_counts[m] + 1) * (current_tick - last_access_tick[m])
// Higher = colder = better spill candidate

// Phase 1: Compute coldness scores (parallel)
for m in tid..M:
    float freq = 1.0f / (access_counts[m] + 1.0f);
    float recency = (float)(current_tick - last_access_tick[m]);
    coldness[m] = freq * recency;

// Phase 2: Find coldness threshold via binary search on percentile
// Target: spill enough entries to free target_free_bytes
int entries_to_spill = (target_free_bytes + entry_size_bytes - 1) / entry_size_bytes;

// Sort-free threshold finding:
// Count entries above each candidate threshold
// Binary search for threshold that gives entries_to_spill

// Phase 3: Mark entries above threshold
for m in tid..M:
    spill_mask[m] = (coldness[m] >= threshold) ? 1 : 0;
```

**Integration:** Health monitoring. Called when VRAM usage > 80%. Daemon mode (Phase C) runs this periodically.

---

### 3.11 `gre_cognitive_executive` — Swarm Chain Trust Evaluation

**Current stub:** `input * weight + bias` — weighted sum.

**Real design:** Meta-reasoning kernel that evaluates which swarm chains to trust for the current query. Uses the resonance matrix from the swarm to detect coherent vs divergent chains.

**Input:** `resonance_matrix[8 × 8]`, `chain_norms[8]`, `chain_outputs[8 × D]`
**Output:** `trust_weights[8]`, `coherence_score` (float, overall swarm agreement)

**Kernel logic:**
```
// Analyze resonance matrix for coherence structure

// Per-chain coherence: how much does each chain agree with others?
for c in 0..8:
    float mean_resonance = 0;
    for j in 0..8:
        if j == c: continue
        mean_resonance += resonance_matrix[c * 8 + j];
    mean_resonance /= 7.0f;

    // Trust = coherence * energy
    // High coherence + high norm = confident chain
    // High coherence + low norm = weak but consistent
    // Low coherence + high norm = divergent (reduce trust)
    trust_weights[c] = fmaxf(mean_resonance, 0.0f) * (1.0f + logf(chain_norms[c] + 1.0f));

// Normalize trust weights (softmax)
softmax(trust_weights, 8);

// Overall coherence: mean off-diagonal resonance
float total = 0;
for c in 0..8:
    for j in c+1..8:
        total += resonance_matrix[c * 8 + j];
coherence_score = total / 28.0f;  // 8 choose 2 = 28 pairs
```

**Integration:** Post-swarm, before synthesis. Replace the swarm's fixed resonance-based weights with trust-evaluated weights. The halting gate can also use `coherence_score` as an additional convergence signal.

**Benchmark impact:** All benchmarks (more reliable swarm composition, fewer divergent-chain-dominated answers).

---

## Part 4: Implementation Order

### Phase 1: Post-Swarm Refinement (Highest benchmark ROI)

**Why first:** The swarm already produces reasonable output. Post-swarm refinement can improve it without changing the swarm itself.

1. **`gre_graph_crystallizer`** (real multi-hop message passing) — Targets LHE 6→8/10
2. **`gre_resonance_field`** (cross-galaxy interference) — Targets MMLU 16→20/50
3. **`gre_cognitive_executive`** (swarm trust evaluation) — Targets all benchmarks

### Phase 2: Pre-Swarm Enrichment (Benchmark-specific gains)

4. **`gre_arc_reasoner`** (grid feature extraction) — Targets ARC expanded 20→30/50
5. **`gre_geometry_router`** (spatial relationships) — Targets ARC + MMLU geometry
6. **`gre_temporal_reasoning`** (sequence patterns) — Targets LHE + GSM8K

### Phase 3: Post-Swarm Composition

7. **`gre_fractal_emitter`** (self-similarity) — Targets ARC + Math recursive
8. **`gre_atomic_fission_fusion`** (compositional verification) — Targets GSM8K
9. **`gre_vector_resonator`** (attention blending) — Targets all benchmarks

### Phase 4: Write-Back + Health

10. **`galaxy_memory_updater`** (verified write-back) — Amortized improvement
11. **`gre_oom_spill`** (memory management) — Phase C prerequisite

### Implementation Per Kernel

For EACH kernel:

1. **Write the `.cu` file** — Replace the stub entirely. Real CUDA with shared memory, proper thread coordination, actual computation.
2. **Compile to `.ptx`** — `nvcc -ptx -arch=sm_86 -O3 --use_fast_math`
3. **Update the bridge class** in `sovereign_bridges.py` — Match new input/output contract.
4. **Wire into pipeline** — Add the kernel call at the correct pipeline stage in `knowledgeverse.py`.
5. **Test sovereignty** — grep for forbidden imports, verify no numpy in hot path.
6. **Benchmark** — Run full quartet, verify non-regression + targeted improvement.

---

## Part 5: Codex Directives

### Critical Rules

1. **No stubs.** Every `.cu` file must contain real CUDA computation. If a kernel has fewer than 50 lines, it's probably still a stub.
2. **No Python fallbacks.** The kernels ARE the computation. If a kernel fails, fix the kernel.
3. **Shared memory.** Real kernels use `__shared__` for data reuse. The existing swarm kernels show the pattern.
4. **Thread coordination.** Use `__syncthreads()`, warp-level operations, atomic operations where appropriate.
5. **Sovereignty.** No numpy, cupy, scipy in the hot path. The bridges call `loader.launch()` and `memcpy_htod/dtoh`. That's it.
6. **Quartet must hold.** After each kernel activation: ARC 10/10, Math 20/20, LHE 6/10, GSM8K 1/10, MMLU 16/50 (navigate=1 at strength=0.5).

### Compilation

```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 --use_fast_math gre_<name>.cu -o gre_<name>.ptx
cp gre_<name>.ptx ../ptx/  # if ptx lives separately
```

### Testing Pattern

```python
# Each kernel gets a sovereign bridge test:
def test_gre_<name>_real_computation():
    """Verify kernel produces non-trivial output."""
    bridge = <BridgeClass>(...)
    input_data = np.random.randn(...).astype(np.float32)
    output = bridge.run(input_data)

    # Verify output is NOT just input * constant
    ratio = output / (input_data + 1e-8)
    assert not np.allclose(ratio, ratio[0]), "Output is trivial scaling"

    # Verify output has structure
    assert np.std(output) > 0.01, "Output has no variation"
```

---

## Part 6: Benchmark Projections (Updated Post-Audit)

| Benchmark | Current (nav=1) | +Post-Swarm (P1) | +Pre-Swarm (P2) | +Composition (P3) | +Write-Back (P4) |
|-----------|----------------|-------------------|-------------------|---------------------|-------------------|
| ARC 10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| ARC 50 | ~20/50 | 22/50 | 30-35/50 | 35/50 | 38/50 |
| Math 20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| GSM8K 10 | 1/10 | 2/10 | 3/10 | 4-5/10 | 5-6/10 |
| LHE 10 | 6/10 | 8/10 | 8/10 | 9/10 | 9-10/10 |
| MMLU 50 | 16/50 | 20/50 | 22/50 | 24/50 | 26/50 |

**Key change from previous roadmap:** Previous projections assumed GRE specialists were ready to activate. They weren't. These projections account for the implementation cost of real kernels.

---

## Daniel's Mandate

> "No Python, no stubs, no placeholders."
> "We fail and fix — this is the goal."

Every stub in `knowledge3d/cranium/kernels/` is sovereignty debt. This specification pays it off with real CUDA designs that match the architectural intent. The kernels were named correctly — they just weren't built yet.
