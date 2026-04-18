# Claude → Codex Spec: Phase B — Native Embedding from meaning_rpn

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Implementer**: Codex
**Phase**: B — Native Embedding (follows Phase A: Transfer Yard + Qwen Qdrant, same date)
**Supersedes**: `sovereign_matryoshka_embedder.py` text-path (replaces trigram-on-surface-form with program-projection-on-meaning_rpn)

---

## 1. Principle — Why a Trained Embedder Is a Sovereignty Violation Inside K3D

Phase A correctly exiled Qwen3-Embedding to the Phenom host for Qdrant / ingestion-path consultation. That leaves one unanswered question: where do the K3D-side embeddings in `star.embeddings` come from? The current answer is `rpn_embedding_engine.py` + `sovereign_matryoshka_embedder.py`, which project **surface-form text strings** (character trigram hashes of words like "cat", "gato") onto a 2048-dim vector. That is wrong for three reasons.

**Reason 1 — Sovereignty.** A trained neural embedder, however small, carries learned weights that encode the training distribution. Those weights are not PTX kernels, not Galaxy programs, not RPN procedures. They are an external data dependency that cannot be reasoned about, debugged, or extended in the sovereign execution environment. K3D's hot path must be reproducible from first principles: `meaning_rpn → float[2048]` using only math cores and ternary opcodes. No gradient descent, no weight files, no external API.

**Reason 2 — Meaning vs. Surface.** The star schema says: `star_id = hash(meaning_rpn)`. Two stars for "cat" in English and "gato" in Portuguese **are the same star** with different `surface_form` references. If the embedding is computed from the surface string, "cat" and "gato" will not be close — they hash to different trigrams. If the embedding is computed from `meaning_rpn`, they are **identical by construction** because they share the same program. Cross-lingual alignment is not a learned property; it falls out structurally.

**Reason 3 — Procedural Transparency.** A trained model is a black box: its output is unpredictable under program changes, weight updates, or new opcode additions. A deterministic projection from `meaning_rpn` is auditable: change the program, observe the embedding shift. Add a new opcode, the projection formula absorbs it. This is the "regenerable from procedures" property the star schema specifies explicitly (`star.embeddings: explicitly marked regenerable from procedures`).

**Sovereignty guarantee**: If embeddings are ever lost or corrupted, re-running the projection kernel over `meaning_rpn` bytes regenerates bit-exact copies. No training run needed. No external service needed.

---

## 2. Contract — meaning_rpn → float[2048]

### 2.1 Input

```
star.meaning_rpn: RPN_Program
    → ordered list of (opcode: uint16, operand: float32|uint32|None)
    → serialized as contiguous uint8 bytes: [opcode_hi, opcode_lo, operand_4bytes, ...] per token
    → max tokens per program: 69 (one Transfer Yard depth)
    → longer programs are chunked and folded (see §3.4)
```

`meaning_rpn` is the Layer 2 canonical center of the star. It is NOT a text string. It is a GPU-executable program that defines what the concept IS.

### 2.2 Output

```
star.embeddings.tier_2048: float[2048]
    → L2 normalized (unit vector)
    → deterministic: same meaning_rpn bytes → same float[2048] across all runs, all hosts
    → stored in Galaxy Universe Region 2 (active working memory VRAM), row-major layout
    → persisted to House (Region 3) JSONL export on sleeptime consolidation

star.embeddings.tier_512:  float[512]  = L2_norm(tier_2048[0:512])
star.embeddings.tier_128:  float[128]  = L2_norm(tier_2048[0:128])
star.embeddings.tier_64:   float[64]   = L2_norm(tier_2048[0:64])
```

**Matryoshka invariant (formal):**

For any tier `k ∈ {64, 128, 512}`:
```
embed_k[i] = tier_2048[i] / ||tier_2048[0:k]||₂    for i in [0, k)
```

This means tier_2048 must encode meaning such that the **first 64 dimensions carry the coarsest, most important semantic signal**, and each successive block `[64:128)`, `[128:512)`, `[512:2048)` adds progressively finer detail. The projection algorithm is responsible for guaranteeing this ordering property.

### 2.3 Storage Region

`star.embeddings` lives in **Knowledgeverse Region 2 (Galaxy Universe)**, the active AI memory VRAM substrate. Specifically, the embedding table is the `float[N][2048]` array the `matryoshka_prefix_dot.cu` kernel already operates on. Region 2 is the correct region: it is volatile VRAM working memory, always loaded, directly addressable by the composed head pipeline. Region 4 (Discoveries) holds new stars created during reasoning; Region 3 (House) holds the persisted JSONL. Embeddings cached to Region 3 on sleeptime for cold-start reload.

---

## 3. Chosen Algorithm — Composable Galaxy-Specific Basis (Candidate D)

### 3.1 Why D Wins

Four candidates were evaluated (sinusoidal hash, Morton spatial spread, folded ternary hash, composable basis). The Kimi architecture swarm concluded: only Candidate D satisfies the conjunction of **matryoshka prefix invariance** and **cross-modal alignment** without a trained model. Summary of why the others fail:

- **A (sinusoidal hash)**: sin/cos across all 2048 dimensions distributes energy uniformly. There is no reason dimension 0 is more important than dimension 63. Prefix truncation loses semantic fidelity non-uniformly. Matryoshka invariant fails.
- **B (Morton spatial spread)**: bit-interleaving scatters sequential opcode information across all dimensions. Taking the first 64 bits returns a subsampling of the Morton curve, not a lower-dimensional semantic summary. Matryoshka invariant fails.
- **C (folded ternary XOR)**: TERNARY_XOR is commutative and mixing. After folding all opcode vectors, every output dimension depends on all input opcodes. There is no dimension ordering by importance. Additionally, XOR hash of a drawing primitive and XOR hash of the same concept expressed as a word will land in uncorrelated regions — cross-modal alignment fails unless the XOR chain is carefully seeded, which reintroduces a form of learned alignment.

**D (composable basis) wins because:**
1. Basis vectors are ordered by semantic granularity — dimensions `[0:64)` hold the coarsest ontological signal, progressively refined in each block. Matryoshka invariant holds by construction.
2. Galaxy-specific encoders map visually distinct representations of the same concept (a drawn circle, the glyph "○", the equation `x²+y²=r²`) to nearby regions via **Semantic Anchor Indices** (dims 0-63 = concept-class anchors, described in §3.3). Cross-modal alignment is an engineering choice, not a learned outcome.
3. Basis functions are analytic (geometric, arithmetic, codepoint-positional) — pure PTX/CUDA, no weights, no Python.
4. Transfer Yard integration is natural: each opcode type maps to a yard bank, contributions accumulate per-bank, final L2-norm pass produces tier_2048.

### 3.2 Projection Algorithm: RPN Program

The embedding kernel `rpn_meaning_project.cu` (new file) executes the following program for each star:

```
// Pseudocode — Codex writes the actual CUDA kernel
FOR each token t in meaning_rpn:
    galaxy_family = CLASSIFY_OPCODE(t.opcode)   // VISUAL | WORD | MATH | META
    bank_id = galaxy_family_to_bank[galaxy_family]  // 0=VISUAL, 1=WORD, 2=MATH, 3=META, 4-8=reserved
    YARD_SELECT(bank_id)
    contribution = BASIS_ENCODE(t.opcode, t.operand, bank_id)  // float[2048] contribution
    YARD_PUSH_BANK(bank_id, contribution)

// Reduction: fold each bank's contributions into one float[2048] accumulator
FOR bank_id in [0, 4):
    bank_vector = YARD_FOLD_SUM(bank_id)      // sum all pushed contributions in this bank
    acc += bank_vector                         // accumulate across banks

// Matryoshka normalization (in-place, 4-pass over tier boundaries)
tier_2048 = L2_NORM_INPLACE(acc)
tier_512  = L2_NORM_INPLACE(acc[0:512])      // slices of already-normalized tier_2048
tier_128  = L2_NORM_INPLACE(acc[0:128])
tier_64   = L2_NORM_INPLACE(acc[0:64])
```

The critical property: `BASIS_ENCODE` maps each `(opcode, operand)` to a contribution vector where **dimension importance is ordered by index**. Dimensions `[0:64)` receive contributions only from the top-level concept class (ontological level). Dimensions `[64:128)` receive structural features. `[128:512)` receive compositional detail. `[512:2048)` receive fine-grained opcode-specific signal.

### 3.3 BASIS_ENCODE — Per Galaxy Family

**VISUAL family** (opcodes: DRAW_LINE, DRAW_CIRCLE, DRAW_RECT, FILL_RGBA, PATH_MOVE, PATH_CLOSE):

| Dim range | Signal | Formula |
|-----------|--------|---------|
| `[0:16)` | Concept-class anchor ("roundness", "linearity", "filledness") | Fixed lookup: DRAW_CIRCLE sets dim 0-3 += 1.0, DRAW_LINE sets dim 4-7 += 1.0, etc. |
| `[16:64)` | Geometric parameters, coarse | `cos(operand_normalized * π * i)` for each dim `i` in [16,64), normalized param in [0,1] |
| `[64:128)` | Edge orientation histogram buckets | Operand (angle/direction) hashed into 64 buckets via `(uint32_t)(angle_rad * 64 / 2π) % 64` |
| `[128:512)` | Compositional: opcode sequence n-gram | Pair (prev_opcode, cur_opcode) hashed: `dim = hash16(prev, cur) % 384 + 128` |
| `[512:2048)` | Fine detail: operand precision band | `sin(operand * 2π * i / 1536)` for `i` in [0, 1536) — full-resolution signal |

**WORD family** (opcodes: CHAR_CODEPOINT, CHAR_COMPOSE, MORPHEME_REF, WORD_REF):

| Dim range | Signal | Formula |
|-----------|--------|---------|
| `[0:16)` | Concept-class anchor ("written", "spoken", "abstract") | CHAR_CODEPOINT sets dim 8-11, WORD_REF sets dim 12-15 |
| `[16:64)` | Unicode block coarse | `codepoint >> 8` mapped to 48 macro-blocks, increment corresponding dim |
| `[64:128)` | Unicode subblock | `(codepoint >> 4) & 0xF` → 16 buckets cycled into [64,128) |
| `[128:512)` | Character sequence position | Position-encoded: `cos(position * π * i / 64)` for dims [128,512) |
| `[512:2048)` | Full codepoint sinusoidal | `sin(codepoint * 2π * i / 1536)` for each output dim |

**MATH family** (opcodes: MATH_ADD, MATH_MUL, MATH_POW, MATH_TRIG, MATH_SERIES, symbolic diff ops):

| Dim range | Signal | Formula |
|-----------|--------|---------|
| `[0:16)` | Concept-class anchor ("algebraic", "geometric", "analytic") | MATH_TRIG sets dim 0-3, MATH_POW sets dim 4-7, MATH_SERIES sets dim 8-11 |
| `[16:64)` | Arity + commutativity + associativity flags | Packed into dims [16,64): arity in [16,32), commutative in [32,48), assoc in [48,64) |
| `[64:128)` | Operator precedence tier | `precedence_bucket = opcode_precedence(opcode) % 64` → increment dim `64 + bucket` |
| `[128:512)` | Expression tree depth contribution | Depth tracked in Transfer Yard active bank; `cos(depth * π * i / 384)` into [128,512) |
| `[512:2048)` | Operand precision | `sin(operand_value * 2π * i / 1536)` |

**META family** (CONCEPT_REF, TAXONOMY_REF, SYMLINK_REF, CONFIDENCE_TRIT):

| Dim range | Signal | Formula |
|-----------|--------|---------|
| `[0:16)` | Concept-class anchor ("structural") | TAXONOMY_REF sets dim 12-15 |
| `[16:64)` | Taxonomy depth | depth % 48 → dim [16,64) |
| `[64:512)` | Referenced star's tier_64 embedding | Load referenced star's existing tier_64 (if available) into [64,128); zero-pad otherwise |
| `[512:2048)` | Symlink count + confidence trit | Packed into a cosine series across [512,2048) |

### 3.4 Semantic Anchor Index Contract (dims 0-15)

Dims 0-15 are **reserved concept-class anchors** shared across all galaxy families:

| Dims | Concept class | Activated by |
|------|--------------|-------------|
| 0-3 | Roundness / curvature | DRAW_CIRCLE, MATH_TRIG (sin/cos), codepoint U+25CB (○) |
| 4-7 | Linearity / directionality | DRAW_LINE, MATH_ADD (sequential), any 1D morpheme |
| 8-11 | Written / symbolic | CHAR_CODEPOINT, WORD_REF, MATH_SYMBOL |
| 12-15 | Structural / relational | TAXONOMY_REF, SYMLINK_REF, CONCEPT_REF |

**This is the cross-modal alignment mechanism**: a circle drawn via `DRAW_CIRCLE`, written as "○" (U+25CB), and expressed as `x²+y²=r²` all activate dims 0-3. After L2 normalization, they will have cosine similarity > 0.7 in the tier_64 slice. No training required — it is an **analytic property of the basis function definitions**.

The anchor table lives in CUDA `__constant__` memory (`rpn_meaning_project.cu`). It is a `uint16_t[256]` mapping opcode → anchor_dim_base. This is **not a learned weight file** — it is a fixed semantic ontology baked into the kernel.

### 3.5 Chunk Folding for Programs Longer Than 69 Tokens

Transfer Yard depth is 69. Programs with more than 69 tokens are processed in 69-token chunks:

```
acc = float[2048]{0.0}
FOR chunk in split(meaning_rpn, chunk_size=69):
    chunk_vec = PROJECT_CHUNK(chunk)   // per §3.2-3.3
    acc = acc + chunk_vec              // accumulate (not replace)
tier_2048 = L2_NORM(acc)
```

Accumulation preserves the first-chunk's low-frequency signal (dims 0-64) with high weight because each chunk produces full-range contributions and the L2 norm at the end equalizes. The ordering invariant for prefix truncation is maintained because each chunk contributes to ALL dimension ranges, and the low-frequency bands (0-64) accumulate faster (higher per-opcode contribution).

---

## 4. Matryoshka Invariants — Formal Properties and Tests

### 4.1 Formal Properties

**P1 (Determinism):** For any fixed `meaning_rpn` program `P`:
```
embed(P) = embed(P)     // identical bytes, every call, every restart
```
Implementation: the projection uses no random state. All constants are in `__constant__` memory or derived from opcode values via fixed arithmetic. No global RNG, no session-dependent seeds.

**P2 (Prefix):** For tiers k ∈ {64, 128, 512}:
```
embed_k[i] = tier_2048[i] / ||tier_2048[0:k]||₂    for all i ∈ [0, k)
```

**P3 (Meaning identity):** Same concept in any language → same `meaning_rpn` (by symlink architecture) → same `embed_2048` → bit-identical.

**P4 (Regenerability):** If `star.embeddings` is dropped from VRAM or corrupted on disk:
```
embeddings = project_meaning_rpn(star.meaning_rpn)   // restores bit-exact
```
No network call, no model download, no training.

### 4.2 Tests Codex Must Add

File: `tests/test_native_embedding.py`

```python
# T1 — Determinism: same program → same bytes across 3 independent calls
def test_embedding_determinism():
    rpn = sample_meaning_rpn("cat_concept")
    e1 = project_meaning_rpn(rpn)
    e2 = project_meaning_rpn(rpn)
    e3 = project_meaning_rpn(rpn)
    assert e1 == e2 == e3   # bit-exact list[float] comparison

# T2 — Prefix invariant: tier_64 == L2_norm(tier_2048[:64])
def test_matryoshka_prefix_invariance():
    rpn = sample_meaning_rpn("cat_concept")
    e2048 = project_meaning_rpn(rpn)
    tier_64_derived = l2_norm(e2048[:64])
    tier_64_native = project_meaning_rpn_tier(rpn, 64)
    for a, b in zip(tier_64_derived, tier_64_native):
        assert abs(a - b) < 1e-6

# T3 — Cross-modal alignment: circle drawn vs "○" codepoint vs circle equation
def test_cross_modal_circle_alignment():
    rpn_circle_visual = make_rpn([("DRAW_CIRCLE", 10.0)])
    rpn_circle_word   = make_rpn([("CHAR_CODEPOINT", 0x25CB)])
    rpn_circle_math   = make_rpn([("MATH_TRIG", "cos"), ("MATH_TRIG", "sin")])
    e_v = project_meaning_rpn(rpn_circle_visual)[:64]
    e_w = project_meaning_rpn(rpn_circle_word)[:64]
    e_m = project_meaning_rpn(rpn_circle_math)[:64]
    sim_vw = cosine(e_v, e_w)
    sim_vm = cosine(e_v, e_m)
    assert sim_vw > 0.5, f"visual-word circle alignment too low: {sim_vw}"
    assert sim_vm > 0.5, f"visual-math circle alignment too low: {sim_vm}"

# T4 — Unit norm: all tiers are unit vectors after normalization
def test_unit_norm_all_tiers():
    rpn = sample_meaning_rpn("cat_concept")
    for tier in [64, 128, 512, 2048]:
        e = project_meaning_rpn_tier(rpn, tier)
        norm = sum(x*x for x in e) ** 0.5
        assert abs(norm - 1.0) < 1e-5

# T5 — Bit-exact reload: embed, write to disk, reload, re-embed, compare
def test_embed_reload_bitexact(tmp_path):
    rpn = sample_meaning_rpn("cat_concept")
    e1 = project_meaning_rpn(rpn)
    (tmp_path / "embed.bin").write_bytes(pack_f32(e1))
    e2 = unpack_f32((tmp_path / "embed.bin").read_bytes())
    e3 = project_meaning_rpn(rpn)   # re-project from program
    for a, b in zip(e2, e3):
        assert abs(a - b) < 1e-6

# T6 — No external model loaded: grep check embedded in test
def test_no_external_model_import():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import knowledge3d.cranium.rpn_meaning_projector"],
        capture_output=True, text=True
    )
    assert "SentenceTransformer" not in result.stdout + result.stderr
    assert "sentence-transformers" not in result.stdout + result.stderr
```

---

## 5. FOV/POV Usage — Tier Selection in the Composed Head Pipeline

The composed head pipeline is:
```
Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate
```

Each stage uses a different embedding tier. Tier selection is made by the `meta_select_matryoshka_tier` RPN meta-rule, which writes to the `tier_signal` shared-memory register in `matryoshka_prefix_dot.cu`. No Python selects the tier. No host-side if/else.

| Pipeline stage | Tier used | Rationale |
|---|---|---|
| **Morton Octree** (spatial index rebuild) | tier_64 | Coarse spatial position needs only ontological class. 64-dim dot product: 2 MACs per lane. Sub-10µs per batch. |
| **LED-A* path scoring** | tier_128 | Structural similarity between path candidates: "is this shelf the right category?" |
| **Frustum Culling** | tier_64 | FOV = field-of-view. Check if star's concept-class is relevant to current query topic. Fast reject on coarse embedding. |
| **Dynamic LOD selection** | tier_128 or tier_512 | Zoom level: distant stars use 128, stars in focus use 512. |
| **Nine-Chain Swarm workers** | tier_512 | Fine-grained matching within candidate set: workers score candidates using 512-dim cosine via `matryoshka_prefix_dot.cu`. |
| **Halting Gate convergence** | tier_2048 | Deduplication check: is this answer a repeat of a prior discovery? Full-fidelity fingerprint. |
| **Discoveries persistence** | tier_2048 | New stars stored with full embedding for future federation/dedup. |

**Integration point**: `matryoshka_prefix_dot.cu` already implements fused variable-width prefix dot with warp-level butterfly reduction. Phase B feeds it with embeddings derived from `meaning_rpn` instead of surface-form trigrams. The kernel is unchanged; only the embedding generation upstream changes.

---

## 6. Cross-Modal Alignment — The Symlink Guarantee

**Language alignment is free by architecture.** `star_id = hash(meaning_rpn)`. "Cat" (en) and "gato" (pt) are the **same star** (same `meaning_rpn`). Their embeddings are bit-identical by P3. No proximity metric needed — they ARE the same point.

**Cross-galaxy alignment** (e.g., the concept "circle" appearing as a Drawing Galaxy star, a Character Galaxy star, and a Math Galaxy star) is handled by the Semantic Anchor Indices (§3.3, dims 0-15). All three representations activate the same anchor dims proportionally. Bidirectional symlinks (per `feedback_bidirectional_symlinks_norm.md`) ensure TRM can traverse from any galaxy entry to any related entry. The embedding proximity provides the first-pass spatial filter; symlinks provide the traversal path.

**The invariant**: if star A and star B are bidirectionally symlinked, their tier_64 embeddings will have cosine similarity > 0.4 (because they share concept-class anchor activation). Stars that are NOT symlinked and belong to unrelated concept classes will have similarity < 0.2 in tier_64.

This means Frustum Culling at tier_64 will naturally pre-filter to semantically related neighborhoods before LED-A* traverses symlink edges. The spatial structure and the graph structure are coherent.

---

## 7. Storage in Galaxy Universe — Region 2

`star.embeddings` is stored in **Region 2 (Galaxy Universe)** of the 7-region Knowledgeverse substrate.

The VRAM layout is a row-major float array:
```
galaxy_embed_table: float[N_stars][2048]    // Region 2, galaxy VRAM
```

Smaller tiers are **views into this table** (prefix slices), not separate allocations. The `matryoshka_prefix_dot.cu` kernel already reads this layout.

Tier-specific caches (Region 2 sub-tables) for frequently-queried stars may be added as an optimization by Codex, but the canonical source is always `tier_2048`.

**Persistence boundary**: On sleeptime, the embedding table is serialized to the star JSONL in House (Region 3). On boot, it is reloaded from House JSONL or regenerated from `meaning_rpn` if missing (regenerability guarantee).

---

## 8. Migration — Retiring the Surface-Form Path

### 8.1 What Exists Today

| File | Current behavior | Problem |
|------|-----------------|---------|
| `knowledge3d/cranium/rpn_embedding_engine.py` | Trigram hash of surface-form text strings | Operates on text, not `meaning_rpn`; surface-form language-dependent |
| `knowledge3d/cranium/sovereign_matryoshka_embedder.py` | Wraps `rpn_embedding_engine`, calls `embed_sentence_gpu(text)` | Input is a string, not a `meaning_rpn` program |
| `knowledge3d/cranium/ptx/trigram_embed.cu` + `.ptx` | Trigram hash kernel operating on raw character bytes | Hash of surface chars, not opcode semantics |
| `knowledge3d/cranium/bridges/trigram_embed_bridge.py` | Bridge to `trigram_embed.ptx` | Same problem |
| `knowledge3d/cranium/ptx_kernels/matryoshka_prefix_dot.cu` | Fused prefix dot product | KEEP — used downstream. Input data changes, kernel stays. |
| `knowledge3d/cranium/embedding_generator.py` | Unknown (Codex: grep and inspect) | Likely calls surface-form path |

### 8.2 Migration Plan

**Step 1**: Write `knowledge3d/cranium/kernels/rpn_meaning_project.cu` per §3. Compile to `knowledge3d/cranium/ptx/rpn_meaning_project.ptx`.

**Step 2**: Write `knowledge3d/cranium/bridges/rpn_meaning_project_bridge.py` — ctypes bridge to the new PTX. Accepts `meaning_rpn` as a byte buffer (opcode+operand pairs), returns `float[2048]`.

**Step 3**: Write `knowledge3d/cranium/rpn_meaning_projector.py` — thin Python module (hot-path boundary: load kernel + dispatch only). Replaces `rpn_embedding_engine.py` in callers that need `meaning_rpn → embedding`. Does NOT accept text strings.

**Step 4**: Update `knowledge3d/ingestion/star_crafter.py` — when building a new star, compute embeddings by calling `rpn_meaning_projector.project(star.meaning_rpn)`, not by calling surface-form embedder.

**Step 5**: Update `knowledge3d/knowledgeverse/knowledgeverse.py` — any call site that computes embeddings at query time must use `rpn_meaning_projector` not `SovereignMatryoshkaTextEmbedder`.

**Step 6**: Retain `rpn_embedding_engine.py` and `sovereign_matryoshka_embedder.py` ONLY if they are used for Qdrant ingestion-path text queries (Phase A domain). If they are used in the hot path (query time, swarm scoring, LOD decisions), those call sites must be migrated before Phase B is declared complete.

**Step 7**: Grep gates (see §9) confirm no surface-form embedder calls remain in the hot path.

### 8.3 Grep Patterns Codex Runs

```bash
# Find all hot-path callers of surface-form embedder
grep -rn "embed_sentence_gpu\|SovereignMatryoshkaTextEmbedder\|RPNEmbeddingEngine" \
    knowledge3d/knowledgeverse/ knowledge3d/cranium/bridges/ knowledge3d/tablet/

# Find any remaining text-to-embedding calls in hot path
grep -rn "embed_sentence\|embed_text\|encode_text" \
    knowledge3d/knowledgeverse/ knowledge3d/cranium/ \
    --include="*.py" | grep -v "ingestion\|qwen_matryoshka_client"

# Confirm rpn_meaning_projector has no import numpy
grep -n "import numpy\|from numpy" knowledge3d/cranium/rpn_meaning_projector.py  # → 0

# Confirm new kernel exists and compiled
test -e knowledge3d/cranium/ptx/rpn_meaning_project.ptx && echo "PTX exists"
```

---

## 9. Acceptance Gates

### §9.1 New kernel and bridge exist
```bash
test -e knowledge3d/cranium/kernels/rpn_meaning_project.cu    || exit 1
test -e knowledge3d/cranium/ptx/rpn_meaning_project.ptx       || exit 1
test -e knowledge3d/cranium/bridges/rpn_meaning_project_bridge.py || exit 1
test -e knowledge3d/cranium/rpn_meaning_projector.py           || exit 1
```

### §9.2 No numpy in new projector
```bash
grep -n "import numpy\|from numpy\|np\." knowledge3d/cranium/rpn_meaning_projector.py  # → 0
grep -n "import numpy\|from numpy\|np\." knowledge3d/cranium/bridges/rpn_meaning_project_bridge.py  # → 0
```

### §9.3 No surface-form embedder in hot path
```bash
grep -rn "embed_sentence_gpu\|SovereignMatryoshkaTextEmbedder" \
    knowledge3d/knowledgeverse/ knowledge3d/tablet/wine/ knowledge3d/cranium/bridges/  # → 0
```

### §9.4 Determinism test passes (T1)
```bash
python -m pytest tests/test_native_embedding.py::test_embedding_determinism -xvs
# → PASSED
```

### §9.5 Matryoshka prefix invariance test passes (T2)
```bash
python -m pytest tests/test_native_embedding.py::test_matryoshka_prefix_invariance -xvs
# → PASSED
```

### §9.6 Cross-modal alignment test passes (T3)
```bash
python -m pytest tests/test_native_embedding.py::test_cross_modal_circle_alignment -xvs
# tier_64 cosine(circle_visual, circle_word) > 0.5 and cosine(circle_visual, circle_math) > 0.5
```

### §9.7 Anchor constant table in kernel — grep
```bash
grep -n "anchor_dim\|concept_class_anchor\|uint16_t anchor" \
    knowledge3d/cranium/kernels/rpn_meaning_project.cu  # → ≥1 hit
grep -n "__constant__" knowledge3d/cranium/kernels/rpn_meaning_project.cu  # → ≥1 hit
```

### §9.8 No trained model loaded by projector module
```bash
python -c "import knowledge3d.cranium.rpn_meaning_projector" 2>&1 | \
    grep -i "sentence-transformer\|SentenceTransformer\|MiniLM"  # → empty
```

---

## 10. Codex Handoff Checklist (Ordered)

1. Read Phase A spec (`CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md`) §5 — understand the existing matryoshka_prefix_dot.cu kernel and its VRAM row-major layout.
2. Read `knowledge3d/cranium/ptx_kernels/matryoshka_prefix_dot.cu` lines 1-50 — the kernel this spec feeds into is already built and correct. Phase B only changes what populates the embedding table.
3. Grep: `grep -rn "embed_sentence_gpu\|SovereignMatryoshkaTextEmbedder" knowledge3d/` — map all current call sites before touching anything.
4. Grep: `grep -rn "import numpy" knowledge3d/cranium/specialists/ knowledge3d/cranium/` — confirm Phase A numpy removal is done first (prerequisite).
5. Write `knowledge3d/cranium/kernels/rpn_meaning_project.cu` per §3.2–3.4: opcode classifier, galaxy-family BASIS_ENCODE per §3.3, Semantic Anchor Index table in `__constant__` memory, chunk folding for programs > 69 tokens, 4-pass in-place L2 normalization for tier boundaries.
6. Compile: `nvcc -arch=sm_86 -ptx -o knowledge3d/cranium/ptx/rpn_meaning_project.ptx knowledge3d/cranium/kernels/rpn_meaning_project.cu` — commit PTX artifact.
7. Write `knowledge3d/cranium/bridges/rpn_meaning_project_bridge.py` — ctypes bridge, accepts `bytes` (serialized opcode+operand pairs), returns `list[float]` of length 2048. No numpy.
8. Write `knowledge3d/cranium/rpn_meaning_projector.py` — thin sovereign Python wrapper. Exposes `project(meaning_rpn: bytes, tier: int = 2048) -> list[float]`. No text string inputs. No surface-form.
9. Write `tests/test_native_embedding.py` with T1–T6 from §4.2.
10. Run T1 (determinism) — if it fails, the kernel has non-deterministic state (check `__shared__` initialization, uninitialized accumulator).
11. Run T2 (prefix invariance) — if it fails, `BASIS_ENCODE` is not loading low-frequency dims preferentially; fix dim-range assignment in §3.3.
12. Run T3 (cross-modal alignment) — if cosine < 0.5, check Semantic Anchor Index table: `DRAW_CIRCLE`, `CHAR_CODEPOINT 0x25CB`, and `MATH_TRIG cos` must all write to dims 0-3.
13. Update `knowledge3d/ingestion/star_crafter.py` — route star embedding generation to `rpn_meaning_projector.project(star.meaning_rpn)`.
14. Audit `knowledge3d/knowledgeverse/knowledgeverse.py` (currently ~4000 lines) — identify and migrate any embedding calls that go through surface-form path. Document file:line in PR.
15. Run §9 acceptance gates in order. Report pass/fail with evidence. Do not declare Phase B complete until all 8 gates pass.

---

## 11. Must-NOT-Do List

- **No Qwen inside K3D path.** Qwen3-Embedding is for Qdrant / ingestion-path consultation on the Phenom host (Phase A). It NEVER runs during inference, LOD selection, swarm scoring, or halting gate evaluation.
- **No gradient training.** The anchor constant table is a design choice, not a learned outcome. If you find yourself running a training loop to calibrate the basis, stop and re-read §3.3. The geometric and codepoint mappings are analytic — they are derived from the mathematical structure of the domains.
- **No Python fallbacks.** If `rpn_meaning_project.ptx` fails to load (CUDA init error, wrong architecture), the system does NOT fall back to the trigram-hash path. It fails loudly. Fix the kernel. Do not add `try/except` around the PTX load.
- **No numpy in projector or bridge.** The projector chain (`.cu → .ptx → bridge.py → projector.py`) is numpy-free. ctypes only. Tests may use numpy for comparison math (cosine similarity check in T3) — that is test infrastructure, not hot path.
- **Do not replace matryoshka_prefix_dot.cu.** It is correct and already deployed. Phase B changes what data feeds it, not how it computes.
- **Do not compute embeddings from `surface_forms`.** The only valid input to `rpn_meaning_projector.project()` is `meaning_rpn` bytes. If a caller wants to embed a text string, they must first look up or build the star, retrieve its `meaning_rpn`, and project that. No shortcuts.
- **Do not raise tier_2048 above 2048.** The composed head pipeline, matryoshka_prefix_dot.cu, and the Galaxy VRAM table are all sized for 2048. If wider embeddings are needed in the future, that is a new spec.
- **Do not invent new opcode families** beyond VISUAL / WORD / MATH / META without updating `RPN_DOMAIN_OPCODE_REGISTRY.md` first and getting architecture sign-off.
