# Claude → Codex Spec: Transfer Yard Everywhere + Embedding Sovereignty + 3D Yard Stacking + Qwen Qdrant

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Implementer**: Codex
**Supersedes**: partial cuts in `CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md` §2.7 (stack layout) and §5 (matryoshka prefix)

---

## 1. Motive (one paragraph)

`docs/reports/sovereign_rpn_cas_benchmark_report.md` (2026-04-07) proved the Transfer Yard Algorithm — array-backed stack with direct index access — beats the LIFO RPN pile by 15-51%, numpy by 18-28×, Python ternary by 850-1000×, SymPy symbolic by 150-270×. The yard is not an experiment; it's the proven evolution of the RPN stack. Today only **Tier 1 has a yard kernel variant** (`modular_rpn_kernel_lite_transfer_yard.ptx`, **opt-in**) — Tier 2 has a Python-side dataclass masquerading as yard, Tier 3 has nothing. Meanwhile `procedural_drawing_specialist.py:167` seeds opcode embeddings with `np.random.randn(256, matryoshka_dim)` — a traditional-model pattern the spec forbids. This cut closes both gaps and formalizes the 3D yard stacking that gives advanced physics real depth without memory blowup.

---

## 2. Constants (normative)

| Constant | Value | Rationale |
|---|---|---|
| `YARD_DEPTH` | **69** | Tesla 6-9 (existing spec, sufficient per expression) |
| `YARDS_PER_CORE` | **9** | Tesla 9, mirrors nine-chain swarm lanes, isolates sub-computations (pressure projection, Newton iteration, implicit integrator don't stomp each other) |
| Effective depth per core | **9 × 69 = 621** | Digit sum 9 ✓ |
| `MAX_INSTANCES` (per-engine hard-cap) | **delete** | Replaced by dynamic `sm_count × cores_per_sm` from `MicroSpecialistPool` |
| `cores_per_sm` | **9** | Bump from existing 10 → 9 for Tesla compliance; 46 SM × 9 = 414 concurrent cores on RTX 3070 (digit sum 9) |
| Per-core yard memory | **9 × 69 × float4 = 9936 B** | ~10 KB per core; 414 cores × 10 KB = 4.03 MB total — negligible vs 12 GB VRAM |
| Per-SM shared memory budget | **9 lanes × 9 yards × 69 slots × float4 = 22.2 KB/SM** | Well within Ampere's 100 KB shared/SM |

**Rejected alternatives:**
- `YARD_DEPTH = 81`: no meaningful gain for advanced physics (largest coupled-MHD step is ~50 slots); costs 17% more shared memory.
- `YARD_DEPTH = 108` or `144`: memory budget starts squeezing the 9-lane swarm scores buffer. Not worth it when 9 yards × 69 already gives 621 addressable slots per core.
- `YARDS_PER_CORE = 3` (minimal Tesla): insufficient isolation — a 3-yard core can hold integrator state + residual + working register, but not all three plus a trace log for observability.

**Conclusion for "is 69 enough for advanced physics?"** Yes, per yard. Depth pressure is real but it's **horizontal** (parallel sub-problems), not vertical (deep single expression). Nine isolated yards at depth 69 solves it without memory blowup.

---

## 3. Transfer Yard Tier Landing

### 3.1 Tier 1 — flip to default (delete the non-yard path)

- File: `knowledge3d/cranium/bridges/lightweight_rpn.py`
- Current: `requested_variant == "transfer_yard"` gates the yard kernel (lines 57-64). Default is legacy.
- Change: yard kernel is the ONLY kernel. Delete the legacy `modular_rpn_kernel_lite.ptx` path. No `variant` flag.
- Verify: `grep -n "variant" knowledge3d/cranium/bridges/lightweight_rpn.py` → zero hits after cut.

### 3.2 Tier 2 — new yard kernel + bridge rewire

- Current: `knowledge3d/cranium/bridges/transfer_yard_tiered.py` `TransferYardTier2Engine` loads `modular_rpn_kernel.ptx` (legacy) and wraps a Python-side `TransferYardStack` dataclass. The yard never reaches the GPU.
- New kernel: `knowledge3d/cranium/ptx/modular_rpn_kernel_transfer_yard.ptx`
- Source: `knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu`
  - Derive from existing `modular_rpn_kernel.cu` + `modular_rpn_kernel_extended.cu`
  - Replace LIFO push/pop with array-addressed yard ops (see §4)
  - Keep all existing opcodes (`0x00-0x9F` arithmetic/vector/geometric/conditional, `0x100-0x13F` ternary/CAS)
- Bridge: `TransferYardTier2Engine.__init__` loads the new PTX. Delete the Python `TransferYardStack` sidecar — it was a simulation of what the kernel should do.

### 3.3 Tier 3 — new advanced yard kernel

- Current: `knowledge3d/cranium/bridges/advanced_rpn.py` has zero yard references.
- New kernel: `knowledge3d/cranium/ptx/advanced_rpn_kernel_transfer_yard.ptx`
- Source: `knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu`
  - Tier 3 opcodes: TRM matvec (`OP_TRM_*`), symbolic diff (`OP_SYMBOLIC_DIFF`), gradient (`OP_GRADIENT`), series sum (`OP_SERIES_SUM`), divergence/curl/laplacian, Groebner basis
  - All ride the yard substrate — no dedicated LIFO fallback
- Bridge: `AdvancedRPNEngine` loads the new PTX, removes any remaining legacy execute paths.

---

## 4. 3D Yard Stacking (new opcodes)

### 4.1 Addressing model

```
yard_addr = (bank_id ∈ [0,9), slot_id ∈ [0,69))
```

`bank_id` = which isolated yard. `slot_id` = position within that yard's 69-slot array. Each yard has its own `sp[bank_id]` (stack pointer) so independent push/pop on different yards never race.

Third dimension: `instance_id ∈ [0, cores_per_sm × sm_count)` selects the core. This is the full 3D address: `(instance, bank, slot)`.

### 4.2 New opcodes (reserve 0x170-0x17F in `RPN_DOMAIN_OPCODE_REGISTRY.md`)

| Opcode | Name | Semantics |
|---|---|---|
| `0x170` | `YARD_SELECT` | Pop `bank_id` (scalar 0-8), set active bank for subsequent push/pop/peek. Default bank = 0. |
| `0x171` | `YARD_PUSH_BANK` | Pop `bank_id`, pop value, push value into yard `bank_id` (independent of active bank). Stack unchanged elsewhere. |
| `0x172` | `YARD_POP_BANK` | Pop `bank_id`, pop from yard `bank_id`, push popped value onto active bank. |
| `0x173` | `YARD_PEEK_ADDR` | Pop `slot_id`, pop `bank_id`, push yard[bank_id][slot_id] onto active bank. True random-access read. |
| `0x174` | `YARD_TRANSFER` | Pop `dst_bank`, pop `src_bank`, pop `n_slots`, move top-n from src to dst. Atomic (no stomping). |
| `0x175` | `YARD_SP` | Pop `bank_id`, push current `sp[bank_id]` onto active bank (introspection). |
| `0x176` | `YARD_CLEAR` | Pop `bank_id`, reset `sp[bank_id] = 0` (drop-all on one yard). |
| `0x177-0x17F` | reserved | Future: `YARD_FOLD` (tier-reduce), `YARD_SIMD_MAP` (warp-cooperative yard op) |

### 4.3 Kernel shared-memory layout (per block, 9 lanes × 9 yards × 69 slots × float4)

```cuda
// Replaces the current single-stack layout in modular_rpn_kernel.cu
__shared__ float4 yards[9][9][69];    // [lane][bank][slot] = 22.2 KB
__shared__ uint8_t  sp[9][9];          //  [lane][bank]       =  81 B
__shared__ uint8_t  active_bank[9];    //  [lane]             =   9 B
```

Per Ampere sm_86: 22.3 KB shared per block fits within the 100 KB/SM budget, leaves room for the swarm-score buffer (2304 B) and halting state (128 B) from the previous closure spec.

### 4.4 Ternary-first bank selector

The `bank_id` operand is a small integer 0-8. Encode it as a **3-trit balanced-ternary field** (-1/0/+1 across three trits maps 27 distinct values, we use 9 — room for future). Use `TQUANT` (0x106) to normalize incoming scalars into trit form before `YARD_SELECT`. This keeps the bank selector in the ternary arithmetic ecosystem rather than binary-float round-tripping.

---

## 5. Embedding Sovereignty (purge numpy + external models)

### 5.1 Hot fixes (cranium/specialists/)

| File:line | Violation | Fix |
|---|---|---|
| [procedural_drawing_specialist.py:167](knowledge3d/cranium/specialists/procedural_drawing_specialist.py#L167) | `np.random.randn(256, matryoshka_dim) * 0.01` for opcode seed embeddings | Derive opcode seed from opcode-number projected through `matryoshka_prefix_dot.cu` on a unit basis vector. Opcode N → basis e_N (length 1024), apply tier-truncation via the prefix dot kernel, normalize. Deterministic and RPN-native. |
| [procedural_drawing_specialist.py:214-220](knowledge3d/cranium/specialists/procedural_drawing_specialist.py#L214-L220) | `codes = [ord(c) for c in semantic[:matryoshka_dim]]` tiled | Semantic → Character Galaxy star_id sequence → execute each star's `meaning_rpn` on a blank yard → reduce yards to tier vector via `matryoshka_prefix_dot`. No `ord()` anywhere. |
| [batch_optimizer.py](knowledge3d/cranium/specialists/batch_optimizer.py) | `import numpy` present | Replace array ops with `rpn_math_core` primitives via `TransferYardTier3Engine`. |

### 5.2 Ingestion path (training_pipelines/)

12 files under `knowledge3d/tools/training_pipelines/` and `knowledge3d/models/` call `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')`. These are not hot path but they produce the embeddings that eventually land in Galaxy — so they're upstream of sovereignty.

**Phase A (this cut)**: redirect all 12 sites to the Phenom host Ollama endpoint. Add `knowledge3d/ingestion/embedders/qwen_matryoshka_client.py` — a thin ctypes-style ingestion client (NOT for hot path) that:
- Reads `OLLAMA_EMBEDDER_HOST` env (default `192.168.0.60:11434`)
- Calls `POST /api/embeddings` with model `qwen3-embedding:0.6b`
- Requests the four matryoshka tiers {64, 128, 512, 2048} and returns them as a `dict[int, list[float]]`
- One call returns all tiers (Qwen3-Embedding produces 4096-dim; we truncate to each tier and L2-normalize)

**Phase B (next cut, not this spec)**: generate embeddings natively from `meaning_rpn` via math cores — no external model at all. This spec lays the foundation (matryoshka_prefix_dot kernel, RPN opcode seeds) so Phase B is a short follow-up.

### 5.3 Forbidden from ingestion too

- `numpy` is OK in ingestion-only scripts (spec already permits this) BUT not in `cranium/`. The existing `feedback_no_numpy_no_bulk_libraries_sovereign_only.md` applies: `knowledge3d/cranium/**` is numpy-free.
- Add grep gate to CI: `grep -rn "import numpy\|from numpy" knowledge3d/cranium/ --include="*.py"` → zero hits.

---

## 6. Qdrant Local Consultation — Qwen3-Embedding on RTX 970

### 6.1 Why Qwen (not nomic)

- Qwen3-Embedding-0.6B: native matryoshka at {64, 128, 256, 512, 768, 1024, 1536, 4096}. Covers our 64/128/512/2048 tiers exactly (2048 = truncate 4096).
- nomic-embed-text-v2-moe: matryoshka only up to 768. Fails the 2048 tier.
- Ollama pull size: 0.6B ≈ 1.2 GB VRAM on RTX 970 — fits comfortably alongside any local LLM inference if ever needed.

### 6.2 Qdrant named-vector collection shape

```python
from qdrant_client.http.models import VectorParams, Distance, HnswConfigDiff

qdrant.create_collection(
    "k3d_specifications",
    vectors_config={
        "tier_64":   VectorParams(size=64,   distance=Distance.COSINE),
        "tier_128":  VectorParams(size=128,  distance=Distance.COSINE),
        "tier_512":  VectorParams(size=512,  distance=Distance.COSINE),
        "tier_2048": VectorParams(size=2048, distance=Distance.COSINE),
    },
    # same shape for k3d_ptx and k3d_canonical
)
```

Query flow: `qdrant-find` picks the smallest tier that fits the precision budget — room-level = 64, shelf-level = 128, star-level = 512, dedup = 2048. Four tiers in one collection, no duplicate collections.

### 6.3 Files to change

- [scripts/ingest_canonical_to_qdrant.py:35](/K3D/GitHub/Knowledge3D/scripts/ingest_canonical_to_qdrant.py#L35) — replace `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"` with Qwen endpoint call
- [scripts/ingest_ptx_corpus.py:19](/K3D/GitHub/Knowledge3D/scripts/ingest_ptx_corpus.py#L19) — same
- [knowledge3d/ingestion/canonical_lookup.py:183](/K3D/GitHub/Knowledge3D/knowledge3d/ingestion/canonical_lookup.py#L183) (planned, not yet implemented) — same (canonical_lookup.py does not yet exist; create per spec)
- Re-run ingestion for all three collections on the fresh named-vector schema
- MCP servers `k3d-knowledge`/`k3d-ptx` need no code change — Qdrant named vectors are transparent; the server just picks `tier_128` by default (or tier passed in query params)

---

## 7. Ternary-first audit trail

Every opcode in this spec that takes a small-integer operand (`bank_id` 0-8, `n_slots`, `slot_id` 0-68) MUST be TQUANT-normalized on entry. No raw float comparisons for "is bank 3 or bank 5" — use the trit representation. This honors `feedback_ternary_first_where_cheaper.md` and avoids the 850-1000× Python penalty for emulated ternary logic.

Also: delete any residual Python branch like `if bank_id == 3 or bank_id == 6 or bank_id == 9` in existing specialist code — replace with `TERNARY_EQ` at kernel level.

---

## 8. Deletion list (grep-verifiable)

```bash
# Tier variant flag — must be gone
grep -rn "requested_variant\|variant.*transfer_yard" knowledge3d/cranium/bridges/lightweight_rpn.py   # → 0

# Python TransferYardStack sidecar — must be gone
grep -rn "class TransferYardStack\b" knowledge3d/cranium/bridges/                                     # → 0

# numpy in specialists — must be gone
grep -rn "import numpy\|from numpy" knowledge3d/cranium/specialists/                                  # → 0

# np.random in cranium — must be gone
grep -rn "np\.random" knowledge3d/cranium/                                                             # → 0

# Hard-coded MAX_INSTANCES = 18 in bridges — must be replaced with dynamic SM-derived sizing
grep -rn "MAX_INSTANCES\s*=\s*18" knowledge3d/cranium/bridges/                                         # → 0

# Old STACK_DEPTH in device header — must be 69
grep -n "RPN_STACK_DEPTH" knowledge3d/cranium/cuda/rpn_execute_device.cuh                              # → all lines show 69

# SentenceTransformer in ingestion — must be Qwen endpoint
grep -rn "SentenceTransformer\|sentence-transformers/all-MiniLM" knowledge3d/tools/training_pipelines/ # → 0
grep -rn "SentenceTransformer" scripts/ingest_canonical_to_qdrant.py scripts/ingest_ptx_corpus.py      # → 0
```

---

## 9. Acceptance gates (§9 style — grep + file existence, no scores)

### §9.1 — New kernel files exist and compile
```bash
test -e knowledge3d/cranium/ptx/modular_rpn_kernel_transfer_yard.ptx        || exit 1
test -e knowledge3d/cranium/ptx/advanced_rpn_kernel_transfer_yard.ptx       || exit 1
test -e knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu     || exit 1
test -e knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu    || exit 1
```

### §9.2 — Shared-memory layout matches spec §4.3
```bash
grep -n "float4 yards\[9\]\[9\]\[69\]" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # → 1 hit
grep -n "uint8_t  sp\[9\]\[9\]"        knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # → 1 hit
```

### §9.3 — New opcodes registered
```bash
grep -nE "YARD_(SELECT|PUSH_BANK|POP_BANK|PEEK_ADDR|TRANSFER|SP|CLEAR)" docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
# → 7 hits minimum
```

### §9.4 — No numpy in specialists or cranium subtree
```bash
grep -rn "import numpy\|from numpy\|np\.random\|np\.zeros\|np\.asarray" knowledge3d/cranium/specialists/  # → 0
```

### §9.5 — Qwen endpoint client exists, all 3 ingesters use it
```bash
test -e knowledge3d/ingestion/embedders/qwen_matryoshka_client.py || exit 1
grep -l "qwen_matryoshka_client\|OLLAMA_EMBEDDER_HOST" scripts/ingest_canonical_to_qdrant.py scripts/ingest_ptx_corpus.py knowledge3d/ingestion/canonical_lookup.py
# → 3 files
```

### §9.6 — Qdrant collections have 4 named vectors
```bash
curl -s http://192.168.0.4:6333/collections/k3d_specifications | jq '.result.config.params.vectors | keys'
# → ["tier_128", "tier_2048", "tier_512", "tier_64"]
```

### §9.7 — Dynamic core spawning (MAX_INSTANCES=18 gone, SM-derived sizing in place)
```bash
grep -rn "sm_count\s*\*\s*cores_per_sm\|query_sm_count" knowledge3d/cranium/bridges/
# → ≥3 files (lightweight, advanced, tiered)
```

---

## 10. Codex handoff checklist (ordered)

1. Read `docs/reports/sovereign_rpn_cas_benchmark_report.md` — internalize the 15-51% Transfer Yard baseline.
2. Read `feedback_transfer_yard_is_the_addressable_matrix.md` and `feedback_ternary_first_where_cheaper.md` in memory.
3. Write `knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu` (derive from existing + yard layout §4.3).
4. Write `knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu` (tier-3 opcodes on yard substrate).
5. Compile both to PTX (nvcc -arch=sm_86 -ptx) — commit .ptx artifacts to `knowledge3d/cranium/ptx/`.
6. Patch `bridges/lightweight_rpn.py`: delete variant flag, make yard the only path.
7. Patch `bridges/transfer_yard_tiered.py`: swap Tier 2 engine to new kernel, delete `TransferYardStack` sidecar.
8. Patch `bridges/advanced_rpn.py`: point Tier 3 to new kernel.
9. Replace `MAX_INSTANCES = 18` hard-caps with `MicroSpecialistPool.query_sm_count() * 9` at engine construction.
10. Fix `cuda/rpn_execute_device.cuh` — `RPN_STACK_DEPTH 16 → 69`.
11. Update `tests/test_rpn_semantic_depth.py` — assert `STACK_DEPTH == 69`.
12. Extend `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` with opcodes 0x170-0x177 per §4.2.
13. Replace `np.random.randn` + `ord()` paths in `procedural_drawing_specialist.py` with yard-based matryoshka prefix dot.
14. Purge numpy from `specialists/batch_optimizer.py`.
15. Write `knowledge3d/ingestion/embedders/qwen_matryoshka_client.py`.
16. On Phenom host: `ollama pull qwen3-embedding:0.6b`.
17. Patch the 3 Qdrant ingesters (§6.3) to use the Qwen client with 4 tiers.
18. Drop + recreate the 3 Qdrant collections with named-vector schema (§6.2); re-ingest.
19. Run all §9 acceptance gates; report pass/fail with evidence.

---

## 11. Must-NOT-do list

- ❌ Don't keep the legacy LIFO kernel alongside yard — "we fail and fix," not "we keep a fallback."
- ❌ Don't reintroduce numpy anywhere under `knowledge3d/cranium/`.
- ❌ Don't let the Python `TransferYardStack` dataclass survive — it's a simulation of what the kernel should do; delete it.
- ❌ Don't use `SentenceTransformer` directly in ingestion scripts — only through the Qwen client.
- ❌ Don't add `YARD_*` opcodes beyond 0x177 without first checking `RPN_DOMAIN_OPCODE_REGISTRY.md` for collisions (WINE_* reserved 0x180+, PHYSICS_EMIT_VISUAL = 0x190).
- ❌ Don't raise `YARD_DEPTH` above 69 or `YARDS_PER_CORE` above 9 without a new spec — the shared-memory budget is tuned for these values.
- ❌ Don't benchmark the yard against the linear pile to "verify" — it's settled. Benchmark the whole pipeline.

---

## 12. Stack depth answer (one line)

**69 per yard, 9 yards per core, dynamic cores_per_sm = 9 × sm_count.** On RTX 3070 that's 414 cores × 9 × 69 = **257,094 addressable slots concurrently — enough for full-universe coupled physics without memory blowup** (4 MB VRAM total).
