# CLAUDE → CODEX — Phase 6.B.3: Sovereign Matryoshka RPN Text Embedder — 2026-04-11

## Daniel's correction

> "Use math cores and embed anything."

The crafter's toy embedder was the obvious symptom, but the deeper problem is that K3D already has the pieces for a **sovereign text → Matryoshka-stack embedder running on the RPN math cores**, and nothing is stitching them into one canonical path. The existing pieces:

- [rpn_embedding_engine.py](../knowledge3d/cranium/rpn_embedding_engine.py) `RPNEmbeddingEngine` — trigram → per-trigram learned vector → aggregate. Already has `embed_word`, `embed_sentence`, `embed_trigrams`, `embed_sentence_gpu`, a host-side trigram table, and a GPU bridge attach point. Native RPN math-core substrate.
- [matryoshka_bridge.py](../knowledge3d/cranium/bridges/matryoshka_bridge.py) `MatryoshkaProjectionBridge` — `project_device` / `project_host` for vector→tier projection. Already exists.
- [matryoshka_trm.py](../knowledge3d/cranium/matryoshka_trm.py) `MatryoshkaTRM` — prefix-self-contained matrix semantics, same pattern we want for embeddings.
- [docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §7](../docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md#7-matryoshka-embeddings-search-lod) — the 64/128/512/2048 tier stack is literally prefix slices of one projection.

Everything is present. Nothing is wired. 6.B.3 wires them.

## Governing principle

**One projection, many tiers, running on the math cores.** The crafter does not own an embedder. The query path does not own an embedder. The sovereign Matryoshka RPN embedder is a single canonical entry point that consumes text (or, long-term, any modality) and emits a max-dim vector; consumers prefix-slice to the tier they need. It runs natively through `RPNEmbeddingEngine` on the math cores.

This is ingestion-path-allowed — but the same entry point is used at query time too, so both paths share state. No two-embedder split. No sentence-transformer dependency.

## Scope for 6.B.3 (text only, minimal, correct)

### 1. Canonical entry point

New module or extension inside the existing engine: `SovereignMatryoshkaTextEmbedder`, living alongside `RPNEmbeddingEngine`. It wraps `RPNEmbeddingEngine` with a Matryoshka contract:

```python
class SovereignMatryoshkaTextEmbedder:
    # The canonical tier set, per schema §7.
    TIER_DIMS = (64, 128, 512, 2048)

    def __init__(self, engine: RPNEmbeddingEngine, *, max_dim: int = 2048):
        self._engine = engine  # runs on math cores
        self._max_dim = max_dim

    def embed_max(self, text: str) -> list[float]:
        """Produce the full max-dim projection via the RPN math core."""
        ...

    def embed_stack(self, text: str) -> dict[int, list[float]]:
        """Return {tier: prefix_slice_normalized} for every tier in TIER_DIMS."""
        ...

    def embed_tier(self, text: str, tier: int) -> list[float]:
        """Return a single tier (prefix-slice + renormalize)."""
        ...
```

- `embed_max` runs the engine's sentence/word embed path on the math cores and returns a max_dim vector.
- `embed_stack` computes `embed_max` once, then produces every tier as **`normalize(vector[:tier])`**. That is the entire Matryoshka contract. Do not re-run the engine per tier.
- `embed_tier` is the convenience path consumers hit when they only want one slice.

The wrapper has **no sentence-transformer call anywhere**. The engine is the only math-core substrate it touches.

### 2. Max-dim decision

`RPNEmbeddingEngine`'s `embedding_dim` is currently whatever its constructor sets. For 6.B.3:

- If the engine's natural max dim is already ≥ 2048, use that.
- If it is smaller, extend the engine's trigram-vector table dim to 2048 via the engine's own sizing path (do not invent a new table). `RPNEmbeddingEngine._initialise_embedding` and the trigram table use a deterministic seed — lift the dim and the existing initialization path produces a 2048-wide random projection.
- The engine's per-trigram random projection is a **deterministic, seeded** source. Both crafter and query path must instantiate the engine with the **same seed**, or get handed the same engine instance through a singleton. Pick instance-sharing: one `SovereignMatryoshkaTextEmbedder` constructed at Knowledgeverse boot, reused everywhere.

### 3. Why a random-seeded projection works for bootstrap

At first glance a random trigram projection can't distinguish `"plus"` from `"minus"` semantically. Two things rescue us:

- **Shared trigrams across rich text.** The crafter already stamps `keywords`, `aliases`, `description`, and `meaning_rpn` onto each star. `math_operator_addition` carries `{plus, addition, sum, add, +}`. A `"plus"` query's trigrams (`_pl, plu, lus, us_`) overlap the star's `plus` and `plus`-adjacent keywords directly. Random trigram vectors cosine-near-zero against non-overlapping text and cosine-high against overlapping text. That is enough to flip rank-5880 to rank-sub-10 for direct keyword queries.
- **Prefix Matryoshka structure holds under random projection.** The Johnson-Lindenstrauss lemma says random projection preserves cosine structure up to distortion bounded by 1/√dim. Prefix slices *further* shrink the dim, so tier_64 will be noisier than tier_512, but direction of cosine (positive/near-zero) survives. We're not asking for fine-grained semantics here — just for the `"plus"` query to beat 41,000 unrelated stars by a keyword-overlap margin.

If the probe after 6.B.3 still shows bad ranks at tier_64, the fix is to **step up to tier_128 or tier_256** for the navigation pass, not to retrain the projection. Learned projections are sleeptime territory, not bootstrap.

### 4. Crafter wiring

In [star_crafter.py](../knowledge3d/ingestion/star_crafter.py), the crafter already stamps rich text (name, domain, meaning_rpn, description, aliases, keywords) onto each crafted entry. Drop 3's partial fix — "remove toy embedder, set `embedding=[]`" — was right about removing the toy but wrong about the fallback. The correct shape:

```python
# At craft time — one embedder, constructed once per crafter run, shared across all stars.
self._embedder = SovereignMatryoshkaTextEmbedder(engine=rpn_engine_singleton())

# Per crafted star — use the rich semantic text the crafter already assembles.
semantic_text = " ".join(filter(None, [
    entry.name,
    entry.meaning_star.domain,
    entry.meaning_star.meaning_rpn,
    entry.metadata.get("description", ""),
    " ".join(entry.metadata.get("aliases", [])),
    " ".join(entry.metadata.get("keywords", [])),
]))
stack = self._embedder.embed_stack(semantic_text)
row["embedding_max"] = stack[max(stack)]          # full max-dim vector
row["embedding"] = stack[64]                       # coarse tier for the GVRAM 64D slot
row["embedding_tier_128"] = stack[128]             # optional: stash higher tiers for later consumers
row["embedding_tier_512"] = stack[512]
row["embedding_tier_2048"] = stack[2048]
```

The 64D slot at [galaxy_vram_table.py:13](../knowledge3d/knowledgeverse/galaxy_vram_table.py#L13) takes `row["embedding"]` = tier_64 as it already does. Higher tiers are stashed on the host dict for future consumers (Phase 6.C's TRM navigation, sleeptime consolidation, Bathtub introspection) without touching the current VRAM layout.

### 5. Query path wiring

Wherever the cosine routing probe embeds a query string (currently `_entry_embedding64` at [knowledgeverse.py:4148](../knowledge3d/knowledgeverse/knowledgeverse.py#L4148) and `_precomputed_entry_embedding64_raw` at [knowledgeverse.py:3977](../knowledge3d/knowledgeverse/knowledgeverse.py#L3977)), replace the existing path with a call to the **same** `SovereignMatryoshkaTextEmbedder` instance. `embed_tier(text, 64)` for the coarse GVRAM cosine pass.

If `_entry_embedding64` has other callers that rely on its current behavior, migrate them. There should only be one canonical text embedder. If migration is too entangled for 6.B.3, land a feature-flagged call that routes the crafter's query path through the sovereign embedder and file a follow-up to migrate the rest. Do not allow two embedders to coexist silently.

### 6. Engine singleton

Add a `get_sovereign_text_embedder()` accessor near the Knowledgeverse boot path that:

- Constructs one `RPNEmbeddingEngine` with a deterministic seed and max_dim=2048.
- Wraps it in one `SovereignMatryoshkaTextEmbedder`.
- Returns the same instance on every subsequent call.
- Is the only construction path used by both the star crafter and the query embedder.

Instance sharing is the cheapest way to guarantee both paths observe the same random projection, which is the cheapest way to guarantee cosine comparability.

### 7. "Embed anything" future-compat

Daniel's "embed anything" points at the fact that the same Matryoshka-stack contract should eventually handle images, audio, and RPN programs — not just text. For 6.B.3 this is explicitly out of scope, but the API must not be text-specific. The `SovereignMatryoshkaTextEmbedder` interface is the thin text front; long-term we add siblings:

- `SovereignMatryoshkaVisualEmbedder` (Drawing Galaxy programs → vectors)
- `SovereignMatryoshkaAudioEmbedder` (spectrogram RPN → vectors)
- `SovereignMatryoshkaProgramEmbedder` (RPN bytecode → vectors)

All sharing the Matryoshka tier contract and running on the same math cores. 6.B.3 lands only the text path; the naming and return shape must allow the siblings to slot in without reshaping consumers.

## Validation

Extend `tests/test_phase6b2_live_integration.py` (or add `tests/test_phase6b3_sovereign_matryoshka_embedder.py`) under `K3D_PYTEST_PROBE_CUDA=1`:

1. **Instance sharing** — crafter's embedder and Knowledgeverse query embedder are the *same object* (`id(a) == id(b)`). If not, the test fails before any cosine math. This catches the two-engine split at the source.
2. **Tier contract** — for a fixed text input, `embed_stack(text)[64] == normalize(embed_stack(text)[2048][:64])`. Prefix property proven.
3. **Determinism** — same text embedded twice returns identical vectors.
4. **Live cosine ranks at tier_64**:
   - `"plus"` → `math_operator_addition` rank ≤ 10
   - `"addition"` → `math_operator_addition` rank ≤ 10
   - `"two"` → `concept_digit_two` rank ≤ 20 (digits are shorter trigram targets, rank is slightly looser)
   - `"3"` → `concept_digit_three` rank ≤ 20
5. **Tier step-up rescue** — if any tier_64 rank is outside the gate above, re-run at tier_128 and record the rank. The test is informational, not gating, for tier_128 — but the diagnostic feeds the rank-vs-tier curve we need to decide navigation tier for 6.C.
6. **Sovereign grep** — no new `import (numpy|cupy|scipy|sympy)` anywhere under `knowledge3d/cranium/` or `knowledge3d/bridge/`. Ingestion-side `knowledge3d/ingestion/star_crafter.py` remains free to import what it needs, but should not reach for sentence-transformers in the embedding path.
7. **Regression** — Phase 6.B / 6.B.1 / 6.B.2 suites all stay green.

## What this does NOT do

- Does not train the trigram projection. Random seeded projection is bootstrap. Learned refinement is sleeptime consolidation in a later phase.
- Does not introduce a new VRAM star record layout. Higher tiers live on the host-side star dict for now; the GVRAM record still holds tier_64 in its existing 64D slot.
- Does not wire sibling modality embedders (visual, audio, program). Text-only, with an API that leaves room.
- Does not fix any unrelated embedding call site outside the crafter→query cosine path. File follow-ups for anything that resists migration.
- Does not start 6.C. HANDLING_QUERY wiring is blocked on this passing the rank gate.

## Order of operations

1. Add `SovereignMatryoshkaTextEmbedder` wrapper on `RPNEmbeddingEngine`. Ship with unit tests for tier prefix contract + determinism. No other wiring yet.
2. Add the singleton accessor. Unit test for instance sharing.
3. Point the crafter at the singleton, delete the `embedding=[]` placeholder, stamp tier_64 + full stack onto the row. Rerun `tests/test_phase6b_star_crafter.py` — must stay green structurally; embedding shape changed but semantics preserved.
4. Point the query path at the same singleton. Rerun cosine routing probe against the live 41,134-star table. Report the ranks per tier.
5. If tier_64 ranks land the gate, greenlight 6.C. If not, report the tier-vs-rank curve and stop for direction.

## One principle

*The Matryoshka RPN embedder is a math-core operation.* It does not borrow semantics from sentence-transformers. It does not borrow structure from a numpy projection. It runs through `RPNEmbeddingEngine`, produces one max-dim vector, and the tier stack is prefix slices of that vector. Every consumer that needs a cosine hit against the Galaxy goes through the same entry point at the same seed. One projection, many tiers, one path.

Report per step. If the rank gate misses at tier_64 but recovers at tier_128, that is a *routing-tier* decision for 6.C, not a failure of 6.B.3. Raw ranks beat passing tests at this stage.
