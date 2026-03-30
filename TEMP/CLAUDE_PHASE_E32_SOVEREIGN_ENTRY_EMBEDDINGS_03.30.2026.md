# Claude -- Phase E.32: Sovereign Entry Embeddings (Pre-Compute, Stop Recomputing)

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL -- this is the CURRENT boot blocker (E.31 KNN build is fixed but unreachable)

---

## Daniel's Recurring Correction

> "Why are we calculating with numpy when we do have our own matryoshka rpn embedding standard?"

E.31 fixed the KNN graph build (numpy matmul → PTX kernel). But `bind_gpu_galaxy_runtime`
never REACHES the GPU KNN build because `_flatten_galaxies_for_gpu()` stalls first.

**Root cause:** `_entry_embedding16()` (line 2351) is called 248,078 times, and EVERY call
hits the CPU fallback because ZERO entries have pre-computed `embedding16` fields:

```python
# Line 2351-2375: _entry_embedding16() call chain
# 1. entry.get("embedding16")   → None (not in ANY entry)
# 2. entry.get("embedding")     → None (not in ANY entry)
# 3. metadata.get("embedding16") → None (not in ANY entry)
# 4. metadata.get("embedding")   → None (not in ANY entry)
# 5. FALLBACK: embed_sentence(text)  ← CPU Python, called 248K times
# 6. LAST RESORT: character hash     ← even slower
```

**Verified:** Checked JSONL files (meaning_layer_stars, Language, Math) AND the warm boot
checkpoint (1.56 GB, 248,078 entries across 19 galaxies). NONE have `embedding16`.

---

## What Exists Sovereign

| Component | File | What it does |
|-----------|------|-------------|
| `RPNEmbeddingEngine.embed_sentence_gpu()` | `cranium/rpn_embedding_engine.py:260` | GPU trigram embedding for one sentence |
| `RPNEmbeddingEngine.embed_sentences_gpu()` | `cranium/rpn_embedding_engine.py:273` | **BATCH GPU embedding for multiple sentences** |
| `TrigramEmbedBridge.embed_indices()` | `cranium/bridges/trigram_embed_bridge.py:73` | GPU trigram lookup + average |
| `trigram_lookup_average` | `cranium/ptx/trigram_embed.cu:13` | PTX kernel: trigram → embedding on GPU |
| `l2_normalize_embedding` | `cranium/ptx/trigram_embed.cu:46` | PTX kernel: L2 normalize on GPU |

The batch GPU path (`embed_sentences_gpu`) already exists. It:
1. Collects unique tokens across all sentences
2. Embeds each unique token via GPU trigram kernel
3. Mean-pools per sentence
4. Returns `List[Float32Vector]`

---

## Architectural Principle: Embeddings Belong WITH the Entry

From `KNOWLEDGEVERSE_SPECIFICATION.md` §2.1:
> "Galaxy Universe is always loaded in VRAM"

Embeddings are part of the Galaxy entry — they're how the entry lives in the VRAM
embedding space. Computing them at bind-time is like computing a word's spelling
every time you open a book. They should be computed ONCE at ingestion and stored.

From `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` §1.2:
> "Form + Meaning composable contracts"

The embedding IS the entry's meaning-space coordinates. It's part of the entry's
foundational representation, not a derived quantity to recompute.

From `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` §1.6:
> "Save Information Principle: DON'T duplicate what exists"

Computing the same embedding 248K times on every boot duplicates work that should
be done once.

---

## Two-Part Fix

### Part 1: Pre-Compute Embeddings at Ingestion (One-Time Enrichment)

Write a one-time enrichment script that:
1. Loads all 248K entries from the checkpoint
2. For entries missing `embedding16`, computes it using `RPNEmbeddingEngine`
3. Saves the enriched entries back to the checkpoint

```python
# scripts/enrich_embeddings.py (one-time, ingestion path)
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

engine = RPNEmbeddingEngine(embedding_dim=16)

# For each entry missing embedding16:
text = entry.get("content") or entry.get("name") or entry.get("id", "")
embedding = engine.embed_sentence(text)  # CPU is fine for one-time ingestion
entry["embedding16"] = list(embedding)
```

Using GPU batch (`embed_sentences_gpu`) would be faster but requires GPU init.
For a one-time script, even CPU is acceptable — ingestion path is flexible per
CLAUDE.md: "Ingestion Path = Flexible: Can use any tools/libraries."

**After this runs:** Every entry in the checkpoint has `embedding16`. Bind-time
reads it directly — zero computation.

### Part 2: GPU Batch Fallback in `_flatten_galaxies_for_gpu()`

For any entries that still lack `embedding16` (new entries added after enrichment),
use the batch GPU path instead of per-entry CPU:

**Replace the per-entry pattern:**
```python
# CURRENT (line 2119): per-entry CPU embedding in _append_flattened_entry
embedding = self._entry_embedding16(entry)  # Called 248K times, CPU fallback each time
```

**With a two-pass approach in `_flatten_galaxies_for_gpu()`:**

```python
def _flatten_galaxies_for_gpu(self, *, galaxy_names=None):
    names = self._resolve_live_galaxy_names(galaxy_names)

    # Pass 1: Collect all entries, identify which need embeddings
    all_entries = []
    needs_embedding = []  # (index, text) pairs
    for galaxy_name in names:
        galaxy = self.galaxy_manager.get_galaxy(galaxy_name)
        for entry in getattr(galaxy, "entries", []):
            idx = len(all_entries)
            all_entries.append((galaxy_name, entry))
            if not _has_precomputed_embedding16(entry):
                text = self._entry_embedding_text(entry) or json.dumps(entry)[:256]
                needs_embedding.append((idx, text))

    # Pass 2: Batch-compute missing embeddings on GPU
    if needs_embedding:
        engine = self.get_text_embedding_engine()
        if engine.has_gpu_bridge():
            texts = [text for _, text in needs_embedding]
            vectors = engine.embed_sentences_gpu(texts)
            for (idx, _), vec in zip(needs_embedding, vectors):
                all_entries[idx][1]["embedding16"] = list(vec)[:16]
        else:
            # If no GPU bridge yet, compute on CPU (boot-time only)
            for idx, text in needs_embedding:
                vec = engine.embed_sentence(text)
                all_entries[idx][1]["embedding16"] = list(vec)[:16]

    # Pass 3: Flatten (now all entries have embedding16, fast path)
    flat, catalog = [], []
    for galaxy_name, entry in all_entries:
        self._append_flattened_entry(flat=flat, catalog=catalog,
                                     galaxy_name=galaxy_name, entry=entry)
    return flat, catalog
```

**Key change:** Instead of 248K individual CPU calls, ONE batch GPU call for all
entries that need embeddings. The GPU trigram kernel processes them all in parallel.

### Part 3: Save Enriched Checkpoint After First Bind

After `_flatten_galaxies_for_gpu()` enriches entries with `embedding16`, save
the enriched checkpoint. Next boot: ALL entries have pre-computed embeddings,
zero computation needed.

```python
# At end of bind_gpu_galaxy_runtime(), after flattening:
if needs_enrichment_count > 0:
    self._save_consolidated_state()  # Persist enriched entries
    print(f"[K3D] Saved enriched checkpoint ({needs_enrichment_count} new embeddings)")
```

---

## Execution Sequence

1. **Write `scripts/enrich_embeddings.py`** — one-time script to add `embedding16` to all
   248K entries in the checkpoint. Run once.

2. **Modify `_flatten_galaxies_for_gpu()`** — two-pass: collect entries, batch-embed
   missing on GPU, then flatten. Replace per-entry CPU embedding with batch GPU.

3. **Add enrichment save** — after first bind enriches entries, save checkpoint so
   embeddings persist for future boots.

4. **Run the enrichment script** — populates embeddings for all 248K entries.

5. **Run local ARC3 benchmark** — should now boot in seconds (pre-computed embeddings
   → fast flatten → GPU KNN build from E.31 → benchmark starts).

6. **Run full benchmark** — real numbers with 248K stars.

---

## Performance Impact

| Operation | Before | After |
|-----------|--------|-------|
| `_entry_embedding16()` × 248K | ~248K CPU `embed_sentence()` calls | Read from entry dict (0 computation) |
| `_flatten_galaxies_for_gpu()` | Minutes (CPU embedding bottleneck) | Seconds (just dict reads + list append) |
| Total bind time | >5 minutes (embedding + KNN stall) | <5 seconds (pre-computed + GPU KNN) |
| GPU utilization during bind | 0% | >50% (GPU KNN from E.31) |

---

## Future: Ingestion Pipeline Enrichment

After this fix, ALL ingestion paths should compute and store `embedding16`:

| Ingestion Path | Action |
|----------------|--------|
| `normalize_meaning_stars.py` | Add `embedding16` to each entry |
| `populate_*.py` scripts | Add `embedding16` to each entry |
| `runtime_ingest.py` | Add `embedding16` to each entry |
| `_save_consolidated_state()` | Preserve `embedding16` in checkpoint |
| `galaxy_manager._read_entries_from_disk()` | Read `embedding16` from JSONL |

This ensures embeddings are computed once at ingestion and carried forward forever.
Bind-time never needs to compute them again.

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/enrich_embeddings.py` | One-time: add `embedding16` to all 248K checkpoint entries |

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Two-pass flatten: batch GPU embed for missing, save enriched checkpoint |

---

## Matryoshka Alignment

The `RPNEmbeddingEngine` supports `embedding_dim=128` (full Matryoshka capacity).
Current entries use 16-dim (coarsest LOD). The enrichment stores the 16-dim prefix.

Future enhancement: store full 128-dim embeddings, use 16-dim prefix for coarse
KNN graph, 64-dim for medium, 128-dim for fine scoring. This IS the Matryoshka
principle: nested dimensions for nested precision levels.

For now: 16-dim sovereign embeddings, pre-computed, stored with entries. Zero
recomputation at bind time.

---

## Success Criteria

- [ ] All 248K entries have `embedding16` in checkpoint after enrichment
- [ ] `_flatten_galaxies_for_gpu()` completes in <10 seconds (was minutes)
- [ ] `_entry_embedding16()` hits the pre-computed fast path for all entries
- [ ] Missing embeddings use batch GPU path (`embed_sentences_gpu`), not per-entry CPU
- [ ] Enriched checkpoint saved for future boots
- [ ] `bind_gpu_galaxy_runtime()` reaches the GPU KNN build (E.31)
- [ ] Total bind time < 15 seconds (was >5 minutes)
- [ ] Local ARC3 benchmark boots and runs
- [ ] Full benchmark boots and runs
