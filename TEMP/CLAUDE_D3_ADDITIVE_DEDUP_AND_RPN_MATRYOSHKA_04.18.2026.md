# D3 Architecture Spec — Additive Deduplication + RPN Matryoshka via Standalone Math Cores (2026-04-18)

**Follows:** `TEMP/CODEX_D2_NORMALIZATION_REPORT_04.18.2026.md` (D2 byte-stable across two full runs; not clean enough for swap).
**Opens:** D3 — the pass that clears the last four violation classes so D3 can swap for the live Galaxy.
**Author role boundary:** Claude writes this spec. Codex implements as executor + orchestrator of well-defined Python-heavy work.

---

## 0. Targets from D2 re-audit (what D3 must clear)

| Violation class | Count | D3 disposition |
|---|---:|---|
| `duplicate_row_count` | 82,672 | **§1 Additive Deduplication** — merge into single meaning-star |
| `missing_matryoshka` | 67,947 | **§2 RPN Matryoshka via Math Cores** — generate prefix-nested embeddings |
| `raw_payload` | 1,995 | **§3 Real K3D RPN Wiring** — convert raw payload to RPN program |
| `unidirectional_site_count` | 144 | **§4 Bidirectional Fix** — trivial closure pass |
| `missing_id` / `ad_hoc_id` | 0 / 0 | ✅ cleared in D2 |

After D3 the re-audit must show `duplicate_row_count = 0`, `missing_matryoshka = 0`, `raw_payload = 0`, `unidirectional_site_count = 0`.

---

## 1. Additive Deduplication

### 1.1 Principle — Daniel's directive verbatim

> "pick the best when two versions of same info/meta or other parameter joining parameters to form the single star by meaning we aim for"

**This is NOT drop-second-copy deduplication.** It is **meaning-star join**: two rows that represent the same concept become ONE star with the union of their parameters, where each parameter slot is filled by the *best* value across all sources.

### 1.2 What "same meaning" means (the grouping key)

Duplicates are grouped by **meaning hash**, NOT by surface hash:

```
meaning_hash = H(
    star_type,                          # e.g., "character", "word", "math_symbol", "concept"
    normalized_form(primary_label),     # NFKC + lowercase + strip accents for Latin; per-script rule for others
    primary_language_family,            # so "cat" (en) and "gato" (pt) DO group if mapped to same concept_id
    concept_id_if_present               # authoritative group key when present
)
```

Two rows sharing `meaning_hash` are candidates for join. Rows with `concept_id` present win the group; rows without fall back to `normalized_form` grouping.

### 1.3 Per-parameter "best value" ranking (the picker)

For each parameter slot in the merged star, apply this preference cascade:

| Parameter slot | Preference rule |
|---|---|
| `id` (canonical) | Keep the smallest lexicographic `k3d-<galaxy>/<hash16>`; record others as `aliases[]` |
| `embedding_64` / `_128` / `_512` / `_2048` | Highest-dimensional present wins; lower dims regenerated from it via matryoshka prefix (see §2) |
| `rpn_program` | Procedural RPN beats raw payload; shorter valid RPN beats longer; deterministic beats stochastic |
| `metadata.source` | Curated spec > W3C CG > PM-KR registry > auto-harvested > heuristic |
| `metadata.confidence` | Max across sources; record `confidence_sources: [(src, c), ...]` |
| `symlinks.languages[]` | Union (deduplicated by language-code + canonical surface form) |
| `symlinks.glyphs[]` | Union with glyph-content dedup |
| `symlinks.audio[]` | Union with audio-hash dedup |
| `edges.bidirectional[]` | Union; every outbound must have a matching inbound — §4 enforces |
| `provenance.sources[]` | Append-only list of all contributing row_ids (NOT overwritten) |
| `timestamps.first_seen` | Min across sources |
| `timestamps.last_updated` | Max across sources |
| Any other scalar | Source-authority cascade: curated > CG > registry > auto > heuristic |

**Invariant:** the merged star must be lossless — every parameter value from every contributing row must be either (a) present in the merged star, or (b) present in `aliases[]` / `symlinks[]` / `provenance.sources[]`. **Never discard information.** This is Daniel's "additive" requirement.

### 1.4 Algorithm sketch (Codex implements)

```python
# Ingestion-side; may use stdlib + minimal numpy for hash/lookup
# Hot-path kernels called ONLY for RPN validation (§3)

def additive_dedup(normalized_rows: Iterable[Row]) -> Iterable[MergedStar]:
    groups = defaultdict(list)
    for row in normalized_rows:
        groups[meaning_hash(row)].append(row)

    for mh, group in groups.items():
        if len(group) == 1:
            yield to_star(group[0])
            continue
        # N ≥ 2: merge
        star = MergedStar(id=pick_canonical_id(group))
        for slot_name in SLOT_ORDER:          # see §1.3 table
            picker = SLOT_PICKERS[slot_name]
            star[slot_name] = picker(group)
        star["provenance.sources"] = [r.id for r in group]
        star["aliases"] = [r.id for r in group if r.id != star.id]
        yield star
```

### 1.5 Expected output

- `merged_stars.jsonl` — one row per meaning-star (pre-D2 row count − merged-away duplicates)
- `dedup_join_map.jsonl` — for each merged_star: the source row_ids it absorbed + which slot came from which source (auditable)
- New re-audit check: `dedup_coverage` — every original D2 row_id must appear as either a `merged_star.id` or a `merged_star.aliases[*]` or a `merged_star.provenance.sources[*]`. Zero orphaned row_ids post-merge.

### 1.6 Acceptance gate

- `duplicate_row_count = 0`
- `|merged_stars| ≤ |normalized rows| − 82,672` (absorbs all 82,672 duplicates)
- `dedup_coverage = 100%` of D2 row_ids accounted for

---

## 2. RPN Matryoshka Embeddings via Standalone Math Cores

### 2.1 Principle — Daniel's directive verbatim

> "enable proper RPN matryoshka embeddings using standalone math cores — this last part is yours before Codex can advance as executor only"

This section is **my (Claude's) architecture contribution** before Codex executes. Intent: the 67,947 missing-matryoshka rows are filled by generating embeddings via K3D's **sovereign math core path** (the PTX math kernels + `rpn_math_core` + `loader.py`), NOT via numpy/torch.

### 2.2 Matryoshka nesting invariant (what we guarantee)

For every star `s`, the embedding set `{e_64(s), e_128(s), e_512(s), e_2048(s)}` must satisfy the **prefix property**:

```
e_64(s)  == e_2048(s)[:64]
e_128(s) == e_2048(s)[:128]
e_512(s) == e_2048(s)[:512]
```

Only `e_2048(s)` is stored. Lower dimensions are **views**, not separate vectors. This is Kusupati et al. 2022 applied through K3D's RPN execution model.

### 2.3 Meaning-RPN → embedding (the embedding function)

The embedding of a star is **a deterministic function of its meaning-RPN program**, not of any language surface form:

```
embed(s) = MathCore( meaning_rpn(s) ) ∈ ℝ^2048
```

Where `MathCore` is a sovereign PTX-backed pipeline:

1. **Token-level projection** — each RPN opcode maps to a fixed 2048-dim basis vector (one-hot into a learned basis matrix `B ∈ ℝ^{|opcodes|×2048}`). Basis matrix is ternary-quantized (BitNet b1.58 pattern: 5 trits per byte, 1.6 bits/weight).
2. **Stream accumulation** — RPN program is executed in a **matryoshka-accumulator kernel** (new; see §2.4) that applies per-token basis vectors with position-aware ternary-contrastive reweighting (see `docs/vocabulary/TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md`).
3. **L2 normalization** — final vector normalized via `VEC_NORM_L2_INT8 scale=64` (per Daniel's BitNet/attention-margin rulings in `feedback_attention_margin_dual_path_rulings.md` — scale=64 for headroom, NOT unit sphere scale=127).
4. **Return** — single 2048-vector; lower-dim views computed as prefixes.

### 2.4 New kernel: `matryoshka_accumulator` (spec — for Codex to implement)

**Input:**
- `rpn_program: [u32]` — token stream (opcode IDs)
- `basis: ternary-packed [|opcodes| × 2048 / 5 bytes]` — BitNet b1.58 format
- `output: [2048 × int8]`

**Semantics:**
```
acc = zeros(2048, int32)
for each token t in rpn_program:
    contrib = unpack_trits(basis[t])     # 2048-trit vector
    acc += position_weight(i) * contrib  # int32 accumulation, no overflow
acc = l2_normalize_int8(acc, scale=64)   # VEC_NORM_L2_INT8
output = acc
```

**Sovereignty:** PTX-only. No numpy/cupy/torch. Fits the `rpn_math_core` + `loader.py` tool stack already on the sovereign path.

**Weight function — MEANING-GRAVITY, not position** *(Daniel ruling 2026-04-18):*

In the House, physical position carries semantic weight (intentional librarian placement). In the **Knowledgeverse**, position is irrelevant — **meaning** is the weight. Even when the galaxy is serving as TRM's working memory, the force that governs accumulation is meaning-centric gravity (Christoph Dorn's ternary-force formula `F = T(s₁,s₂)·M(s₁)·M(s₂)/d²`), not stream position.

```
meaning_weight(t, rpn_program) =
    M(t) * sum over predecessors p in rpn_program of T(p, t) * M(p) / d²(p, t)
```

Where:
- `M(x) ∈ [0, 1]` is the meaning-mass of opcode `x` (learned per-star; for D3 use a default `M(t) = 1` for all canonical opcodes and `M(t) = 0.5` for deferred/uncertain opcodes)
- `T(a, b) ∈ {-1, 0, +1}` is the ternary affinity between opcodes (derived from the canonical opcode registry adjacency: +1 same-block, 0 unrelated, -1 opposite-block — defined per `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`)
- `d(p, t)` is RPN distance (token-stream gap between predecessor `p` and current token `t`), clamped at `≥1` to avoid singularity

This makes the accumulator meaning-centric: two semantically aligned opcodes contribute more to the star's embedding than two unrelated ones, even when stream-adjacent. Position-only weighting (uniform or linear) is explicitly rejected.

### 2.5 Why matryoshka prefix holds by construction

Because `acc` is accumulated dimension-independently (each opcode contributes to all 2048 dims simultaneously, dims do not interact during accumulation), truncating `acc` to any prefix length yields the same value you would have gotten if you had accumulated into only that prefix length. **No retraining, no dimension-specific projection — the invariant falls out of the accumulator's linearity.**

This is the specific reason to use a **linear matryoshka accumulator** rather than a non-linear embedder (which would NOT preserve the prefix property under truncation). Daniel's "standalone math cores" constraint is what makes this possible — non-linearity would require torch.

### 2.6 Acceptance gate

- For every one of the 67,947 missing-matryoshka rows: `e_2048` populated; `e_64`, `e_128`, `e_512` computed as views.
- **Prefix-property test:** random-sample 1,000 merged stars; for each, verify `e_N == e_2048[:N]` for N ∈ {64, 128, 512}. Zero failures.
- **Determinism test:** re-run embedding on the same RPN program twice; byte-identical outputs (int8 accumulator, no float nondeterminism).
- **Sovereignty test:** `scripts/sovereignty_preflight.sh` stays green after kernel lands.

### 2.7 What Claude has NOT specified (intentionally, for Codex autonomy)

- Exact PTX register allocation for `matryoshka_accumulator` — Codex's call.
- Whether to batch across stars or per-star — Codex benchmarks and decides.
- Host-side orchestrator (Python) — Codex implements; must stay in `knowledge3d/ingestion/` (not hot path).

---

## 3. Real K3D RPN Wiring (raw_payload → RPN)

### 3.1 Principle — Daniel's directive verbatim

> "after this, proper wiring of missing metadata and real K3D RPN"

D2 deferred 1,995 procedural upgrades (all of the raw_payload rows). D3 wires them.

### 3.2 What "raw payload" means and what "real K3D RPN" means

- **Raw payload row:** a star whose content is stored as an opaque blob (JSON, text, bytes) rather than as an executable RPN program. Invisible to the sovereign hot path.
- **Real K3D RPN row:** a star whose content is an executable RPN program over the canonical opcode registry (`docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`), runnable on PTX via `loader.py`.

### 3.3 Conversion rules by raw_payload type

For each of the 1,995 rows, apply one of these transforms based on `payload_type`:

| `payload_type` | Target RPN form | Opcode range | Fallback if un-convertible |
|---|---|---|---|
| `math_expression` (infix string) | Infix → RPN via Transfer Yard (`knowledge3d/skills/infix_to_rpn.py`) | Math block (existing) | Flag `conversion_failed: math_parse` |
| `glyph_bezier` (SVG path) | Bézier → segment RPN (Drawing galaxy) | Drawing block (existing) | Flag `conversion_failed: bezier_parse` |
| `text_sequence` (word/char list) | Character symlink chain | Character/Word galaxy opcodes | Flag `conversion_failed: unknown_char` |
| `grammar_rule` (natural language rule) | Defeasible-logic RPN (`gre_defeasible_resolver.cu` contract) | Reasoning block 0xA0-0xF1 | Flag `conversion_failed: rule_parse` |
| `audio_spec` (frequency/duration) | Temporal pattern RPN | Audio galaxy opcodes | Flag `conversion_failed: audio_parse` |
| `unknown` | — | — | Flag `conversion_failed: unknown_type`; log for human review |

### 3.4 Acceptance gate — DISCARD-AND-MARK policy *(Daniel ruling 2026-04-18)*

- `raw_payload = 0` in D3 re-audit (all 1,995 rows leave the raw-payload class)
- **Success path:** converted RPN validates via sovereign loader smoke-test (load + dry-run, no execution) → row is promoted to `rpn_upgrades.jsonl` with `conversion_status: success`
- **Failure path (incomplete conversion):** row is **discarded from the normalized tree** AND emitted to `pending_proceduralization_queue.jsonl` with fields:
  ```json
  {
    "source_row_id": "...",
    "payload_type": "math_expression | glyph_bezier | ...",
    "failure_reason": "math_parse | bezier_parse | unknown_type | ...",
    "target_phase": "next_proceduralization_phase",
    "queued_at": "2026-04-18T...",
    "original_payload_hash": "..."
  }
  ```
- No "conversion_failed" rows carried in the live Galaxy. Queue is the explicit target for the next proceduralization phase (post-D3, separate spec).
- Success rate is **observed, not gated** — we discard what we can't proceduralize; we don't halt D3 on imperfect coverage.

---

## 4. Bidirectional Edge Closure (trivial)

The 144 unidirectional sites are fixed by a single pass:

```python
for every edge (a → b) lacking reverse (b → a):
    emit (b → a) with inverse_of: (a → b)
```

**Acceptance gate:** `unidirectional_site_count = 0`.

---

## 5. Execution order (batches)

| Batch | Contents | Depends on |
|---|---|---|
| B1 | §4 Bidirectional closure | nothing |
| B2 | §1 Additive dedup (meaning-star join) | B1 (so merged stars inherit complete edge sets) |
| B3 | §3 Raw_payload → RPN | B2 (dedup'd stars may absorb raw_payload siblings) |
| B4 | §2 RPN Matryoshka embedding | B3 (embeddings need the final RPN program to hash against) |
| B5 | D3 re-audit + acceptance gates | B4 |

**Codex runs these serially; each batch produces an artifact + hash; each batch must be byte-stable across two runs before the next starts (same discipline as D2).**

---

## 6. Deliverables (what D3 emits)

- `merged_stars.jsonl` — post-dedup star table (replaces D2's normalized/*.jsonl for the live Galaxy swap)
- `dedup_join_map.jsonl`
- `rpn_upgrades.jsonl` — raw_payload → RPN conversion log
- `matryoshka_embeddings.bin` — int8 2048-dim vectors per merged_star, memory-mapped (views give lower dims)
- `re_audit_d3/` — same structure as `re_audit/` in D2; all acceptance gates pass
- `D3_FINAL_REPORT.md` — summary, hashes, gate-pass evidence

---

## 7. Daniel's rulings (2026-04-18 — all three resolved)

1. **§2.4 weight function** — **meaning-gravity, not position**. In the House position matters; in the Knowledgeverse meaning does. Accumulator uses Christoph Dorn's ternary force `F = T·M·M/d²` with opcode-adjacency affinity. Specified inline in §2.4.
2. **§1.3 source-authority cascade** — `curated > CG > registry > auto > heuristic` endorsed as adequate. No change.
3. **§3.3 conversion-failed policy** — **discard incomplete, mark as target for next proceduralization phase**. Failures leave the normalized tree and go to `pending_proceduralization_queue.jsonl` for explicit re-attack later. D3 does NOT halt on imperfect coverage; success rate is observed, not gated. Specified inline in §3.4.

Codex may now proceed with **B1 → B2 → B3 → B4 → B5** sequentially.

---

## 8. Relationship to Paper A

The `matryoshka_accumulator` kernel (§2.4) is a **concrete deliverable** that strengthens Paper A's C1 (sovereignty) and C2 (TRM-as-Avatar) claims — it demonstrates that even embedding generation, which conventionally requires torch, runs on the sovereign substrate. Add to Paper A §5 Results as a sub-result of C1. Do NOT promote it to a fourth contribution — it belongs under C1.

The additive-dedup meaning-star join (§1) is a **Paper D (Form → Meaning) contribution** — preview it in Paper A §6 Conclusion as future work; full treatment in Paper D.
