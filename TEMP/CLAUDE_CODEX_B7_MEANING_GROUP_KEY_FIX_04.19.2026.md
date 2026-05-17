# Codex Spec — B7 Surgical Fix for `_meaning_group_key()` (benchmark-vs-canonical scope)

**Date**: 2026-04-19
**Owner**: Claude (spec author) — acting on Codex's B7 diagnostic
**Severity**: P0 for D3 dedup correctness (Gate 1 is still open at 68,659 duplicate rows)
**Scope**: One narrow edit to `_meaning_group_key()` + two helper tweaks in
`scripts/ingestion/d3/galaxy_d3.py`. No registry touches, no kernel touches, no
ingestion-pipeline restructure.

---

## Why this spec exists

Codex's post-B6 B7 diagnostic sampled 100 duplicate-group deltas and broke them
down by axis:

| Axis | Sample share | Description |
|---|---|---|
| `benchmark-vs-canonical scope` | **68 / 100** | Same concept, but one copy has `star_type = "benchmark_<X>"` and another has `star_type = "<X>"`, so the SHA-256 over `basis` diverges. |
| coarse procedural-template reuse | 24 / 100 | Many rows share a near-identical procedural template (stencil / placeholder fill-in) and get grouped as duplicates by the B7 audit but are legitimately distinct instances. Audit false positives — NOT a `_meaning_group_key()` bug. |
| language-tag divergence | 5 / 100 | Two rows for the same concept, different surface languages, treated as duplicates in the audit. Hit `_primary_language_family()` ID-namespace fallback. |
| surface slug punctuation | 3 / 100 | `hello-world` vs `hello_world` vs `hello.world` land in distinct groups due to `_normalize_label()` not collapsing slug punctuation. |

**This spec covers the 68 + 5 + 3 = 76 samples (~76%)** that are genuine
`_meaning_group_key()` bugs. The remaining 24% (coarse-template audit false
positives) is a separate work item — audit tightening, not grouping tightening.

---

## The fix — three narrow edits in `scripts/ingestion/d3/galaxy_d3.py`

### Edit 1 — Strip `benchmark_` prefix from `star_type` (addresses 68/100)

**Location**: `_meaning_group_key()` at line 396, in both `basis` dict
constructions (lines 407-411 and 430-434).

**Current code**:
```python
basis = {
    "star_type": _non_placeholder_text(row.get("category") or row.get("kind") or row.get("type") or row.get("domain") or row.get("galaxy")),
    "concept_id": concept_id,
    "language": _primary_language_family(row),
}
```

**Problem**: When the same concept shows up under both a canonical ingestion
lane (e.g., `star_type = "lexeme"`) and a benchmark-scoped ingestion lane
(e.g., `star_type = "benchmark_lexeme"`), the SHA-256 over `basis` diverges
even though concept_id and language match. This produces the 68/100 dominant
axis Codex reported.

**Fix**: Introduce a helper near the existing text helpers (immediately above
`_meaning_group_key` at line 396) that normalizes `star_type` by stripping the
`benchmark_` prefix.

```python
def _canonical_star_type(row: dict[str, Any]) -> str:
    raw = _non_placeholder_text(
        row.get("category") or row.get("kind") or row.get("type")
        or row.get("domain") or row.get("galaxy")
    )
    lowered = raw.lower()
    # Benchmark lanes prefix their star_type with "benchmark_" to tag ingestion
    # provenance. That prefix MUST NOT fragment the canonical meaning group.
    # See TEMP/CLAUDE_CODEX_B7_MEANING_GROUP_KEY_FIX_04.19.2026.md.
    if lowered.startswith("benchmark_"):
        return raw[len("benchmark_"):]
    return raw
```

Then replace BOTH occurrences of `"star_type": _non_placeholder_text(row.get("category") or …)`
with `"star_type": _canonical_star_type(row)`.

**Exact replacement text** (apply twice — once at the `concept_id` branch,
once at the `semantic_text` branch):

```python
# BEFORE
"star_type": _non_placeholder_text(row.get("category") or row.get("kind") or row.get("type") or row.get("domain") or row.get("galaxy")),
# AFTER
"star_type": _canonical_star_type(row),
```

**What this must NOT do**:
- Must not strip any other prefix (`arc3_`, `math_`, `mmlu_`, etc.) — those
  tag legitimately-distinct domains, not provenance lanes. Only `benchmark_`
  is a pure provenance tag.
- Must not mutate the row in place. `_canonical_star_type()` is pure.
- Must not touch `_payload_type_for_row()` at line 439 — that function consumes
  `category` for a different purpose (shape dispatch) and benefits from the
  prefix.

### Edit 2 — Drop the ID-namespace fallback in `_primary_language_family()` (addresses 5/100)

**Location**: `_primary_language_family()` at line 380, specifically lines
388-392.

**Current code**:
```python
def _primary_language_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        language_profile = metadata.get("language_profile")
        if isinstance(language_profile, dict):
            language = _non_placeholder_text(language_profile.get("language"))
            if language:
                return language
    row_id = _non_placeholder_text(row.get("id"))
    if row_id.startswith("word_"):
        parts = row_id.split("_")
        if len(parts) >= 3:
            return parts[1]
    return "unknown"
```

**Problem**: When metadata.language_profile.language is missing, the function
falls back to pulling the language from the row_id namespace (e.g.,
`word_en_hello` → "en"). This creates *false separation* because two copies of
the same concept under different ingestion namespaces (one with a proper
language profile, one without) get assigned different language families by
different code paths — even when they are in fact the same language.

**Fix**: Drop the row_id fallback entirely. If the metadata doesn't declare a
language, return "unknown" — which is a single consistent sentinel across all
rows that lack a language profile, so they group together instead of
fragmenting across id-derived pseudo-languages.

**Exact replacement**:

```python
def _primary_language_family(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        language_profile = metadata.get("language_profile")
        if isinstance(language_profile, dict):
            language = _non_placeholder_text(language_profile.get("language"))
            if language:
                return language
    # Removed row_id-namespace fallback (2026-04-19 B7 fix): id-derived language
    # produced false separation when two copies of the same concept landed under
    # different ingestion namespaces. Absence of a language profile means
    # "language unknown", not "language = whatever's in the id slug".
    # See TEMP/CLAUDE_CODEX_B7_MEANING_GROUP_KEY_FIX_04.19.2026.md.
    return "unknown"
```

**What this must NOT do**:
- Must not remove the metadata.language_profile.language read — that is the
  authoritative source and covers most rows.
- Must not add a new fallback. "unknown" is the correct sentinel here.

### Edit 3 — Slug-punctuation normalization in `_normalize_label()` (addresses 3/100)

**Location**: `_normalize_label()` — find it with
`grep -n '_normalize_label' scripts/ingestion/d3/galaxy_d3.py`. It normalizes
text for the `semantic_text` branch of `_meaning_group_key()`.

**Problem**: `hello-world`, `hello_world`, `hello.world` currently hash to
distinct `semantic_text` values because `_normalize_label` preserves
punctuation. For slug-style labels, hyphen / underscore / period should be
collapsed to a single separator before hashing.

**Fix**: In `_normalize_label()`, after the existing lowercase/strip logic,
add a slug-collapse pass that replaces `-`, `_`, `.` with ` ` and then
collapses runs of whitespace to a single space.

**Recommended implementation** (Codex adapts to the exact current body):

```python
# at the end of _normalize_label, just before the final return
text = text.replace("-", " ").replace("_", " ").replace(".", " ")
text = " ".join(text.split())
```

**What this must NOT do**:
- Must not touch the content-hash path (`_content_hash`). Content hashing must
  preserve exact bytes.
- Must not run this normalization on `concept_id` — concept_ids are opaque
  identifiers, not human labels.
- Must not be applied to `_canonical_star_type()` output. Star types are
  already alphanumeric / underscore-structured and do not suffer from slug
  punctuation variance.

---

## What NOT to do in this commit

- **Do NOT** attempt to fix the coarse-template 24/100 axis in this commit.
  That's an audit-tightening problem (the B7 audit is flagging legitimately
  distinct template instantiations as duplicates). File a separate spec once
  this commit lands and the remaining duplicate_row_count is re-measured.
- **Do NOT** bump the D3 pipeline version or change the `.bin` hash convention.
  The matryoshka producer path is unaffected.
- **Do NOT** modify `_content_hash`, `_collect_procedural_payload`, or the
  `row_identity_values` chain. Those govern byte-level identity and are
  out-of-scope for the meaning-group axis.
- **Do NOT** touch the `knowledgeverse` hot path or any PTX kernel. This is
  ingestion-path Python only.

---

## Execution order

Single commit. Three narrow edits + unit test confirmation. Before/after
`duplicate_row_count` in the commit body.

### Step 1 — Apply edits 1, 2, 3

All three in `scripts/ingestion/d3/galaxy_d3.py`. No other file changes.

### Step 2 — Rerun the B6 recovery state (already healthy) and the B7 audit

```bash
bash scripts/ingestion/d3/run.sh
```

This reruns the full pipeline (B1-B5, matryoshka re-embed, B6 edge closure,
B7 audit). Expected delta:

| Gate | Before this commit | After this commit (target) |
|---|---|---|
| Gate 2 (unidirectional_site_count) | 38 | ≤ 38 (no regression) |
| Gate 3 (missing_target) | 0 | 0 (no regression) |
| Gate 1 (duplicate_row_count) | 68,659 | ≤ ~16,500 (approx 76% reduction) |

The ~16,500 residual is the coarse-template 24% axis which this commit
intentionally does not address.

### Step 3 — Spot-check the expected-to-merge groups

On the post-commit merged_stars.jsonl, sample 10 rows where the OLD audit
saw `benchmark_lexeme` vs `lexeme` as distinct. After this commit they must
share a `group_key`. Paste five sample pairs + their shared `group_key` into
the commit body.

### Step 4 — Cross-check the .bin is unchanged by this commit

The matryoshka embedding path is input-stable under these changes IF and ONLY
IF the fixed grouping causes rows to merge (fewer rows out), NOT if it mutates
any serialized field. Confirm:

```bash
sha256sum scripts/ingestion/staging/D3_dedup/matryoshka_embeddings.bin
```

Should still be deterministic across two runs. The hash value itself will
legitimately change relative to the pre-commit baseline because the row set
changed (fewer rows emitted). That is expected.

### Step 5 — Commit

Commit message template:

```
d3: tighten _meaning_group_key() on benchmark-vs-canonical scope (B7 Gate 1 fix)

Addresses 76% of the duplicate_row_count=68,659 delta surfaced by the B7
diagnostic. Three narrow edits in scripts/ingestion/d3/galaxy_d3.py:

1. New _canonical_star_type() helper strips the "benchmark_" provenance
   prefix from star_type before hashing into the meaning group basis.
   Addresses 68/100 sample deltas.

2. _primary_language_family() drops the row_id-namespace fallback (was
   pulling "en" out of "word_en_hello" when metadata.language_profile was
   absent). Absence now collapses to "unknown", a single consistent sentinel.
   Addresses 5/100 sample deltas.

3. _normalize_label() now collapses slug punctuation (-, _, .) to whitespace
   before hashing into the semantic_text basis. Addresses 3/100 sample deltas.

BEFORE:
  duplicate_row_count  = 68,659
  unidirectional_site_count = 38
  missing_target       = 0

AFTER (target):
  duplicate_row_count  = ~16,500 (residual coarse-template audit FPs, separate work)
  unidirectional_site_count = 38 (unchanged)
  missing_target       = 0 (unchanged)

Spec: TEMP/CLAUDE_CODEX_B7_MEANING_GROUP_KEY_FIX_04.19.2026.md
```

---

## Related files

- `scripts/ingestion/d3/galaxy_d3.py` — sole target of edits
- `scripts/ingestion/d3/recover_b6.py` — unaffected (pure edge closure)
- `scripts/ingestion/d3/matryoshka_bin_producer.cu` — unaffected (pure embedding)
- Prior B7 diagnostic (Codex's post-B6-reapply report, in-chat 2026-04-19) —
  source of the 100-sample axis breakdown

---

**Estimated effort**: 30-45 min (3 edits + run pipeline + spot-check + commit).
**Blocks**: Gate 1 close-out → D3 dedup declared green → sovereign procedural
symlinked architecture live end-to-end.
**Blocked by**: Nothing. Ready to execute.
**Location**: `TEMP/CLAUDE_CODEX_B7_MEANING_GROUP_KEY_FIX_04.19.2026.md`
