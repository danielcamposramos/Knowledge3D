# CODEX BATCH 8.1 — Canonical Math Alias Seed (Phase 7.A.1 Unblock)

**Date:** 2026-04-14
**Author:** Claude (architecture)
**Status:** Approved — ready for Codex
**Depends on:** Batch 8 (landed)
**Unblocks:** `ingest_hs_math_cluster1.py --write`
**Scope:** ingestion path only (no hot-path, no PTX, no numpy/cupy/scipy)

---

## 1. Problem

`scripts/ingest_phase7a1_seed_audit.py` reports 6/38 Phase 7.A.1 aliases present in `k3d_canonical`. The printed remedy — `python scripts/populate_math_symbols.py --pinned` — cannot resolve it:

- `populate_math_symbols.py` writes to local JSONL (`/K3D/Knowledge3D.local/galaxies/Character.jsonl`) via `galaxy_population_utils.upsert_entries`. It **never touches Qdrant `k3d_canonical`**.
- Its star IDs (`char_math_summation`, `char_op_plus`, …) do **not match** the scheme Batch 8 aliases expect (`math_symbol_n_ary_summation` via `math_symbol_star_id()`, `char_u002b` via `canonical_char_star_id()`, `concept_reciprocal`).
- `--pinned` is a no-op compatibility shim.

Net: the runbook is a dead end. A dedicated seed script is required.

## 2. Deliverable

### 2.1 New file: `scripts/seed_batch8_canonical_math_aliases.py`

Iterate `SYMBOL_ALIASES ∪ CONSTANT_ALIASES` (from `knowledge3d/ingestion/math_semantic_aliases.py`, 30 + 8 = 38 entries) and register each in `k3d_canonical` via `CanonicalLookup.register()`.

**Kind dispatch by star_id prefix:**

| star_id prefix        | kind          |
|-----------------------|---------------|
| `math_symbol_`        | `math_symbol` |
| `char_`               | `char`        |
| anything else         | `concept`     |

**Key:** the alias name (the dict key — `"plus"`, `"pi"`, `"reciprocal"`, …). Stable, human-readable, matches the audit lookup surface.

**Metadata (per entry):**
```python
{"context_id": 0, "ethical_trit": 0, "source": "batch8_seed"}
```

**Idempotency:** `canonical_entry_id(kind, key)` is uuid5-deterministic — reruns upsert the same point, never duplicate.

**Sketch:**
```python
from __future__ import annotations

import argparse
from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
from knowledge3d.ingestion.math_semantic_aliases import (
    CONSTANT_ALIASES,
    SYMBOL_ALIASES,
)


def _dispatch_kind(star_id: str) -> str:
    if star_id.startswith("math_symbol_"):
        return "math_symbol"
    if star_id.startswith("char_"):
        return "char"
    return "concept"


def seed(lookup: CanonicalLookup | None = None) -> dict[str, int]:
    lookup = lookup or CanonicalLookup()
    lookup.ensure_collection()
    aliases = {**SYMBOL_ALIASES, **CONSTANT_ALIASES}
    counts = {"math_symbol": 0, "char": 0, "concept": 0}
    for alias_name, star_id in aliases.items():
        kind = _dispatch_kind(star_id)
        lookup.register(
            kind=kind,
            key=alias_name,
            star_id=star_id,
            metadata={"context_id": 0, "ethical_trit": 0, "source": "batch8_seed"},
        )
        counts[kind] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Batch 8 canonical math aliases into k3d_canonical.")
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()
    counts = seed(CanonicalLookup(url=args.url))
    total = sum(counts.values())
    print(f"seeded: total={total} math_symbol={counts['math_symbol']} char={counts['char']} concept={counts['concept']}")


if __name__ == "__main__":
    main()
```

### 2.2 Fix audit error message

In `scripts/ingest_phase7a1_seed_audit.py`, replace the printed remedy:

- **From:** `python scripts/populate_math_symbols.py --pinned`
- **To:**   `python scripts/seed_batch8_canonical_math_aliases.py`

### 2.3 Update Batch 8 spec §5.3 runbook

In `TEMP/CODEX_BATCH8_HS_MATH_CLUSTER1_INGESTION_2026-04-14.md` §5.3, replace the `populate_math_symbols.py --pinned` line with the new script. Leave `populate_math_symbols.py` untouched — it remains the Character-galaxy bootstrap, a separate concern.

### 2.4 Test: `tests/test_batch8_1_canonical_alias_seed.py`

Unit tests (fake CanonicalLookup stub):
1. `seed()` issues exactly 38 `register()` calls.
2. Kind dispatch is correct for each of the three prefix families (spot-check: `plus`→`math_symbol`, `power`→`char`, `reciprocal`→`concept`).
3. All 38 `(kind, key, star_id)` tuples match the aliases dict.
4. Idempotency: calling `seed()` twice on the same fake issues 76 register calls with identical payloads (real Qdrant collapses via uuid5; the script itself is a pure iteration, so this just guards against accidental dedup logic).

Integration test (gated on `K3D_QDRANT_INTEGRATION=1`):
5. Run `seed()` against real Qdrant, then assert `ingest_phase7a1_seed_audit.py` exits 0.

## 3. Acceptance

- `python scripts/seed_batch8_canonical_math_aliases.py` completes without error against a running `k3d_canonical`.
- `python scripts/ingest_phase7a1_seed_audit.py` reports **38/38 present**, exit 0.
- `K3D_QDRANT_INTEGRATION=1 python scripts/ingest_hs_math_cluster1.py --write` now proceeds past the seed gate and writes Cluster 1 meaning_stars + symlinks.
- Rerunning the seed is a no-op (uuid5 upsert).
- New pytest file passes; existing Batch 8 suite still green.

## 4. Sovereignty

- Ingestion path only. No hot-path, no PTX, no Galaxy VRAM writes.
- Dependencies: `CanonicalLookup` (fastembed + qdrant-client) — already allowed on ingestion path.
- No numpy / cupy / scipy / sympy. No Python fallbacks.

## 5. Out of scope

- `populate_math_symbols.py` Character-galaxy bootstrap — independent concern, leave alone.
- Cluster 2 / Cluster 3 drivers — Batches 9–10.
- KIMI reasoning-taxonomy truncation (AML_AND_SOLVERS, AUTOMATED_REASONING) — Batch 7.5 parallel track.

---

**Hand-off:** Codex implements §2.1–§2.4, runs the seed, reruns the audit (must exit 0), then runs the real Cluster 1 `--write`. Report back with audit output + Cluster 1 write counts.
