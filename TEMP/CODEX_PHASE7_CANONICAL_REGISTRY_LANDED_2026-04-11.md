# CODEX Phase 7 Canonical Registry Landed — 2026-04-11

## Scope

This report covers Phase 7.0 plus the first Phase 7.A slice that landed before the canonical knowledge-seed work:

- canonical ID helpers
- file-backed Qdrant credential resolution
- bidirectional symlink helpers
- multilingual meaning-builder migration to canonical word/char IDs
- canonical Qdrant bootstrap
- vocabulary spec for the canonical registry

## What Shipped

- Added strict canonical registry overlay in `knowledge3d/ingestion/canonical_lookup.py`
- Added file-backed Qdrant credential resolver in `knowledge3d/ingestion/qdrant_credentials.py`
- Added bidirectional symlink helpers in `knowledge3d/ingestion/symlink_helpers.py`
- Added canonical Qdrant seed script in `scripts/ingest_canonical_to_qdrant.py`
- Migrated multilingual meaning ingestion to canonical word and character IDs
- Updated House/foundational surface helpers to consume canonical char IDs
- Wrote `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md`
- Registered the canonical spec in `docs/vocabulary/README.md`

## Canonical Registry State

Verified live:

- collection: `k3d_canonical`
- vector: `fast-all-minilm-l6-v2`
- entries: `54`

Verified strict lookups:

- `star_id / char::a -> char_a`
- `drawing_primitive / line -> drawing_primitive_line`
- `grammar_template / en:copula -> grammar_template_en_copula`

Credential path:

- env override: `QDRANT_API_KEY`
- local secret file: `/K3D/Knowledge3D.local/secrets/qdrant_api_key.txt`

No secret literal remains in tracked repo code.

## Green Validation

Focused Phase 7 batch:

- `tests/test_phase7_multilingual_symlinks.py`
- `tests/test_multilingual_meanings.py`
- `tests/test_math_zero_fix.py`

Result:

- `28 passed`

Additional checks:

- `python3 -m py_compile` clean for touched ingestion/spec scripts
- `git diff --check` clean
- canonical seed script completes and returns `k3d_canonical: 54 canonical entries`

## Behavioral Notes

- `CanonicalLookup` is fail-and-fix only. Missing canonical rows raise `KeyError("canonical_lookup_miss:{kind}:{key}")`.
- `canonical_entry_id()` now uses deterministic UUIDv5 because Qdrant rejects raw short hex IDs as point IDs.
- `canonical_slug()` preserves non-Latin content deterministically through `uXXXX` segments instead of collapsing to `unknown`.
- `symlink_helpers.append_ref()` now covers taxonomy, meta, grammar, reality, visual, audio, component, composite, and `surface_forms.{lang}.word_ref/char_refs`.

## Not Done Yet

This landing does **not** implement the Phase 7.A.1 content shelves:

- no font ingestion yet
- no `font_glyph` registration yet
- no letter galaxy builder yet
- no math symbol registry expansion yet
- no UD grammar ingestion yet
- no DBnary/Kaikki word-lemma pass yet

## Next

Next slice starts with Phase 7.A.1 Slice 1:

- choose/download text font set into `/K3D/Knowledge3D.local/assets/fonts/text/`
- write `MANIFEST.json`
- implement `knowledge3d/ingestion/fonts/glyph_to_rpn.py`
- register `font_glyph` canonical entries

The canonical librarian is in place. The next step is filling the shelves.
