# Codex Phase 7.A.1 Slice 5 Report - Word Galaxy Population - 2026-04-11

## Shipped

- Added `knowledge3d/ingestion/universal_knowledge/dbnary_ingester.py`.
  - Parses one DBnary OntoLex TTL/TTL.BZ2 file at a time.
  - Normalizes lexical entries into `LexicalRecord`.
  - Merges records into OMW by `(language, lemma)`.
  - Creates canonical word stars only for lemmas OMW missed.

- Added `knowledge3d/ingestion/universal_knowledge/kaikki_ingester.py`.
  - Parses one Kaikki JSONL/JSONL.GZ file at a time.
  - Reuses the same `LexicalRecord` merge path as DBnary.
  - Carries glosses, IPA, and etymology into ingestion metadata.

- Added `scripts/run_phase7a1_unified_ingestion.py`.
  - Reads system-font letter stars in refs-only mode.
  - Reads math symbol stars.
  - Loads UD grammar artifacts from `/K3D/Knowledge3D.local/assets/grammar/`.
  - Builds OMW word/meaning stars, then merges DBnary and Kaikki.
  - Runs a final bidirectional symlink sweep.
  - Emits JSONL, canonical entries, and dangling-ref reports under `/K3D/Knowledge3D.local/assets/phase7a1/`.
  - Fails if dangling refs remain.

- Exported the new ingestion APIs from `knowledge3d/ingestion/universal_knowledge/__init__.py`.

- Added `tests/test_phase7a1_word_galaxy_population.py`.

## Validation

- `CUDA_VISIBLE_DEVICES=0 K3D_PYTEST_PROBE_CUDA=1 pytest -q tests/test_phase7a1_word_galaxy_population.py`
  - `6 passed`

- `CUDA_VISIBLE_DEVICES=0 K3D_PYTEST_PROBE_CUDA=1 pytest -q tests/test_phase7_multilingual_symlinks.py tests/test_math_symbol_symlinks.py tests/test_multilingual_meanings.py tests/test_math_zero_fix.py tests/test_phase7a1_glyph_to_rpn.py tests/test_phase7a1_letter_galaxy_builder.py tests/test_phase7a1_math_symbol_builder.py tests/test_phase7a1_ud_grammar_builder.py tests/test_phase7a1_word_galaxy_population.py`
  - `61 passed, 78 warnings`

- `python3 -m py_compile knowledge3d/ingestion/universal_knowledge/dbnary_ingester.py knowledge3d/ingestion/universal_knowledge/kaikki_ingester.py scripts/run_phase7a1_unified_ingestion.py`
  - passed

- `git diff --check`
  - passed

- Live source probe:
  - DBnary yielded English entries: `dictionary noun`, `dictionary verb`, `free adjective`
  - Kaikki yielded Portuguese entries: `thesaurus noun`, `frei noun`, `gratis adverb`

## Sovereignty

- New Slice 5 code is ingestion-path only.
- New Slice 5 files introduce no `numpy`, `cupy`, `scipy`, `sympy`, `rdflib`, `conllu`, or `pyconll` imports.
- Broad repo grep still reports historical `numpy`/`cupy` imports in `cranium/` and `knowledgeverse/`; those are pre-existing and outside this slice.

## Not Run

- Full no-cap unified ingestion over all OMW, DBnary, Kaikki, letters, math, and grammar was not run in this turn. The runner is implemented and writes bulk outputs to `/K3D/Knowledge3D.local/`, but the full pass should be launched as a dedicated long job because it will process large local datasets and fail on any remaining dangling refs by design.
