# Codex Phase 7.A.1 Slice 4 Report — UD Grammar Galaxy

Date: 2026-04-11

## Scope

Implemented Slice 4 from
`TEMP/CLAUDE_PHASE7A1_CANONICAL_KNOWLEDGE_SEED_2026-04-11.md`.

Generated artifacts live in the local runtime workspace:

- `/K3D/Knowledge3D.local/assets/grammar/UD_GRAMMAR_STARS.jsonl`
- `/K3D/Knowledge3D.local/assets/grammar/UD_GRAMMAR_STARS.jsonl.canonical.json`
- `/K3D/Knowledge3D.local/assets/grammar/UD_GRAMMAR_STARS.jsonl.meta.json`

## Built

- Added `knowledge3d/ingestion/grammar/ud_grammar_builder.py`.
- Added `knowledge3d/ingestion/grammar/__init__.py`.
- Parser is stdlib-only CoNLL-U, no runtime/hot-path imports.
- Selected the nine Phase 7 treebanks:
  `en:EWT`, `pt:Bosque`, `es:GSD`, `fr:GSD`, `de:GSD`, `it:ISDT`,
  `ja:GSD`, `zh:GSD`, `ru:SynTagRus`.
- Emits grammar templates:
  `copula`, `periphrastic_explanation`, and observed core constructions
  (`nsubj_obj`, `det_noun`, `amod_noun`, `case_oblique`, `aux_verb`).
- Emits grammar rules for high-frequency UPOS classes and observed word order.
- Registers canonical `grammar_template` and `grammar_rule` entries.
- Updated the canonical seed script to include 9-language `copula` and
  `periphrastic_explanation` templates.

## Local Build

Command:

```bash
TMPDIR=/K3D/Knowledge3D.local/tmp PYTHONPATH="$(pwd)" \
  python3 -m knowledge3d.ingestion.grammar.ud_grammar_builder \
  --ud-root /K3D/K3D_llama_cpp/datasets/ud/ud-treebanks-v2.14 \
  --out /K3D/Knowledge3D.local/assets/grammar/UD_GRAMMAR_STARS.jsonl
```

Result:

- `grammar_stars=198`
- `canonical_entries=198`
- Languages: `de,en,es,fr,it,ja,pt,ru,zh`

Treebank stats:

- `de`: `13814` sentences, `263791` tokens
- `en`: `12544` sentences, `204578` tokens
- `es`: `14187` sentences, `382435` tokens
- `fr`: `14450` sentences, `354584` tokens
- `it`: `13121` sentences, `276014` tokens
- `ja`: `7050` sentences, `168333` tokens
- `pt`: `7018` sentences, `171776` tokens
- `ru`: `69630` sentences, `1204640` tokens
- `zh`: `3997` sentences, `98616` tokens

Canonical registration:

- Registered `198` grammar entries in `k3d_canonical`.
- Verified lookup for:
  `en:periphrastic_explanation`,
  `pt:periphrastic_explanation`,
  `ja:copula`,
  `ru:word_order_svo`.

## Tests

Command:

```bash
TMPDIR=/K3D/Knowledge3D.local/tmp PYTHONPATH="$(pwd)" \
  python3 -m pytest -q \
  tests/test_phase7a1_glyph_to_rpn.py \
  tests/test_phase7a1_letter_galaxy_builder.py \
  tests/test_phase7a1_math_symbol_builder.py \
  tests/test_phase7a1_ud_grammar_builder.py \
  tests/test_phase7_multilingual_symlinks.py \
  tests/test_multilingual_meanings.py \
  tests/test_math_zero_fix.py \
  --basetemp /K3D/Knowledge3D.local/tests/pytest_phase7a1_slices1234
```

Result:

- `53 passed, 78 warnings in 155.78s`
- Warnings are existing Python deprecation warnings from `multiprocessing` and
  `ast.Num`.

## Notes

- No new RPN opcodes were introduced.
- No Python was added to the runtime hot path.
- The `saudades` path is represented by per-language
  `periphrastic_explanation` templates; synthesis wiring remains a later phase.
