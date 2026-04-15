# Codex Phase 7.A.1 Slice 1 Report — System Font Glyph RPN

Date: 2026-04-11

## Scope

Implemented the system-font ingestion slice from
`TEMP/CLAUDE_PHASE7A1_CANONICAL_KNOWLEDGE_SEED_2026-04-11.md`.

The generated artifact lives in the local runtime workspace, not the NAS repo:

- `/K3D/Knowledge3D.local/assets/fonts/text/MANIFEST.json`

## Built

- Added `knowledge3d/ingestion/fonts/glyph_to_rpn.py`.
- Scans `/usr/share/fonts` for `.ttf`, `.otf`, and `.ttc` files with `fontTools`.
- Excludes symbol/pictogram fonts by the hardcoded Phase 7 list:
  Wingdings, Webdings, FontAwesome, PowerlineSymbols, StandardSymbolsPS,
  D050000L, Symbol, Symbola, POV-Ray Logo, stmary10, wasy10, rsfs10,
  Dingbats, and Emoji families.
- Preserves `STIX Two Math` as a text/math-outline font.
- Emits glyph outlines as existing Drawing Galaxy textual RPN:
  `MOVE`, `LINE`, `QUAD`, `CUBIC`, `CLOSE`, `STROKE`.
- Normalizes glyph coordinates and metrics to a 2048-unit em square.
- Provides canonical `font_glyph` helpers:
  `glyph_key`, `glyph_star_id`, `font_glyph_metadata`, `register_font_glyph`.
- Provides CLI:
  `python -m knowledge3d.ingestion.fonts.glyph_to_rpn --font-root /usr/share/fonts --out /K3D/Knowledge3D.local/assets/fonts/text/MANIFEST.json`

## Local Manifest

Full scan result:

- `font_count=3054`
- `excluded_count=19`
- `unreadable_count=0`
- Scripts found:
  `arab,beng,common,cyrl,deva,ethi,geor,grek,gujr,guru,hang,hani,hebr,hira,kana,latn,letter_other,math,mlym,number,taml,thai,tibt`
- Symbol-font violations in retained manifest: `0`
- `STIX Two Math` retained: `True`

## Tests

Focused command:

```bash
TMPDIR=/K3D/Knowledge3D.local/tmp PYTHONPATH="$(pwd)" \
  python3 -m pytest -q tests/test_phase7a1_glyph_to_rpn.py \
  --basetemp /K3D/Knowledge3D.local/tests/pytest_phase7a1_fonts
```

Result:

- `6 passed in 1.89s`

Regression command:

```bash
TMPDIR=/K3D/Knowledge3D.local/tmp PYTHONPATH="$(pwd)" \
  python3 -m pytest -q \
  tests/test_phase7_multilingual_symlinks.py \
  tests/test_multilingual_meanings.py \
  tests/test_math_zero_fix.py \
  tests/test_phase7a1_glyph_to_rpn.py \
  --basetemp /K3D/Knowledge3D.local/tests/pytest_phase7a1_regression
```

Result:

- `35 passed, 78 warnings in 163.69s`
- Warnings are existing Python deprecation warnings from `multiprocessing` and
  `ast.Num` in the math benchmark helper.
- `git diff --check` clean for Slice 1 touched files.

## Notes

- No Python was added to the cranium/runtime hot path.
- No new drawing opcode numbers were introduced.
- The repo stores source/tests only; generated font manifest data stays under
  `/K3D/Knowledge3D.local`.
