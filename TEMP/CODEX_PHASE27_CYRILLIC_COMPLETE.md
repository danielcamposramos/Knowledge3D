# Phase 2.7 Cyrillic - Procedural Dataset Upgrade (2025-11-19)

## What Changed
- Installed additional open-source Cyrillic-friendly fonts to broaden glyph diversity:
  ```bash
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    fonts-oldstandard fonts-sil-charis fonts-sil-charis-compact \
    fonts-sil-doulos fonts-sil-doulos-compact fonts-sil-gentiumplus \
    fonts-sil-gentiumplus-compact fonts-uralic
  ```
- Rewrote `scripts/harvest_cyrillic_simple.py` into a true procedural harvester:
  - Traces actual glyph outlines via `fontTools` + `extract_glyph_cached`.
  - Converts outlines into MOVE/LINE/QUAD RPN strings (`segments_to_rpn`).
  - Emits normalized font metadata (`font_family`, `font_weight`, `font_style`, etc).
  - Annotates each glyph with ISO 639-1 language coverage (`get_character_languages`).
  - Outputs to `/K3D/Knowledge3D.local/datasets/atomic/fonts_cyrillic_procedural.jsonl` and hard-links to the legacy `fonts_cyrillic_simple.jsonl` for compatibility.
- `scripts/test_atomic_formation_limited.py` now prefers the procedural Cyrillic dataset while falling back to the legacy file when needed.

## New Dataset Snapshot
- Command:
  ```bash
  env PYTHONPATH=. python3 scripts/harvest_cyrillic_simple.py --scripts all --max-fonts 200
  ```
- Result (multi-script): 30 JSONL datasets written under `/K3D/Knowledge3D.local/datasets/atomic/`:
  - Arabic (134,580 glyphs / 768 chars / 184 fonts)
  - Bengali (17,632 glyphs / 96 chars / 184 fonts)
  - Braille (42,483 glyphs / 256 chars / 166 fonts)
  - Canadian Aboriginal syllabics (90,601 glyphs / 512 chars / 186 fonts)
  - CJK sample (187,647 glyphs / 1,024 chars / 184 fonts)
  - Cyrillic (47,823 glyphs / 304 chars / 190 fonts) mirrored to the legacy simple file
  - ... plus Greek, Hebrew, Indic (Devanagari, Gujarati, Gurmukhi, etc.), Hangul, Hiragana, Katakana, Lao, Latin (basic + extended), Malayalam, Myanmar, Oriya, Sinhala, Tamil, Telugu, Thai, Tibetan.
- Every entry now carries `script` and `script_label` metadata so downstream tooling can filter or merge per script.
- Each JSONL row now includes:
  ```json
  {
    "char": "Ђ",
    "visual_rpn": "... MOVE ... LINE ... STROKE",
    "font_family": "Doulos SIL Compact",
    "font_style": "normal",
    "font_weight": 400,
    "font_variant": "regular",
    "font_path": "/usr/share/fonts/truetype/doulos/DoulosSILCompact-R.ttf",
    "languages": ["bs", "sr"],
    "segments": 438,
    "unicode_codepoint": "U+0402",
    "category": "cyrillic",
    "type": "cyrillic_glyph"
  }
  ```

## Validation & Reproduction
- Limited atomic formation run (Latin + Math + new Cyrillic):
  ```bash
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/test_atomic_formation_limited.py
  ```
  - Confirms procedural Cyrillic glyphs are ingested (Cyrillic samples show SIL fonts + correct language aggregation).
- Regression suites:
  ```bash
  env PYTHONPATH=. pytest tests/test_character_languages.py -v
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/test_rpn_sovereignty_phase2.py -v -s
  ```
  - Character-language mapping sanity preserved.
  - Phase 2 sovereignty checks still pass (RPN vs CPU parity, ternary gate, validate-and-commit).

## Extra Font Coverage
- Debian font set expanded with Indic, Southeast Asian, CJK, Tibetan, Georgian, etc. packages:
  `fonts-lohit-*`, `fonts-samyak-*`, `fonts-khmeros`, `fonts-lao`, `fonts-sil-padauk`, `fonts-sil-abyssinica`, `fonts-lklug-sinhala`, `fonts-arphic-ukai/uming`, `fonts-ipafont-*`, `fonts-hosny-amiri`, `fonts-tibetan-machine`, `fonts-bpg-georgian`, `fonts-lg-aboriginal`, and more (61 packages total).
- Additional open fonts fetched outside Debian and stored under `/K3D/Knowledge3D.local/fonts/external/` via `scripts/download_open_fonts.py`:
  - Atkinson Hyperlegible (OFL)
  - ADLaM Display (covers Adlam script)
  - Noto Sans Cherokee (variable weight)
  - Noto Sans Canadian Aboriginal (variable weight)
- `scripts/harvest_cyrillic_simple.py` now automatically scans `/K3D/Knowledge3D.local/fonts` so these families are included during harvesting.

## Character-Language Mapping Upgrades
- `get_character_languages` now returns ISO 639-1 language coverage (or script-specific tags where ISO codes do not exist, e.g., `chr`) for:
  Greek, Hebrew, Arabic (extended), Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala, Thai, Lao, Tibetan, Myanmar, Georgian, Armenian, Ethiopic, Cherokee, Canadian syllabics, Braille, Bopomofo, Hiragana, Katakana, Hangul, and the sampled CJK blocks.
- Stats now expose counts for each new script plus a `total_scripts_tracked` field so downstream analytics know how many writing systems are active.

## Follow-ups / Notes
- Script still supports legacy workflows by mirroring the new dataset to `fonts_cyrillic_simple.jsonl`; future phases can retire the mirror when all call sites migrate.
- Additional CLI knobs (`--charset-file`, `--chars`, `--no-supplement`) make it easy to harvest smaller or extended sets (e.g., Cyrillic Extended-B) without editing code.
- Warnings such as “'created' timestamp seems very low” originate from certain SIL fonts and are harmless (fontTools informational messages).
- Next candidates: wire the procedural dataset into `scripts/train_atomic_procedural_full.py` and expand to other scripts (Arabic, CJK) using the same harvesting pattern.
