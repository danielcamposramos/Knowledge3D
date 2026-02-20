# Ingestion Pipeline Reorg (Meaning-First, Procedural-First)

Status: working notes for redoing ingestion and upsert wiring.

## Operator Manuals
- `docs/ingestion/CURRENT_STACK_COMMAND_MANUAL.md`
- `docs/ingestion/CURRENT_STACK_COMMANDS_QUICKREF.md`

## Current Assets (local paths)
- **Fonts/letters (procedural)**: `/K3D/Knowledge3D.local/datasets/atomic/fonts_*_procedural.jsonl` (~26 GB) — per script (Latin, Cyrillic, Arabic, CJK, Indic/SE Asian, Hangul/Kana, Braille). Multi-glyph procedural visual_rpn + font metadata; no packing applied.
- **Words (UD-derived)**: `/K3D/Knowledge3D.local/datasets/word_stars_all.jsonl` (merged UD v2.14) — meaning_id/sense now explicit (`scripts/ingest_ud_word_stars.py`), `meaning_program`, `morph_rpn`. Lexique/DBnary/OMW staged but not merged.
- **Audio stars (phoneme seeds)**: `/K3D/Knowledge3D.local/datasets/audio_stars_all.jsonl` (en/es/pt-br/ja/zh) — procedural harmonic/envelope payloads, loader stub `scripts/load_audio_stars_into_galaxy.py`.
- **Math symbols**: procedural glyphs/RPN exist (kernels/codecs) but no dedicated ingestion builder yet.

## Gaps / Redo List
1) **Letter Meaning Galaxies (per script)**  
   - Build meaning-first stars: one per letter meaning per script (Latin A/a grouped; Cyrillic А separate), with glyph variants (upper/lower/italic/bold), compositional rules (case selection, kerning, baseline), procedural programs primary (visual_rpn, audio if available, meaning_rpn), Matryoshka embeddings secondary/regenerable.  
   - Outputs: `/K3D/Knowledge3D.local/galaxy/language_<script>.glb`

2) **Sublexical Galaxies (syllables/morphemes)**  
   - Syllables: pattern metadata (CV/CVC...), letter_refs composition, optional phonetic_rpn; identity by (lang, syllable, pattern). Segmenter implemented (heuristic CV split) for pt/es/en.  
   - Morphemes: prefix/suffix/root/inflection; letter_refs composition; morph_rpn + brief meaning_rpn; identity by meaning/function. Greedy longest-match segmenter implemented for pt/es/en using curated affix lists.  
   - Outputs: `/K3D/Knowledge3D.local/galaxy/syllables_<lang>.glb`, `/K3D/Knowledge3D.local/galaxy/morphemes_<lang>.glb`.

3) **Word Meaning Galaxy (sense-disambiguated)**  
   - Use `sense`/`meaning_id` (now emitted by UD ingest) as dedup key.  
   - Hierarchical refs: prefer morpheme_refs or syllable_refs; letter_refs only for leftover positions to reduce edge crossing.  
   - Procedural programs primary: meaning_rpn, morph_rpn, phonetic_rpn, syntactic hints; embeddings secondary.  
   - Output: `/K3D/Knowledge3D.local/galaxy/meaning_words.glb` (default-loaded).

4) **Phrase Meaning Galaxies (curated + user)**  
   - Curated idioms/phrases: phrase_id, word_refs, meaning_rpn, usage/register metadata; identity by meaning.  
   - User phrase galaxy: ships empty, runtime-writable; same schema; separates user additions from curated core.  
   - Outputs: `/K3D/Knowledge3D.local/galaxy/phrase_meanings.glb`, `/K3D/Knowledge3D.local/galaxy/user_phrase_meanings.glb`.

5) **Math Symbol Galaxy (separate)**  
   - One star per operator/constant; fields: visual_rpn variants (size/font), math_rpn (execution/stack effect), optional audio_rpn (verbalization); NO case variants, NO word-composition rules.  
   - Output: `/K3D/Knowledge3D.local/galaxy/math_symbols.glb` (default-loaded).

6) **Audio attachment**  
   - Attach phoneme/pronunciation programs to letter meaning stars (not math).  
   - Keep procedural payloads intact; embeddings derived if needed.

7) **Galaxy/House upsert bridges**  
   - Implement actual upsert for letters/words/audio/math into ProceduralGalaxy/House GLBs with `extras.k3d` procedural-first schema.  
   - Current bridge writes ProceduralGalaxy + minimal `.gltf` snapshots (extras only) and uses ProceduralCompiler when raw embeddings are present; swap to ctypes/GLB pipeline when available.

8) **On-demand loading policy**  
   - Tablet/Galaxy loader to default-load: base (text/visual/audio/reasoning) + word meaning + math symbols + punctuation.  
   - On-demand: letter meaning galaxies by detected script; unload when idle to stay ~200 MB VRAM.

## Immediate Code Tasks (high priority)
- [x] Add meaning_id/sense fields and meaning-first dedup in word ingest/merge/load (done).  
- [x] Add math symbol galaxy builder skeleton.  
- [x] Add letter meaning galaxy builder skeleton (procedural-first, variant grouping).  
- [x] Add word meaning galaxy builder to create GLB with letter_refs and procedural programs.  
- [x] Add sublexical builders (syllables, morphemes) and hierarchical linking in words.  
- [x] Implement upsert bridge (pending swap to GLB/ctypes).  
- [ ] Update tablet/loader runtime to honor default-load + on-demand script galaxies (incl. phrase/user phrase, sublexical).
- [x] Add auto-segmentation script (syllables/morphemes) from word stars (heuristic).
- [x] Export minimal glTF snapshots with `extras.k3d` (procedural-first); upgrade to full GLB+ctypes later.
- [x] Add segmenters (syllables/morphemes) for pt/es/en/fr/it/de (heuristic CV + affix tables).
- [ ] Extend segmentation dictionaries/rules to additional languages (CJK, Arabic, etc.) as needed.
- [x] Add drawing grammar builder (primitives→strokes→shapes→scenes→collections) with JSONL/GLB export.
- [ ] Optional richer syllabification via pyphen (if installed) now supported in segmenter; consider adding to env when allowed.

## Data Rebuild Order (recommended)
1. Rebuild letter meaning galaxies per script from `fonts_*_procedural.jsonl`.  
2. Build sublexical galaxies (syllables, morphemes) per language.  
3. Rebuild math symbol galaxy (procedural glyphs + math_rpn).  
4. Rebuild word meaning galaxy using updated `word_stars_all.jsonl` (sense-aware) and linking to morphemes/syllables (letters only for leftovers).  
5. Attach audio stars to letter galaxies.  
6. Run validation: compositional rendering (letters→syllables/morphemes→words), math execution, tablet load policy smoke tests.

## Notes
- Procedural-first: store visual_rpn/audio_rpn/math/meaning_rpn as canonical; embeddings are regenerable Matryoshka tiers for search/LOD.  
- Meaning-first: same meaning → one star with many glyph variants; different meanings (even with similar glyphs) → separate stars/galaxies.  
- Math symbols/operators stay in their own galaxy; never merged with letter/word galaxies.  
- Case variants are compositional partners inside letter stars; word construction selects variants per position/context (sentence start, proper noun, continuation, ASCII art roles).  
- Sublexical hierarchy: words → morphemes/syllables → letters; only link letters directly when not already covered by sublexical refs to reduce edge crossings.
