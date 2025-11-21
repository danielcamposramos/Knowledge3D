# Word Galaxy — Atomic Plan (Multilingual, Procedural, Grammar-Aware)

**Goal**: Build “word galaxies” as composed stars (words/lexemes) that link to character stars (letters/glyphs) and carry grammar + language metadata. Default-load alongside the character galaxy. No raw pixels or opaque weights; weights = logic, House/Galaxy = data.

## Construction Targets
1) **Composed Stars (Words/Sequences)**
   - Each word star links to its character stars (symbolic-link style) and stores:
     - Surface form(s) per language
     - Grammar: POS (noun/verb/adj/etc.), gender, number, tense/aspect/mood, person, case, etc. (language-specific)
     - Pronunciations: IPA + optional audio (procedural seeds already available for letters/phonemes)
     - Meanings: synsets/definitions (anchors to concept/emoji galleries)
     - Morphology: lemmas, inflections, clitics, compounds; per-language patterns
   - **Procedural meaning and grammar**:
     - `meaning_program`: PD-compressed embedding of a gloss/definition (Text→RPN embedding → PD04), giving a procedural semantic program.
     - `morph_rpn`: tiny RPN encoding POS/morph features, e.g., `POS_NOUN GENDER_NEUTER NUMBER_SING PACK_GRAMMAR`.
   - Procedural embeddings: word-level embeddings compressed via Matryoshka/PD; no raw text blobs.

2) **Default-Load Galaxy**
   - Load word galaxy by language partitions (index by ISO 639-1/BCP47).
   - Semantic clusters (synsets, hypernyms, etc.) for fast query; leverage existing GalaxyResonanceEngine.

3) **Ingestion Sources (Raw → /K3D/K3D_llama_cpp/datasets/)**
   - Multilingual dictionaries/WordNets (definitions, synsets, POS):
     - Princeton WordNet (en) + Open Multilingual WordNet (OMW)
     - Wiktionary/DBnary dump (multilingual, grammar, pronunciations)
     - BabelNet (if licensing permits; otherwise skip)
   - Morphology/grammar annotated corpora:
     - Universal Dependencies (UD) treebanks (POS, morphology) — multilingual
     - Wikidata lexemes export (forms, senses) — multilingual
   - Pronunciation/IPA mapping:
     - Wiktionary/DBnary (IPA fields)
     - CMUdict (en); Lexique (fr); JPron (ja); CC-CEDICT (zh pinyin)
   - Script-specific/underrepresented languages: use UD + DBnary/Wikidata lexemes for coverage (Arabic, Indic, CJK, Cyrillic, etc.)

4) **Transform Spec (Raw → Stars)**
   - Parse dictionaries → word entries with POS, lemma, senses, translations, IPA.
   - Build composed-word stars:
     - `characters`: list of character star IDs
     - `pos/morph`: POS tags + language-specific features (UD-compatible)
     - `language`: ISO 639-1/BCP47
     - `pronunciations`: IPA + optional audio link (if available)
     - `meanings`: synset IDs/definitions
     - `embedding_proc`: procedural embedding (Matryoshka/PD)
   - Store as JSONL; compress embeddings to PD04; link to character galaxy and concept/emoji galaxies.

5) **Sovereign Processing**
   - All heavy lifting on GPU where applicable (compression, embedding ops).
   - No NumPy-only hot path; rely on existing PTX codecs and compression kernels.
   - No raw PDFs/images; text ingested as structured entries.

## Implementation Steps
1) **Data fetch (to /K3D/K3D_llama_cpp/datasets/)** — pending approval due to size:
   - DBnary/Wiktionary dumps (multilingual)
   - Open Multilingual WordNet (OMW)
   - UD treebanks (all languages)
   - CMUdict (en), CC-CEDICT (zh), JPron or IPA sources (ja), Lexique (fr), etc.
2) **Parsers/Normalizers**
   - Build readers for DBnary/Wiktionary, WordNet, UD; extract POS/morph/IPA; map to ISO codes.
   - Construct word stars linking characters and meanings.
3) **Embeddings/Compression**
   - Generate procedural embeddings (existing text embedding → PD compressor).
   - Store compressed programs + metadata; attach IPA/pronunciations where available.
4) **Galaxy Load**
   - Default-load by language partitions; enable cross-lingual links via synsets/translations.
5) **Validation**
   - Check coverage: #words per language, POS distribution, morph coverage.
   - Spot-check link integrity (characters present; IPA present where expected).

## Notes
- Word stars are compositional: they reference character stars and concepts/emoji; they do not duplicate glyph data.
- Grammar-rich languages (Arabic, Finnish, Hindi, etc.) rely on UD + lexeme sources for morphology.
- Licensing: ensure OMW/DBnary/UD allow local use; skip BabelNet unless licensed.

## Hierarchy & Emergence (Atoms → Molecules → Phrases)
- **Atomic layer**: character stars (letters/emoji/math/tactile) with procedural glyphs, phoneme/name audio, math meanings.
- **Molecular layer**: word stars as compositions of character stars, enriched with procedural meaning_program + morph_rpn; grammar-aware.
- **Phrase/sentence layer**: composed from word stars; can carry a higher-level meaning_program (compressed sentence embedding) and structural RPN (e.g., dependencies/constituency encoded as small programs). This preserves the dual-client contract: humans see words/phrases; RPN mind executes procedural programs and composition links.
