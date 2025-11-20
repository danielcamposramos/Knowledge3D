## Open Audio Sources for Letters and Phonemes (Gemini Findings)

**Key distinction:** store both letter names (e.g., “A” → /eɪ/) and phonemes (contextual sounds: /æ/, /ɑː/, /ə/).

### Primary multi-language source
- **Lingua Libre (Wikimedia France):** isolated phonemes and letters; CC-BY-SA; query via SPARQL for items `instance of phoneme (Q708031)` or `letter (Q9779)` filtered by language. Best single open source.

### Language-specific sources
- **English (en):**
  - Letter names: ISOLET (UCI ML repo).
  - Phonemes: UCLA Phonetic Corpus (GitHub).
- **Portuguese (pt-br/pt):**
  - Phonemes: Brazilian Portuguese Phonemes (Kaggle) ~31 WAVs (a, b, ch, d, etc.).
  - Alphabet: Wikimedia Commons “Portuguese pronunciation” category for letter names.
- **Chinese (zh-cn):**
  - Pinyin syllables: audio-cmn (GitHub hugolpz/audio-cmn), syllabs/ folder ~1700 pinyin MP3s.
- **Japanese (ja):**
  - Kana syllables (hiragana/katakana): use Lingua Libre filtered for Kana classes; alternative GitHub “Japanese-Alphabet-Audio” style repos; Common Voice–derived kana sets on HF. (dar5in repo 404; need a valid kana source)
- **Spanish (es):**
  - Alphabet/phonemes: Wikimedia Commons Spanish alphabet (OGG split or individual files); CC Anki decks with media (check licenses).

### Strategy
1) Use Lingua Libre SPARQL to bulk-fetch phoneme/letter clips per language.
2) Supplement with language-specific sets above for coverage and quality.
3) Ingest both letter-name audio and phoneme audio; label accordingly in manifests for proceduralization.***
