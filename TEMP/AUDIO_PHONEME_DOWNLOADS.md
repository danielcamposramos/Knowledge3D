## Audio phoneme/letter downloads (current state)

Downloaded/placed under `/K3D/K3D_llama_cpp/datasets/audio/phoneme_external/`:
- **ISOLET**: isolet.zip unpacked (feature vectors only, no PCM). Manifest placeholder `isolet_manifest.csv` maps labels A–Z to rows (useful only if synthesizing audio later).
- **audio-cmn**: full Mandarin pinyin syllable repo cloned (real audio across 18k/24k/64k/96k folders).
- **Alphabet-with-sounds**: English letter-name clips (m4a) under `Alphabet-with-sounds/sounds/*.m4a`.
- **SUCSpeech**: cloned; Spanish voicepacks present (e.g., `Sucspeech/Voicepacks_source_pro/es_default_hq/*.wav`) with various syllables/letters.
- **UCLA phonetic corpus**: GitHub sample cloned (abk); full release link stale.
- **Lingua Libre downloader**: adjusted SPARQL still returns 0 entries; needs a better Commons/Lexeme query or manual pull.
- **Japanese kana**: pending (previous repo 404); need a valid kana source.

Notes:
- Repo downloader script (`scripts/download_letter_audio_repos.sh`) clones Alphabet-with-sounds and SUCSpeech; kana clone commented out due to 404.
- Lingua Libre fetch still empty; consider Commons scraping or corrected SPARQL for lexemes/forms.
