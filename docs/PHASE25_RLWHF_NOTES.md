# Phase 25 RLWHF Training Notes

## Teacher Prompt Update
- The RLWHF teacher now addresses the controller as **K3D Automatic RLWHF Training Program**.
- All references to “AI response” were renamed to “student response” so evaluation feedback focuses on the student being trained.
- The teacher always receives the original question and the expected answer alongside the student’s reply.

## Corpus Generation Hygiene
- All Phase 25 corpus builders skip sentences containing authorship, publishing, or copyright boilerplate (e.g., “ISBN”, “All rights reserved”, publisher names).
- This keeps drill material focused on conceptual content, avoiding trivia about publication metadata.

## Training Loop Adjustments
- `AlgorithmicThinkingTrainer` now supplies question + expected answer to both exaone models for each RLWHF check.
- Teacher evaluations therefore operate with full context, eliminating guesses when scoring.

## Next Steps
1. Rebuild the corpora with the updated filters (already done as part of this change).
2. Relaunch the PTX + RLWHF training loop and monitor `logs/phase25_thinking_train.log` once exaone is ready.

## Timeout Adjustments
- TeacherEvaluator default timeout increased to 5 minutes (initial) / 2.5 minutes (subsequent) so exaone-deep evaluations no longer time out mid-run.

## Sleep-Time Consolidation
- Algorithmic Thinking trainer now triggers  automatically after processing roughly two-thirds of the corpus (mimicking an 8h sleep window in a 24h cycle).
- Sleep writes materialised artifacts to  and logs adjustments under .

## Sleep-Time Consolidation
- Algorithmic Thinking trainer now triggers `SleepTimeCompute` automatically after processing roughly two-thirds of the active corpus (mirroring an 8h sleep window in a 24h cycle).
- Sleep writes materialised artifacts to `viewer/public/house/materialized_objects/` and logs adjustments under `logs/sleep_time_adjustments.json`.

## Lexicon Resources
- Downloaded English WordNet (2024 edition), OpenWordNet-PT, and CC-CEDICT into `/home/daniel/K3D_llama_cpp/datasets/lexicons/` and mirrored under `/K3D/Knowledge3D.local/datasets/lexicons/` for onboarding lexical meaning stars.

## Lexicon Builder Scripts
- New CLI utilities convert each lexicon into Galaxy-ready stars under `viewer/public/galaxy/working/`:
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_en --out viewer/public/galaxy/working/lexicon_en_wordnet.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_pt --out viewer/public/galaxy/working/lexicon_pt_openwordnet.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_es --out viewer/public/galaxy/working/lexicon_es_kaikki.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_zh --out viewer/public/galaxy/working/lexicon_zh_cedict.jsonl`
- Pass `--limit` while iterating to generate quick samples (e.g., `--limit 200`). Omitting the flag emits the full corpus.
- Each star stores lemma, POS, definition(s), synonyms, pronunciations, and relations so the Galaxy can stitch lexical concepts into Phase 25 reasoning drills.

## Pronunciation Audio Builder
- `env PYTHONPATH=. python -m knowledge3d.tools.pronunciation_audio_builder --metadata <manifest.tsv> --audio-root <clips_dir> --language en --source commonvoice --out viewer/public/galaxy/working/lexicon_audio_en_commonvoice.jsonl`
- Defaults expect Common Voice style TSVs (`path` + `sentence` columns). Override column names with `--path-field`, `--text-field`, and `--ipa-field` when manifests differ.
- Output stars fuse text and audio modalities to keep pronunciation drills close to the lexicon stars; trainer auto-detects them when present.

## Speech Dataset Acquisition
- `knowledge3d.tools.fetch_common_voice_subset` now wraps both Common Voice (once access is granted) and the open `PolyAI/minds14` spoken intent corpus.
  ```bash
  env PYTHONPATH=. python -m knowledge3d.tools.fetch_common_voice_subset \
    --dataset minds14 --language en-US \
    --out-dir /home/daniel/K3D_llama_cpp/datasets/audio \
    --manifest /home/daniel/K3D_llama_cpp/datasets/audio/en_us/minds14/manifest.csv \
    --mirror-root /K3D/Knowledge3D.local/datasets/audio \
    --count 150
  ```
- Repeat for `pt-PT`, `es-ES`, and `zh-CN` to seed multilingual speech clips; manifests land under `../datasets/audio/<lang>/<dataset>/manifest.csv` and are mirrored into `Knowledge3D.local` for builders.
- Common Voice remains the long-term target; once licensing access is approved, rerun the same command with `--dataset common_voice` to swap in the official corpora without changing downstream tooling.

## Trainer Integration
- `AlgorithmicThinkingTrainer` now ingests lexicon prompts automatically. When any `lexicon_*.jsonl` file exists in `viewer/public/galaxy/working/`, the trainer samples per-language definition, synonym, and IPA questions before spawning RLWHF loops.
- Lexicon prompts join the existing corpora in `generate_rpn_queries`, ensuring every session interleaves lexical mastery with algorithmic drills.
