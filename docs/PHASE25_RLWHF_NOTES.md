# Phase 25 RLWHF Training Notes

## Teacher Prompt Update
- The RLWHF teacher now addresses the controller as **K3D Automatic RLWHF Training Program**.
- All references to “AI response” were renamed to “student response” so evaluation feedback focuses on the student being trained.
- The teacher always receives the original question and the expected answer alongside the student’s reply.

## Corpus Generation Hygiene
- All Phase 25 corpus builders skip sentences containing authorship, publishing, or copyright boilerplate (e.g., “ISBN”, “All rights reserved”, publisher names).
- This keeps drill material focused on conceptual content, avoiding trivia about publication metadata.

## Training Loop Adjustments
- `AlgorithmicThinkingTrainer` verifies the CUDA PTX geometry head before each session; if the kernel cannot load, training aborts immediately with guidance.
- RLWHF scoring now routes through `exaone-deep:latest` (Ollama); if the teacher is unavailable, the fallback remains the house honesty evaluator. Standard telemetry still lands in `logs/phase25_pt_br_train.log` thanks to the stdout/stderr tee.

## Next Steps
1. Rebuild the corpora with the updated filters (already done as part of this change).
2. Relaunch the PTX + RLWHF training loop and monitor `logs/phase25_thinking_train.log` once exaone is ready.

## Timeout Adjustments
- TeacherEvaluator default timeout increased to 5 minutes (initial) / 2.5 minutes (subsequent) so exaone-deep evaluations no longer time out mid-run.

## RLWHF Prompt Refresh (2025‑09‑21)
- Rebuilt RLWHF drills with `exaone3.5` only to avoid exaone-deep thinking-tag injections.
- Artifact: `viewer/public/galaxy/working/rlwhf_exaone3p5.jsonl` (60 question/answer/feedback triples).
- Regeneration command:
  ```bash
  PYTHONPATH=. conda run -n k3d-cranium python3 -m knowledge3d.tools.gen_rlwhf_exaone \
    --gltf viewer/public/galaxy.v8.glb \
    --out viewer/public/galaxy/working/rlwhf_exaone3p5.jsonl \
    --n 60 --ollama http://127.0.0.1:11434 --model exaone3.5:latest
  ```

## Sleep-Time Consolidation
- Algorithmic Thinking trainer now schedules three `SleepTimeCompute` passes per corpus (≈33 %, 66 %, and final completion) so consolidation brackets the full drill window.
- Each pass appends adjustments to `logs/sleep_time_adjustments.json` and emits reflection diaries under `viewer/public/house/materialized_objects/reflection_diary_cycle_*.json`.
- The 2025‑09‑21 run produced cycles `_1758436355`, `_1758443093`, and `_1758450824`, confirming persistence on every pass.
- PTX fallback paths were removed; the geometry head must execute on the GPU, and any failure now surfaces before consolidation begins.

## Lexicon Resources
- Downloaded English WordNet (2024 edition), OpenWordNet-PT, and CC-CEDICT into `/home/daniel/K3D_llama_cpp/datasets/lexicons/` and mirrored under `/K3D/Knowledge3D.local/datasets/lexicons/` for onboarding lexical meaning stars.

## Lexicon Builder Scripts
- New CLI utilities convert each lexicon into Galaxy-ready stars under `viewer/public/galaxy/working/`:
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_en --out viewer/public/galaxy/working/lexicon_en_wordnet.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_pt --out viewer/public/galaxy/working/lexicon_pt_openwordnet.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_es --out viewer/public/galaxy/working/lexicon_es_kaikki.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_zh --out viewer/public/galaxy/working/lexicon_zh_cedict.jsonl`
  - `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_pt_br --out viewer/public/galaxy/working/lexicon_pt_br_kaikki.jsonl`
- `env PYTHONPATH=. python -m knowledge3d.tools.lexicon_builder_pt_grammar --out viewer/public/galaxy/working/lexicon_pt_pt_grammar.jsonl`
- Pass `--limit` while iterating to generate quick samples (e.g., `--limit 200`). Omitting the flag emits the full corpus.
- Each star stores lemma, POS, definition(s), synonyms, pronunciations, and relations so the Galaxy can stitch lexical concepts into Phase 25 reasoning drills.
  - The Kaikki Portuguese dump (`kaikki.org-dictionary-Portuguese.jsonl.gz`) sits under `/home/daniel/K3D_llama_cpp/datasets/lexicons/portuguese_br/`; it produces `viewer/public/galaxy/working/lexicon_pt_br_kaikki.jsonl` (~GBs). Keep this artifact out of git—regenerate on demand with the command above.
  - Grammar scaffolding for pt-PT is curated in-code (see `knowledge3d/tools/lexicon_builder_pt_grammar.py`) so the variant keeps its structural rules.
- PTX-first galaxy packaging:
  - `env PYTHONPATH=. python -m knowledge3d.tools.language_galaxy_builder \
      --input viewer/public/galaxy/working/lexicon_pt_br_kaikki.jsonl \
      --input viewer/public/galaxy/working/lexicon_audio_pt_br_librispeech9h.jsonl \
      --language-id pt-BR \
      --label "Portuguese (BR) Language Galaxy" \
      --out viewer/public/galaxy/language_pt_br.glb \
      --manifest viewer/public/galaxy/language_pt_br.json`
  - Repeat for EN/ES/ZH, swapping the language corpora and IDs. The builder emits binary GLBs with embedded bufferViews so `PTXGeometrySession` can load them directly.
  - Legacy JSON `.gltf` assets must be rebuilt via `env PYTHONPATH=. python -m knowledge3d.tools.rebuild_house_glb viewer/public/houses/<id>/memory_house.gltf viewer/public/houses/<id>/memory_house.glb` before SleepTimeCompute or trainer runs; CPU fallbacks have been removed.

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
    --count -1  # use -1 to pull the full split
  ```
- Repeat for `pt-PT`, `es-ES`, and `zh-CN` to seed multilingual speech clips; manifests land under `../datasets/audio/<lang>/<dataset>/manifest.csv` and are mirrored into `Knowledge3D.local` for builders.
- Common Voice remains the long-term target; once licensing access is approved, rerun the same command with `--dataset common_voice` to swap in the official corpora without changing downstream tooling.
- Current `minds14` pulls yield: EN 563 clips (≈1.34 h), PT 604 (≈2.69 h), ES 486 (≈1.53 h), ZH 502 (≈1.26 h); average utterance lengths stay below 17 seconds which keeps audio builder runtime fast.
- Brazilian Portuguese audio uses the 9-hour split of `facebook/multilingual_librispeech` (`env PYTHONPATH=. python -m knowledge3d.tools.fetch_common_voice_subset --dataset multilingual_librispeech --language portuguese --split 9_hours --count -1 ...`). The manifest lives at `/home/daniel/K3D_llama_cpp/datasets/audio/pt_br/multilingual_librispeech_9h/manifest.csv` and mirrors into `/K3D/Knowledge3D.local/datasets/audio/pt_br/multilingual_librispeech_9h/`.
- Build the pt-BR pronunciation stars with `env PYTHONPATH=. python -m knowledge3d.tools.pronunciation_audio_builder --metadata /K3D/Knowledge3D.local/datasets/audio/pt_br/multilingual_librispeech_9h/manifest.csv --audio-root /K3D/Knowledge3D.local/datasets/audio --language pt --source multilingual_librispeech_9h --out viewer/public/galaxy/working/lexicon_audio_pt_br_librispeech9h.jsonl`.
- All generated JSONL/WAV assets exceed the 99 MB git ceiling—do **not** commit them. Keep the commands above in this log for reproducibility.

## Trainer Integration
- `AlgorithmicThinkingTrainer` now ingests lexicon prompts automatically. When any `lexicon_*.jsonl` file exists in `viewer/public/galaxy/working/`, the trainer samples per-language definition, synonym, and IPA questions before spawning RLWHF loops.
- Lexicon prompts join the existing corpora in `generate_rpn_queries`, ensuring every session interleaves lexical mastery with algorithmic drills.
- Repetition in the current run is expected—the corpora are finite—but consolidation now succeeds, so new reflection diaries land in the House after every sleep cycle.

## AIME 2024 Baseline (2025‑09‑21)
- Harness: `PYTHONPATH=. conda run -n k3d-cranium python3 -m knowledge3d.tools.phase25.aime_evaluator [--limit N]`
- Current run (limit 1): `0 / 1` correct using the fused head only; see `docs/benchmarks/aime_2024_results.json` for the transcript. Expect a long runtime (~7 min/problem) when scaling to all 30 items.
- Observed failure mode: the fused head emits geometry tokens (e.g., `icosahedron`) instead of numeric answers; integrate RLWHF feedback or retrain the head so arithmetic outputs emerge without assistance. During training, a rotating queue of AIME prompts is now mixed into the RLWHF loop to expose the model to real exam statements.
- The fused head now calls PTX-native helpers (ModularRPNEngine + shape generator) before logits, so GPU computations can yield numerical or spatial answers when the prompt allows.

## Outstanding Issues (2025‑09‑21)
- **Fused Head Calibration:** The AIME baseline shows geometry-token answers; integrate RLWHF feedback or retrain the fused head before retrying the benchmark.
- **Teacher Monitoring:** Track early-session scores to ensure the `exaone-deep` teacher stays responsive and avoid repeated `-1.00` warmup penalties.
- **Corpus Diversity:** Continue the plan to ingest the balanced Wikipedia splits (EN/ES/PT_PT/ZH) so the RLWHF loop sees broader contexts.
- **Thinking Tags:** After each major RLWHF batch, rerun the Phase 10 thinking-tag trainer to surface reasoning labels in UI and logs.
- **AIME Throughput:** The evaluation harness currently takes ~7 minutes/problem; schedule longer runs or parallelise once additional compute is available.
