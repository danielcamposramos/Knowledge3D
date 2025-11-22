# Large Assets Kitchen

Large (>99 MB) or bulk-generated artifacts live outside the repo under `../Knowledge3D.local/`. Use these recipes to rebuild them when you need to run legacy viewers or regression suites.

## House & Viewer Exports
- Target location: `Knowledge3D.local/old_attempts/legacy_fancy_rag/viewer_public/`
- Recipe: run `python -m knowledge3d.tools.house_memory_builder --house-id <id> --output Knowledge3D.local/...` followed by `python -m knowledge3d.tools.publish_local_artifacts`.
- Reference docs: `docs/HOUSE_MEMORY.md`, `docs/TABLET_APPS.md`

## Benchmark Reports
- Target: `Knowledge3D.local/old_attempts/legacy_fancy_rag/docs_reports/`
- Recipe: rerun the evaluation scripts:
  - Math: `python -m knowledge3d.tools.phase25.math_bench_evaluator --auto`
  - ARC/HLE: `python -m knowledge3d.tools.phase23.arc_hle_tester --all`
  - RLWHF: `python -m knowledge3d.tools.phase25.consistency_trainer --replay latest`
- Save outputs to the `.local` path to keep Git clean.

## AI Compendium CSV
- Target: `Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv`
- Recipe: `python -m knowledge3d.tools.build_ai_compendium --vectors-out Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv --limit 80000`

## ARC-AGI Reasoning Cache
- Target: `Knowledge3D.local/datasets/arc_agi/arc_reasoning_pairs.npz`
- Recipe:
  1. `python scripts/train_trm_on_arc_reasoning.py --epochs 0 --rebuild-cache` (build cache only)
     - Optional: add `--limit-pairs <int>` for quick smoke tests.
  2. The script downloads the ARC-AGI dataset under `Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/`
     and stores metadata next to the cache (`arc_reasoning_pairs.json`).
- Notes: Keep the cache in sync with the latest RPN embeddings; rebuild after significant sleep-time consolidation so reasoning aligns with House/Galaxy updates.

## Galaxy GLBs (viewer/dist)
- Target: `Knowledge3D.local/old_attempts/legacy_fancy_rag/viewer_dist/`
- Recipe: `npm run build` inside `viewer/`, then copy the resulting GLBs to `.local`.
- Make sure `K3D_LOCAL_DIR` points to `.local` before building so runtime loads from the right place.

## Word Galaxy Sources (UD + Lexique)
- Targets:
  - UD treebanks archive: `/K3D/K3D_llama_cpp/datasets/ud/ud-treebanks-v2.14.tgz`
  - UD extracted: `/K3D/K3D_llama_cpp/datasets/ud/ud-treebanks-v2.14/`
  - Lexique382 TSV: `/K3D/K3D_llama_cpp/datasets/lexicons/Lexique382.tsv`
- Recipes:
  1. UD v2.14 download (official bitstream):
     ```
     mkdir -p /K3D/K3D_llama_cpp/datasets/ud
     cd /K3D/K3D_llama_cpp/datasets/ud
     wget -O ud-treebanks-v2.14.tgz https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/29f1dc4f-8055-4827-8bc0-904dfbfe2d51/content
     tar -xzf ud-treebanks-v2.14.tgz
     ```
  2. Lexique382 (French lexicon):
     ```
     mkdir -p /K3D/K3D_llama_cpp/datasets/lexicons
     cd /K3D/K3D_llama_cpp/datasets/lexicons
     wget -O Lexique382.tsv.gz http://www.lexique.org/databases/Lexique382/Lexique382.tsv.gz
     gunzip -kf Lexique382.tsv.gz
     ```
  3. Ingest to word stars (UD):
     ```
     CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/ingest_ud_word_stars.py \
       --ud-root /K3D/K3D_llama_cpp/datasets/ud/ud-treebanks-v2.14 \
       --out /K3D/Knowledge3D.local/datasets/word_stars_ud.jsonl

     python3 scripts/merge_word_stars.py \
       --inputs /K3D/Knowledge3D.local/datasets/word_stars_ud.jsonl \
       --output /K3D/Knowledge3D.local/datasets/word_stars_all.jsonl
     ```
  4. Upsert to Galaxy/House: implement bridge in `scripts/load_word_stars_into_galaxy.py` (placeholder).
  5. Meaning/procedural-first guardrails:
     - Words are sense-disambiguated (fruit vs company, etc.); identity by meaning.
     - Letters are referenced as letter-meaning nodes (case variants live inside the letter node); do not merge math symbols/operators.
     - Procedural programs stay primary (meaning_rpn/morph_rpn/phonetic); embeddings are secondary/regenerable.
     - Default loads: base + word meaning + math symbols + punctuation; letter galaxies load on-demand per script.

## Legacy Examples
- Target: `Knowledge3D.local/old_attempts/legacy_fancy_rag/examples/`
- Recipe: regenerate with `python -m k3dgen` per the original README instructions. Update manifests if you add or remove samples.

> Keep this folder up to date whenever you introduce a new large artifact. Describe **where** it should live in `.local` and **how** to rebuild it.
