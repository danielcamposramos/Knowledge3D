# Codex Multi-Curriculum Inventory (Week 21.5 Prep)

Date: 2026-02-10
Scope: Current unified world + available datasets for immediate ingestion planning.

## 1) Unified world status (validated)
- Source files:
  - `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl` (last entry)
  - `../Knowledge3D.local/benchmarks/run_all_benchmarks_history.jsonl` (last entry)
  - `../Knowledge3D.local/results/week21_4_unified_full100/week14_benchmark_summary.json`
- Persistence mode: `unified`
- Shared instance: `true`
- Same instance ID for empty/enriched: `140590496417920`
- Lazy embedding mode: `skip`
- Runtime seed: `false`
- Runtime: `578.418s` (~9.64 min)
- World growth during run: `21823 -> 23311` (`+1488` entries)

## 2) Current galaxy population (enriched world)
Counted from `../Knowledge3D.local/galaxies_enriched/galaxies/*.jsonl`:
- `Grammar.jsonl`: 17,413
- `Drawing.jsonl`: 2,867
- `Reality.jsonl`: 1,584
- `Math.jsonl`: 507
- `3DObjects.jsonl`: 434

Not present as persisted populated files in this storage root:
- `Word.jsonl` (missing)
- `Character.jsonl` (missing)
- `Audio.jsonl` (missing)

Note: `Knowledgeverse.DEFAULT_GALAXIES` includes Drawing/Character/Word/Grammar/Math/Reality/Audio/3DObjects (`knowledge3d/knowledgeverse/knowledgeverse.py`).

## 3) Dataset/source inventory (ready-to-ingest)
Scanned under:
- `../Knowledge3D.local/datasets`
- `../Knowledge3D.local/foundation_curriculum_world_21_1`

### 3.1 High-level assets
- PDF/EPUB/DJVU found: 4 (all TheoremQA solution PDFs)
- Corpus-like files found: 2 (DROP train/dev JSON)
- Math-like files found: 44
- Physics/Science-like files found: 24

### 3.2 Phase1B curated knowledge exists
Path: `../Knowledge3D.local/datasets/knowledge_prep_phase1b`
- Enrichment JSON files exist for:
  - `algorithmic_thinking`, `math_foundations`, `logic_reasoning`, `cs_fundamentals`,
  - `competition_math`, `undergraduate_math`, `geometry_theorems`, `arc_agi_training`,
  - `classical_mechanics`, `grammar_rules`, `problem_solving_strategies`
- Corresponding markdown foundations also exist (substantial sizes, e.g., geometry_theorems.md ~123KB).

### 3.3 Global benchmark datasets available (local)
Path: `../Knowledge3D.local/datasets/global_benchmarks`
- Present with files: `alphageometry`, `bbh`, `big_bench`, `drop`, `gpqa`, `gsm8k`, `hellaswag`, `humaneval`, `mmlu`, `theoremqa`, `truthfulqa`
- Gaps:
  - `piqa` directory exists but appears empty (`files=0`)
  - `math` dataset dir exists but no `repo` subdir (current checker: `exists=False` for `.../math/repo`)

## 4) Implications for next step (before more ARC oracle tuning)
1. Unified world architecture is now correct and stable.
2. Cross-curriculum content is partially available, but ingestion is skewed heavily to Grammar/Drawing.
3. Word/Character/Audio are currently under-populated (effectively absent in persisted enriched files), limiting true all-galaxy synergy.
4. There is enough curated local content to run a structured ingestion pass without waiting on new external downloads.

## 5) Recommended execution order (docs/vocabulary-aligned)
1. **Reinforce missing default galaxies first** (Word, Character, Audio) using existing ingestion scripts and local datasets.
2. **Run Phase1B ingestion refresh** against current unified world (do not fork worlds):
   - `scripts/execute_knowledge_prep_phase1b.py`
   - `scripts/knowledge_prep_ingest.py`
3. **Pin ingestion outputs to K3D standard only** (RPN/Galaxy entries), preserving benchmark scripts as translation-only boundary.
4. **Run SleepTime consolidation** after ingestion before next ARC100.
5. **Then run ARC oracle calibration sweep** (strict/medium/relaxed), now with fuller universe coverage.

## 6) Questions for Claude (proposed)
1. Do we treat `Word/Character/Audio` as mandatory gate before Week 21.6 oracle tuning?
2. Should we prioritize `knowledge_prep_phase1b/*` over large `global_benchmarks/*` ingestion for signal quality vs noise?
3. For missing dataset payloads (`piqa`, `math/repo`), do we patch downloader first or proceed with available corpora?
4. Should SleepTime run once after ingestion or between each strictness sweep run?
5. Do we enforce a minimum cross-galaxy participation metric (e.g., nonzero hits from >=5 default galaxies) before accepting ARC metrics?

