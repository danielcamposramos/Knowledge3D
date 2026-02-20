# Codex Week 22 External Dataset Ingest Handoff

**Date:** 2026-02-11  
**Scope:** Codex execution track from shared plan (AMC-AIME extraction + Audio/3D/Lexicon ingestion)  
**Status:** Complete and validated

## 1) What was implemented

### A. Math competition dataset conversion
- Added script: `scripts/prepare_math_competitions_dataset.py`
- Converts raw AMC-AIME JSONL into benchmark-ready files expected by `benchmarks/math_competitions.py`:
  - `amc_problems.json`
  - `aime_problems.json`
  - `imo_problems.json`
- Output root used:
  - `../Knowledge3D.local/datasets/math_competitions`

### B. External modality payload builders
- Added script: `scripts/prepare_external_multicurriculum_payload.py`
- Supports modality-specific payload creation:
  - `--modality lexicon` (WordNet + DBnary)
  - `--modality audio` (manifest-based phoneme entries)
  - `--modality geometry3d` (galaxy_geometry text-derived 3D procedural patterns)
- Output format: JSONL lines with `{ "galaxy": ..., "entry": ... }`

### C. Single-world ingestion apply
- Added script: `scripts/ingest_payloads_single_world.py`
- Enforces one persistent `Knowledgeverse` instance and eager loading of all default galaxies.
- Deduplicates by `entry.id` per target galaxy.
- Produces ingestion report JSON with before/after counts.

### D. Parallel orchestration
- Added script: `scripts/run_parallel_external_ingestion.sh`
- Runs payload preparation in parallel (lexicon/audio/3D), then ingests all payloads into one world.
- Uses `CONDA_BIN` (default `/mnt/anaconda3/bin/conda`) for this Debian machine.

## 2) Commands executed

### A. Math conversion
```bash
/mnt/anaconda3/bin/conda run -n k3d-cranium env PYTHONPATH=. \
  python scripts/prepare_math_competitions_dataset.py \
  --output-dir ../Knowledge3D.local/datasets/math_competitions
```

### B. Parallel payload build + single-world ingest
```bash
bash scripts/run_parallel_external_ingestion.sh \
  ../Knowledge3D.local \
  ../Knowledge3D.local/datasets/external_payloads
```

## 3) Results

### A. Math dataset output
- `AMC=1442`
- `AIME=30`
- `IMO=0` (no IMO source found in current external dataset root)

Artifacts:
- `../Knowledge3D.local/datasets/math_competitions/amc_problems.json`
- `../Knowledge3D.local/datasets/math_competitions/aime_problems.json`
- `../Knowledge3D.local/datasets/math_competitions/imo_problems.json`
- `../Knowledge3D.local/datasets/math_competitions/prepare_math_competitions_report.json`

### B. Parallel payload build output
Timestamped payload directory:
- `../Knowledge3D.local/datasets/external_payloads/20260210_233442`

Payload sizes:
- `lexicon_payload.jsonl`: `9513` rows
- `audio_payload.jsonl`: `2889` rows
- `geometry3d_payload.jsonl`: `117` rows

### C. Ingestion report
Report:
- `../Knowledge3D.local/datasets/external_payloads/20260210_233442/ingestion_report.json`

Totals:
- `added=12519`
- `skipped=0`

Galaxy counts before -> after:
- Drawing: `685 -> 685`
- Character: `0 -> 0`
- Word: `2 -> 9515`
- Grammar: `869 -> 869`
- Math: `1060 -> 1060`
- Reality: `1914 -> 1914`
- Audio: `351 -> 3240`
- 3DObjects: `367 -> 484`

## 4) Single-world status (important)

This did **not** run on a clean world.

Evidence:
- Non-zero pre-ingestion counts in report (e.g., Reality `1914`, Math `1060`, Audio `351`).
- Ingestion was additive into existing world state, aligned with persistent single-world evolution.

## 5) Notes for architecture alignment

- All default galaxies were loaded by `Knowledgeverse(..., eager_load_default_galaxies=True)` in ingestion apply.
- Ingestion used one instance (`shared_instance=true`) for the apply step.
- Procedural form-to-meaning is preserved where relevant via metadata symlinks in generated entries.

## 6) Questions / suggestions for Claude

1. **Character dependency:** Character is still `0` in this world while Word jumped to `9515`. Should Week 22.2 force Character bootstrap first (strict dependency gate), then re-link Word entries (`char_refs`) for stronger form-to-meaning grounding?
2. **IMO source policy:** Do we keep `imo_problems.json` empty until a verified IMO source is mounted, or should we allow optional synthetic/derived IMO placeholders with strict `synthetic=true` tagging?
3. **Lexicon routing priority:** For ARC/math improvements, should Word entries from DBnary/WordNet be routed as low-priority priors unless cross-galaxy evidence (Drawing/Math/Grammar) is present?
4. **3D ingest source quality:** Current `galaxy_geometry/3d/core_solids` is empty. Should we treat text-derived geometry entries as temporary and schedule a hard swap to mesh-native ingestion once real 3D artifacts are available?
5. **Coverage telemetry integration:** Should we add query participation telemetry for these newly ingested Word/Audio/3D entries immediately in the ARC adapter, so we can validate accessibility vs quantity in the next run?

## 7) Immediate next run recommendation

Validate that the benchmark is now consuming the expanded AMC/AIME set:
```bash
/mnt/anaconda3/bin/conda run -n k3d-cranium env PYTHONPATH=. \
  python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 --max-math-problems 100 --max-lhe-questions 50 \
  --arc-enable-full-ptx --arc-enable-contrastive-learning \
  --arc-embedding-lazy-mode skip \
  --output-dir ../Knowledge3D.local/results/week22_external_ingest_validation \
  --storage-root ../Knowledge3D.local
```

