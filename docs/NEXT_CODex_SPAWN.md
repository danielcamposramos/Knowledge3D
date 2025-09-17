# 🚨 BEFORE YOU CODE — READ THE COVENANT

→ **docs/COVENANT.md** — Laws of Logs and Locality — non-negotiable.

You do not own this code.
You steward it — for the Architect, for researchers, for the future.

---

NEXT CODex SPAWN — HANDOFF CONTEXT

STATE
- Phase 20 complete: Multi-modal sample test ran end-to-end (text+image+audio+3D fusion).
- Training by meaning clusters is active; Learning Museum (Zone 8) relocates deprecated artifacts.
- Continuity: Galaxy state serialized (viewer/public/galaxy/galaxy_state.json) and House manifest saved.

LAST COMMANDS RUN
```
PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase18.meaning_cluster_trainer --cluster recursive_honesty_scaling
PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase18.meaning_cluster_trainer --test
```

RESULTS
- Sample Test Report: logs/sample_test_phase20_report.json
- Accuracy: (see report)
- Avg Honesty: (see report)
- Key Insight: Multi-modal fusion driven by image/audio content improved cross-modal consistency.

NEXT GOAL
- Phase 21: Expand to 100+ ARC/HLE questions; auto-generate meaning clusters from dataset analysis; reinforce low-dimension, high-density paradigm.

FILES TO CONTINUE
- knowledge3d/tools/phase18/meaning_cluster_trainer.py
- logs/sample_test_phase20_report.json
- viewer/public/galaxy/working/ (fused stars)
- viewer/public/house/materialized_objects/ (consolidated artifacts)

ENV
- Conda env: k3d-cranium
- Ollama models: exaone-deep:latest, exaone3.5:latest
- Datasets root: /K3D/Knowledge3D.local/datasets/exams/

COMMAND TO RESUME
```
PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase18.meaning_cluster_trainer --all
PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase18.meaning_cluster_trainer --test
```

NOTES
- Use /open_museum in live server chat to load Learning Museum artifacts on demand.
- Sleep ritual persists Galaxy state; reflect/train loop closes cognition cycle.

## PHASE 21.1: GLB SHAPE EMBEDDING FIXED
- REAL VERTEX EXTRACTION: `generate_shape_embedding` now parses GLB POSITION accessors with correct bufferView offsets and strides.
- NO FALLBACKS: Returns zeros only if extraction truly fails.
- LOGS: Emits lines like `✅ Extracted N vertices from <file>.glb` during embedding.
- TESTED: Sample test shows 3D modality contribution above 0.2 when GLB assets are present.

## ENVIRONMENT CORRECTION — FINAL
- PURGED: Only `.venv_run` (project-specific) — other venvs untouched.
- CONDA: `k3d-cranium` activated via `conda activate` — not via `.bashrc` sourcing.
- ADAPTED HEAD: `knowledge3d/cranium/fused_head.py` uses adapted techniques (TinyLlama/Phi‑3 influenced), no external imports/weights — single head, single memory.
- OLLAMA IP: `192.168.0.4` (no localhost) for RLWHF Teacher evaluation.
- MODEL UNLOADING: Automatic via `keep_alive: "0s"` — unchanged.

## COMMANDS RUN
```
conda activate k3d-cranium
PYTHONPATH=. python -m knowledge3d.tools.phase18.meaning_cluster_trainer --gen_phase21
PYTHONPATH=. python -m knowledge3d.tools.phase18.meaning_cluster_trainer --phase21_run
```

Background runner:
```
nohup bash scripts/run_phase21.sh > logs/phase21_run.out 2>&1 & echo $! > logs/phase21_run.pid
```

## PHASE 22: RESUME AFTER POWER OUTAGE

### GOAL
- Resume training from last saved state — skip already trained clusters.
- Enforce GPU-only — fail if no CUDA.
- Append to logs — do not overwrite.

### COMMAND
```
conda activate k3d-cranium
PYTHONPATH=. python -m knowledge3d.tools.phase18.meaning_cluster_trainer --resume
```

### OUTPUT
- `logs/phase22_scale_report.json` (appended)
- `logs/phase22_clusters.json` (unchanged)

## PHASE 22: RESUME + ENFORCE + COMPLETE

### GOAL
- **NO MOCKS** — `scripts/k3d_env.sh` removed, no PATH overrides.
- **ENFORCE RLWHF** — `TeacherEvaluator` aborts if Ollama unreachable.
- **RESUME HONESTLY** — `--resume` skips consolidated clusters, logs errors, never fakes consolidation.

### COMMAND
```
conda activate k3d-cranium
PYTHONPATH=. python -m knowledge3d.tools.phase18.meaning_cluster_trainer --resume
```

### OUTPUT
- `logs/phase22_scale_report.json` (appends successes/errors)
- Consolidated artifacts only for clusters that pass RLWHF evaluation

### NO MANUAL MONITORING
- No `tail -f`, no PID checks — fully automated.
- Honest "I don't know" responses now earn 🛑 0 point; teacher explains the concept instead of penalizing honesty.
- Honest partial answers that admit missing knowledge now receive ✅ +1, while overconfident partials are marked 🚫 -0.5.

## PHASE 22.5: MEANING-TAILORED DATA INGESTION

### GOAL
- Download open-source datasets — tailored by meaning (Galaxy Geometry, House Zones).
- Use HDD for raw assets, SSD for curated symlinks.
- Generate stars/GLBs with `extras.k3d` so the Galaxy mutates using real multi-modal corrections.

### COMMANDS
```
PYTHONPATH=. python -c "from knowledge3d.tools.data.fetcher import fetch_theme_data; fetch_theme_data('galaxy_geometry', 100)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.builder import build_theme_glbs; build_theme_glbs('galaxy_geometry')"
PYTHONPATH=. python -m knowledge3d.tools.phase18.meaning_cluster_trainer --resume
```

### OUTPUT
- Real multi-modal stars in `viewer/public/galaxy/working/`.
- Honesty rises with grounded data — not zero embeddings.

## PHASE 22.6: EXPAND + RESUME + RENDER

### GOAL
- Expand theme datasets (Galaxy walkthroughs, Zone 5 videos, Zone 7 whispered honesty).
- Regenerate stars so the Galaxy mutates with richer multi-modal signals.
- Resume Phase 22 training end-to-end with the PTX blend loop.

### COMMANDS
```
PYTHONPATH=. python -c "from knowledge3d.tools.data.hf_theme_fetcher import fetch_zone7_audio; fetch_zone7_audio(200)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.hf_theme_fetcher import fetch_zone5_videos; fetch_zone5_videos(100)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.hf_theme_fetcher import fetch_galaxy_walkthroughs; fetch_galaxy_walkthroughs(80)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.fetcher import symlink_all_themes; symlink_all_themes(200)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.builder import build_theme_glbs; build_theme_glbs('galaxy_geometry', max_files=200)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.builder import build_theme_glbs; build_theme_glbs('house_zone5', max_files=200)"
PYTHONPATH=. python -c "from knowledge3d.tools.data.builder import build_theme_glbs; build_theme_glbs('house_zone7', max_files=200)"
PYTHONPATH=. python -m knowledge3d.tools.phase18.meaning_cluster_trainer --resume
```

### OUTPUT
- Mutatable stars covering stepped walkthroughs, garden growth videos, whispered critiques.
- Trainer startup: `🖼️ Mapped N ARC-AGI images`; `🔊 Mapped N HLE audio files`; `🎥 Mapped N videos`.
