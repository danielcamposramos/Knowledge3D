# CODEX: Foundational Knowledge Ingestion — Fix & Run

**Date:** December 10, 2025
**From:** Claude (Architecture)
**To:** Codex (Implementation)
**Priority:** High

---

## Context Recap

We're implementing the 4-layer foundational knowledge ingestion (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md):
- **Layer 1**: Math Galaxy (176 symbols) — COMPLETE
- **Layer 2**: Word Galaxy — scaffolding created (word_galaxy.py)
- **Layer 3**: Grammar Galaxy — existing, needs expansion
- **Layer 4**: Eloquence Galaxy — scaffolding created (eloquence_galaxy.py)

**Progress so far:**
1. ✅ word_galaxy.py created with symlink pattern (char_sequence references)
2. ✅ eloquence_galaxy.py created with rule_refs symlinks
3. ✅ ingest_foundational_pdfs.py created with heuristic extraction
4. ✅ tests/test_knowledge_ingestion.py — 2/2 tests pass
5. ✅ 86 PDFs symlinked to `/K3D/Knowledge3D.local/datasets/foundational_pdfs/`

**Current blocker:**
```
AttributeError: 'GrammarGalaxy' object has no attribute 'add_rule'. Did you mean: 'get_rule'?
```

The ingestion script calls `grammar_galaxy.add_rule(rule)` but GrammarGalaxy doesn't have that method.

---

## Your Tasks

### Task 1: Fix GrammarGalaxy.add_rule

**File:** `knowledge3d/training/arc_agi/grammar_galaxy.py`

Add an `add_rule()` method to GrammarGalaxy that:
1. Accepts a GrammarRule object
2. Validates symlink integrity (symbol_refs exist in Math Galaxy)
3. Stores the rule (avoid duplicates by rule_id)
4. Persists to disk if needed

**Pattern to follow:** Check how Word Galaxy implements `add_word()` and mirror that.

### Task 2: Verify Eloquence Galaxy has add_meta_rule

**File:** `knowledge3d/cranium/eloquence_galaxy.py`

Ensure `add_meta_rule()` method exists and validates `rule_refs` symlinks to Grammar Galaxy.

### Task 3: Run Dry-Run Ingestion

After fixing, run ingestion on a small subset (2-3 PDFs) to verify:
```bash
# Test with just Advanced Mathematics category
PYTHONPATH=. /home/daniel/miniforge/bin/conda run -n k3d-cranium python scripts/ingest_foundational_pdfs.py
```

**Expected output:** Stats showing words/rules/meta-rules extracted per category.

### Task 4: Run Full Ingestion via tmux

Once dry-run passes, run full ingestion in tmux background:

```bash
# Create tmux session for ingestion
tmux new-session -d -s k3d_ingest "bash -lc '
  export PATH=\"/home/daniel/miniforge/bin:/home/daniel/miniforge/condabin:\$PATH\"
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=.
  python scripts/ingest_foundational_pdfs.py 2>&1 | tee /K3D/Knowledge3D.local/logs/ingestion_$(date +%Y%m%d_%H%M%S).log
'"

# Monitor progress
tmux attach -t k3d_ingest
# Or check logs:
tail -f /K3D/Knowledge3D.local/logs/ingestion_*.log
```

### Task 5: Run ARC-AGI Training Post-Ingestion

After ingestion completes, launch hybrid TRM training to measure uplift:

```bash
# Create tmux session for training
tmux new-session -d -s k3d_arc_post_ingest "bash -lc '
  export PATH=\"/home/daniel/miniforge/bin:/home/daniel/miniforge/condabin:\$PATH\"
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python -m knowledge3d.training.arc_agi.sovereign_pipeline \
    --mode hybrid \
    --epochs 162 \
    --batch-size 4 \
    --deep-refinement-gate 0.15 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/arc_post_ingest_$(date +%Y%m%d_%H%M%S).log
'"
```

**Baseline to beat:** 42-51% accuracy (Math Galaxy only, 108 tasks × 162 epochs)

---

## Environment Notes (IMPORTANT - Debian, not Ubuntu)

- **OS:** Debian (strict, not Ubuntu)
- **Conda:** `/home/daniel/miniforge/bin/conda`
- **GPU env:** `k3d-cranium` (CUDA 12.4, cupy-cuda12x 13.6.0)
- **Repo path:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D`
- **PYTHONPATH:** Must be set to `.` for imports to work

**Activate shell pattern:**
```bash
export PATH="/home/daniel/miniforge/bin:/home/daniel/miniforge/condabin:$PATH"
conda activate k3d-cranium
```

**One-shot pattern:**
```bash
PYTHONPATH=. /home/daniel/miniforge/bin/conda run -n k3d-cranium python <script>
```

---

## Success Criteria

1. ✅ `GrammarGalaxy.add_rule()` implemented and tested
2. ✅ Ingestion script runs without errors
3. ✅ Stats logged: X words, Y rules, Z meta-rules extracted from 86 PDFs
4. ✅ ARC-AGI training completes with accuracy reported
5. ✅ Comparison: post-ingestion accuracy vs 42-51% baseline

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/training/arc_agi/grammar_galaxy.py` | Add `add_rule()` method |
| `knowledge3d/cranium/eloquence_galaxy.py` | Verify/add `add_meta_rule()` |
| `scripts/ingest_foundational_pdfs.py` | Bug fixes if needed |

---

## Reference: PDF Categories Ready

```
Advanced Mathematics/: 18 PDFs
Pedagogy & Learning/: 8 PDFs
Language, Grammar & Semantics/: 2 PDFs
Eloquence, Rhetoric & Persuasion/: 8 PDFs
Self-Reflection/: 7 PDFs
Story Telling/: 8 PDFs
Acting - Delivery/: 6 PDFs
Context & Contextual Understanding/: 13 PDFs
Temporal Understanding/: 13 PDFs
Academic Research Methods/: 3 PDFs
─────────────────────────────────
Total: 86 PDFs
```

---

**Hand-off complete. Codex, you have the conn.**
