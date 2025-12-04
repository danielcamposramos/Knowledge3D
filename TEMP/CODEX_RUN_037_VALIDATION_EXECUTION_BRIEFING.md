# CODEX Run 037 Validation — Execution Briefing

**Date:** December 4, 2025
**Phase:** ARC-AGI Regression Recovery Validation
**Agent:** Codex (Implementation)
**Status:** Ready for Execution

---

## CRITICAL: Read CODEX.md First

**BEFORE doing ANY work:**

1. **Read CODEX.md completely** — line by line, verbatim, no snippets
2. **Follow its instructions to the letter** (not CLAUDE.md!)
3. **Then read this briefing in full** — understand the complete validation sequence before starting

**Why:** This is a multi-phase validation sequence where each step depends on the previous one. Partial understanding leads to wrong execution order and wasted compute cycles.

---

## Executive Summary

**Mission:** Execute controlled validation sequence to verify regression fixes before committing to full Run 037 training cycle (24 hours).

**Context:** ARC-AGI training regressed from 46.7% (Run 028-034) to 31.8-31.9% (Run 035-036). Claude identified four root causes, you implemented all fixes. Now we validate before full commitment.

**Your Role:** Execute the validation sequence, monitor metrics, report decision points, and proceed to full Run 037 only if diagnostic passes.

**Timeline:**
- Phase 1 (Fresh Bootstrap): 2 minutes
- Phase 2 (Test Script): 5 minutes
- Phase 3 (Diagnostic): 15 minutes
- Phase 4 (Decision): Based on results
- Phase 5 (Full Run): 24 hours if approved

**Success Criteria:** Diagnostic run shows ≥40% accuracy, scale-invariant primitive usage >0, vocabulary detection >0, procedural bootstrap complete.

**Architectural Principle (Critical):**
This run uses **fresh bootstrap from code-defined procedural primitives** (not restored JSON checkpoints). Following the **Dual Client Reality** principle: primitives are procedurally defined and referenced (symlinked), not duplicated in JSON. Scale-invariant primitives (REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL) are now wired into the generation pipeline.

---

## Background: The Regression

### Baseline Performance (Healthy Variance)
- **Run 028:** 46.7% accuracy (beats Gemini 45.1%)
- **Runs 030-034:** 42-47% accuracy (±3% variance around 45% mean)
- **Training config:** 108 tasks × 162 epochs × 3 attempts per epoch

### Regression Discovery
- **Run 035:** 31.9% accuracy (-14.8% drop, persistent across epochs)
- **Run 036:** 31.8% accuracy (-14.9% drop, confirms not random)
- **Pattern:** Immediate drop at epoch 1, no recovery trajectory

### Root Cause Analysis (Claude + Codex Investigation)

**Issue 1: SleepTime Over-Pruning**
- Pruning threshold 0.60 removed 19 entries without audit trail
- Lost building blocks for Shadow Copy discovery
- No visibility into what was pruned or why

**Issue 2: Scale-Invariant Primitives Not Wired**
- Added REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL to DrawingGalaxy
- Never connected to CandidateGenerator pipeline
- 0 occurrences in generated candidates (confirmed via grep)

**Issue 3: Vocabulary Instrumentation Broken**
- Parser checked semantic rule names ("flip_horizontal_task")
- RPN programs use tokens ("FLIP_V", "rotate")
- Namespace mismatch → 0 grammar/shape detections since inception

**Issue 4: Shadow Copy Growth Stalled**
- Run 035: +2 discoveries (vs healthy growth in earlier runs)
- Run 036: +2 discoveries (persistent stall)
- Hypothesis: Lost primitives from over-pruning broke discovery chains

---

## Fixes Implemented (You Did This Already)

### Fix 1: SleepTime Audit Logging
**Files Modified:**
- `knowledge3d/training/arc_agi/sleeptime_consolidator.py`
- `scripts/run_sleeptime_consolidation.py`

**Changes:**
- Added `_pruned_audit` storage for all pruned entries
- Export JSON + human-readable audit logs
- Relaxed pruning threshold: 0.60 → 0.50
- Log detailed rule/shape stats before and after pruning

**Validation Check:** Audit logs exist at `/K3D/Knowledge3D.local/logs/sleeptime_audit_*.log`

### Fix 2: Wire Scale-Invariant Primitives
**Files Modified:**
- `knowledge3d/training/arc_agi/candidate_generator.py`
- `knowledge3d/training/arc_agi/parallel_generator.py`

**Changes:**
- Added DrawingGalaxy parameter to CandidateGenerator
- Implemented `_generate_scale_invariant_candidates()` with CPU fallbacks
- Pass drawing_galaxy through ParallelCandidateGenerator workers
- Support for REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL

**Validation Check:** Scale-invariant primitive usage >0 in logs

### Fix 3: Vocabulary Token Parsing
**Files Modified:**
- `knowledge3d/training/arc_agi/sovereign_pipeline.py`
- `scripts/analyze_run_diagnostics.py`

**Changes:**
- Parse RPN tokens directly (FLIP_V, rotate, RECT, LINE)
- Cache grammar/shape token indices for fast lookup
- Fix regex bug in diagnostic analyzer
- Token-based detection in `_log_vocabulary_quality()`

**Validation Check:** Vocabulary detection >0 in logs

### Fix 4: Test Infrastructure
**Files Created:**
- `scripts/test_regression_fixes.py`

**Tests:**
- Scale-invariant primitive generation (all 4 types)
- Vocabulary token parsing (grammar + shapes)
- SleepTime audit plumbing
- Export verification

**Validation Check:** All tests pass

---

## Execution Sequence

### User's Decisions (From Claude's Strategic Questions)

**Q1 - Checkpoint State:**
- Decision: **Fresh Bootstrap (Revised)**
- **Architectural Discovery:** Checkpoints are overwritten each run (no run_034/run_036 directories exist)
- **User Guidance:** "Start from basic drawing primitives - this time symlinked, and not json spliced"
- **Execution:** Fresh bootstrap from code-defined primitives with new audit logging

**Q2 - Success Criteria:**
- Agreed with architectural goals
- Flexible on exact thresholds ("ML can take time and sometimes regresses a little to progress back")
- Primary signal: Evidence of all three fixes working

**Q3 - Future Plans:**
- "Let's see what happens, then we decide the future in the future"
- Decision point AFTER diagnostic results

### Phase 1: Fresh Bootstrap with Audit Logging

**Purpose:** Start fresh from bootstrap primitives (code-defined) with new audit logging enabled.

**Checkpoint Architecture Discovery:**
```bash
# ACTUAL checkpoint paths (from train_arc_sovereign_loop.py):
/K3D/Knowledge3D.local/checkpoints/arc_agi/
├── drawing_galaxy.json        # 24 shapes (from Run 036)
├── grammar_galaxy.json         # 220 rules (from Run 036)
├── shadow_copy.json            # 145 entries (from Run 036)
└── deduplication_index.json    # 145 unique programs (from Run 036)

# These files are OVERWRITTEN each run (no per-run directories)
# Run 036 overwrote Run 035 which overwrote Run 034
# Cannot restore Run 034 - those checkpoints no longer exist
```

**Strategy:**
1. Backup Run 036 state for reference
2. Delete checkpoints to force bootstrap from code
3. Bootstrap will use default_grammar_rules() and default drawing primitives
4. New audit logging will track all changes from baseline

**Commands:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Backup Run 036 state for reference
mkdir -p /K3D/Knowledge3D.local/checkpoints/arc_agi_run036_backup
cp /K3D/Knowledge3D.local/checkpoints/arc_agi/*.json \
   /K3D/Knowledge3D.local/checkpoints/arc_agi_run036_backup/

# Delete checkpoints to force fresh bootstrap
rm /K3D/Knowledge3D.local/checkpoints/arc_agi/drawing_galaxy.json
rm /K3D/Knowledge3D.local/checkpoints/arc_agi/grammar_galaxy.json
rm /K3D/Knowledge3D.local/checkpoints/arc_agi/shadow_copy.json
rm /K3D/Knowledge3D.local/checkpoints/arc_agi/deduplication_index.json

# Verify clean slate
ls -lh /K3D/Knowledge3D.local/checkpoints/arc_agi/
```

**Expected Bootstrap Behavior:**
- DrawingGalaxy: "No checkpoint at {path}, using bootstrap" → loads default_shapes()
- GrammarGalaxy: "No checkpoint at {path}, using bootstrap" → loads default_grammar_rules()
- DualShadowCopy: "No checkpoint at {path}, starting fresh" → empty shadow
- ContentDeduplicator: "No checkpoint at {path}, starting fresh" → empty dedup index

**Success Check:**
- Backup directory exists: `/K3D/Knowledge3D.local/checkpoints/arc_agi_run036_backup/*.json`
- Checkpoint directory empty or missing files (will be recreated by training)
- Training will start with bootstrap primitives from code

**Report Back:**
```
[PHASE 1 COMPLETE] Fresh bootstrap preparation
- Run 036 backed up to: /K3D/Knowledge3D.local/checkpoints/arc_agi_run036_backup/
- Checkpoints cleared for fresh start
- Bootstrap will use:
  - Drawing primitives from code (default_shapes)
  - Grammar rules from code (default_grammar_rules)
  - Empty shadow copy
  - Scale-invariant primitives NOW WIRED (REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL)
```

### Phase 2: Test Script Validation (Before Training)

**Purpose:** Verify all three fixes work in isolation before full training.

**Commands:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Run regression fix test suite
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/test_regression_fixes.py
```

**What to Monitor:**
- Test 1: Scale-invariant primitive generation (REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL)
- Test 2: Vocabulary token parsing (FLIP_V, rotate, RECT, LINE)
- Test 3: SleepTime audit plumbing
- All tests must pass

**Success Check:**
- All tests pass
- Scale-invariant primitives generate valid candidates
- Vocabulary parser detects RPN tokens
- Audit logs readable

**Report Back:**
```
[PHASE 2 COMPLETE] Test script validation
- Test scale-invariant primitives: PASS/FAIL
- Test vocabulary parsing: PASS/FAIL
- Test SleepTime audit: PASS/FAIL
- Overall: ALL PASS
```

**If Any Test Fails:**
- STOP immediately
- Report failure details
- Wait for architectural review before proceeding

### Phase 3: Diagnostic Training Run (10 Tasks × 3 Epochs)

**Purpose:** Quick validation that fixes work in actual training loop before committing 24 hours.

**Setup:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Set up tmux session for monitoring
tmux new-session -d -s arc_diagnostic
tmux send-keys -t arc_diagnostic "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" Enter

# Confirm GPU available
nvidia-smi
```

**Training Command:**
```bash
tmux send-keys -t arc_diagnostic "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/validate_arc_reasoning.py --n-samples 10 --n-epochs 3 --run-id diagnostic_037 2>&1 | tee /K3D/Knowledge3D.local/logs/diagnostic_037.log" Enter
```

**What to Monitor (Every 5 Minutes):**
```bash
# Check progress
tail -n 50 /K3D/Knowledge3D.local/logs/diagnostic_037.log

# Look for these signals:
# 1. Scale-invariant primitive usage: "scale_invariant candidates generated: X" (X > 0)
# 2. Vocabulary detection: "grammar rules detected: Y" or "drawing shapes detected: Z" (Y or Z > 0)
# 3. Accuracy progression: "Accuracy: A.AA%" per task
# 4. Shadow Copy growth: "+N new discoveries" (N > 0)
```

**GPU Monitoring:**
```bash
# Check GPU utilization (expect >182 MiB VRAM usage)
watch -n 5 nvidia-smi
```

**Success Criteria:**
- **Primary:** Final accuracy ≥40% (recovery signal)
- **Secondary:** Scale-invariant usage >0 (Fix 2 working)
- **Secondary:** Vocabulary detection >0 (Fix 3 working)
- **Secondary:** Shadow Copy growth >2 (Fix 4 working)
- **Tertiary:** No PTX crashes or fallback spikes

**Flexible Thresholds (Per User Guidance):**
- 40-47%: Strong recovery → Proceed to Run 037
- 30-39%: Partial recovery → Review logs, may proceed if fixes visible
- <30%: No recovery → STOP, full architectural review

**Estimated Time:** 15 minutes

**Report Back:**
```
[PHASE 3 COMPLETE] Diagnostic training results (10 tasks × 3 epochs)
- Final accuracy: XX.X%
- Scale-invariant usage: Y instances (>0 = PASS)
- Vocabulary detection: Z instances (>0 = PASS)
- Shadow Copy growth: +W discoveries (>2 = PASS)
- PTX success rate: AA.A%
- Recommendation: PROCEED/REVIEW/STOP
```

### Phase 4: Decision Point

**If Diagnostic Passes (≥40% + All Fixes Visible):**
- Wait for user confirmation to proceed
- Report: "Diagnostic passed, ready for Run 037. Awaiting authorization."

**If Diagnostic Partial (30-39% + Fixes Visible):**
- Report results with detailed logs
- Recommendation: "Partial recovery, fixes are working. ML may need time. Recommend proceed with monitoring."
- Wait for user decision

**If Diagnostic Fails (<30% or Fixes Not Working):**
- STOP immediately
- Report: "Diagnostic failed. Not proceeding to Run 037."
- Archive logs for architectural review
- Wait for instructions

### Phase 5: Full Run 037 (Execute Only If Approved)

**Setup:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Create new tmux session
tmux new-session -d -s arc_run_037
tmux send-keys -t arc_run_037 "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" Enter

# Confirm GPU available
nvidia-smi
```

**Training Command:**
```bash
tmux send-keys -t arc_run_037 "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/validate_arc_reasoning.py --n-samples 108 --run-id 037 2>&1 | tee /K3D/Knowledge3D.local/logs/run_037.log" Enter
```

**Monitoring Schedule:**
- First hour: Check every 15 minutes for crashes
- Hours 2-6: Check every hour
- Hours 7-24: Check every 3 hours
- Final: Full analysis when complete

**What to Monitor:**
```bash
# Progress check
tail -n 100 /K3D/Knowledge3D.local/logs/run_037.log

# Look for:
# 1. Epoch progression (1-162)
# 2. Task accuracy trends
# 3. Scale-invariant usage consistency
# 4. Vocabulary detection consistency
# 5. Shadow Copy growth trajectory
# 6. GPU utilization (expect >182 MiB)
```

**Success Metrics (Same as Run 028-034):**
- **Target:** 40-47% accuracy range (recovery to baseline)
- **Minimum:** 38% (Run 034 low end)
- **Regression flag:** <35% (investigate if below this)

**Estimated Time:** 24 hours

**Report Back When Complete:**
```
[PHASE 5 COMPLETE] Run 037 training finished
- Final accuracy: XX.X%
- Scale-invariant usage: Y total instances
- Vocabulary detection: Z total instances
- Shadow Copy discoveries: +W new
- PTX success rate: AA.A%
- Recovery status: FULL/PARTIAL/NONE
- Detailed logs: /K3D/Knowledge3D.local/logs/run_037.log
```

---

## Environment Setup

### GPU Configuration
```bash
# Expose GPU (RTX 3060, 12GB VRAM)
export CUDA_VISIBLE_DEVICES=0

# Verify GPU available
nvidia-smi

# Expected output: GPU 0, RTX 3060, ~11 GB available
```

### Python Environment
```bash
# Use K3D Cranium environment
export PYTHONPATH="/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Python interpreter
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python
```

### Tmux Sessions
```bash
# Create session
tmux new-session -d -s <session_name>

# Send commands
tmux send-keys -t <session_name> "<command>" Enter

# Attach to view progress
tmux attach -t <session_name>

# Detach: Ctrl+B, then D
```

### Log Locations
- Training logs: `/K3D/Knowledge3D.local/logs/run_*.log`
- Audit logs: `/K3D/Knowledge3D.local/logs/sleeptime_audit_*.log`
- Checkpoints: `/K3D/Knowledge3D.local/checkpoints/run_*/`

---

## Monitoring Guide

### Key Log Signals

**1. Scale-Invariant Primitive Usage (Fix 2)**
```
[PARALLEL GEN] scale_invariant candidates generated: X
```
- Expected: X >0 (confirms primitives wired into generation)
- Regression signal: X = 0 (primitives not being used)

**2. Vocabulary Detection (Fix 3)**
```
[TRAIN] grammar rules detected: Y (AA.A%)
[TRAIN] drawing shapes detected: Z (BB.B%)
```
- Expected: Y >0 and Z >0 (confirms token parsing working)
- Regression signal: Y = 0 and Z = 0 (vocabulary instrumentation broken)

**3. Shadow Copy Growth (Fix 4)**
```
[DISCOVERY] +N new discoveries this epoch
```
- Expected: Steady growth across epochs
- Regression signal: +0 or +1-2 persistent stalls

**4. SleepTime Audit (Fix 1)**
```
[SLEEPTIME] Pruned X entries (threshold 0.50)
[SLEEPTIME] Audit log: /K3D/Knowledge3D.local/logs/sleeptime_audit_*.log
```
- Expected: X < 19 (more conservative than Run 035)
- Audit logs exist and readable

**5. Accuracy Progression**
```
Task abc12345: Accuracy: AA.AA% (epoch 1)
Task abc12345: Accuracy: BB.BB% (epoch 50)
Task abc12345: Accuracy: CC.CC% (epoch 162)
```
- Healthy: Upward trend or stable high
- Regression: Flat low or downward trend

**6. GPU Utilization**
```bash
nvidia-smi
# Look for:
# - VRAM usage > 182 MiB (Run 035-036 used only 182 MiB)
# - GPU-Util > 0% during training
```

### Progress Checkpoints

**First 15 Minutes (Diagnostic or Full Run):**
- Confirm all three fixes show evidence in logs
- Verify GPU utilization increased
- Check no PTX crashes

**First Hour (Full Run Only):**
- Confirm epoch progression
- Check accuracy trends across tasks
- Monitor Shadow Copy growth

**Every 3 Hours (Full Run Only):**
- Spot-check accuracy progression
- Verify no crashes or fallback spikes
- Confirm GPU still active

**Final Analysis:**
- Compare final accuracy to baseline (40-47% target)
- Aggregate fix metrics across all epochs
- Generate completion report

---

## Decision Framework

### Proceed to Next Phase If:
- Phase 1: Bootstrap preparation complete, Run 036 backed up
- Phase 2: All tests pass
- Phase 3: Accuracy ≥40% OR accuracy 30-39% with all fixes visible
- Phase 4: User authorization received
- Phase 5: N/A (final phase)

### STOP and Report If:
- Any phase fails validation checks
- Diagnostic accuracy <30% (no recovery signal)
- Any of the three fixes show 0 usage in diagnostic
- PTX crashes or fallback rate >50%
- GPU not utilized (VRAM stuck at 182 MiB)

### Flexible Interpretation (Per User Guidance):
- "ML can take time and sometimes regresses a little to progress back"
- 30-39% diagnostic result: Review logs, may proceed if fixes are clearly working
- Small accuracy fluctuations: Expected, focus on fix evidence
- Shadow Copy growth: +2-3 is OK if primitives/vocabulary working

---

## Reporting Template

### Phase Completion Report
```
[PHASE X COMPLETE] <Phase Name>
- Objective: <What this phase validated>
- Result: PASS/FAIL
- Key Metrics:
  - Metric 1: Value
  - Metric 2: Value
- Next Step: Proceed to Phase X+1 / STOP / Await authorization
- Logs: [paths to relevant logs]
```

### Decision Point Report
```
[DECISION POINT] Phase 4 Diagnostic Results
- Final Accuracy: XX.X%
- Fix Evidence:
  - Scale-invariant usage: Y instances (PASS/FAIL)
  - Vocabulary detection: Z instances (PASS/FAIL)
  - Shadow Copy growth: +W discoveries (PASS/FAIL)
- Recommendation: PROCEED/REVIEW/STOP
- Reasoning: <Why this recommendation>
- Awaiting user authorization to proceed to Run 037
```

### Final Run Report
```
[RUN 037 COMPLETE] ARC-AGI Training Validation
- Duration: XX hours
- Final Accuracy: XX.X%
- Baseline Comparison: Run 028 (46.7%) vs Run 037 (XX.X%) = +/- Y.Y%
- Recovery Status: FULL/PARTIAL/NONE
- Fix Validation:
  - Scale-invariant primitives: Z total uses
  - Vocabulary detection: W total detections
  - Shadow Copy growth: +V discoveries
  - SleepTime audit: Pruned U entries (vs 19 in Run 035)
- Architectural Assessment: <Claude will provide>
- Next Steps: <User will decide>
- Detailed logs: /K3D/Knowledge3D.local/logs/run_037.log
```

---

## Troubleshooting

### GPU Not Available
```bash
# Check GPU status
nvidia-smi

# If busy, wait or identify process
fuser -v /dev/nvidia0

# If needed, wait for current process to finish
```

### Tmux Session Lost
```bash
# List sessions
tmux ls

# Reattach
tmux attach -t <session_name>

# If session died, check logs directly
tail -f /K3D/Knowledge3D.local/logs/<log_file>
```

### Test Script Failures
- Read error output carefully
- Check if it's a test bug vs actual regression
- Report to Claude for architectural review
- DO NOT proceed to diagnostic if tests fail

### Diagnostic Accuracy Unexpectedly Low
- Check if all three fixes show evidence (maybe one isn't working)
- Review last 200 lines of log for errors
- Check GPU actually being used (nvidia-smi)
- Report to user/Claude for decision

### PTX Crashes
- Check for CUDA errors in logs
- Report immediately (sovereignty violation signal)
- DO NOT proceed to full run if PTX crashing

---

## Success Criteria Summary

### Diagnostic Run (Phase 3)
**Must Pass:**
- Scale-invariant usage >0
- Vocabulary detection >0
- No PTX crashes
- Bootstrap from code successful

**Target:**
- Accuracy ≥40%

**Flexible:**
- 30-39% acceptable if all fixes clearly working
- User guidance: "ML can take time, may regress before progress"

### Full Run 037 (Phase 5)
**Target:**
- 40-47% accuracy (recovery to Run 028-034 baseline)

**Minimum Acceptable:**
- 38% (Run 034 low end)

**Regression Flag:**
- <35% (stop and investigate)

**Fix Evidence Required:**
- All three fixes show consistent usage throughout training
- Procedural primitives (symlinked, not JSON spliced) working correctly
- Shadow Copy growth trajectory healthy

---

## Critical Reminders

1. **Follow CODEX.md Protocol:** Read it completely before starting
2. **Execute Phases in Order:** Bootstrap → Test → Diagnostic → Decision → Full Run
3. **Report at Every Phase:** Don't batch reports, communicate progress and blockers
4. **STOP if Validation Fails:** Don't proceed to next phase if current phase fails
5. **Monitor GPU Usage:** VRAM should be >182 MiB during training
6. **Check Fix Evidence:** All three fixes must show usage in logs
7. **Procedural Architecture:** Bootstrap uses code-defined primitives (symlinked references, not JSON duplicates)
8. **Await Authorization:** Don't launch Run 037 until user approves after diagnostic
9. **Be Flexible on Thresholds:** Focus on fix evidence, not just accuracy number
10. **Report Completion:** Full report when Run 037 finishes

---

## Context Files for Reference

**If you need more details:**
- `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/CODEX.md` — Your protocol
- `TEMP/CLAUDE_ARC_REGRESSION_FIX_SPECIFICATION.md` — Full architectural specification (1247 lines)
- `docs/ROADMAP.md` — Phase 3 ARC-AGI context
- `BRIEFING.md` — Dual-client reality, sovereignty principles

**Do not re-read unless stuck.** This briefing contains everything needed for execution.

---

## Start Command

```bash
# Begin Phase 1: Restore Run 034 Checkpoints
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Report to user: "Starting Phase 1: Restoring Run 034 checkpoints"
```

---

**End of Briefing — Execute in order, report at each phase, STOP if validation fails. Good luck, Codex.**
