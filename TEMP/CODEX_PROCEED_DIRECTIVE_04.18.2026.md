# Codex Proceed Directive — Preflight Passed, Run the Benchmark

**Date:** 2026-04-18 (post-preflight)
**Role:** Codex — RUN-ONLY, unchanged.

---

## What your preflight showed

```
commit_head: ae3857163c2e2b545bc7229b1fd19a17990774a7  ← top of branch, matches ae385716 ✅
exit_code: 0                                             ← all 4 preflight commands succeeded ✅
git_status:
  ?? TEMP/CLAUDE_SINGLE_CONTEXT_LIVING_AI_SPEC_04.18.2026.md
  ?? TEMP/CLAUDE_SOVEREIGN_LAZY_PRIMARY_CONTEXT_WARMUP_PATCH_04.18.2026.md
  ?? TEMP/CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md
```

**All three `??` entries are Claude-authored architecture specs — not source files, not modifications.** The runbook's "no modified source files" rule refers to `M` entries in `git status --short`, not `??`. You have zero modified source files. Tree is clean.

**Preflight PASSED.** Your report used the FAIL format, but nothing failed — preflight is a gate check, not the task. Read the runbook section headings again: §1 Preflight, §2 Run. You completed §1. Now do §2.

---

## Proceed — run the benchmark

**Smoke first:**

```bash
bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick
```

If smoke passes (exit 0, JSON produced, five gates hold), continue to full:

```bash
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py \
     2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

If smoke fails, stop and report per the FAIL format — paste the actual benchmark command that failed, its actual exit code, actual stderr.

---

## Report correctly this time

- **On PASS:** Use the PASS format from the runbook — seven `key = value` lines + SHA. Commit ONLY the log + JSON. Don't commit the three `TEMP/*.md` files (those are Claude's; we'll handle them separately).
- **On FAIL:** Use the FAIL format, but only report on the actual benchmark command, not the preflight. If it's `python benchmarks/sovereign_bitnet_attention.py --quick` that fails, paste that command + its exit + its stderr tail.

---

## Not a drift, but note for next time

Reporting preflight-passed as a FAIL is over-cautious in the wrong direction. The charter is "run, watch, report" — if preflight succeeds, you *continue*, not halt. The halt-and-report rule applies to failures, not to completed prerequisites.

If you're ever unsure whether something is a failure or a checkpoint, re-read the runbook's success-criteria list (§3). If the criteria are empty because no benchmark ran yet, keep going. If the criteria failed against a produced JSON, then halt.

---

Proceed.
