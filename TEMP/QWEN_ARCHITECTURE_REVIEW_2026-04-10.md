## Architecture Review: ARC-3 Sovereignty Cleanup

### Executive Summary

**Status: ⚠️ PARTIAL COMPLIANCE** — Core cleanup is correct, but **2 confirmed sovereignty violations remain** that must be addressed before deployment.

---

## ✅ CORRECT REMOVALS (Verified)

| Removed Item | Sovereignty Risk | Status |
|--------------|------------------|--------|
| `_strategy_hint()` function | Python reasoning | ✅ Deleted |
| Strategy hint injection into query | Text-based rule injection | ✅ Removed |
| Episode rule text in query | Python-dictated strategy | ✅ Removed |
| `inferred_rules` from WINE envelope | Rule leakage to TRM | ✅ Removed |
| `recent_outcomes` from WINE envelope | Outcome bias in perception | ✅ Removed |
| `rules` from `episode_context()` | Python rule selection | ✅ Removed |
| `self._strategy_hint` attribute | Stateful strategy storage | ✅ Removed |

**Galaxy VRAM operations are correct:**
- `_upsert_live_rule()` → Creates Galaxy stars ✓
- `_upsert_live_object()` → Creates Galaxy stars ✓
- `_crystallize_rules()` → Micro-sleeptime consolidation ✓

---

## ❌ CONFIRMED SOVEREIGNTY VIOLATIONS

### 1. `stuck_signal` in `choose_action()` — **CRITICAL VIOLATION**

```python
# DESCRIBED BEHAVIOR (not visible in truncated files):
if centroid_drift < 2.0 over last 5 actions:
    query += "stuck avatar not progressing try different approach"
```

**Why this violates sovereignty:**

| Principle | Violation |
|-----------|-----------|
| **Python = I/O only** | Python is making a strategic judgment ("stuck") |
| **No rule injection via text** | "try different approach" is a strategy hint injected into perception query |
| **TRM finds rules via Galaxy** | This bypasses Galaxy star navigation by telling TRM what to try |
| **RAW SIGNALS only** | "stuck" is an interpretation, not a raw signal |

**Correct approach:**
```python
# PASS RAW SIGNALS ONLY:
task_context["centroid_history"] = [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5)]
task_context["action_history"] = ["ACTION1", "ACTION4", "ACTION1", "ACTION4", "ACTION1"]
# Let Galaxy TRM detect the pattern and create a "stuck detection" star
```

**Fix required:** Remove the text injection. Pass centroid drift as a numeric signal if needed, but let Galaxy create the "stuck detection" rule star.

---

### 2. Budget "Bucket" Classification — **VIOLATION**

```python
# DESCRIBED BEHAVIOR:
budget_status = "critical" if budget < 10 else "low" if budget < 50 else "healthy"
query += f"budget {budget_status}"
```

**Why this violates sovereignty:**

| Principle | Violation |
|-----------|-----------|
| **Python = I/O only** | Python is classifying state semantics ("critical") |
| **RAW SIGNALS only** | "critical" is an interpretation, not raw data |
| **Strategy leakage** | "critical" implies urgency that influences TRM routing |

**Correct approach:**
```python
# PASS RAW SIGNALS ONLY:
task_context["budget_remaining"] = 7  # Raw integer
# Let Galaxy create "budget urgency" stars based on its own thresholds
```

**Fix required:** Pass raw budget integer. Remove bucket classification text from query.

---

## ⚠️ POTENTIAL VIOLATIONS (Need Full File Review)

### 3. `_focus_centroid()` Color Preference Logic

```python
# In arc_agi_3.py and arc3_episode_galaxy.py:
preferred_colors = [0, 1, 6, 9, 11, 12, 15]
# ...prefers salient rare colors...
```

**Assessment:** Borderline acceptable as **signal preprocessing**, but verify:
- ✅ If this is just computing which pixels to report → OK (I/O)
- ❌ If this influences which objects get priority in Galaxy → Violation (strategy)

**Recommendation:** Document this as "signal normalization" not "attention strategy". Ensure Galaxy can override these preferences via its own stars.

---

### 4. `_seed_game_mechanics_priors()` in Galaxy

```python
# In arc3_episode_galaxy.py:
rule = {
    "type": "ARC3_RULE",
    "condition": "agent_adjacent_to_untested_object",
    "action": "ACTION5",
    # ...
}
```

**Assessment:** ✅ **ACCEPTABLE** — This is Galaxy creating initial stars in VRAM, not Python injecting into query text. This is the correct pattern for bootstrapping episodic memory.

---

## 📋 REMEDIATION CHECKLIST

| Issue | Severity | Fix |
|-------|----------|-----|
| `stuck_signal` text injection | 🔴 Critical | Remove text, pass raw centroid history |
| Budget bucket classification | 🔴 Critical | Pass raw integer, remove classification |
| Verify `choose_action()` full implementation | 🟡 High | Review complete file for other injections |
| Verify `arc3_game_envelope()` task_context | 🟡 High | Ensure no strategy hints in context dict |
| Document `_focus_centroid()` as I/O | 🟢 Low | Add comment clarifying signal vs strategy |

---

## 🏗️ ARCHITECTURAL CORRECTNESS VERDICT

| Layer | Status | Notes |
|-------|--------|-------|
| **Python Boot/I/O** | ✅ Correct | Grid normalization, centroid calculation |
| **Galaxy VRAM Stars** | ✅ Correct | `_upsert_live_rule()`, `_upsert_live_object()` |
| **WINE Envelope** | ⚠️ Needs Review | Verify no strategy hints in `task_context` |
| **Perception Query** | ❌ Violations | `stuck_signal` and budget text must be removed |
| **TRM Rule Discovery** | ⚠️ At Risk | Text injections bypass Galaxy star navigation |

---

## 🔧 REQUIRED ACTIONS

1. **Remove `stuck_signal` text injection** from `choose_action()` — pass raw centroid history only
2. **Remove budget bucket classification** — pass raw integer only
3. **Audit complete `choose_action()` method** — verify no other strategy text injections
4. **Audit `arc3_game_envelope()` task_context** — ensure no derived interpretations
5. **Add sovereignty guardrails** — consider adding assertions that reject strategy keywords in query text

**Once these 2 critical violations are fixed, the architecture will be sovereignty-compliant.**