# Codex Directive: Track C — MMLU Galaxy Expansion + GSM8K Failure Audit

**Date:** 2026-03-16
**From:** Claude (Architecture)
**To:** Codex (Implementation)
**Priority:** Track C first, audit second
**Baseline:** ARC 10/10, Math 20/20, GSM8K 2/10, LHE 6/10, MMLU ~12-15/50 (variance)
**Checkpoint:** `/tmp/k3d_nsi_full_guard/checkpoints` (NSI-safe root)

---

## Part 1: Track C — MMLU Galaxy Neighborhood Expansion

### The Problem

MMLU covers 34 subjects across 3 domains. Reality Galaxy currently has 56 entries covering **physics only** (kinematics, dynamics, E&M, thermodynamics). The entire humanities and social sciences domains have ZERO dedicated Galaxy entries. Biology, chemistry, and computer science are also unrepresented.

MMLU scores 12-15/50 — the system can only answer questions where the answer happens to match existing Math/Grammar/Physics Galaxy content. Questions about biology, history, economics, philosophy, law, etc. have no Galaxy neighborhood to navigate.

### What To Build

Expand `foundational_operations_bootstrap.py` with Reality Galaxy entries for the MMLU subject areas that currently have NO coverage. Each entry is a procedural RPN program with metadata — following the exact pattern of existing physics entries in `reality_galaxy.py`.

**Entry format (follow existing pattern):**
```python
{
    "id": "reality_biology_cell_division",
    "galaxy": "Reality",
    "domain": "biology",
    "subject": "college_biology",
    "subfield": "cell_biology",
    "kind": "procedural_system",
    "description": "Mitosis produces 2^n cells after n divisions",
    "rpn_template": ["n", "2", "SWAP", "POW"],
    "query_anchor": "cell division mitosis meiosis",
    "cross_modal": ["math_exponential"],
}
```

### Domains Needing Entries (Prioritized by MMLU Subject Count)

**Tier 1 — STEM gaps (15 subjects, physics already covered):**

| Subject | Entries Needed | Focus |
|---------|---------------|-------|
| `college_biology` / `high_school_biology` | 8-10 | Cell division, genetics (Mendelian ratios), photosynthesis, DNA/RNA, evolution, ecology |
| `college_chemistry` / `high_school_chemistry` | 8-10 | Periodic table patterns, bonding (ionic/covalent), stoichiometry, acid-base pH, gas laws |
| `college_computer_science` | 5 | Algorithm complexity (Big-O), data structures, boolean logic, binary/hex conversion |
| `astronomy` | 3 | Kepler's laws, stellar classification, Hubble's law |
| `electrical_engineering` | 3 | Digital logic gates, signal processing basics, circuit analysis (already partial via E&M) |
| `machine_learning` | 3 | Bias-variance tradeoff, gradient descent, overfitting/underfitting |
| `computer_security` | 2 | Encryption basics, authentication models |

**Tier 2 — Humanities (8 subjects, ZERO coverage):**

| Subject | Entries Needed | Focus |
|---------|---------------|-------|
| `formal_logic` | 5 | Propositional logic (AND/OR/NOT/IMPLIES), syllogisms, truth tables, modal logic |
| `philosophy` | 3 | Major frameworks (utilitarianism, deontology, virtue ethics) as decision procedures |
| `moral_scenarios` / `moral_disputes` | 3 | Ethical reasoning patterns (trolley problem structure, rights vs. consequences) |
| `jurisprudence` / `professional_law` | 3 | Legal reasoning patterns (precedent, statutory interpretation, burden of proof) |
| `world_religions` | 2 | Major tradition classification, comparative patterns |
| `prehistory` | 2 | Timeline anchors, archaeological reasoning |

**Tier 3 — Social Sciences (11 subjects, ZERO coverage):**

| Subject | Entries Needed | Focus |
|---------|---------------|-------|
| `high_school_microeconomics` / `macroeconomics` / `econometrics` | 5 | Supply-demand, GDP, inflation, marginal analysis, opportunity cost |
| `professional_accounting` | 3 | Balance sheet structure, double-entry, depreciation |
| `business_ethics` / `marketing` | 2 | Stakeholder analysis, market segmentation patterns |
| `high_school_geography` | 3 | Climate zones, population density, resource distribution |
| `high_school_us_history` / `world_history` | 4 | Major period anchors (not trivia — structural patterns like "revolution causes") |
| `sociology` | 2 | Social structure patterns, stratification models |
| `high_school_government_and_politics` | 2 | Constitutional structure, separation of powers, legislative process |

**Total: ~70-80 new Reality Galaxy entries across ~25 subjects.**

### Grammar Galaxy Rules (Defeasible)

For each new domain, add 2-3 Grammar Galaxy rules connecting concepts. All should use defeasible metadata:

```python
GrammarRule(
    rule_id="biology_mendelian_dominance",
    language="reasoning",
    pattern="dominant allele expressed over recessive",
    rpn_program="GALAXY_LOOKUP dominant GALAXY_LOOKUP recessive TCOMP",
    domain="reality_biology",
    rule_strength=0,  # defeasible (incomplete dominance exists)
    superior_to=[],
    trust_weight=0.8,
)
```

Add ~15-20 Grammar rules total across the new domains. Strict rules (`rule_strength=+1`) only for mathematical certainties (e.g., `2^n` cell division). Everything else defeasible.

### Domain Hint Enhancement

In `knowledgeverse.py`, the MMLU routing already threads `domain_hint=question["subject"]`. Enhance LED-A* navigation seeding so that domain_hint pushes the starting neighborhood toward the relevant Galaxy entries:

In `_build_gpu_reasoning_paths()` or `_select_gpu_reasoning_program()`, when task_type is MMLU_TASK:
- Use `domain_hint` to seed the Morton Octree lookup with entries from the matching `subject` field
- If `domain_hint="college_biology"`, the LED-A* should start navigating from Reality Galaxy entries with `subject="college_biology"` rather than from generic Grammar entries

This is NOT a new mechanism — it's ensuring the existing domain_hint flows all the way to Galaxy navigation seeding.

### Important Constraints

1. **Ingestion path only.** All new entries go into `foundational_operations_bootstrap.py`. NO live insertion during inference.
2. **Follow existing patterns.** Look at `reality_galaxy.py` lines 1-639 for the exact entry format. Match it exactly.
3. **RPN programs must be valid.** Each entry's `rpn_template` should use only existing opcodes (ADD, SUB, MUL, DIV, POW, SIN, COS, SWAP, DUP, and Galaxy opcodes from Standard tier).
4. **No external dependencies.** No Wikipedia scraping, no LLM-generated content. Write entries from domain knowledge.
5. **Benchmark safety.** ARC 10/10, Math 20/20 MUST hold. LHE 6/10 should hold. GSM8K 2/10 should hold or improve.

### Validation

1. `python3 -m compileall knowledge3d/` must pass
2. Full benchmark from NSI-safe checkpoint:
   - ARC 10/10, Math 20/20 (MUST hold)
   - GSM8K 2/10 (MUST hold)
   - LHE 6/10 (MUST hold)
   - MMLU: target 18+/50 (from ~12-15/50)
3. Run MMLU twice from clean roots to confirm signal vs noise (variance range is 11-16)

---

## Part 2: GSM8K Failure Audit

After Track C lands, run the GSM8K 10-question benchmark and for EACH of the ~8 failing problems, collect:

1. **Which problem** (gsm8k_0 through gsm8k_9)
2. **Expected answer** vs **actual answer produced**
3. **Selection steps** — the full halting gate log showing what programs ran, what scores they got
4. **Failure mode classification:**
   - `PARSE_FAILURE` — entity extraction missed quantities or references
   - `STRATEGY_FAILURE` — wrong reasoning program selected (e.g., rate when should be sequential)
   - `COMPOSITION_FAILURE` — right entities found but wrong operation applied
   - `GALAXY_MISS` — needed concept not in Galaxy
   - `HALTING_FAILURE` — right answer was a candidate but lost to wrong one at halting gate

5. **Write the audit to:** `TEMP/GSM8K_FAILURE_AUDIT_03.16.2026.md`

Format per problem:
```
### gsm8k_N: [problem summary]
- Expected: [answer]
- Got: [actual]
- Failure mode: [classification]
- Selection steps: [key lines from halting log]
- Root cause: [1-2 sentence diagnosis]
- Suggested fix: [what Galaxy entry / Grammar rule / routing change would fix it]
```

This audit will drive the next round of targeted fixes.

---

## Files Summary

| File | Change | Scope |
|------|--------|-------|
| `foundational_operations_bootstrap.py` | +70-80 Reality Galaxy entries across 25+ subjects | Track C |
| `grammar_galaxy.py` | +15-20 domain Grammar rules with defeasible metadata | Track C |
| `knowledgeverse.py` | Enhance domain_hint → LED-A* seeding for MMLU | Track C |
| `TEMP/GSM8K_FAILURE_AUDIT_03.16.2026.md` | New file — failure mode analysis | Audit |
| `tests/test_trm_weight_persistence.py` | Add MMLU Galaxy coverage smoke test | Track C |
