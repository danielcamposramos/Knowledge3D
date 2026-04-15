# Codex Report — Live Benchmark 50-Slice Strict WINE/Tablet Run

**Date:** 2026-04-15  
**Author:** Codex  
**Audience:** Claude + Daniel  
**Status:** Completed run, report for ingestion prioritization

---

## 0. Scope

This report summarizes the first **significant strict live benchmark sweep**
after the WINE/tablet session wiring fix:

- **One resident `Knowledgeverse`**
- **One tablet session path**
- **No per-item Python orchestration**
- **Strict `knowledgeverse_dispatch_session` / live game path**

Run used **50 items per suite** for:

- ARC-AGI 2
- MMLU
- GSM8K
- Last Humanity Exam
- Unified Math
- AMC/AIME
- Omni-MATH
- IMO Bench

`arc3_local` was correctly archived/skipped by design.

Artifacts:

- `Knowledge3D.local/batch10_full50_logs/summary.execution.json`
- `Knowledge3D.local/batch10_full50_logs/full_results.execution.json`
- `Knowledge3D.local/batch10_full50_results.json`

---

## 1. Core Finding

The **live wiring is now correct**.

This run is the first one in which:

- every active benchmark suite returned through the **tablet translator**
- every active suite reported **`gpu_result_packets = 50 / 50`**
- route-family accounting was no longer `UNKNOWN`
- result rows carried real sovereign fields like:
  - `solver = knowledgeverse_gpu_query`
  - `runtime = knowledgeverse_gpu_query`
  - `program_id = gpu_task_dispatch_sovereign`
  - structured `task_result.route_family`

So the benchmark path is no longer lying to us. The numbers below reflect
the current live system, not a Python orchestration artifact.

---

## 2. Final 50-Slice Results

| Suite | Correct / Total | Accuracy | GPU packets | Route family |
|---|---:|---:|---:|---|
| ARC-AGI 2 | 0 / 50 | 0.0000 | 50 / 50 | `GAME_2D` |
| MMLU | 8 / 50 | 0.1600 | 50 / 50 | `MMLU` |
| GSM8K | 1 / 50 | 0.0200 | 50 / 50 | `MATH` |
| Last Humanity Exam | 2 / 50 | 0.0400 | 50 / 50 | `LHE`=40, `MMLU`=10 |
| Unified Math | 1 / 50 | 0.0200 | 50 / 50 | `MATH` |
| AMC/AIME | 0 / 50 | 0.0000 | 50 / 50 | `MATH` |
| Omni-MATH | 0 / 50 | 0.0000 | 50 / 50 | `MATH` |
| IMO Bench | 0 / 50 | 0.0000 | 50 / 50 | `MATH` |

Operationally this means:

- **execution path = green**
- **knowledge/reasoning quality = still weak**

---

## 3. What The Results Mean

### 3.1 What is now proven

Proven by this run:

- the **single live head** is being exercised through the tablet/WINE path
- the benchmark runner is using a **resident world**, not per-item boot
- the benchmark suites are not bypassing the live system with Python-side solves
- the form/meaning/routing stack is loaded enough to produce routed answers

### 3.2 What is still broken

What is not proven:

- that the current knowledge payload is sufficient
- that the current math reasoning programs are converging well
- that ARC has the right pattern/program substrate

The accuracy profile says the system is **running correctly** but **thinking poorly**.

---

## 4. Domain-by-Domain Interpretation

### 4.1 ARC-AGI 2 — 0 / 50

Route is correct (`GAME_2D`), GPU path is correct, but output quality is zero.

Interpretation:

- this does **not** look like a missing tablet/wiring issue anymore
- it looks like missing or insufficient **ARC reasoning primitives / transform priors**
- next ARC work should be **knowledge/program ingestion**, not transport work

Most likely next knowledge asset:

- `TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md`

Recommendation:

- ingest ARC reasoning primitives / transform families / program priors
- do not expect HS curriculum ingestion alone to move ARC materially

### 4.2 MMLU — 8 / 50 (16%)

This is the best signal in the current run because it spans broad factual and
curricular knowledge.

Observed misses include:

- `abstract_algebra`
- `clinical_knowledge`
- `college_biology`
- `college_mathematics`
- `electrical_engineering`
- `high_school_government_and_politics`
- `high_school_macroeconomics`
- `high_school_microeconomics`
- `high_school_psychology`
- `high_school_us_history`

Observed hits include isolated wins in:

- `astronomy`
- `college_physics`
- `high_school_biology`
- `high_school_geography`

Interpretation:

- current ingested reasoning taxonomy + HS math clusters are not enough
- the missing signal maps directly to the **remaining HS curriculum waves**

### 4.3 Last Humanity Exam — 2 / 50 (4%)

LHE is currently weak across most advanced domains.

Observed domains with misses:

- Mathematics
- Physics
- Computer Science
- Electrical Engineering
- Applied Mathematics
- Linguistics
- Chess / Game Design

Observed small positive signal:

- `Chemistry`: 1 / 5
- `Law`: 1 / 1

Interpretation:

- LHE confirms the same picture as MMLU:
  broad curriculum/domain knowledge is still thin
- the system has enough knowledge to route, but not enough depth to solve

### 4.4 Math suites — almost zero

Current math-family results:

- GSM8K: 1 / 50
- Unified Math: 1 / 50
- AMC/AIME: 0 / 50
- Omni-MATH: 0 / 50
- IMO: 0 / 50

Common pattern in wrong outputs:

- repeated stereotyped answers like `98`, `20`, `[0,3)`, `260`, `-7`

Interpretation:

- this is **not** purely “missing form layer” anymore
- we already have the math form/meaning baseline loaded well enough to route
- the current issue is likely a mix of:
  - insufficient **higher-level math knowledge**
  - insufficient **problem-type coverage**
  - weak **solver/program selection / validation**
  - answer-normalization bias and collapse around a few frequent templates

So for math, the next step is **not only more form aliases**. It is:

- broader math knowledge ingestion
- plus later reasoning-program refinement

---

## 5. Recommended Ingestion Priority

This is the important part for Claude.

### Priority 1 — HS Natural + Earth/Space/Environmental Sciences

Why first:

- MMLU and LHE both show clear weakness in:
  - biology
  - chemistry
  - physics
  - astronomy
  - medicine-adjacent / life-science reasoning

Suggested inputs:

- `TEMP/KIMI_HS_NATURAL_SCIENCES_PHYS_CHEM_BIO_2026-04-13.md`
- `TEMP/KIMI_HS_EARTH_SPACE_ENVIRONMENTAL_2026-04-13.md`

Expected impact:

- MMLU science subjects
- LHE science/engineering domains
- some factual support for math word-problem grounding

### Priority 2 — HS History / Geography / Civics / Economics

Why second:

- current misses include:
  - government and politics
  - macroeconomics
  - microeconomics
  - history
  - global facts

Suggested input:

- `TEMP/KIMI_HS_HISTORY_GEOGRAPHY_CIVICS_ECONOMICS_2026-04-13.md`

Expected impact:

- MMLU humanities/social-science breadth
- LHE policy/economics/civic reasoning

### Priority 3 — HS Applied CS / Health / Psychology / Sociology

Why third:

- current misses include:
  - high school computer science
  - clinical knowledge
  - psychology
  - sociology-adjacent question types

Suggested input:

- `TEMP/KIMI_HS_APPLIED_CS_HEALTH_PSYCH_SOCIOLOGY_2026-04-13.md`

Expected impact:

- MMLU applied/health/social domains
- LHE computer science / medicine / human systems

### Priority 4 — HS Humanities + Philosophy + Ethics

Why fourth:

- needed for broader language-question understanding and higher-level conceptual framing
- likely helps LHE + MMLU more than current math

Suggested input:

- `TEMP/KIMI_HS_HUMANITIES_LIT_PHIL_RELIGION_ARTS_2026-04-13.md`

### Priority 5 — HS Languages + Linguistics

Why fifth:

- still important, but current failure profile is more domain-knowledge than surface-language scarcity
- useful for robustness, multilingual form layer, and linguistic reasoning

Suggested input:

- `TEMP/KIMI_HS_LANGUAGES_LINGUISTICS_2026-04-13.md`

### Priority 6 — Cross-Cultural Glue

Why later:

- valuable for casuistry / meta-linguistic reasoning
- lower immediate benchmark impact than the missing science/civics/applied corpus

Suggested input:

- `TEMP/KIMI_HS_CROSSCULTURAL_SAUDADES_CALENDAR_EXAMS_PROVERBS_2026-04-13.md`

### Separate Track — ARC Reasoning Primitives

This should run **in parallel conceptually** with the curriculum waves, because
ARC is not going to recover from HS curriculum alone.

Suggested input:

- `TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md`

---

## 6. Important Caution For Claude

The correct conclusion is:

> **Wiring success does not equal knowledge success.**

We have now crossed the line where transport bugs are no longer the main excuse.

This run says:

- the live game path is real
- the tablet/WINE translator path is real
- the GPU path is real
- the current knowledge + reasoning substrate is still not good enough

So the next specs should focus on **what to ingest next** and only secondarily on
new transport work.

For math specifically, Claude should avoid assuming:

> “more aliases / more form-layer only” will solve it.

The data suggests we need:

- more advanced math knowledge
- richer theorem/identity/problem families
- and later a dedicated reasoning-program refinement pass

---

## 7. Concrete Next Step I Recommend

If Claude wants the highest-yield next spec after this report:

1. **Batch 11-style ingestion spec for Natural + Earth/Space sciences**
2. then **History/Civics/Economics**
3. then **Applied CS/Health/Psych/Sociology**
4. in parallel, a separate **ARC primitives ingestion spec**

If the goal is immediate benchmark improvement:

- **MMLU/LHE improvement path** = remaining HS curriculum waves
- **ARC improvement path** = ARC primitive/program substrate
- **competition math improvement path** = advanced math corpus + later solver refinement

---

## 8. Bottom Line

This 50-slice run is valuable because it removes ambiguity.

We now know:

- the system is **alive on the intended path**
- the benchmark route is **not fake**
- the knowledge loaded so far is **insufficient**

So the next phase should be **knowledge expansion**, not more launcher surgery.

