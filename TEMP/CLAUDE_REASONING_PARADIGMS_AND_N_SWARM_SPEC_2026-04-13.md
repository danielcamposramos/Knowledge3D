# Claude Spec — Reasoning Paradigms Mapping + N-Scalable Internal Swarm

**Date:** 2026-04-13
**Author:** Claude (architecture partner)
**Status:** Spec for Codex — DO NOT implement without Daniel sign-off
**Role reminder:** Claude writes specs. Codex implements.

---

## 0. Context & Scope

Investigation of the reasoning-paradigm survey in
`TEMP/Claude and Kimi_Swarm investigation and gathering - 04.13.2026.md`
(lines ~400 → EOF), discarding topics already covered by the prior
`KIMI_KNOWLEDGE_*` outputs (automated reasoning baseline, heuristics,
metaheuristics, AML/solvers, extensions).

Remaining paradigms mapped here:

1. **Semantic reasoners / rule engines** (Cyc, KAON2, Cwm, Drools, Evrete,
   EYE, D3web, Flora-2, Jena, NRules, Prova, DIP/defeasible, S-LOR).
2. **Case-Based Reasoning** (retrieve / reuse / revise / retain).
3. **Casuistry** (case-based ethical / analogical reasoning).
4. **Abductive reasoning** (Peirce IBE, set-cover, ALP, subjective-logic,
   bi-abduction / Infer).
5. **Automated Theorem Proving** (resolution, superposition, tableaux,
   DPLL/SMT, model checking, ATP systems).
6. **N-scalable internal swarm** — the final directive at EOF
   ("spawn N times within hardware constraints, starting from the base").

Deep research outputs live in sibling files (do not duplicate here):

- `TEMP/KIMI_REASONING_SEMANTIC_CBR_CASUISTRY_2026-04-13.md`
- `TEMP/KIMI_REASONING_ABDUCTION_ATP_2026-04-13.md`
- `TEMP/KIMI_N_SCALABLE_INTERNAL_SWARM_2026-04-13.md`

This spec is the **architectural synthesis** + **Codex hand-off**.

---

## 1. Unifying Principle

All six paradigms collapse to the same K3D primitive stack:

> **Galaxy (VRAM graph) + RPN program + PTX micro-kernel + TRM tick +
> Halting gate.**

They differ only in:

- **What is retrieved** from Galaxy (rules, cases, hypotheses, clauses).
- **How candidates are composed** in RPN.
- **When the halting gate fires** (coverage / proof / saturation / timeout).

There is **no new runtime system** to introduce. We extend the existing
composed-head pipeline with (a) a handful of new ternary RPN opcodes and
(b) small PTX micro-kernels reusing current infrastructure.

---

## 2. Paradigm → K3D Mapping Summary

Condensed table. Full derivations in the KIMI_* companion files.

| Paradigm | Retrieve | Compose (RPN) | Halt | New opcodes | New kernel? |
|---|---|---|---|---|---|
| Rete / Drools / NRules | Galaxy rule-cluster (alpha/beta index) | `RETE_ALPHA_TEST` → `RETE_BETA_JOIN` → `AGENDA_INSERT` | agenda empty | 3 | reuse existing match kernels |
| Cyc (microtheory) | Galaxy cluster tagged by context-id | `CTX_SWITCH` → Rete chain | same as Rete | 1 | `cyc_heuristic_filter.ptx` (thin) |
| KAON2 (DL tableaux) | Word/Grammar symlink DAG | `DL_SATURATE`, `BLOCKING_CHECK` | clash or saturation | 2 | `dl_tableau_expand.ptx` |
| Cwm / EYE (N3 / Euler paths) | RDF-style Galaxy | `EULER_COMPLETE` | transitive closure done | 1 | `euler_path_closure.ptx` |
| DIP / defeasible | already sovereign | existing | existing | 0 | reuse `gre_defeasible_resolver.cu` (3 stages) |
| S-LOR (IoT rules) | Reality Galaxy sensor subdomain | reuse Rete opcodes | n/a | 0 | reuse |
| CBR | Galaxy neighborhood via Morton + LED-A* + frustum | `CASE_FETCH` → `CASE_REBIND` → nine-chain critique → shadow append | halting gate convergence | 4 | reuse composed-head + shadow-append |
| Casuistry | CBR pathway filtered by `ethical_sig` mask | same as CBR, projected across Reality Galaxy ethical subdomain | defeasible stage 3 | 0 (reuses CBR + defeasible) | 0 |
| Peirce abduction | Hypothesis Galaxy by observation key | `ABDUCE` → `EXPLAIN` → `SUSPECT` | coverage + simplicity | 3 | `abduce_peirce_kernel` |
| Set-cover abduction | ternary-masked hypothesis subset | `SCUNION` (warp popcount union) | uncovered-mask = 0 | 1 | `set_cover.cu` |
| Abductive LP | backward-chain over Horn Galaxy | `ALPCHAIN`, `ICHECK`, `ABDRES`, `ABDNEG` | goal resolved ∧ IC satisfied | 4 | `alp_solver.cu` |
| Subjective-logic abduction | opinions stored as TQUANT triples | `EBELIEF` (Bayes inverse) | uncertainty below threshold | 1 | extends ternary_quant |
| Bi-abduction (Infer-style) | Reality Galaxy pre/post frames | `BIDUCE`, `FRAME` | frame inferred | 2 | `frame_infer.cu` (new) |
| Resolution + unification (ATP) | discrimination tree in Galaxy | `TUNIFY`, `TRESOLVE`, `TSUBSUME` | empty clause | 3 | `resolution.cu` |
| Superposition / rewriting | Math Galaxy rewrite rule set | `TORDER` (KBO), `TSUPERPOS`, `TREWRITE` | confluence | 3 | `superpos.cu` |
| Tableaux | branch structs Morton-indexed | `TSPLIT`, `TCLOSE`, `TEXPAND` | all branches closed | 3 | fits natively on swarm |
| DPLL / SMT | SAT-branch kernel + Galaxy T-lemmas | `TBCP`, `TLEARNT`, reuse `TUNIFY` | ⊥ or satisfying assignment | 2 | `dpll_sat.cu` |
| Model checking | frustum-culled state graph | reuse LED-A* + Morton | property held / violated | 0 | reuse composed-head |

Total net-new RPN opcodes across the whole bundle: **~32**, all fitting
into the Extended tier (0xA0–0xD4 block). Total net-new PTX kernels:
**~10**, all thin (< 400 lines each), most reusing existing unification,
Morton index, LED-A*, frustum, and nine-chain infrastructure.

---

## 3. Galaxy Schema Additions

Consolidated — Codex should land these as one ingestion patch.

```
Hypothesis Galaxy        (new)  — AbductiveNode {h_id, effects, simplicity, status}
Frame Galaxy             (new)  — separation-logic pre/post for Reality Galaxy
Microtheory index         (new)  — Cyc-style context-id → cluster-root
Ethical subdomain mask   (tag)  — bitmask on Reality Galaxy stars for casuistry
Rule-cluster tag         (tag)  — alpha/beta Rete memories (VRAM region id)
Tableau heap             (tmp)  — per-tick scratch pool in VRAM arena
```

All additions are **ingestion-time** Galaxy writes. Hot-path only reads.

---

## 4. Opcode Block — Unified Map (Extended Tier)

Reserved range: **0xA0 – 0xD4**. All ternary-packed. Do NOT collide with
existing TADD/TMUL/TNOT/TCOMP/TQUANT/TPACK/TUNPACK (0x70–0x76).

```
0xA0 ABDUCE          0xA4 SCUNION
0xA1 EXPLAIN         0xA5 ICHECK
0xA2 SUSPECT         0xA6 ABDRES
0xA3 ABDUCE_HALT     0xA7 ABDNEG

0xB0 EBELIEF         0xB3 EULER_COMPLETE
0xB1 BIDUCE          0xB4 DL_SATURATE
0xB2 FRAME           0xB5 BLOCKING_CHECK
                     0xB6 CTX_SWITCH
                     0xB7 ALPCHAIN

0xC0 TUNIFY          0xC3 TSUBSUME
0xC1 TRESOLVE        0xC4 TSUPERPOS
0xC2 TORDER          0xC5 TREWRITE

0xD0 TSPLIT          0xD3 TBCP
0xD1 TCLOSE          0xD4 TLEARNT
0xD2 TEXPAND

0xE0 RETE_ALPHA_TEST
0xE1 RETE_BETA_JOIN
0xE2 AGENDA_INSERT

0xF0 HALT_SET        0xF1 HALT_SYNC      (system)
```

Codex: please update `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`
with these reservations **before** any kernel lands, so the registry
remains the authority.

---

## 5. N-Scalable Internal Swarm — Architectural Spec

### 5.1 Principle

Nine-Chain Swarm becomes **N-Chain Swarm**, where `N ∈ [9, N_max(hw)]`
is chosen **per TRM tick** from free VRAM and deadline budget.
Consistent with `feedback_resource_aware_dispatch`: scale by **free GPU**,
not by question complexity. The base **9** is the floor, never reduced
below except in the Tier-3 degradation case.

### 5.2 Per-lane cost budget (baseline)

Derived from current 132 MiB / 9 lanes:

- TPACK belief stack:        3.5 MiB   (shared, bank-interleaved)
- LoRA specialist (A+B r=64): 8.2 MiB  (global atlas + local scratch)
- RPN registers + scratch:    2.0 MiB
- Cross-ref bitmap:           1.0 MiB
- **Total per lane:          14.68 MiB**

With 12 GB VRAM and 10% driver headroom → **N_max ≈ 735–755** lanes.
Baseline 9 uses < 2 % capacity. Massive headroom is real, not hypothetical.

### 5.3 N selection formula (per tick)

```
N_vram     = floor( (VRAM_free * 0.90) / 14.68 MiB )
N_entropy  = N_vram * (1 + H_belief / H_max)            # diversity boost
N_deadline = floor( T_remaining / T_per_lane )          # T_per_lane ≈ 45–50 μs
N_cand     = frustum_culled_candidate_count             # don't exceed candidates

N_tick     = clamp( min(N_entropy, N_deadline, N_cand), 9, N_hard_max )
```

`N_hard_max = 1024` to keep lane index in 10 bits for atomics.

**N is fixed intra-tick** (barrier stability, bar.sync count cannot change
mid-kernel). **Elastic inter-tick**: re-evaluated at the tick boundary
(`grid.sync` → re-read host-mapped `d_n_active`).

### 5.4 Launch pattern — persistent cooperative kernel

Reject CUDA Dynamic Parallelism: non-deterministic latency and runtime
malloc violate sovereignty. Adopt a single **persistent cooperative
kernel** (`k3d_swarm_sovereign`) launched once with
`cuLaunchCooperativeKernel`.

- Grid over-provisioned to `N_hard_max` physical lanes.
- Each tick, lanes with `phys_lane_id < N_tick` execute;
  `phys_lane_id ≥ N_tick` fall straight to the barrier (virtual masking).
- Outer `while` loop until `halting_gate = 1`.
- Per-block: 128 threads = 4 warps. 256 blocks → 1024 max lanes.
- Shared memory: 15 KB/block, bank-interleaved so lane `j`'s stack is
  readable by lane `i` (the cross-reference substrate).

### 5.5 Halting gate — tree reduction with atomics

Lane-local `HLT` via RPN sets a `p_halt_local` predicate. Active lanes
atomically increment `g_halting_counter`; lane 0 sets the global
`halting_gate` when the counter reaches `N_tick`. `bar.sync 0` ensures
visibility. No cooperative-groups runtime dependency — pure atomics + bar.

### 5.6 Specialist (LoRA) attachment per lane

- Galaxy Atlas: read-only global memory, `[specialist_id][layer][A|B]`.
- Lane loads its LoRA pair via `ld.global.nc` (non-coherent cache) to
  minimise bank pressure.
- **Streaming**: weights are fetched on demand, not pre-resident, so
  spawning more lanes doesn't multiply the resident weight footprint.
- Hyper-parallel topology (per memory):
  - N lanes      = `gridDim.x * warps_per_block`
  - RPN cores   = warps (SIMT)
  - Weights     = texture/LoRA atlas via `tex.1d.v4.f32`
  - Stacks      = shared-mem banks 0–15 with bitmask interleave

### 5.7 Shadow-copy performance logging (runs = training)

Each lane writes a `LanePerf` struct at tick end into a 1 M-entry ring
buffer in device memory:

```
struct LanePerf {
  u32 n_active;       // N this tick
  u32 entropy_input;
  f32 belief_delta;   // L2 norm of belief update
  u32 cycles_consumed;
  u8  specialist_id;
};
```

Utility metric: `belief_delta / cycles_consumed`. Sleep-time consolidator
reads the ring, computes moving averages of `utility[N]`, and:

- If `avg_utility[N+1] > 1.15 * avg_utility[N]` → raise `N_base`.
- If VRAM page faults observed → reduce `N_max` by 10%.

This is the **runs = training** loop from the feedback memory, now
parameterised by N.

### 5.8 Three-tier degradation when GPU saturates

- **Tier 1 — Weight sparsification** (VRAM > 90 %):
  Drop LRU LoRA B matrices, reconstruct `A · 0` (higher compute, stable N).
- **Tier 2 — Lane consolidation** (deadline miss imminent):
  Force `N = 9` (baseline). Queue excess frustum candidates to next tick.
- **Tier 3 — Host fallback**: `trap;` with `0xDEAD_N`, CPU 9-chain swarm
  engaged, sleep-time tuner permanently reduces `N_max` by 20%.

Tier 3 is the only place a host fallback exists, and it is a **failure
signal** (logged, scored, used by sleep-time), not a "mock" — consistent
with the "we fix or we fix" rule: the CPU path is recovery instrumentation,
not a steady-state option.

---

## 6. Sovereignty Checklist (spans all paradigms + N-swarm)

- [ ] No numpy / cupy / scipy / sympy anywhere in the hot path.
- [ ] No Python regex / string ops inside RPN execution.
- [ ] All new opcodes compile to ternary-packed PTX, dispatched by the
      existing TRM tick — no new Python orchestrator.
- [ ] Galaxy writes only at ingestion or sleep-time (shadow-append ring
      for runtime logging is the sole exception; the consolidate pass
      runs under sleep).
- [ ] `gre_defeasible_resolver.cu` remains the single defeasible kernel
      reused across 3 pipeline stages (no duplicates).
- [ ] Persistent kernel, not CUDA Dynamic Parallelism.
- [ ] N selection consumes free VRAM, entropy, deadline, candidate count
      — never question text / complexity heuristics.

---

## 7. Codex Hand-off — Suggested Landing Order

Each slice is independently testable; no big-bang.

1. **Opcode registry update** — reserve 0xA0–0xD4 in
   `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`.
2. **Persistent N-chain kernel shell** — over-provisioned, virtual-mask
   active lanes, barrier + halting gate. Ship with `N=9` only; all
   infrastructure in place for dynamic N.
3. **Shadow-copy perf ring** — `LanePerf` struct + ring writer + a
   sleep-time consumer that logs `utility[N]`.
4. **Dynamic N selector** — host-mapped volatile `d_n_active` + formula
   (§5.3). Gate behind a sovereign flag; benchmarks must stay ≥ baseline.
5. **Tableaux opcodes** (`TSPLIT`, `TCLOSE`, `TEXPAND`) — tableaux fits
   the swarm natively, so this is the cheapest ATP addition with LHE
   multi-hop payoff.
6. **Resolution + unification** (`TUNIFY`, `TRESOLVE`, `TSUBSUME`) —
   shares the Galaxy discrimination tree; unblocks Math and LHE.
7. **Abductive opcodes** (`ABDUCE`/`EXPLAIN`/`SUSPECT`/`SCUNION`/`ICHECK`)
   — unblocks MMLU hypothesis exploration once Galaxy coverage is up.
8. **Bi-abduction** (`BIDUCE`, `FRAME`) — for Reality Galaxy physics
   pre/post. Last because it depends on Frame Galaxy ingestion.
9. **Superposition + DL tableaux + Euler paths + Microtheory context** —
   incremental, per-benchmark need.
10. **Rete tag for Drools-style rules** — reuse the same alpha/beta match
    kernels already in the 46; only new piece is the agenda priority
    queue on GPU (bitonic sort).

### Tests Codex must ship with each slice

- Sovereignty grep: no `numpy|cupy|scipy|sympy|re\.` in modified code.
- Benchmark non-regression: ARC 10/10, Math 20/20 unchanged.
- New: **N-sweep test** — run N ∈ {9, 18, 36, 72} on the existing
  benchmark set; record `LanePerf` ring; assert utility curve
  non-decreasing up to the current hardware plateau.

---

## 8. Architectural Decisions (Daniel, 2026-04-13)

1. **Cross-reference topology — FULL.** LOD and FOV are the only cuts;
   every lane sees every other lane's summary. The world of knowledge is
   linked; no artificial locality gate in the swarm.
2. **Ethical mask — ternary.** Values: `ok / defeasible / forbidden`,
   packed into a single trit per star. `gre_defeasible_resolver.cu`
   reads the trit directly at stage 3; `forbidden` short-circuits with
   no further search, `defeasible` enters conflict-resolution, `ok`
   passes through.
3. **Sleep-time N = max hardware.** Sleep runs only on idle or on explicit
   command, so maximum throughput is the right default. Consolidation,
   re-embedding, symlink materialisation, and contrastive updates all
   run at `N = N_max_physical`.
4. **Microtheory / context = star metadata, single source of truth.**
   No new `MT_INDEX` galaxy. Add a new metadata field on the star itself:
   `context_id : u32` (0 = universal / global; non-zero = scoped).
   Knowledgeverse remains the single source of truth; context is a
   filter, not a separate index.

### 8.1 Derived adjustments to §2–§4

- Drop proposed `MT_INDEX` from §3 (Galaxy Schema Additions).
- Rename opcode `CTX_SWITCH` → **operates on star metadata**, not on a
  separate microtheory index. Semantics: "filter subsequent Galaxy reads
  to stars whose `context_id` matches top-of-stack, or `0`."
- Add star-schema field `context_id : u32` to the canonical star struct
  (Codex: land this as an ingestion-time migration, not a hot-path
  change).
- Ethical trit: add star-schema field `ethical_trit : u2` (ternary
  {ok=0, defeasible=1, forbidden=-1}). `gre_defeasible_resolver.cu`
  gains a `stage3_ethical_gate` that consults this trit.
- N-swarm §5.3 unchanged for runtime; sleep-time scheduler is now
  authorised to use `N_max_physical` directly (skip §5.3's entropy /
  deadline clamps during sleep).

---

## 9. Prior Research Inventory (pre-compaction swarms)

These 14 kimi_swarm outputs in `TEMP/` landed before the current session
and must be integrated with §2–§7. They are **ingestion-plan artefacts**:
meaning-star catalogues with `star_id`, `is_a` parent, RPN sketch, and
symlink lists, ready to be fed through the canonical registry pipeline.

### 9.1 Reasoning-infrastructure plans (automated reasoning tier)

| File | Scope |
|---|---|
| `KIMI_KNOWLEDGE_AML_AND_SOLVERS_2026-04-13.md` | Algebraic Modeling Languages (AMPL, GAMS, Pyomo, JuMP, MiniZinc, Mosel, OPL, Zimpl) + solvers (CPLEX, Gurobi, HiGHS, GLPK, CBC, SCIP, IPOPT, Bonmin, Couenne, BARON, Knitro, LocalSolver, Z3/MILP). Meaning-star plan. |
| `KIMI_KNOWLEDGE_HEURISTICS_AND_METAHEURISTICS_2026-04-13.md` | Heuristics, metaheuristics, matheuristics: GA/ES/EP, SA, TS, ACO, PSO, DE, CMA-ES, VNS, GRASP, ILS, Memetic, Scatter Search, Path Relinking, Bee/Firefly/Bat/Whale/Grey-Wolf, Harmony Search, Cuckoo, Hyper-heuristics, matheuristics families. |
| `KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md` | Automated reasoning + reasoning systems baseline: propositional + FOL, SAT/SMT, resolution, tableaux, natural deduction, sequent calculus, modal/temporal/dynamic/deontic/epistemic logics, description logics, non-monotonic, defeasible, probabilistic. |
| `KIMI_KNOWLEDGE_EXTENSION_AML_HEURISTICS_REASONING_2026-04-13.md` | Appendix: every remaining variant/sub-discipline across the three above. Filling gaps the first three didn't cover. |

### 9.2 High-school knowledge catalogues (meaning-star plans)

| File | Scope |
|---|---|
| `KIMI_MATH_HS_CLUSTER1_ARITHMETIC_ALGEBRA_2026-04-13.md` | Arithmetic, number theory, pre-algebra, elementary + intermediate algebra, sequences & series. All formulae/identities as meaning stars. |
| `KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md` | Plane + solid geometry, trigonometry, analytic/coordinate geometry, HS vectors, transformations. |
| `KIMI_MATH_HS_CLUSTER3_STATS_DISCRETE_APPLIED_2026-04-13.md` | Pre-calculus (limits intro, continuity intro), statistics, probability, combinatorics, discrete math, applied HS topics. |
| `KIMI_HS_NATURAL_SCIENCES_PHYS_CHEM_BIO_2026-04-13.md` | HS physics + chemistry + biology at full world-curriculum depth (IB, A-Levels, AP, ENEM, Abitur, Bac, Gaokao, CBSE, JEE). |
| `KIMI_HS_EARTH_SPACE_ENVIRONMENTAL_2026-04-13.md` | Geology, astronomy, meteorology, oceanography, climatology, environmental science. |
| `KIMI_HS_HISTORY_GEOGRAPHY_CIVICS_ECONOMICS_2026-04-13.md` | History + geography + civics/government + economics, world curriculum, multi-regional. |
| `KIMI_HS_HUMANITIES_LIT_PHIL_RELIGION_ARTS_2026-04-13.md` | World literature, philosophy/ethics/logic, religious studies/comparative religion, arts (visual / music / drama / film). |
| `KIMI_HS_LANGUAGES_LINGUISTICS_2026-04-13.md` | World language families, grammar systems across major languages, phonology, morphology, syntax, semantics, pragmatics. |
| `KIMI_HS_APPLIED_CS_HEALTH_PSYCH_SOCIOLOGY_2026-04-13.md` | HS Computer Science, health / nutrition / PE, psychology + sociology + anthropology, study skills / critical thinking. |
| `KIMI_HS_CROSSCULTURAL_SAUDADES_CALENDAR_EXAMS_PROVERBS_2026-04-13.md` | Cross-curricular glue: saudades catalog of untranslatables as first-class stars, world measurement systems, date/calendar systems, exam systems, proverbs. |

### 9.3 Integration with this spec

The reasoning-paradigm mapping in §2 **consumes** the §9.1 plans:

- §9.1 AML/solvers → feeds ATP and abductive-LP opcodes (§4) with the
  taxonomy of solver interfaces and MILP/MINLP/CP paradigms as stars;
  the solver/AML meaning stars are what `TUNIFY` / `ALPCHAIN` dispatch
  against.
- §9.1 heuristics/metaheuristics → feeds the N-swarm specialist roster:
  each metaheuristic is a candidate LoRA-style specialist attached to
  a lane (§5.6). The 9-lane baseline maps naturally to the 9 canonical
  metaheuristic families; additional N → additional variants.
- §9.1 automated reasoning → is the parent taxonomy for §2's 16
  paradigms; the paradigms in §2 are **stars** inside the catalogue
  produced by this plan.
- §9.1 extension → fills gaps in paradigm coverage; any variant that
  requires a new opcode must be appended to §4.

The §9.2 HS plans are the **Knowledgeverse population payload** that
will exercise the paradigms at query time:

- HS math clusters populate the Math Galaxy with the RPN templates the
  superposition/rewriting opcodes (`TORDER`, `TSUPERPOS`, `TREWRITE`)
  operate on.
- HS natural + earth/space sciences populate the Reality Galaxy with
  procedural systems that bi-abduction (`BIDUCE`, `FRAME`) frames
  pre/post conditions against.
- HS history / civics / economics populate a civic/historical subdomain
  of the Reality Galaxy; `context_id` (per §8) tags era / region.
- HS humanities (philosophy / ethics) populates the ethical subdomain;
  the ternary `ethical_trit` (per §8) is assigned at ingestion from the
  philosophy stars.
- HS languages populates the Word/Grammar Galaxy; multilingual symlinks
  (already spec'd in Phase 7) tie every meaning star to its language
  surface forms.
- HS applied / CS / health / psych populates Tool + Reality galaxies.
- Cross-cultural glue (saudades, calendars, units, proverbs) populates
  **meta-linguistic** stars — these are the cases where casuistry
  (§2) has real data to reason analogically over.

### 9.4 Codex ingestion order (prior plans)

Codex should ingest the §9 catalogues in this order, after the §4 opcode
registry update:

1. Reasoning taxonomy (§9.1: automated reasoning, then AML/solvers,
   then heuristics, then extension). This lights up §2's mapping.
2. HS math clusters 1 → 2 → 3 (already partly in Math Galaxy; this
   completes world-curriculum depth).
3. HS natural + earth/space sciences (Reality Galaxy).
4. HS languages + linguistics (Word/Grammar Galaxy, with §8
   `context_id` per region / era).
5. HS history + civics + economics (Reality Galaxy civic subdomain).
6. HS humanities + philosophy + ethics. **Ingestion must set
   `ethical_trit` per §8.**
7. HS applied / CS / health / psych / sociology.
8. Cross-cultural glue (saudades, calendars, units, proverbs) — casuistry
   substrate.

Each stage reuses the canonical-registry pipeline landed in Phase 7
(`knowledge3d/ingestion/canonical_lookup.py`,
 `scripts/ingest_canonical_to_qdrant.py`) with the new metadata fields
(`context_id`, `ethical_trit`) added to the ingestion schema.

---

## 10. Remaining Open Questions

None for this spec — all four prior questions resolved per §8.
Claude is ready to expand any §9 plan into a per-slice Codex spec when
you're ready.

---

## 9. What This Spec Does Not Do

- It does **not** implement anything. Codex implements.
- It does **not** re-derive the automated-reasoning baseline, AML, or
  heuristics — those live in the earlier `KIMI_KNOWLEDGE_*` outputs.
- It does **not** propose changes to the TRM game loop itself; N-swarm
  is a capacity dimension of the existing `trm_step_fused.ptx`.

---

**End of spec.**
