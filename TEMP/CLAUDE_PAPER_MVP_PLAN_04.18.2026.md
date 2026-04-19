# Paper MVP Plan — ARC Prize 2026 Paper Track (2026-04-18, v2)

> **⚠ SUPERSEDED BY:** [CLAUDE_PAPER_SERIES_AND_ATTRIBUTIONS_04.18.2026.md](CLAUDE_PAPER_SERIES_AND_ATTRIBUTIONS_04.18.2026.md)
> — Daniel directed a re-evaluation on 2026-04-18 after flagging that ternary logic, form→meaning, OSI-inspired layering, hyper-modular architecture, and semantic gravity are each standalone paper subjects. The successor spec narrows Paper A to C1/C2/C3 and schedules Papers B-F as a companion series. This v2 file is kept for reference only; do not execute §§3-11 without reading the successor first.

**Supersedes v1** (MLSys/NeurIPS-oriented). Retargeted to
[arcprize.org/competitions/2026/paper](https://arcprize.org/competitions/2026/paper)
+ [kaggle.com/competitions/arc-prize-2026-paper-track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track).

> **Thesis for this submission:**
> The pivot *out of Python orchestration* is not an optimization — it is what
> made K3D measurable. For six months we tuned navigation over a poisoned
> execution path; only after the April 2026 absolute-sovereignty purge did
> ARC-shaped failures become legible. That is the paper.

---

## 1. Competition facts (authoritative, from the Kaggle & ARC Prize pages)

| Field | Value |
|-------|-------|
| **Paper deadline** | November 8, 2026 |
| **Linked code deadline** | November 2, 2026 |
| **Results announced** | December 4, 2026 |
| **Top Paper pool** | **$75K guaranteed** — 1st $50K, 2nd $20K, 3rd $5K |
| **Outstanding Papers pool** | $375K discretionary (score ≥ 4.5 rubric avg) |
| **Total pool** | $450K |
| **Linkage rule** | Paper MUST link to a Kaggle code submission in ARC-AGI-2 *or* ARC-AGI-3. Code need not win; it must **exist** and demonstrate the approach. |
| **License** | Open-source (permissive: CC0 / MIT-0 per 2025 rules). |
| **Required sections** | Abstract · Intro · Prior work · Approach · Results · Conclusion. |
| **Format guidance (verbatim from page)** | *"Shorter and clearer is always better. No filler, no unnecessary equations."* No page limit stated. |

### 1.1 Official 2026 rubric (six axes, 0–5 each, straight avg)

| # | Criterion | Question judges ask | Our defensible score |
|---|-----------|---------------------|----------------------|
| 1 | **Accuracy** | How accurate is the submission on the leaderboard? | 2–3/5 (low-but-honest) |
| 2 | **Universality** | How general is the approach beyond this competition? | **5/5** |
| 3 | **Progress** | How much does the paper increase the overall chance of hitting 85% on ARC-AGI? | **5/5** |
| 4 | **Theory** | How well does the paper describe *why* the approach works? | **4–5/5** |
| 5 | **Completeness** | How thoroughly does the paper cover the submission? | 3–4/5 (bounded by ablation hours) |
| 6 | **Novelty** | How novel is the approach relative to existing public research? | **5/5** |

**Target average ≥ 4.3** — puts us inside the Outstanding Papers pool band
(≥ 4.5 is the stretch target) and competitive for the top-three tier.

### 1.2 Why accuracy is not the deciding axis

TRM won 1st place in the 2025 Paper Prize with ~8 % on ARC-AGI-2. SOAR
won 2nd. CompressARC won 3rd. All three had modest leaderboard numbers.
The rubric weights novelty + theory + progress-toward-85% as first-class,
and explicitly frames Accuracy as *one of six* not the gate.
**Low accuracy is survivable; unclear novelty or ornamental theory is not.**

---

## 2. Linked code track: ARC-AGI-3 (not ARC-AGI-2)

We anchor to the interactive track because:

- K3D completed ARC-AGI-3 *Level 1 live* on `ls20-9607627b` (13 actions,
  March 30 2026) — only K3D-class architecture known to have a live result.
- ARC-AGI-3 is the avatar/world-model benchmark; TRM-as-Avatar maps 1:1.
  ARC-AGI-2 is grid-transformation and rewards heavy test-time-training
  ensembles — a game we don't play.
- The 2026 rubric's *Universality* + *Progress* axes reward architectures
  that naturally extend from grid puzzles to interactive worlds. K3D does;
  retrofitted LLM-agent stacks don't.

The paper can *mention* ARC-AGI-2 numbers (curated 10/10, expanded 10/50)
as a control, but the linked Kaggle code submission is the ARC-3 agent.

---

## 3. Central claim (two paragraphs, memorized before prose begins)

Current ARC-reasoning systems converge on two categories: (a) large
language-model orchestrations that tokenize grids and refine via
chain-of-thought, and (b) TTT ensembles that fit per-task learners. Both
treat reasoning as a *Python orchestration problem over floating-point
tensors*, with numpy / PyTorch / kernel libraries mediating every step.
Our experience is that this substrate is what masked ARC failures: for
six months we tuned router precision and swarm wiring over a ring whose
middle still re-entered Python regex to rebuild query envelopes, and the
classifier stayed flat at 0.125 across all four task classes because its
signal had already been flattened by the Python-side feature extraction.

We pivoted. Across Phases 6.C → 7.6 (April 11–18 2026) we removed every
`numpy`/`cupy`/`scipy`/`sympy`/`torch` import from the hot path — not as
optimization but as an architectural declaration. The reasoning loop now
runs as **PTX kernels + RPN + Galaxy Universe VRAM + a 7M-parameter TRM
avatar**, dispatched through a ~300-line pure-`ctypes`/`libcuda.so`
launcher. Python retains boot, I/O, and display. Once the hot path was
sovereign, ARC-shaped errors became localisable, fixable, and — for
the first time — *legible*. That legibility, not the headline
accuracy number, is what this paper contributes to the ARC-AGI community.

---

## 4. Three contributions (the paper has exactly these — no more)

**C1. Sovereignty as a publishable system property.**
A grep-testable definition of the hot-path boundary (six banned names;
three-directory scope) + a commit-blocking preflight hook + a failure
invariant (zero silent fallback). Shows how `knowledgeverse.py` moved
from ~4 000 lines of Python orchestration in March 2026 toward ~200
lines in April 2026 (boot + I/O + display only), with every removed
line replaced by a PTX kernel call, not rewritten in Python.

**C2. TRM-as-Avatar as an ARC-AGI-3 native architecture.**
TRM (~7M params, two-layer SwiGLU MLP) is not a model we call — it
is the avatar entity. It runs as `trm_step_fused.ptx` (one game tick =
perceive → navigate → reason → decide → act → learn) inside a
unified Galaxy Universe VRAM workspace. Nine-chain swarm = parallel
cognitive channels, not worker pool. Live ARC-AGI-3 Level 1 completion
on `ls20-9607627b` is the existence proof.

**C3. The ActionBuffer 288-byte binary contract.**
A fixed struct (24 typed fields + trailing pad, aliased to 72 u32 words
for PTX writes) lets the host post queries and PTX kernels write
results in the same buffer without host-side deserialization. Enables
the ARC-3 action loop to live inside the GPU tick rather than a Python
event loop. Appendix prints the byte-accurate layout.

---

## 5. Section plan — exactly the six required sections, nothing extra

| § | Title | ≈ length | Owner | Why |
|---|-------|----------|-------|-----|
| 1 | Abstract + Introduction | 0.75 p | Sonnet | The whole argument in nuce. |
| 2 | Prior Work | 0.5 p | ollama `ask_cloud` (lit sweep) → Sonnet prose | TRM 2025, SOAR, CompressARC, NVARC TTT, BitNet b1.58, SPINdle defeasible logic, Method of Loci. |
| 3 | Approach — Sovereign Substrate | 3 p | Sonnet, long-form | §3.1 hot-path boundary (C1) · §3.2 Galaxy Universe · §3.3 TRM game loop (C2) · §3.4 ActionBuffer (C3) · §3.5 the Python-exit arc (March → April 2026, the pivot narrative). |
| 4 | Results | 1.5 p | Haiku (tables) + Sonnet (interpretation) | ARC-AGI-3 Level 1 live completion transcript · curated ARC-AGI-2 10/10 · math 20/20 · Phase-7.6 preflight clean on 197 hot-path files · ablation (Python-dispatch ON vs OFF). |
| 5 | Conclusion | 0.25 p | Sonnet | One paragraph; no future-work bullet list. |
| — | Appendices (electronic only) | — | mixed | A: 88-kernel inventory · B: ActionBuffer bytes · C: preflight reproducer · D: Kaggle notebook for ARC-3 agent. |

**Target body:** ~6 pages before appendices. The Kaggle page's explicit
guidance *"shorter and clearer is always better"* is a design constraint,
not a courtesy note. A ruthless first draft beats a padded polish.

---

## 6. The Python-exit arc (§3.5 — spine of the paper)

This is the crucial evidence Daniel asked about. These five dated
specs + one live benchmark form the timeline we cite:

| Date | Artifact | What it established |
|------|----------|---------------------|
| 2026-03-23 | `TEMP/CLAUDE_PHASE_D_TRM_GAME_LOOP_MIGRATION_SPEC_03.23.2026.md` | **Diagnosis.** GPU utilization ~2 %. `knowledgeverse.py` = ~4 000 lines of Python orchestration doing the job TRM should do. Target: ~200 lines. |
| 2026-04-11 | Phase 6.C / `docs/reports/SOVEREIGN_MATH_ANSWER_2026-04-11.md` + qdrant entry | **First sovereign GPU math answer.** `"2+3?" → 5` with zero Python arithmetic. Proof of concept. |
| 2026-04-17 | `TEMP/CLAUDE_KILL_PYTHON_DISPATCH_04.18.2026.md` (authored 04-17) | **The one cut.** Round B empirical proof: classifier flat at 0.125 across MULTI_HOP/NUMERIC/FACTUAL/ARC; one CPU core pinned; `nvidia-smi util = 0`. Python dispatch **is** the problem, not a line item. |
| 2026-04-18 | `TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md` | **Daniel's absolute ruling.** Every `numpy`/`cupy`/`torch` site on the hot path moves to `Old_Attempts/`. Not delta-scoped; absolute. Gate 0C grep must return **zero**. |
| 2026-04-18 | `TEMP/CLAUDE_PHASE_7_6_LIVE_SERVER_PURGE_SPEC_04.18.2026.md` | **Final carve-out lift.** 17 numpy+torch sites in `bridge/live_server.py` purged across 6 batches; preflight `live_server.py` carve-out removed; full-tree scan (197 files) clean. |
| 2026-04-18 | Math round-trip reproducer `/tmp/verify_math_roundtrip.py` | **Survival check.** After full purge: `7*6?` → PTX writes `42` into words[61] of ActionBuffer → host decodes via `ctypes.c_int32.from_buffer`. Sovereign path still answers correctly. |

**Why this belongs in the paper, not the appendix:** every other ARC
paper describes *what they added*. We describe *what we subtracted*, and
the subtraction is what changed the measurement. It is the specific
thing no 2025 winner did and no 2026 competitor (visible in public
Kaggle notebooks) has done. A reviewer who wants to score us 5/5 on
Novelty or Progress needs this specific narrative to cite.

---

## 7. Labor routing — agent-by-agent, per section

**Axis:** Claude Sonnet = argumentation & judgment. Claude Haiku =
mechanical polish. `ollama-specialists` = delegated heavy thinking.
MCP qdrant = zero-copy spec retrieval.

| Agent | Where they shine in this paper | Rationale |
|-------|--------------------------------|-----------|
| **Claude Sonnet (me)** | §1 Intro · §3 Approach (all five subsections) · §5 Conclusion · all rubric-facing judgment calls (theory clarity, novelty framing). | Long-form coherent prose; respects K3D vocabulary; the rubric's Theory axis rewards careful prose that other agents flatten. |
| **Claude Haiku (sub-agent)** | §4 result tables · ActionBuffer byte table (Appendix B) · reference-list formatting · 88-kernel inventory table (Appendix A) · notation-consistency pass in Week 4. | Mechanical polish is cheap per-token and parallelizable; the rubric's Completeness axis rewards hygiene Haiku produces reliably. |
| `mcp__ollama-specialists__ask_cloud` (default deepseek-v3.2, escalate to qwen3.5:397b-cloud or kimi-k2.5 for architecture-depth) | §2 Prior Work lit sweep · §3.5 similar-system comparison (was anyone else sovereign?) · sanity-check on novelty claim. | Web-enabled; cloud-planner class. Prevents us claiming novelty that someone already published. |
| `mcp__ollama-specialists__kimi_swarm` (think=True) | Two-angle critique passes: Week-1 "strongest/weakest sovereignty claim"; Week-4 "correctness/clarity review." | Native parallel sub-agent pattern; produces A/B views Sonnet integrates. |
| `mcp__ollama-specialists__plan_task` | §4.5 ablation plan (Python-dispatch ON vs OFF reproducer). | Structured implementation plans for Codex. |
| `mcp__ollama-specialists__extract_facts` | Mine `docs/ROADMAP.md`, `TEMP/*COMPLETE*.md`, commit messages for numeric claims. | Prevents re-typing benchmark numbers incorrectly (rubric Completeness). |
| `mcp__ollama-specialists__summarize` | Compress spec files into paper paragraphs (e.g., THREE_BRAIN_SYSTEM → 2-sentence Background). | Keeps the main context clean. |
| `mcp__ollama-specialists__web_search` | Primary-source URLs for Prior Work citations (TRM arXiv, SOAR, CompressARC, BitNet b1.58). | DuckDuckGo; no API-key drama. |
| `mcp__k3d-knowledge__qdrant-find` | All vocabulary-spec excerpts without reading files. | Paragraph-shaped results; already embedded; saves tokens. |
| `mcp__k3d-ptx__qdrant-find` + `qdrant-store` | PTX ISA citations (§3.3 footnotes) + retrieval of the 5 sovereign patterns we deposited 2026-04-18. | Authoritative PTX references + our own crafted patterns. |

**Rule of thumb:** mechanical → Haiku or ollama. Argumentative or
synthetic → Sonnet inline. Parallel perspectives → `kimi_swarm`.

---

## 8. Timeline — 29 weeks to Nov 8; three campaigns

### Campaign A — Paper skeleton (weeks 17–20 = April 21 → May 18)

*Goal: arXiv v1 preprint by end of May. Gives us a citeable timestamp
and a draft that survives 24 weeks of review without a scramble.*

- **Wk 17** (Sonnet inline): lock title + abstract + C1/C2/C3. Draft §1
  Intro + §3.1 sovereignty definition + §3.5 Python-exit arc. These
  three subsections are the paper's argument.
- **Wk 18** (delegated): `ask_cloud` + `web_search` assemble the §2
  Prior Work bibliography. `kimi_swarm` runs the strongest/weakest
  sovereignty-claim dual pass; I integrate.
- **Wk 19** (Sonnet): §3.2 Galaxy Universe + §3.3 TRM game loop + §3.4
  ActionBuffer. Haiku produces the kernel-inventory table (Appendix A)
  and the 288-byte layout (Appendix B) in parallel.
- **Wk 20** (Sonnet + Haiku): §4 Results skeleton — Haiku formats the
  curated numbers, I write the interpretation. Ship arXiv v1 draft.

### Campaign B — Ablation + code-track lockdown (weeks 21–28 = May 19 → July 13)

*Goal: June 30 ARC-AGI-3 Milestone-1 open-source submission + ablation
results in the paper before the September milestone.*

- **Wk 21–22**: Codex ships `arc3_kaggle_agent.py` + `arc3_action_bridge.py`
  per the June-30 R0 plan in `TEMP/ARC_PRIZE_2026_COMPETITIVE_ASSESSMENT.md`
  §R0. Paper-side: I draft §4.3 "Ablation — Python dispatch ON vs OFF"
  from the Round-B evidence + the post-purge round-trip reproducer.
- **Wk 23–25**: Codex tightens ARC-3 agent against the preview
  environment set (`ls20`, `ft09`, `vc33`). I update §4 results with
  action-count and efficiency numbers as they land.
- **Wk 26–27**: `kimi_swarm` correctness-review pass. Haiku reference
  hygiene pass. `extract_facts` re-mines commit messages for any
  numbers that drifted.
- **Wk 28**: arXiv v2 with ablation. Freeze §3 prose.

### Campaign C — Revision + competition packaging (weeks 29–42 = July 14 → Nov 8)

*Goal: the Nov-2 Kaggle code submission, the Nov-8 Paper Writeup, an
MIT-0 GitHub release tag, and a 90-second demo video.*

- **Wk 29–34**: Codex runs Phase R2 (cross-level memory, world-model
  kernel). I fold any result-shifts into §4; otherwise no new prose.
- **Wk 35–38**: full-paper read-through; `kimi_swarm` A/B correctness +
  clarity pass. Haiku final notation-consistency pass.
- **Wk 39**: record demo video (math round-trip + ARC-3 live +
  preflight passing).
- **Wk 40**: MIT-0 release tag `paper-v1.0` on GitHub; Kaggle code
  submission live on ARC-AGI-3 track.
- **Wk 41 (Oct 26 – Nov 1)**: slack week. **No scope creep.**
- **Wk 42 (Nov 2 – Nov 8)**: Kaggle Writeup live; paper PDF attached.

---

## 9. Risk register (four reviewer attacks, pre-empted)

| Risk | Mitigation |
|------|-----------|
| "Accuracy 2–3/5 makes Outstanding-pool unreachable." | The 2026 rubric is a straight average. 5/5 on three axes (Novelty, Universality, Progress) + 4/5 on Theory + 3/5 on Accuracy + 3/5 on Completeness = 3.83, which is under the 4.5 stretch but well inside the top-three band ($50K/$20K/$5K). TRM won 1st at ~8 %. Anchor to that datum publicly and in the rebuttal. |
| "Sovereignty is a buzzword — show me the boundary." | §3.1 presents the exact grep pattern (`^[[:space:]]*(import\|from)[[:space:]]+(numpy\|cupy\|scipy\|sympy\|torch)([[:space:]]\|$\|\.)`) and the three-directory scope. The preflight script is reproducible in ~30 lines. Appendix C ships it. |
| "7M-param TRM is too small to be interesting." | We are *not* claiming modeling SOTA. The contribution is an *existence proof of a sovereign substrate under which 7M still reasons*. Lead paragraphs of §1 and §5 state this explicitly. This framing won TRM $50K in 2025. |
| "Python-exit is a hygiene story, not a research contribution." | §3.5 cites the classifier-flat-at-0.125 Round-B evidence. Before purge: no signal. After purge: signal legible, ablation measurable. That is a research result about what the Python orchestration layer was *hiding*, not about what it was costing. Progress and Theory axes both reward this. |

---

## 10. Non-goals (enforced at weekly review)

- No modeling claims. No "we beat Gemini 3 Pro on ARC-2." We don't.
- No SHGI / multi-TRM results. SHGI is cited as horizon, not result.
- No ARC-AGI-2 code submission. One linked track, not two.
- No benchmark expansion push (stay on curated numbers; 10/50 is not
  a headline).
- No new TEMP specs written from inside this paper campaign. All
  architecture specs predate the paper. The paper describes existing
  work; it does not design new work.
- No venue-hedging. We write for the ARC Prize 2026 rubric, not
  NeurIPS or MLSys. Post-competition reuse (MLSys 2027 extended
  journal) is a later decision.

---

## 11. Artifacts that drop back into the repo

- `docs/papers/arc-prize-2026/` — LaTeX source + rendered PDF + demo MP4
- `docs/papers/arc-prize-2026/reproducers/` — math round-trip + preflight + ARC-3 launcher
- `docs/vocabulary/SOVEREIGN_HOT_PATH_DEFINITION.md` — §3.1 promoted to a
  first-class vocabulary entry (referenced by CLAUDE.md / AGENTS.md)
- `knowledge3d/benchmarks/arc3_kaggle_agent.py` (Codex)
- Additional k3d-ptx qdrant entries for any new sovereign patterns
  crafted during ablation work

---

**Authored by:** Claude (architecture partner), 2026-04-18 (v2 retarget).
**Pivot trigger:** Daniel, 2026-04-18 — *"let's direct the paper plan to
[ARC Prize 2026 Paper Track] … find what we have already planned to
see our movement out of python was crucial."* That movement is §3.5.
**Execution:** starts when Codex D2 + Phase 7.6 PR sync to main and
Daniel greenlights Campaign A.
